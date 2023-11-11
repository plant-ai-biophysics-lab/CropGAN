import torch
import tqdm
import wandb
import numpy as np

from torch import nn
from terminaltables import AsciiTable
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchmetrics.classification import BinaryAccuracy
from pytorchyolo.utils.loss import compute_loss
from pytorchyolo.utils.utils import to_cpu, ap_per_class, get_batch_statistics, non_max_suppression, xywh2xyxy
from models import Upsample

# for loss calculations
# cross_entropy = nn.CrossEntropyLoss()
bce = nn.BCELoss()
binary_accuracy = BinaryAccuracy(threshold=0.5).to('cuda')

def print_eval_stats(metrics_output, class_names, verbose):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean():.5f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")
        
def _evaluate(model, dataloader, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose):
    """Evaluate model on validation dataset.

    :param model: Model to evaluate
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images with targets
    :type dataloader: DataLoader
    :param class_names: List of class names
    :type class_names: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param iou_thres: IOU threshold required to qualify as detected
    :type iou_thres: float
    :param conf_thres: Object confidence threshold
    :type conf_thres: float
    :param nms_thres: IOU threshold for non-maximum suppression
    :type nms_thres: float
    :param verbose: If True, prints stats of model
    :type verbose: bool
    :return: Returns precision, recall, AP, f1, ap_class
    """
    model.eval()  # Set model to evaluation mode
    print(f"model training: {model.training}")

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for _, imgs, targets in tqdm.tqdm(dataloader, desc="Validating"):
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(
        true_positives, pred_scores, pred_labels, labels)

    print_eval_stats(metrics_output, class_names, verbose)

    return metrics_output

def discriminator_step(
      discriminator,
      map_features, 
      labels,
      mini_batch_size
    ) -> None:

    """ 
    Discriminator step performed between the source and targer domain. 
    Input arguments:
      map_features: Tensor = feture map obtained from the feature extractor
      labels: Tensor = ground truth
    Return:
      Tensor = cross entropy loss between the prediction and the ground truth.
    """
    outputs = discriminator(map_features)
    
    # calculate accuracy
    # pred_labels = outputs[:, 0, 0, 0]
    discriminator_acc = binary_accuracy(outputs, labels)
    
    # calculate loss
    # outputs = outputs.view(mini_batch_size, -1)
    # discriminator_loss = cross_entropy(outputs, labels.float())
    discriminator_loss = bce(outputs, labels.float())
    
    return discriminator_loss, discriminator_acc

def train(
    model: nn.Module,
    discriminator: nn.Module,
    source_dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    optimizer_classifier: torch.optim.Optimizer,
    mini_batch_size: int,
    target_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    verbose: bool = False,
    epochs: int = 10,
    evaluate_interval: int = 1,
    class_names: list = None,
    iou_thresh: float = 0.5,
    conf_thresh: float = 0.5,
    nms_thresh: float = 0.5,
):
    # upsample_4 = Upsample(scale_factor=4, mode="nearest")
    # upsample_2 = Upsample(scale_factor=2, mode="nearest")
    downsample_2 = Upsample(scale_factor=0.5, mode="nearest")
    downsample_4 = Upsample(scale_factor=0.25, mode="nearest")
    
    for epoch in range(1, epochs+1):
        
        print("\n---- Training Model ----")

        # set to training mode
        model.train() # set yolo model to training mode
        discriminator.train() # set discriminator to training mode

        for batch_i, (data_source, data_target) in enumerate(
            tqdm.tqdm(zip(source_dataloader, target_dataloader), desc=f"Training Epoch {epoch}")
        ):
            # Reset gradients
            optimizer.zero_grad()
            optimizer_classifier.zero_grad()

            batches_done = len(source_dataloader) * (epoch-1) + batch_i
            
            # get imgs from data
            _, imgs_s, targets = data_source
            _, imgs_t, _ = data_target
            if len(imgs_s) < mini_batch_size or len(imgs_t) < mini_batch_size:
                break
            source_imgs = imgs_s.to(device)
            target_imgs = imgs_t.to(device)
            targets = targets.to(device)
            
            # run source pass, upsample features and calculate yolo loss
            source_outputs, source_features = model(source_imgs)
            # source_features[0] = upsample_4(source_features[0])
            # source_features[1] = upsample_2(source_features[1])
            source_features[1] = downsample_2(source_features[1])
            source_features[2] = downsample_4(source_features[2])
            yolo_loss, loss_components = compute_loss(source_outputs, targets, model)
            
            # run target pass upsample features
            zeros_label = torch.zeros(mini_batch_size, dtype=torch.long, device=device)
            ones_label = torch.ones(mini_batch_size, dtype=torch.long, device=device)
            target_outputs, target_features = model(target_imgs)
            # target_features[0] = upsample_4(target_features[0])
            # target_features[1] = upsample_2(target_features[1])
            target_features[1] = downsample_2(target_features[1])
            target_features[2] = downsample_4(target_features[2])
            
            # concatenate source and target features
            # source_features = torch.cat(source_features, dim=1).to(device)
            # target_features = torch.cat(target_features, dim=1).to(device)
            source_features = source_features[0]+source_features[1]+source_features[2]
            target_features = target_features[0]+target_features[1]+target_features[2]

            # Combine source and target batches for discriminator
            features = torch.cat([source_features,target_features],axis=0)
            labels = torch.cat([zeros_label,ones_label],axis=0)
            
            # Shuffle batch
            idx = torch.randperm(features.shape[0])
            features_shuffled = features[idx]
            labels_shuffled = labels[idx]

            discriminator_loss, discriminator_acc = discriminator_step(discriminator, features_shuffled, labels_shuffled, 2*mini_batch_size)

            # discriminator step and calculate discriminator loss
            # discriminator_source_loss, discriminator_source_acc = discriminator_step(discriminator, source_features, zeros_label, mini_batch_size)
            # discriminator_target_loss, discriminator_target_acc = discriminator_step(discriminator, target_features, ones_label, mini_batch_size)
            # discriminator_loss = discriminator_source_loss + discriminator_target_loss

            # run backward propagation
            # loss = yolo_loss + discriminator_loss
            loss = discriminator_loss
            loss.backward()

            # run optimizer
            # if batches_done % model.hyperparams['subdivisions'] == 0:
            # adapt learning rate
            lr = model.hyperparams['learning_rate']
            if batches_done < model.hyperparams['burn_in']:
                lr *= (batches_done / model.hyperparams['burn_in'])
            else:
                for threshold, value in model.hyperparams['lr_steps']:
                    if batches_done > threshold:
                        lr *= value
            # log the learning rate
            wandb.log({"lr": lr})
            # set leraning rate
            for g in optimizer.param_groups:
                g['lr'] = lr
                
            # Run optimizer
            optimizer.step()
            optimizer_classifier.step()
            
        
            # log progress
            if verbose:
                print(AsciiTable(
                        [
                            ["Type", "Value"],
                            ["IoU loss", float(loss_components[0])],
                            ["Object loss", float(loss_components[1])],
                            ["Class loss", float(loss_components[2])],
                            ["Loss", float(loss_components[3])],
                            # ["Source loss", float(discriminator_source_loss)],
                            # ["Target loss", float(discriminator_target_loss)],
                            ["YOLO Batch loss", to_cpu(yolo_loss).item()]
                        ]).table)
            wandb.log({
                "iou_loss": float(loss_components[0]),
                "obj_loss": float(loss_components[1]),
                "cls_loss": float(loss_components[2]),
                "yolo_loss": float(loss_components[3]),
                # "dscm_src_loss": float(discriminator_source_loss),
                # "dscm_trgt_loss": float(discriminator_target_loss),
                # "dscm_src_acc": float(discriminator_source_acc),
                # "dscm_trgt_acc": float(discriminator_target_acc),
                "dscm_acc": float(discriminator_acc),
                "dscm_loss": float(discriminator_loss)
                })
            model.seen += imgs_s.size(0)
            
        # save model to checkpoint file
        # if epoch % args.checkpoint_interval == 0:
        #     checkpoint_path = f"checkpoints/yolov3_ckpt_{epoch}.pth"
        #     print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
        #     torch.save(model.state_dict(), checkpoint_path)
        
        # evaluate
        if epoch % evaluate_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = _evaluate(
                model,
                validation_dataloader,
                class_names,
                img_size=model.hyperparams['height'],
                iou_thres=iou_thresh,
                conf_thres=conf_thresh,
                nms_thres=nms_thresh,
                verbose=verbose
            )

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                wandb.log({
                    "precision": precision.mean(),
                    "recall": recall.mean(),
                    "f1": f1.mean(),
                    "mAP": AP.mean()
                })
    
    return model