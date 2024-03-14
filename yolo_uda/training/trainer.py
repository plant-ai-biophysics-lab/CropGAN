import os
from datetime import datetime
from typing import Callable, Union, Dict, List, Any

import torch
import tqdm
import wandb
import numpy as np
from torch import nn
import torch.nn.functional as F
from terminaltables import AsciiTable
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchmetrics.classification import BinaryAccuracy
from pytorchyolo.utils.loss import compute_loss
from pytorchyolo.utils.utils import to_cpu, ap_per_class, get_batch_statistics, non_max_suppression, xywh2xyxy
from models import Upsample
from metrics import FeatureMapCosineSimilarity, FeatureMapEuclideanDistance, MMDLoss, RBF


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
    for _, imgs, targets, _ in tqdm.tqdm(dataloader, desc="Validating"):
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
        global_discriminator,
        local_discriminator,
        map_features,
        labels,
        mini_batch_size,
        global_discriminator_loss_function,
        local_discriminator_loss_function,
        context = False
    ):

    """
    Discriminator step performed between the source and targer domain.
    Input arguments:
      map_features: Tensor = feature map obtained from the feature extractor
      labels: Tensor = ground truth
    Return:
      Tensor = cross entropy loss between the prediction and the ground truth.
    """
    if context:
        global_outputs, global_context = global_discriminator(map_features['global_features'])
        local_outputs, local_context = local_discriminator(map_features['local_features'])
    else:
        global_outputs = global_discriminator(map_features['global_features'])
        local_outputs = local_discriminator(map_features['local_features'])

    # calculate accuracy
    global_discriminator_acc = binary_accuracy(global_outputs, labels['global_labels'])
    local_discriminator_acc = binary_accuracy(local_outputs, labels['local_labels'])
    discriminator_acc = {"global_discriminator_acc":global_discriminator_acc, "local_discriminator_acc":local_discriminator_acc}

    # calculate loss
    global_discriminator_loss = global_discriminator_loss_function(global_outputs, labels['global_labels'].float())
    local_discriminator_loss = local_discriminator_loss_function(local_outputs, labels['local_labels'].float())

    if context:
        return global_discriminator_loss, local_discriminator_loss, discriminator_acc, global_context, local_context
    return global_discriminator_loss, local_discriminator_loss, discriminator_acc


def compose_discriminator_batch(source_features: torch.Tensor, target_features: torch.Tensor,
                                mini_batch_size: int, downsample_2: nn.Module, downsample_4: nn.Module,
                                labels_source: torch.Tensor, labels_target: torch.Tensor,
                                device: torch.device, shuffle: bool = True):
    # source_features[1] = downsample_2(source_features[1])
    # target_features[1] = downsample_2(target_features[1])

    # only used for yolov3.cfg, not yolov3-tiny.cfg
    if len(source_features) == 3:
        source_features[2] = downsample_4(source_features[2])
    # only used for yolov3.cfg, not yolov3-tiny.cfg
    if len(target_features) == 3:
        target_features[2] = downsample_4(target_features[2])

    # Create pixel-wise labels
    activation_dims = (source_features[1].shape[2], source_features[1].shape[3], 1)
    labels_source_pixelwise = labels_source.repeat(activation_dims).permute(2,0,1)
    labels_target_pixelwise = labels_target.repeat(activation_dims).permute(2,0,1)

    # Combine source and target batches for discriminator
    features = {
        "global_features":torch.cat([source_features[0], target_features[0]],axis=0).to(device),
        "local_features":torch.cat([source_features[1], target_features[1]],axis=0).to(device)
        }
    labels = {
        "global_labels": torch.cat([labels_source, labels_target],axis=0).to(device),
        "local_labels": torch.cat([labels_source_pixelwise, labels_target_pixelwise],axis=0).to(device)
        }

    if shuffle:
        # Shuffle batch
        idx = torch.randperm(features['global_features'].shape[0])
        features_shuffled = {key:value[idx] for key,value in features.items()}
        labels_shuffled = {key:value[idx] for key,value in labels.items()}
        return features_shuffled, labels_shuffled
    return features, labels


def train(
    model: nn.Module,
    global_discriminator: nn.Module,
    local_discriminator: nn.Module,
    source_dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    optimizer_global_classifier: torch.optim.Optimizer,
    optimizer_local_classifier: torch.optim.Optimizer,
    mini_batch_size: int,
    target_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    discriminator_loss_function: Union[Callable, nn.Module],
    save_dir: str,
    run: wandb.run,
    lambda_discriminator: float = 0.5,
    lambda_mmd: float = 0.001,
    verbose: bool = False,
    epochs: int = 10,
    class_names: list = None,
    iou_thresh: float = 0.5,
    conf_thresh: float = 0.5,
    nms_thresh: float = 0.5,
    context = False,
    metric_suffix: str = "", # Not used, just mirrors validate interface
):
    # upsample_4 = Upsample(scale_factor=4, mode="nearest")
    # upsample_2 = Upsample(scale_factor=2, mode="nearest")
    downsample_2 = Upsample(scale_factor=0.5, mode="nearest")
    downsample_4 = Upsample(scale_factor=0.25, mode="nearest")
    batches_done = 0

    best_map, map_ckpt_name = 0.0, ""
    best_precision, precision_ckpt_name = 0.0, ""
    best_recall, recall_ckpt_name = 0.0, ""
    best_f1, f1_ckpt_name = 0.0, ""
    for epoch in range(1, epochs+1):
        print("\n---- Training Model ----")
        wandb.log({'epoch': epoch}, step=batches_done)

        # set to training mode
        model.train() # set yolo model to training mode
        global_discriminator.train() # set discriminator to training mode
        local_discriminator.train()
        # Collect discriminator accuracy over training batches
        # Note: total is the sum of batch-level accuracy, not sample-level accuracy.
        # To get the average for the dataset, divide by the batch count.
        global_discriminator_acc = {"total": 0, "batch_count": 0, "batch_size": mini_batch_size*2}
        local_discriminator_acc = {"total": 0, "batch_count": 0, "batch_size": mini_batch_size*2}

        ## Feature map similarity metrics
        # Cosine similarity metrics
        cosine_similarity_metrics_l15 = FeatureMapCosineSimilarity(layer="15")
        cosine_similarity_metrics_l22 = FeatureMapCosineSimilarity(layer="22")

        # Euclidean distance metrics
        euclidean_distance_metrics_l15 = FeatureMapEuclideanDistance(layer="15")
        euclidean_distance_metrics_l22 = FeatureMapEuclideanDistance(layer="22")

        # MMD calculation
        mmd_metric = MMDLoss()

        # tracker
        updated_lr_this_epoch = False

        print(model)

        for batch_i, contents in enumerate(
            tqdm.tqdm(zip(source_dataloader, target_dataloader), desc=f"Training Epoch {epoch}")
        ):
            (data_source, data_target) = contents

            # Reset gradients
            optimizer.zero_grad()
            optimizer_global_classifier.zero_grad()
            optimizer_local_classifier.zero_grad()

            batches_done = len(target_dataloader) * (epoch-1) + batch_i

            # get imgs from data
            _, imgs_s, targets, labels_source = data_source
            _, imgs_t, _, labels_target = data_target
            if len(imgs_s) < mini_batch_size or len(imgs_t) < mini_batch_size:
                break
            source_imgs = imgs_s.to(device)
            target_imgs = imgs_t.to(device)
            targets = targets.to(device)

            # with context, we need to get the global/local features from the yolo model,
            # pass them through the discriminator to get the discriminator outputs and context
            # vectors, and then pass the context vectors back into the yolo model to get the
            # final yolo output -> this requires multiple steps
            if context:
                # run source pass
                source_features = model.forward_features(source_imgs)
                # Run target pass to encode features for classifier
                target_features = model.forward_features(target_imgs)

                features, labels = compose_discriminator_batch(
                    source_features=source_features,
                    target_features=target_features,
                    mini_batch_size=mini_batch_size,
                    downsample_2=downsample_2,
                    downsample_4=downsample_4,
                    labels_source=labels_source,
                    labels_target=labels_target,
                    device=device
                )

                # discriminator_step handles both global and local
                (global_discriminator_loss, local_discriminator_loss, batch_discriminator_acc,
                 global_context, local_context) = discriminator_step(
                    global_discriminator=global_discriminator,
                    local_discriminator=local_discriminator,
                    map_features=features,
                    labels=labels,
                    mini_batch_size=2 * mini_batch_size,
                    global_discriminator_loss_function=discriminator_loss_function,
                    local_discriminator_loss_function=nn.MSELoss(),
                    context=True
                )
                
                # get the source outputs with the context
                source_outputs = model.forward_with_context(source_imgs, global_context, local_context)
                
                # yolo loss
                yolo_loss, loss_components = compute_loss(source_outputs, targets, model)

            else:
                # run source pass
                source_outputs, source_features = model(source_imgs)
                # Run target pass to encode features for classifier
                target_outputs, target_features = model(target_imgs)
                # yolo loss
                yolo_loss, loss_components = compute_loss(source_outputs, targets, model)

                features, labels = compose_discriminator_batch(
                    source_features=source_features,
                    target_features=target_features,
                    mini_batch_size=mini_batch_size,
                    downsample_2=downsample_2,
                    downsample_4=downsample_4,
                    labels_source=labels_source,
                    labels_target=labels_target,
                    device=device
                )

                # discriminator_step handles both global and local
                global_discriminator_loss, local_discriminator_loss, batch_discriminator_acc = discriminator_step(
                    global_discriminator=global_discriminator,
                    local_discriminator=local_discriminator,
                    map_features=features,
                    labels=labels,
                    mini_batch_size=2*mini_batch_size,
                    global_discriminator_loss_function=discriminator_loss_function,
                    local_discriminator_loss_function=nn.MSELoss(),
                    context=False
                )

            # Calculate average MMD loss per batch
            mmd_loss = mmd_metric(source_features[1], target_features[1])

            # run backward propagation
            discriminator_loss = 0.05 * global_discriminator_loss + 0.95 * local_discriminator_loss
            loss = yolo_loss + lambda_discriminator * discriminator_loss + lambda_mmd * mmd_loss
            loss.backward()

            # run optimizer
            # if batches_done % model.hyperparams['subdivisions'] == 0:
            # adapt learning rate
            lr = model.hyperparams['learning_rate']
            if batches_done < model.hyperparams['burn_in']:
                lr *= (batches_done / model.hyperparams['burn_in'])
            else:
                # manually select which steps to decay the LR, and by what value
                if 'lr_steps' in model.hyperparams:
                    for threshold, value in model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value

                # decay every N steps or every N epochs
                elif 'lr_gamma' in model.hyperparams:
                    if 'lr_step' in model.hyperparams:
                        if batches_done % model.hyperparams['lr_step'] == 0:
                            print("Decaying learning rate ({}) by gamma {}".format(lr, model.hyperparams['lr_gamma'])
                                  + " to {}".format(lr * model.hyperparams['lr_gamma']))
                            lr = lr * model.hyperparams['lr_gamma']
                            model.hyperparams['learning_rate'] = lr
                    elif 'lr_epoch' in model.hyperparams:
                        if epoch % model.hyperparams['lr_epoch'] == 0:
                            if not updated_lr_this_epoch:
                                updated_lr_this_epoch = True
                                print("Decaying learning rate ({}) by gamma {}".format(lr, model.hyperparams['lr_gamma'])
                                      + " to {}".format(lr * model.hyperparams['lr_gamma']))
                                lr = lr * model.hyperparams['lr_gamma']
                                model.hyperparams['learning_rate'] = lr

            # log the learning rate
            wandb.log({"lr": lr}, step=batches_done)

            # set learning rate
            for g in optimizer.param_groups:
                g['lr'] = lr

            # Run optimizer
            optimizer.step()
            optimizer_global_classifier.step()
            optimizer_local_classifier.step()

            # Metrics
            # Track discriminator accuracy
            global_discriminator_acc["total"] += batch_discriminator_acc["global_discriminator_acc"]
            global_discriminator_acc["batch_count"] += 1
            local_discriminator_acc["total"] += batch_discriminator_acc["local_discriminator_acc"]
            local_discriminator_acc["batch_count"] += 1

            # Update cosine similarity metrics
            # *_features[0] and *_features[1] are the feature maps of different yolo layers.
            cosine_similarity_metrics_l15.update(source_features=source_features[0],target_features=target_features[0])
            cosine_similarity_metrics_l22.update(source_features=source_features[1],target_features=target_features[1])

            # Update euclidean distance metrics
            euclidean_distance_metrics_l15.update(source_features=source_features[0],target_features=target_features[0])
            euclidean_distance_metrics_l22.update(source_features=source_features[1],target_features=target_features[1])


            # log progress
            if verbose:
                print(AsciiTable(
                        [
                            ["Type", "Value"],
                            ["IoU loss", float(loss_components[0])],
                            ["Object loss", float(loss_components[1])],
                            ["Class loss", float(loss_components[2])],
                            ["Loss", float(loss_components[3])],
                            ["Discriminator batch loss", float(discriminator_loss)],
                            ["YOLO batch loss", to_cpu(yolo_loss).item()]
                        ]).table)
            wandb.log({
                "iou_loss": float(loss_components[0]),
                "obj_loss": float(loss_components[1]),
                "cls_loss": float(loss_components[2]),
                "yolo_loss": float(loss_components[3]),
                "glob_dscm_loss": float(global_discriminator_loss),
                "loc_dscm_loss": float(local_discriminator_loss)
            }, step=batches_done)
            model.seen += imgs_s.size(0)

        # Training epoch metrics
        # Discriminator accuracy
        wandb.log({"glob_dscm_acc": global_discriminator_acc["total"] / global_discriminator_acc["batch_count"]}, step=batches_done)
        wandb.log({"loc_dscm_acc": local_discriminator_acc["total"] / local_discriminator_acc["batch_count"]}, step=batches_done)

        # Average cosine similarity within source, within target, and across source-target
        # For both feature layers
        for metric in [cosine_similarity_metrics_l15, cosine_similarity_metrics_l22, euclidean_distance_metrics_l15, euclidean_distance_metrics_l22, mmd_metric]:
            wandb.log(metric.return_metrics(), step=batches_done)
            metric.reset()

        # evaluate
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
            }, step=batches_done)

            # Save the best checkpoint for each metric
            save_date = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
            ckpt_name = "ckpt_best_{mt}.pth"
            if precision.mean() >= best_precision:
                best_precision = precision.mean()
                if precision_ckpt_name:
                    os.remove(os.path.join(save_dir, precision_ckpt_name))
                precision_ckpt_name = ckpt_name.format(
                    mt="precision", value=best_precision, date=save_date, epoch=epoch)
                torch.save(model.state_dict(),
                           os.path.join(save_dir, precision_ckpt_name))
            if recall.mean() >= best_recall:
                best_recall = recall.mean()
                if recall_ckpt_name:
                    os.remove(os.path.join(save_dir, recall_ckpt_name))
                recall_ckpt_name = ckpt_name.format(
                    mt="recall", value=best_recall, date=save_date, epoch=epoch)
                torch.save(model.state_dict(),
                           os.path.join(save_dir, recall_ckpt_name))
            if AP.mean() >= best_map:
                best_map = AP.mean()
                if map_ckpt_name:
                    os.remove(os.path.join(save_dir, map_ckpt_name))
                map_ckpt_name = ckpt_name.format(
                    mt="map", value=best_map, date=save_date, epoch=epoch)
                torch.save(model.state_dict(),
                           os.path.join(save_dir, map_ckpt_name))
            if f1.mean() >= best_f1:
                best_f1 = f1.mean()
                if f1_ckpt_name:
                    os.remove(os.path.join(save_dir, f1_ckpt_name))
                f1_ckpt_name = ckpt_name.format(
                    mt="f1", value=best_f1, date=save_date, epoch=epoch)
                torch.save(model.state_dict(),
                           os.path.join(save_dir, f1_ckpt_name))


    return model

