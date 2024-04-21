import torch
import torch.nn as nn
from yolo_uda.training.models import Upsample
import tqdm
import wandb
import numpy as np
from terminaltables import AsciiTable
from torch.autograd import Variable
from torchmetrics.classification import BinaryAccuracy
from pytorchyolo.utils.utils import ap_per_class, get_batch_statistics, non_max_suppression, xywh2xyxy

binary_accuracy = BinaryAccuracy(threshold=0.5).to('cuda')

# def discriminator_step(
#         global_discriminator,
#         local_discriminator,
#         map_features,
#         labels,
#         global_discriminator_loss_function,
#         local_discriminator_loss_function,
#     ):

#     """
#     Discriminator step performed between the source and targer domain.
#     Input arguments:
#       map_features: Tensor = feature map obtained from the feature extractor
#       labels: Tensor = ground truth
#     Return:
#       Tensor = cross entropy loss between the prediction and the ground truth.
#     """
#     global_outputs, global_context = global_discriminator(map_features['global_features'])
#     local_outputs, local_context = local_discriminator(map_features['local_features'])

#     # calculate accuracy
#     global_discriminator_acc = binary_accuracy(global_outputs, labels['global_labels'])
#     local_discriminator_acc = binary_accuracy(local_outputs, labels['local_labels'])
#     discriminator_acc = {"global_discriminator_acc": global_discriminator_acc, "local_discriminator_acc":local_discriminator_acc}

#     # calculate loss
#     global_discriminator_loss = global_discriminator_loss_function(global_outputs, labels['global_labels'].float())
#     local_discriminator_loss = local_discriminator_loss_function(local_outputs, labels['local_labels'].float())

#     return global_discriminator_loss, local_discriminator_loss, discriminator_acc, global_context, local_context

# def compose_discriminator_batch_evaluation(source_features: torch.Tensor,
#                                            downsample_2: nn.Module, downsample_4: nn.Module,
#                                            labels_source: torch.Tensor,
#                                            device: torch.device, shuffle: bool = True):
#     # source_features[1] = downsample_2(source_features[1])
#     # target_features[1] = downsample_2(target_features[1])

#     # Create pixel-wise labels
#     activation_dims = (source_features[0].shape[2], source_features[0].shape[3], 1)
#     labels_source_pixelwise = labels_source.repeat(activation_dims).permute(2,0,1)

#     # Combine source and target batches for discriminator
#     features = {
#         "global_features": source_features[1].to(device),
#         "local_features": source_features[0].to(device)
#         }
#     labels = {
#         "global_labels": labels_source.to(device),
#         "local_labels": labels_source_pixelwise.to(device)
#         }

#     if shuffle:
#         # Shuffle batch
#         idx = torch.randperm(features['global_features'].shape[0])
#         features_shuffled = {key:value[idx] for key,value in features.items()}
#         labels_shuffled = {key:value[idx] for key,value in labels.items()}
#         return features_shuffled, labels_shuffled

#     return features, labels


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


def _evaluate(
        model,
        # global_discriminator,
        # local_discriminator,
        # discriminator_loss_function,
        dataloader, 
        class_names, 
        img_size, 
        # iou_thres, 
        # conf_thres, 
        # nms_thres, 
        step, 
        verbose,
        # device,
        # mini_batch_size,
        num_imgs_to_log=0,
    ):
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
    :param step: Training step, for logging bboxes
    :type step: int
    :param num_imgs_to_log: Number of images to log with predictions, if any. 
    :type num_imgs_to_log: int
    :param verbose: If True, prints stats of model
    :type verbose: bool
    :return: Returns precision, recall, AP, f1, ap_class
    """
    
    model.eval()  # Set model to evaluation mode
    
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    
    # Log bboxes to W&B
    imgs_to_log = []
    
    for _, imgs, targets, labels_source in tqdm.tqdm(dataloader, desc="Validating"):
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            batch = {"imgs":imgs, "labels_source":labels_source}
            outputs = model(batch)
            # source_features = model.forward_features(imgs, return_feature_maps=True)

            # features, disc_labels = compose_discriminator_batch_evaluation(
            #     source_features=source_features,
            #     downsample_2=downsample_2,
            #     downsample_4=downsample_4,
            #     labels_source=labels_source,
            #     device=device
            # )

            # # discriminator_step handles both global and local
            # (global_discriminator_loss, local_discriminator_loss, batch_discriminator_acc,
            #  global_context, local_context) = discriminator_step(
            #     global_discriminator=global_discriminator,
            #     local_discriminator=local_discriminator,
            #     map_features=features,
            #     labels=disc_labels,
            #     # mini_batch_size=2 * mini_batch_size,
            #     global_discriminator_loss_function=discriminator_loss_function,
            #     local_discriminator_loss_function=nn.MSELoss()
            # )

            # # duplicate along the first dimension for the global and local context
            # global_context = global_context.repeat(2, 1)
            # local_context = local_context.repeat(2, 1, 1, 1)

            # # print('evaluate shape', imgs.shape, global_context.shape, local_context.shape)

            # # get the source outputs with the context
            # outputs = model.forward_with_context(imgs, global_context, local_context)
            # outputs = non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=model.iou_thresh)
        if num_imgs_to_log > 0 and len(imgs_to_log) < num_imgs_to_log:
            for img, output in zip(imgs,outputs):
                all_boxes = []
                for box_ix in range(output.shape[0]):
                    box = output[box_ix]
                    box_data = {"position" : {
                        "minX" : box[0].item(),
                        "maxX" : box[2].item(),
                        "minY" : box[1].item(),
                        "maxY" : box[3].item()},
                        "class_id" : 0,
                        "box_caption" : "Conf: (%.3f)" % (box[4].item()),
                        "domain" : "pixel",
                        "scores" : { "confidence" : box[4].item() }
                        }
                    all_boxes.append(box_data)
                box_image = wandb.Image(img, boxes={"predictions": {"box_data": all_boxes, "class_labels": {0: "grapes"}}})
                if len(imgs_to_log) < num_imgs_to_log:
                    imgs_to_log.append(box_image)
    if num_imgs_to_log > 0:
        wandb.log({"Predictions": imgs_to_log},step=step)

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