import torch
import tqdm
import wandb
import numpy as np
from terminaltables import AsciiTable
from torch.autograd import Variable
from torchmetrics.classification import BinaryAccuracy
from pytorchyolo.utils.utils import ap_per_class, get_batch_statistics, non_max_suppression, xywh2xyxy

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


def _evaluate(
        model,
        dataloader, 
        class_names, 
        img_size, 
        step, 
        verbose,
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