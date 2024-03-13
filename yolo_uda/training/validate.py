from typing import Optional
import torch
import tqdm
import wandb
import numpy as np

from torch import nn
from terminaltables import AsciiTable
from torch.utils.data import DataLoader
from torch.autograd import Variable
from pytorchyolo.utils.utils import ap_per_class, get_batch_statistics, non_max_suppression, xywh2xyxy

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
    

def validate(
    model: nn.Module,
    device: torch.device,
    validation_dataloader: DataLoader,
    run: wandb.run,
    class_names: list = ["0"],
    iou_thresh: float = 0.5,
    conf_thresh: float = 0.5,
    nms_thresh: float = 0.5,
    # The rest are here so that the train() and validate() interfaces are the same
    discriminator: Optional[nn.Module] = None,
    source_dataloader: Optional[DataLoader] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    optimizer_classifier: Optional[torch.optim.Optimizer] = None,
    mini_batch_size: Optional[int] = 1,
    target_dataloader: Optional[DataLoader] = None,    
    lambda_discriminator: Optional[float] = 0.5,
    verbose: Optional[bool] = False,
    epochs: Optional[int] = 10,
    evaluate_interval: Optional[int] = 1,
    metrics_suffix: Optional[str] = "",
):
    
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
        run.log({
            f"test_precision_{metrics_suffix}": precision.mean(),
            f"test_recall_{metrics_suffix}": recall.mean(),
            f"test_f1_{metrics_suffix}": f1.mean(),
            f"test_mAP_{metrics_suffix}": AP.mean()
        })
