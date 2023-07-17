from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

def adapt_batchnorm(model, data_folder, path, img_size, batch_size):
    model.eval()

    # Set running_mean and running_var to requires_grad=True
    for module_idx, sequential in enumerate(model.module_list):
        for layer_idx, layer in enumerate(sequential):
            if isinstance(layer,torch.nn.BatchNorm2d):
                layer.reset_running_stats()
                layer.training = True
                # Re-freeze the parameters
                for param in layer.parameters():
                    param.requires_grad = False
                print(f"Batchnorm {module_idx}.{layer_idx} reset.")

    # Get dataloader
    dataset = ListDataset(path, data_folder=data_folder, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    for _, imgs, targets in tqdm.tqdm(dataloader, desc="Detecting objects"):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        # print("imgs", imgs.shape)

        with torch.no_grad():
            outputs = model(imgs)
    
    model.eval()
    
    return model



def evaluate(model, data_folder, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, data_folder=data_folder, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        # print("imgs", imgs.shape)

        with torch.no_grad():
            outputs = model(imgs)
            # print("outputs", outputs.shape)
            # print("conf_thres: ", conf_thres)
            # print("nms_thres: ", nms_thres)

            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            # print("outputs_nms", len(outputs))

        # print("nms output: ", outputs[0])
        # print("targets: ", targets)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)


    # Concatenate sample statistics
    if len(sample_metrics) > 0:
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    else:
        print("sample_metrics == []")
        return np.asarray(0), np.asarray(0), np.asarray(0), np.asarray(0), None


    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--adaptive_batchnorm", default=False, action='store_true', help="If true, do not use BatchNorm running_mean and _var in testing.")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["test"]
    class_names = load_classes(data_config["names"])
    data_folder = data_config["data_folder"]

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    if opt.adaptive_batchnorm:
        print("Running Adaptive BatchNorm...")
        model = adapt_batchnorm(model,
            data_folder=data_folder,
            path=valid_path,
            img_size=opt.img_size,
            batch_size=8,
        )

    print("Compute mAP...")
    precision, recall, AP, f1, ap_class = evaluate(
        model,
        data_folder=data_folder,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
