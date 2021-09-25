"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import util.util_yolo as util_yolo
import time
import tqdm

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        # convert it into a numpy array
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / \
            2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def save_image_tensor(img_tensor, filename):
    image = np.asarray(img_tensor.detach().cpu().squeeze(
        0).permute([1, 2, 0])*0.5+0.5)*255
    image = image.astype(np.uint8)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    cv.imwrite(filename, image)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

# --------------------------
# ZFei
# --------------------------


def parse_loss_log(loss_file):
    header = []
    loss_data = []
    with open(loss_file, "r") as infile:
        for line in infile:
            if "D_A" in line:
                data = line.split(")")
                meta = data[0]
                loss = data[1]
                loss_split = loss.split(" ")
                header = []
                losses = []
                for text in loss_split:
                    if (text != '') and (text != '\n'):
                        if '_' in text:
                            header.append(text.replace(":", ''))
                        else:
                            losses.append(float(text))
                loss_data.append(losses)
    loss_data = np.asarray(loss_data)
    loss_data
    return header, loss_data


def plot_loss(loss_file, plot_headers=None, moving_average_window=None):
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    header, losses = parse_loss_log(loss_file)
    print("header: ",header)
    print("header: ",losses.shape)
    for i in range(len(header)):
        if plot_headers is None or header[i] in plot_headers:
            if moving_average_window is not None:
                plot_loss = moving_average(losses[:, i], moving_average_window)
            else:
                plot_loss = losses[:, i]
            plt.plot(plot_loss, label=header[i])
    plt.legend()


def plot_image_with_detections(img_tensor,
                               detections_bbox=None,
                               groun_truth_bbox=None,
                               figsize=[5, 5],
                               normalize=True,
                               save_name=None,
                               ax=None):
    """
    args:
    detections_bbox are from yolo model output after non-max-suppression, format are [[x1, y1, x2, y2, conf, cls_conf, cls_pred]] in pixels
    groun_truth_bbox are in yolo data format:  [[x, y, w, h] in relative coordinates
    """
    if ax is None:
        fig, (ax) = plt.subplots(1, 1, figsize=figsize)
    if normalize:
        ax.imshow(img_tensor.detach().cpu().squeeze(
            0).permute([1, 2, 0])*0.5+0.5)
    else:
        ax.imshow(img_tensor.detach().cpu().squeeze(0).permute([1, 2, 0]))

    image_shape = img_tensor.detach().cpu().squeeze(0).shape
    # Plot detection bbox
    if detections_bbox is not None:
        detections_bbox = detections_bbox[0]
        if detections_bbox is not None:
            for detect in detections_bbox:
                if detect is not None:
                    detect = detect.squeeze(0)
                    x1, y1, x2, y2, conf, cls_conf, cls_pred = detect
                    box_w = x2 - x1
                    box_h = y2 - y1
                    color = (1., 0., 0.)
                    bbox = patches.Rectangle(
                        (x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    ax.add_patch(bbox)

    # Plot ground truth bbox
    if groun_truth_bbox is not None:
        if 'Tensor' in str(type(groun_truth_bbox)):
            groun_truth_bbox = groun_truth_bbox.detach().cpu().numpy()
        if groun_truth_bbox is not None:
            for detect in groun_truth_bbox:
                if detect is not None:
                    # print(detect)
                    _, _, x, y, w, h = detect
                    x1 = image_shape[2] * (x - w/2)
                    x2 = image_shape[2] * (x + w/2)
                    y1 = image_shape[1] * (y - h/2)
                    y2 = image_shape[1] * (y + h/2)
                    box_w = x2 - x1
                    box_h = y2 - y1
                    color = (0., 1., 0.)
                    bbox = patches.Rectangle(
                        (x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    ax.add_patch(bbox)

    if save_name is not None:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig(save_name)
        plt.close(fig)


def plot_analysis(model, data, figsize=[12, 12],
                  iou_thres=0.5,
                  conf_thres=0.5,
                  nms_thres=0.1,
                  img_size=416,
                  plot_detections=True,
                  reverse=False,
                  save_name=None, title="Cycle GAN"):

    model.set_input(data)  # unpack data from data loader
    model.forward()           # run inference

    if reverse:
        fig, ax_grids = plt.subplots(2, 3, figsize=figsize)
    else:
        fig, ax_grids = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14)

    for axs in ax_grids:
        for ax in axs:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    ax_grids[0][0].set_title("Real A")
    ax_grids[0][1].set_title("Fake B")
    ax_grids[1][0].set_title("Real B")
    ax_grids[1][1].set_title("Fake A")

    if plot_detections:
        # Real A
        # de-normalize the image before feed into the yolo net
        loss_yolo_b, bbox_outputs = model.netYolo(
            model.real_A*0.5+0.5, model.A_label)
        detections_nms = util_yolo.non_max_suppression(
            bbox_outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        plot_image_with_detections(
            model.real_A, detections_nms,  model.A_label,  ax=ax_grids[0][0])

        # Fake B
        # de-normalize the image before feed into the yolo net
        loss_yolo_b, bbox_outputs = model.netYolo(
            model.fake_B*0.5+0.5, model.A_label)
        detections_nms = util_yolo.non_max_suppression(
            bbox_outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        plot_image_with_detections(
            model.fake_B, detections_nms,  model.A_label, ax=ax_grids[0][1])

        # Real B
        # de-normalize the image before feed into the yolo net
        loss_yolo_b, bbox_outputs = model.netYolo(
            model.real_B*0.5+0.5, model.A_label)
        detections_nms = util_yolo.non_max_suppression(
            bbox_outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        plot_image_with_detections(model.real_B, detections_nms,  ax=ax_grids[1][0])

        # Fake A
        # de-normalize the image before feed into the yolo net
        loss_yolo_b, bbox_outputs = model.netYolo(
            model.fake_A*0.5+0.5, model.A_label)
        detections_nms = util_yolo.non_max_suppression(
            bbox_outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        plot_image_with_detections(model.fake_A, detections_nms,  ax=ax_grids[1][1])
    else:
        plot_image_with_detections(model.real_A, ax=ax_grids[0][0])
        plot_image_with_detections(model.fake_B, ax=ax_grids[0][1])
        plot_image_with_detections(model.real_B, ax=ax_grids[1][0])
        plot_image_with_detections(model.fake_A, ax=ax_grids[1][1])

    if reverse:
        ax_grids[0][2].set_title("Labeled B")
        ax_grids[1][2].set_title("Fake Labeled A")
        if plot_detections:
            # labeled_B
            # de-normalize the image before feed into the yolo net
            loss_yolo_b, bbox_outputs = model.netYolo(
                model.labeled_B*0.5+0.5, model.labeled_B_label)
            detections_nms = util_yolo.non_max_suppression(
                bbox_outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            plot_image_with_detections(model.labeled_B, detections_nms,  model.labeled_B_label, ax=ax_grids[0][2])
            # fake_labeled_A
            # de-normalize the image before feed into the yolo net
            loss_yolo_b, bbox_outputs = model.netYolo(
                model.fake_labeled_A*0.5+0.5, model.labeled_B_label)
            detections_nms = util_yolo.non_max_suppression(
                bbox_outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            plot_image_with_detections(model.fake_labeled_A, detections_nms,  model.labeled_B_label, ax=ax_grids[1][2])
        else:
            plot_image_with_detections(model.labeled_B, ax=ax_grids[0][2])
            plot_image_with_detections(model.fake_labeled_A, ax=ax_grids[1][2])


    if save_name is not None:
        plt.savefig(save_name)
        plt.close(fig)

def plot_analysis_double_task(model, data, figsize=[12, 12],
                            iou_thres=0.5,
                            conf_thres=0.5,
                            nms_thres=0.1,
                            img_size=416,
                            plot_detections=True,
                            reverse=False,
                            save_name=None, title="Cycle GAN"):

    model.set_input(data)  # unpack data from data loader
    model.forward()           # run inference

    if reverse:
        fig, ax_grids = plt.subplots(2, 3, figsize=figsize)
    else:
        fig, ax_grids = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14)

    for axs in ax_grids:
        for ax in axs:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    ax_grids[0][0].set_title("Real A")
    ax_grids[0][1].set_title("Fake B")
    ax_grids[1][0].set_title("Real B")
    ax_grids[1][1].set_title("Fake A")

    if plot_detections:
        # Real A
        # de-normalize the image before feed into the yolo net
        loss_yolo_b, bbox_outputs = model.netYoloA(
            model.real_A*0.5+0.5, model.A_label)
        detections_nms = util_yolo.non_max_suppression(
            bbox_outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        plot_image_with_detections(
            model.real_A, detections_nms,  model.A_label,  ax=ax_grids[0][0])

        # Fake B
        # de-normalize the image before feed into the yolo net
        loss_yolo_b, bbox_outputs = model.netYoloB(
            model.fake_B*0.5+0.5, model.A_label)
        detections_nms = util_yolo.non_max_suppression(
            bbox_outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        plot_image_with_detections(
            model.fake_B, detections_nms,  model.A_label, ax=ax_grids[0][1])

        # Real B
        # de-normalize the image before feed into the yolo net
        loss_yolo_b, bbox_outputs = model.netYoloB(
            model.real_B*0.5+0.5, model.A_label)
        detections_nms = util_yolo.non_max_suppression(
            bbox_outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        plot_image_with_detections(model.real_B, detections_nms,  ax=ax_grids[1][0])

        # Fake A
        # de-normalize the image before feed into the yolo net
        loss_yolo_b, bbox_outputs = model.netYoloA(
            model.fake_A*0.5+0.5, model.A_label)
        detections_nms = util_yolo.non_max_suppression(
            bbox_outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        plot_image_with_detections(model.fake_A, detections_nms,  ax=ax_grids[1][1])
    else:
        plot_image_with_detections(model.real_A, ax=ax_grids[0][0])
        plot_image_with_detections(model.fake_B, ax=ax_grids[0][1])
        plot_image_with_detections(model.real_B, ax=ax_grids[1][0])
        plot_image_with_detections(model.fake_A, ax=ax_grids[1][1])

    if reverse:
        ax_grids[0][2].set_title("Labeled B")
        ax_grids[1][2].set_title("Fake Labeled A")
        if plot_detections:
            # labeled_B
            # de-normalize the image before feed into the yolo net
            loss_yolo_b, bbox_outputs = model.netYoloB(
                model.labeled_B*0.5+0.5, model.labeled_B_label)
            detections_nms = util_yolo.non_max_suppression(
                bbox_outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            plot_image_with_detections(model.labeled_B, detections_nms,  model.labeled_B_label, ax=ax_grids[0][2])
            # fake_labeled_A
            # de-normalize the image before feed into the yolo net
            loss_yolo_b, bbox_outputs = model.netYoloA(
                model.fake_labeled_A*0.5+0.5, model.labeled_B_label)
            detections_nms = util_yolo.non_max_suppression(
                bbox_outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            plot_image_with_detections(model.fake_labeled_A, detections_nms,  model.labeled_B_label, ax=ax_grids[1][2])
        else:
            plot_image_with_detections(model.labeled_B, ax=ax_grids[0][2])
            plot_image_with_detections(model.fake_labeled_A, ax=ax_grids[1][2])


    if save_name is not None:
        plt.savefig(save_name)
        plt.close(fig)


# ------------------------------
# Evaluation
# ------------------------------

def further_train_yolo_a(model, dataset, yolo_epochs, yolo_gradient_accumulations=1, curoff_batch=np.inf,
                         rec_A=False,
                         fake_labeled_A=False,
                         real_A=False,
                         fake_B=False,
                         iou_thres=0.5,
                         conf_thres=0.1,
                         nms_thres = 0.1,
                         evaluate_through_fake_A=None,
                         evaluate_through_real_B=None,
                         dataset_yolo_eval=None):
    yolo_epochs = 1
    batches_done = 0
    yolo_gradient_accumulations = 1
    model.optimizer_yolo.zero_grad()
    evaluate_through_fake_A_results = []
    evaluate_through_real_B_results = []

    for epoch_yolo in range(yolo_epochs):
        model.netYolo.train()
        start_time = time.time()
        for i, data in enumerate(dataset):
            print("Batch: ", i)
            if evaluate_through_fake_A is not None:
                if batches_done % evaluate_through_fake_A == 0:
                    precision, recall, AP, f1 = \
                    util_yolo.evaluate_yolo_through_fakeA(model, dataset_yolo_eval, iou_thres, conf_thres, nms_thres)
                    print("evaluate_through_fake_A: precision %.2f, recall %.2f, AP %.2f, f1 %.2f"%(precision, recall, AP, f1))
                    evaluate_through_fake_A_results.append([batches_done, precision[0], recall[0], AP[0], f1[0]])

            if evaluate_through_real_B is not None:
                if batches_done % evaluate_through_real_B == 0:
                    precision, recall, AP, f1 = \
                    util_yolo.evaluate_yolo_through_realB(model, dataset_yolo_eval, iou_thres, conf_thres, nms_thres)
                    print("evaluate_through_real_B: precision %.2f, recall %.2f, AP %.2f, f1 %.2f"%(precision, recall, AP, f1))
                    evaluate_through_real_B_results.append([batches_done, precision[0], recall[0], AP[0], f1[0]])

            model.netYolo.train()
            model.optimizer_yolo.zero_grad()
            model.set_input(data)  # unpack data from data loader
            model.forward()           # run inference
            imgs = model.rec_A

            # Train on rec A
            if rec_A:        
                imgs = model.rec_A.detach()*0.5+0.5
                targets =  model.A_label
                loss, outputs = model.netYolo(imgs, targets)
                if loss > 0:
                    loss.backward()
                    model.netYolo.seen += imgs.size(0)
                    if batches_done % yolo_gradient_accumulations == 0:
                        # Accumulates gradient before each step
                        model.optimizer_yolo.step()
                        model.optimizer_yolo.zero_grad()
                print("rec_A loss: ", loss.item())
            
            # Train on fake labeled A  
            if fake_labeled_A:      
                imgs = model.fake_labeled_A.detach()*0.5+0.5
                targets =  model.labeled_B_label
                loss, outputs = model.netYolo(imgs, targets)
                if loss > 0:
                    loss.backward()
                    model.netYolo.seen += imgs.size(0)
                    if batches_done % yolo_gradient_accumulations == 0:
                        # Accumulates gradient before each step
                        model.optimizer_yolo.step()
                        model.optimizer_yolo.zero_grad()
                print("fake_labeled_A loss: ", loss.item())

            # Train on real A
            if real_A:        
                imgs = model.real_A*0.5+0.5
                targets =  model.A_label
                loss, outputs = model.netYolo(imgs, targets)
                if loss > 0:
                    loss.backward()
                    model.netYolo.seen += imgs.size(0)
                    if batches_done % yolo_gradient_accumulations == 0:
                        # Accumulates gradient before each step
                        model.optimizer_yolo.step()
                        model.optimizer_yolo.zero_grad()
                print("real_A loss: ", loss.item())
            
            if fake_B:
                imgs = model.fake_B.detach()*0.5+0.5
                targets =  model.A_label
                loss, outputs = model.netYolo(imgs, targets)
                if loss > 0:
                    loss.backward()
                    model.netYolo.seen += imgs.size(0)
                    if batches_done % yolo_gradient_accumulations == 0:
                        # Accumulates gradient before each step
                        model.optimizer_yolo.step()
                        model.optimizer_yolo.zero_grad()
                print("fake_B loss: ", loss.item())
            
            batches_done += imgs.size(0)


            if batches_done > curoff_batch:
                break
    
    return evaluate_through_fake_A_results, evaluate_through_real_B_results
            
def further_train_yolo_a_eval_double(model, dataset, yolo_epochs, yolo_gradient_accumulations=1, curoff_batch=np.inf,
                         rec_A=False,
                         fake_labeled_A=False,
                         real_A=False,
                         fake_B=False,
                         labeled_B=False,
                         iou_thres=0.5,
                         conf_thres=0.1,
                         nms_thres = 0.1,
                         labeled_B_train_ratio=1,
                         evaluate_through_real_B=None,
                         dataset_yolo_eval=None,
                         validation_path=None):
    batches_done = 0
    yolo_gradient_accumulations = 1
    model.optimizer_yolo_b.zero_grad()
    evaluate_through_fake_A_results = []
    evaluate_through_real_B_results = []

    for epoch_yolo in range(yolo_epochs):
        model.netYoloB.train()
        start_time = time.time()
        for i, data in enumerate(dataset):
            print("Batch: ", batches_done)
            if evaluate_through_real_B is not None and dataset_yolo_eval is not None:
                if batches_done % evaluate_through_real_B == 0:
                    precision, recall, AP, f1 = \
                    util_yolo.evaluate_yolo_through_realB_double(model, dataset_yolo_eval, iou_thres, conf_thres, nms_thres)
                    print("evaluate_through_real_B: precision %.2f, recall %.2f, AP %.2f, f1 %.2f"%(precision, recall, AP, f1))
                    evaluate_through_real_B_results.append([batches_done, precision[0], recall[0], AP[0], f1[0]])

            if evaluate_through_real_B is not None and validation_path is not None:
                if batches_done % evaluate_through_real_B == 0:
                    precision, recall, AP, f1, _, _ = util_yolo.evaluate_yolo_net(model.netYoloB, 
                                                                                validation_path, 
                                                                                iou_thres,
                                                                                conf_thres, 
                                                                                nms_thres, img_size=416, class_names='grapes')

                    print("evaluate_through_real_B (validation_path): precision %.2f, recall %.2f, AP %.2f, f1 %.2f"%(precision, recall, AP, f1))
                    evaluate_through_real_B_results.append([batches_done, precision[0], recall[0], AP[0], f1[0]])

            model.netYoloB.train()
            model.optimizer_yolo_b.zero_grad()
            model.set_input(data)  # unpack data from data loader
            model.forward()           # run inference
            imgs = model.rec_A

            # Train on rec A
            if rec_A:        
                imgs = model.rec_A.detach()*0.5+0.5
                targets =  model.A_label
                loss, outputs = model.netYoloB(imgs, targets)
                if loss > 0:
                    loss.backward()
                    model.netYoloB.seen += imgs.size(0)
                    if batches_done % yolo_gradient_accumulations == 0:
                        # Accumulates gradient before each step
                        model.optimizer_yolo.step()
                        model.optimizer_yolo.zero_grad()
                    print("rec_A loss: ", loss.item())
            
            # Train on fake labeled A  
            if fake_labeled_A:      
                imgs = model.fake_labeled_A.detach()*0.5+0.5
                targets =  model.labeled_B_label
                loss, outputs = model.netYoloB(imgs, targets)
                if loss > 0:
                    loss.backward()
                    model.netYoloB.seen += imgs.size(0)
                    if batches_done % yolo_gradient_accumulations == 0:
                        # Accumulates gradient before each step
                        model.optimizer_yolo.step()
                        model.optimizer_yolo.zero_grad()
                    print("fake_labeled_A loss: ", loss.item())

            # Train on real A
            if real_A:        
                imgs = model.real_A*0.5+0.5
                targets =  model.A_label
                loss, outputs = model.netYoloB(imgs, targets)
                if loss > 0:
                    loss.backward()
                    model.netYoloB.seen += imgs.size(0)
                    if batches_done % yolo_gradient_accumulations == 0:
                        # Accumulates gradient before each step
                        model.optimizer_yolo.step()
                        model.optimizer_yolo.zero_grad()
                    print("real_A loss: ", loss.item())
            
            if fake_B:
                imgs = model.fake_B.detach()*0.5+0.5
                targets =  model.A_label
                loss, outputs = model.netYoloB(imgs, targets)
                if loss > 0:
                    loss.backward()
                    model.netYoloB.seen += imgs.size(0)
                    if batches_done % yolo_gradient_accumulations == 0:
                        # Accumulates gradient before each step
                        model.optimizer_yolo_b.step()
                        model.optimizer_yolo_b.zero_grad()
                    print("fake_B loss: ", loss.item())

            if labeled_B and batches_done % labeled_B_train_ratio == 0:
                imgs = model.labeled_B*0.5+0.5
                targets =  model.labeled_B_label
                loss, outputs = model.netYoloB(imgs, targets)
                if loss > 0:
                    loss.backward()
                    model.netYoloB.seen += imgs.size(0)
                    if batches_done % yolo_gradient_accumulations == 0:
                        # Accumulates gradient before each step
                        model.optimizer_yolo_b.step()
                        model.optimizer_yolo_b.zero_grad()
                    print("labeled_B loss: ", loss.item())

            batches_done += imgs.size(0)


            if batches_done > curoff_batch:
                break
    
    return evaluate_through_fake_A_results, evaluate_through_real_B_results
            