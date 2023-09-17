#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import sys
sys.path.append("/home/michael/ucdavis/adaptive_teacher")
                
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data.datasets import register_coco_instances

from adapteacher import add_ateacher_config
from adapteacher.engine.trainer import ATeacherTrainer, BaselineTrainer

# import adapteacher.data.datasets.builtin
from cropgan_adapteacher.util.cfg_setup import setup


from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel

register_coco_instances("helios_raw_synthetic", {}, "/home/michael/datasets/cropgan_coco/helios_raw_synthetic/train/_annotations.coco.json", "/home/michael/datasets/cropgan_coco/helios_raw_synthetic/train/")
register_coco_instances("helios_raw_synthetic_val", {}, "/home/michael/datasets/cropgan_coco/helios_raw_synthetic/valid/_annotations.coco.json", "/home/michael/datasets/cropgan_coco/helios_raw_synthetic/valid/")
register_coco_instances("helios_plus_cropgan_n_1", {}, "/home/michael/datasets/cropgan_coco/cropgan_generated_n_1/train/_annotations.coco.json", "/home/michael/datasets/cropgan_coco/cropgan_generated_n_1/train/")
register_coco_instances("helios_plus_cropgan_adaptive", {}, "/home/michael/datasets/cropgan_coco/cropgan_adaptive_n_0_091123/train/_annotations.coco.json", "/home/michael/datasets/cropgan_coco/cropgan_adaptive_n_0_091123/train/")
register_coco_instances("real_borden_night_train", {}, "/home/michael/datasets/cropgan_coco/borden_night_real/train/_annotations.coco.json", "/home/michael/datasets/cropgan_coco/borden_night_real/train")
register_coco_instances("real_borden_night_test", {}, "/home/michael/datasets/cropgan_coco/borden_night_real/test/_annotations.coco.json", "/home/michael/datasets/cropgan_coco/borden_night_real/test/")
register_coco_instances("real_borden_day_train", {}, "/home/michael/datasets/cropgan_coco/borden_day_real/train/_annotations.coco.json", "/home/michael/datasets/cropgan_coco/borden_day_real/train")
register_coco_instances("real_borden_day_test", {}, "/home/michael/datasets/cropgan_coco/borden_day_real/test/_annotations.coco.json", "/home/michael/datasets/cropgan_coco/borden_day_real/test/")
classes = ('grapes')


# TENSORBOARD SETUP
# tensorboard_writer = SummaryWriter('runs/faster-rcnn-cropgan/baseline-helios-plus-cropgan')

#  Tensorboard helper functions
# helper functions

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


def main(args):
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.98)
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "ateacher":
        Trainer = ATeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ateacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )