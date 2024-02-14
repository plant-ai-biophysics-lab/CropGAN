import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import glob
import sys
import os
import copy
import argparse
import random
import tqdm
import time
from PIL import Image
import shutil
import subprocess
from multiprocessing import Pool

import torch
from torch.autograd import Variable

# Import Crop GAN related libs
gan_dir = os.path.abspath("../src/")
sys.path.append(gan_dir)
from options.train_options import TrainOptions
# from options.test_options import TestOptions

from data import create_dataset
from models import create_model
import util.util_yolo as util_detector
from models.yolo_model import Darknet
import util.util as utils
from util.dataset_yolo import ListDataset


""" 1. Setup Model
The most important args to be set are:  
--checkpoints_dir:  # Set to the location of model  
--name: # the folder name 
"""

arguments = "--model double_task_cycle_gan\
             --checkpoints_dir ../data/models/ \
             --name Synthetic2bordenNight_Yolo_adaptive_branch_102223\
             --no_flip\
             --num_threads 0\
             --gpu_ids 0\
             --display_id -1\
             --preprocess resize_and_crop\
             --load_size 416\
             --crop_size 416\
             --batch_size 1 \
             --task_model_def /home/michael/ucdavis/CropGAN/yolov3/config/yolov3-tiny.cfg" 

opt = TrainOptions().parse_notebook(arguments.split())

model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
model.eval()
print("MODEL SETUP")
## 2. Load pretrained model weights

load_suffix = "../data/models/adaptive_teacher/Synthetic2bordenNight_Yolo_adaptive_branch_102223/latest"
model.load_networks_from_folder(load_suffix)

print("WEIGHTS LOADED")
## 3. Load an systhetic (Domain A) image you want to transfer
# Loop through all the synthetic images, generate realistic ones.
def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

out_path = "/home/michael/ucdavis/CropGANData/mhs_cropgan_generated/test/"
in_path = "/home/michael/ucdavis/CropGANData/helios_synthetic_datasets/sytheticVis2bordenNight_n_0/"

synth_images = glob.glob(in_path + "trainA/*.jpg")
print(f"len(synth_images): {len(synth_images)}")
with torch.no_grad():

    for img_path in synth_images:
        print(f"Processing img: {img_path}")
        raw_image = Image.open(img_path).convert('RGB')
        raw_img_tensor, raw_img_np = utils.preprocess_images(raw_image,resize=[416,416])
        fake_img = model.netG_A(raw_img_tensor)
        fake_img_np = fake_img.detach().cpu().squeeze(0).permute([1, 2, 0])*0.5+0.5
        im = tensor_to_image(fake_img_np)
        im.save(out_path + img_path.split('/')[-1])

