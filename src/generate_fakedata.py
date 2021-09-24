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
from tqdm import tqdm
import time
from PIL import Image
import shutil
import subprocess
from multiprocessing import Pool
import torch
from torch.autograd import Variable

# Import cycle gan related libs
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import util.util_yolo as util_yolo
from models.yolo_model import Darknet
import util.util as utils
from models.sem_cycle_gan_model import SemCycleGANModel
from util.dataset_yolo import ListDataset

# Setup options
opt = TrainOptions().parse()   # get training options
print(opt)
opt.num_threads = 0   # test code only supports num_threads = 0
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
opt.batch_size = 1
opt.isTrain=False
dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
print("dataset size ", len(dataset))


# Load GAN model
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
model.eval()
load_suffix = "200"
model.load_networks(load_suffix)

# Save Fake
fake_dataset_save_dir = opt.fake_dataset_save_dir
validation_path = opt.validation_path
test_path = opt.test_path
train_path = validation_path.replace("valid.txt", "train.txt")

image_dir = fake_dataset_save_dir + "/images/"
label_dir = fake_dataset_save_dir + "/labels/"

if not os.path.exists(image_dir):
    os.makedirs(image_dir)
if not os.path.exists(label_dir):
    os.makedirs(label_dir)

print("fake_dataset_save_dir: ", fake_dataset_save_dir)
print("validation_path: ", validation_path)
print("test_path: ", test_path)
print("labeled_train_path: ", train_path)

print("Generating data...")
for i, data in enumerate(tqdm(dataset)):
    image_save_path = image_dir + "/%.05i.jpg"%i
    label_save_path = image_save_path.replace("images", "labels").replace(".jpg", ".txt")
    with torch.no_grad():
        model.set_input(data)  # unpack data from data loader
        model.forward()        # run inference
        image = ((model.fake_B.detach().cpu().squeeze(0).permute([1,2,0])*0.5+0.5).numpy())*255
        image = image.astype(dtype=np.uint8)
        if opt.model == "double_task_cycle_gan":
            label = model.A_label.to("cpu").numpy()
            label = label[:, 1:]
            np.savetxt(label_save_path, label)
        cv.imwrite(image_save_path, cv.cvtColor(image, cv.COLOR_RGB2BGR))
    if i >= dataset.dataset.A_size:
        break
    
with open(fake_dataset_save_dir+"data.data", 'w') as file:
    file.write("classes=1 \n")
    file.write("train=" + fake_dataset_save_dir + "train.txt" + "\n")
    file.write("valid=" + validation_path + "\n")
    file.write("test=" + test_path + "\n")
    file.write("names=" + fake_dataset_save_dir + "classes.names" + "\n")
with open(fake_dataset_save_dir+"classes.names", 'w') as file:
    file.write("grapes\n")

image_all = glob.glob(image_dir + "/*.jpg")
image_all.sort()
print("Total number of images: ", len(image_all))
# Write fake filenames:
print("Saving traning into .. ", fake_dataset_save_dir+"train.txt")
# Write labeled fileanmes:

with open(fake_dataset_save_dir+"train.txt", 'w') as file:
    for i in range(len(image_all)):
        file.write(image_all[i]+"\n")
    with open(train_path, 'r') as file2:
        file.write(file2.read())