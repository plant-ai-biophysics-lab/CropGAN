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


mix_val_ratio = 0.2 # 20 % of the generated image will be added into validation

# Setup options
opt = TrainOptions().parse()   # get training options
print(opt)
opt.num_threads = 0   # test code only supports num_threads = 0
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
opt.batch_size = 1
opt.isTrain=False



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

with open(fake_dataset_save_dir+"data.data", 'w') as file:
    file.write("classes=1 \n")
    file.write("train=" + fake_dataset_save_dir + "train.txt" + "\n")
    file.write("valid=" + fake_dataset_save_dir + "valid.txt" + "\n")
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

train_num = int(len(image_all) * (1-mix_val_ratio))
image_train = image_all[:train_num]
image_val = image_all[train_num:]

print("Total number of images: ", len(image_train))
print("Total number of images: ", len(image_val))



with open(fake_dataset_save_dir+"train.txt", 'w') as file:
    for i in range(len(image_train)):
        file.write(image_train[i]+"\n")
    with open(train_path, 'r') as file2:
        file.write(file2.read())

with open(fake_dataset_save_dir+"valid.txt", 'w') as file:
    for i in range(len(image_val)):
        file.write(image_val[i]+"\n")
    with open(validation_path, 'r') as file2:
        file.write(file2.read())