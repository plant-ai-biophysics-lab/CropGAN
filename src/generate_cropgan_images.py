import argparse
import glob
import sys
import os
from multiprocessing import Pool

from PIL import Image
import numpy as np
import torch

# Import Crop GAN related libs
gan_dir = os.path.abspath("../src/")
sys.path.append(gan_dir)

from models import create_model
from models.double_task_cycle_gan_model import DoubleTaskCycleGanModel
from options.image_gen_options import ImageGenOptions
import util.util as utils

import wandb

# Load a synthetic (Domain A) image you want to transfer
# Loop through all the synthetic images, generate realistic ones.
def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

# This is called at the end of train_cropgan.py
def generate_images_from_source(opt: argparse.Namespace, model: DoubleTaskCycleGanModel, run: wandb.run):
    synth_images = sorted(glob.glob(opt.image_path + "*.jpg"))  # deterministic for WandB
    print("Image path: ", opt.image_path)
    print("Number of synth input images: ", len(synth_images))

    # results
    with torch.no_grad():
        for img_path in synth_images:
            print(f"Processing img: {img_path}")
            raw_image = Image.open(img_path).convert('RGB')
            raw_img_tensor, raw_img_np = utils.preprocess_images(raw_image, resize=[416,416])
            fake_img = model.netG_A(raw_img_tensor)
            fake_img_np = fake_img.detach().cpu().squeeze(0).permute([1, 2, 0]) * 0.5 + 0.5
            im = tensor_to_image(fake_img_np)
            save_path = opt.out_path + img_path.split('/')[-1]
            print("Saving at: ",save_path)
            im.save(save_path)

            # log the synthetic & syn2real images to WandB
            if opt.log:
                print("logging image to WandB")
                run.log({"synthetic": [wandb.Image(raw_image, caption="synthetic")],
                        "syn2real": [wandb.Image(im, caption="syn2real")]})

