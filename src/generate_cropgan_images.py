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
from options.image_gen_options import ImageGenOptions
import util.util as utils

import wandb


# Setup Model
opt = ImageGenOptions().parse()

model = create_model(opt)      # create a model given opt.model and other options
# Load pretrained model weights
model.setup(opt)               # regular setup: load and print networks; create schedulers
model.eval()


# Load a synthetic (Domain A) image you want to transfer
# Loop through all the synthetic images, generate realistic ones.
def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


synth_images = sorted(glob.glob(opt.image_path + "*.jpg"))  # deterministic for WandB
print("Image path: ", opt.image_path)
print("Number of synth input images: ", len(synth_images))


# logging
if opt.log:
    wandb.init(project="cropgan-results", name=opt.name, config=opt)


# results
with torch.no_grad():
    for img_path in synth_images:
        print(f"Processing img: {img_path}")
        raw_image = Image.open(img_path).convert('RGB')
        raw_img_tensor, raw_img_np = utils.preprocess_images(raw_image, resize=[416,416])
        fake_img = model.netG_A(raw_img_tensor)
        fake_img_np = fake_img.detach().cpu().squeeze(0).permute([1, 2, 0]) * 0.5 + 0.5
        im = tensor_to_image(fake_img_np)
        im.save(opt.out_path + img_path.split('/')[-1])

        # log the synthetic & syn2real images to WandB
        if opt.log:
            print("logging image to WandB")
            wandb.log({"synthetic": [wandb.Image(raw_image, caption="synthetic")],
                       "syn2real": [wandb.Image(im, caption="syn2real")]})

