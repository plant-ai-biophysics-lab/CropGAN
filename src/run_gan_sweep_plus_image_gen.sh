#! /bin/bash -l
#SBATCH --job-name=cropgan-run
#SBATCH --output=train.out
#SBATCH --error=train.out
#SBATCH --partition=gpum
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=amnjoshi@ucdavis.edu@
#SBATCH --time=48:00:00 # Change the time accordingly
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=12

# Load module and load environment
source ~/.bashrc
conda activate agml
unset PYTHONPATH
WORKDIR="/group/jmearlesgrp/scratch/amnjoshi/CropGAN/src"
cd $WORKDIR

cropgan_dir="/group/jmearlesgrp/scratch/amnjoshi"
dataroot="/group/jmearlesgrp/intermediate_data/amnjoshi"

## For CropGAN
# Source for CropGAN images (note: different structure than yolo source: trainA, labeledB, etc.)
cropgan_source_images="$dataroot/CropGANData/syntheticVis2bordenNight"

# Yolo Config
task_model_def="$cropgan_dir/CropGAN/yolo_uda/configs/yolov3-tiny-lowlr.cfg"

# Where the model checkpoints are saved
checkpoints_dir="$cropgan_dir/model_checkpoints/cyclegan/"
# make directories
mkdir -p $checkpoints_dir

# Same as the training images for training yolo-uda: something like BordenNight/source/train/images
source_image_path="$dataroot/yolo_grl_data/BordenNight/source/train/images/"

## For Image Generation
# Where you want the CropGAN generated images to go
out_path="$dataroot/yolo_grl_data/BordenNight/cropgan_generated"

# standard arguments
read -r -d '' standard_args << EOM
--dataroot $cropgan_source_images \
--num_threads 4 \
--dataset_mode yolo_task_reverse \
--checkpoints_dir $checkpoints_dir \
--no_flip \
--preprocess aug \
--model double_task_cycle_gan_context \
--load_size 416 \
--crop_size 256 \
--lambda_yolo_b 0.1 \
--lambda_yolo_a 0.01 \
--batch_size 1 \
--yolo_eval_on_real_period 500 \
--task_model_def $task_model_def \
--yolo_a_weights /group/jmearlesgrp/intermediate_data/amnjoshi/yolo_grl_weights_context \
--save_epoch_freq 25 \
--use_grl
--image_path $source_image_path
--out_path $out_path
EOM


python -u train_cropgan.py \
    --name BEST_k=98_mar30_context_apr16_context_test \
    --reverse_task_k 98 \
    --wandb_name BEST_k=98_mar30_context_apr16_context_test \
    --grl_alpha 0.5 \
    --grl_lambda 0.0001 \
    --grl_lmmd 0.0 \
    $standard_args

