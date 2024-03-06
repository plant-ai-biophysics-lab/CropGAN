#! /bin/bash -l
#SBATCH --job-name=paibl
#SBATCH --output=train.out
#SBATCH --partition=gpum
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=michael.hamer.stanley@gmail.com
#SBATCH --time=48:00:00 # Change the time accordingly
#SBATCH --mail-type=ALL
#SBATCH --error=train.err
#SBATCH --cpus-per-task=12

# Load module and load environment
source ~/.bashrc
conda activate yolo-uda
unset PYTHONPATH
WORKDIR="/group/jmearlesgrp/scratch/mhstan/CropGAN/src"
cd $WORKDIR

cropgan_dir="/group/jmearlesgrp/scratch/mhstan"
dataroot="/group/jmearlesgrp/intermediate_data/mhstan"

## For CropGAN
# Source for CropGAN images (note: different structure than yolo source: trainA, labeledB, etc.)
cropgan_source_images="$dataroot/syntheticVis2bordenDayRow"

# Yolo Config
task_model_def="$cropgan_dir/CropGAN/yolo_uda/configs/yolov3-tiny-lowlr.cfg"

# Where the model checkpoints are saved
checkpoints_dir="$cropgan_dir/model_checkpoints/cyclegan/"
# make directories
mkdir -p $checkpoints_dir

# Same as the training images for training yolo-uda: something like BordenNight/source/train/images
source_image_path="$dataroot/yolo_grl_data/BordenDay/source/train/images/"

## For Image Generation
# Inside of checkpoints_dir, this is the specific directory for the run you want to use
run_name="BASELINE_k_24_day"

# Where you want the CropGAN generated images to go
out_path="$dataroot/yolo_grl_data/BordenDay/cropgan_generated/$run_name"

# Weights used for yolo_a and yolo_b
yolo_a_weights="$dataroot/yolo_grl_data/weights"

# standard arguments
read -r -d '' standard_args << EOM
--dataroot $cropgan_source_images \
--num_threads 4 \
--dataset_mode yolo_task_reverse \
--checkpoints_dir $checkpoints_dir \
--no_flip \
--preprocess aug \
--model double_task_cycle_gan \
--load_size 416 \
--crop_size 256 \
--lambda_yolo_b 0.1 \
--lambda_yolo_a 0.01 \
--batch_size 1 \
--cycle_gan_epoch 1 \
--yolo_eval_on_real_period 500 \
--yolo_epochs 0 \
--task_model_def $task_model_def \
--yolo_a_weights /group/jmearlesgrp/intermediate_data/mhstan/yolo_grl_data/weights \
--save_epoch_freq 25 \
--use_grl
--image_path $source_image_path
--out_path $out_path
EOM

python -u train_cropgan.py --name $run_name --reverse_task_k 32 --wandb_name $run_name --grl_alpha 0.0 --grl_lambda 0.0 --grl_lmmd 0.0 $standard_args
python -u train_cropgan.py --name $run_name --reverse_task_k 32 --wandb_name $run_name --grl_alpha 0.1 --grl_lambda 0.0001 --grl_lmmd 0.0005 $standard_args