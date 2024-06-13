#! /bin/bash -l
#SBATCH --job-name=gan-sweep
#SBATCH --output=/group/jmearlesgrp/scratch/amnjoshi/CropGAN/src/result.out
#SBATCH --error=/group/jmearlesgrp/scratch/amnjoshi/CropGAN/src/result.out
#SBATCH --partition=gpu-a100-h
#SBATCH --account=geminigrp
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=joshi.amoghn@gmail.com
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
dataroot="/group/jmearlesgrp/intermediate_data/amnjoshi/CropGANData/syntheticVis2bordenNight"
# Where the model checkpoints are saved

checkpoints_dir="$cropgan_dir/model_checkpoints/cyclegan/"
task_model_def="$cropgan_dir/CropGAN/yolo_uda/configs/yolov3-tiny.cfg"

# make directories
mkdir -p $checkpoints_dir

# standard arguments
read -r -d '' standard_args << EOM
--dataroot $dataroot \
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
--yolo_a_weights /group/jmearlesgrp/intermediate_data/amnjoshi/yolo_grl_weights/ \
--save_epoch_freq 25 \
--use_grl
EOM


python -u train_cropgan.py \
    --name BASELINE_k=40 \
    --reverse_task_k 40 \
    --wandb_name BASELINE_k=40 \
    --grl_alpha 0.0 \
    --grl_lambda 0.0 \
    $standard_args


python -u train_cropgan.py \
    --name BEST_k=40 \
    --reverse_task_k 40 \
    --wandb_name BEST_k=40 \
    --grl_alpha 0.5 \
    --grl_lambda 0.0001 \
    $standard_args



