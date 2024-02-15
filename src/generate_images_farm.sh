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
data_dir="/group/jmearlesgrp/intermediate_data/mhstan/yolo_grl_data/BordenNight/"
task_model_def="$cropgan_dir/CropGAN/yolo_uda/configs/yolov3-tiny.cfg"

# the epoch is either a number or latest, it will be used to pull cyclegan checkpoint filenames.
epoch="latest"

# Same as the training images for training yolo-uda: something like BordenNight/source/train/images
image_path="$data_dir/yolo_grl_data/BordenNight/source/train/images"

# Where you want the CropGAN generated images to go
out_path="$data_dir/cropgan_modified_images/$run_name/$epoch/"

# Where your cyclegan checkpoints are stored (note: run_name is a sub-directory of checkpoints_dir)
checkpoints_dir="$cropgan_dir/model_checkpoints/cyclegan/"

# Inside of checkpoints_dir, this is the specific directory for the run you want to use
run_name="BASELINE_k_4"


python -u generate_cropgan_images.py \
    --checkpoints_dir $checkpoints_dir \
    --name $run_name \
    --epoch latest \
    --no_flip \
    --num_threads 0 \
    --load_size 416 \
    --crop_size 416 \
    --batch_size 1 \
    --task_model_def $task_model_def \
    --image_path $image_path \
    --out_path $out_path