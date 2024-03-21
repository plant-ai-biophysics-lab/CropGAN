#! /bin/bash -l
#SBATCH --job-name=paibl
#SBATCH --output=generate.out
#SBATCH --partition=gpum
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=amnjoshi@ucdavis.edu
#SBATCH --time=48:00:00 # Change the time accordingly
#SBATCH --mail-type=ALL
#SBATCH --error=generate.out
#SBATCH --cpus-per-task=12


# Load module and load environment
source ~/.bashrc
conda activate agml
unset PYTHONPATH
WORKDIR="/group/jmearlesgrp/scratch/amnjoshi/CropGAN/src"
cd $WORKDIR

cropgan_dir="/group/jmearlesgrp/scratch/amnjoshi"
dataroot="/group/jmearlesgrp/intermediate_data/amnjoshi/yolo_grl_data_final/BordenNight/"
task_model_def="$cropgan_dir/CropGAN/yolo_uda/configs/yolov3-tiny.cfg"

# the epoch is either a number or latest, it will be used to pull cyclegan checkpoint filenames.
epoch="latest"

# Same as the training images for training yolo-uda: something like BordenNight/source/train/images
image_path="$dataroot/source/train/images/"

# Inside of checkpoints_dir, this is the specific directory for the run you want to use
run_name="BEST_k=24"

# Where you want the CropGAN generated images to go
out_path="$cropgan_dir/cropgan_modified_images/$run_name/$epoch/"
mkdir -p $out_path

# Where your cyclegan checkpoints are stored (note: run_name is a sub-directory of checkpoints_dir)
checkpoints_dir="$cropgan_dir/model_checkpoints/cyclegan/"


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
    --out_path $out_path \
    --log
