#! /bin/bash -l
#SBATCH --job-name=inference
#SBATCH --account=geminigrp
#SBATCH --output=/group/jmearlesgrp/scratch/amnjoshi/CropGAN/yolo_uda/parent_sweep_1.out
#SBATCH --error=/group/jmearlesgrp/scratch/amnjoshi/CropGAN/yolo_uda/parent_sweep_1.out
#SBATCH --partition=gpu-a100-h
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=amnjoshi@ucdavis.edu
#SBATCH --time=48:00:00 # Change the time accordingly
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=12

# Load module and load environment
source ~/.bashrc
conda activate agml
unset PYTHONPATH
WORKDIR="/group/jmearlesgrp/scratch/amnjoshi/CropGAN/yolo_uda"
cd $WORKDIR


# GLOBAL OPTIONS FOR THE SWEEP
DATASET="BordenDayRow"
CONFIG_NAME="yolov3-tiny-lowlr.cfg"
CHECKPOINT_DIR="/group/jmearlesgrp/intermediate_data/amnjoshi/yolo_grl_weights_final/"
SWEEP_NAME="feb29-sweep"


read -r -d '' standard_args << EOM
-t /group/jmearlesgrp/intermediate_data/amnjoshi/yolo_grl_data_final/$DATASET/source/train/images \
-tt /group/jmearlesgrp/intermediate_data/amnjoshi/yolo_grl_data_final/$DATASET/target/train/images \
-tv /group/jmearlesgrp/intermediate_data/amnjoshi/yolo_grl_data_final/$DATASET/target/valid/images \
-c configs/$CONFIG_NAME \
-p /group/jmearlesgrp/intermediate_data/amnjoshi/yolo_grl_data/weights/yolov3-tiny.weights \
-e 1500 \
--n-cpu 4 \
-s $CHECKPOINT_DIR \
--name $SWEEP_NAME \
EOM


python3 training/main.py -k 12 -a 0.0 -l 0.0 \
                          $standard_args

python3 training/main.py -k 12 -a 0.01 -l 0.01 \
                          $standard_args

python3 training/main.py -k 12 -a 0.5 -l 0.0001 \
                          $standard_args



