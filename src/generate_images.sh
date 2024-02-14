cropgan_dir="/group/jmearlesgrp/scratch/mhstan"
dataroot="/group/jmearlesgrp/intermediate_data/mhstan/syntheticVis2bordenNight"
# Where the model checkpoints are saved
checkpoints_dir="$cropgan_dir/model_checkpoints/cyclegan/"
# yolo a will detect on synthetic-ish images
yolo_a_weights="$cropgan_dir/model_checkpoints/yolo_grl/'BASELINE_k=4_01-21-18-53-29.pth'"
# yolo b will detect on real-ish images
yolo_b_weights="$cropgan_dir/model_checkpoints/yolo_grl/'BASELINE_k=4_01-21-18-53-29.pth'"
task_model_def="$cropgan_dir/CropGAN/yolo_uda/configs/yolov3-tiny.cfg"

python -u generate_cropgan_images.py \
    --checkpoints_dir ../data/models/ \
    --name Synthetic2bordenNight_Yolo_adaptive_branch_102223 \
    --no_flip \
    --num_threads 0 \
    --display_id -1 \
    --load_size 416 \
    --crop_size 416 \
    --batch_size 1 \
    --task_model_def "/home/michael/ucdavis/CropGAN/yolov3/config/yolov3-tiny.cfg"
    # --use_grl
