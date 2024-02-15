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
    --checkpoints_dir /home/michael/ucdavis/CropGAN/data/models/ \
    --name Sythetic2bordenNight \
    --epoch "latest" \
    --no_flip \
    --num_threads 0 \
    --load_size 416 \
    --crop_size 416 \
    --batch_size 1 \
    --task_model_def "/home/michael/ucdavis/CropGAN/yolov3/config/yolov3-tiny.cfg" \
    --image_path "/home/michael/ucdavis/CropGANData/helios_synthetic_datasets/sytheticVis2bordenNight_n_0/trainA/" \
    --out_path "/home/michael/ucdavis/CropGANData/mhs_cropgan_generated/test/"
    # --use_grl
    # --cropgan_weights "/home/michael/ucdaivs/CropGAN/data/models/Sythetic2bordenNight/latest" \
    
    