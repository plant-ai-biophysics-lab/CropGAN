cropgan_dir="/group/jmearlesgrp/scratch/mhstan"
data_dir="/group/jmearlesgrp/intermediate_data/mhstan/"
run_name="BASELINE_k_4"
epoch="latest"

image_path="$data_dir/yolo_grl_data/BordenNight/source/train/images"
out_path="$data_dir/cropgan_modified_images/$run_name/$epoch/"
checkpoints_dir="$cropgan_dir/model_checkpoints/cyclegan/"
task_model_def="$cropgan_dir/CropGAN/yolo_uda/configs/yolov3-tiny.cfg"

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
    # --use_grl
    # --cropgan_weights "/home/michael/ucdaivs/CropGAN/data/models/Sythetic2bordenNight/latest" \
    
    