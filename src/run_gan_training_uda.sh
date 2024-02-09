cropgan_dir="/home/michael/ucdavis"
dataroot="$cropgan_dir/CropGANData/helios_synthetic_datasets/sytheticVis2bordenNight_n_1/"
# Where the model checkpoints are saved
checkpoints_dir="$cropgan_dir/CropGANData/output/BordenNight/adaptive_teacher_091623/"
# yolo a will detect on synthetic-ish images
yolo_a_weights="$cropgan_dir/CropGANData/ckpt_last_k=4_alpha=0.001_lambda=0.01_2024-01-14_11-58-48.pth"
# yolo b will detect on real-ish images
yolo_b_weights="$cropgan_dir/CropGANData/ckpt_last_k=4_alpha=0.001_lambda=0.01_2024-01-14_11-58-48.pth"
task_model_def="$cropgan_dir/CropGAN/yolo_uda/configs/yolov3-tiny.cfg"

python -u train_cropgan.py \
    --dataroot $dataroot\
    --num_threads 4 \
    --name sythetic2bordenNight \
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
    --yolo_a_weights $yolo_a_weights \
    --yolo_b_weights $yolo_b_weights \
    --save_epoch_freq 25 \
    --use_grl