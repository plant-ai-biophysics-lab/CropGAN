cropgan_dir="/home/michael/ucdavis"
dataroot="$cropgan_dir/CropGANData/helios_synthetic_datasets/sytheticVis2bordenNight_n_1/"
checkpoints_dir="$cropgan_dir/CropGANData/output/BordenNight/adaptive_teacher_091623/"
yolo_a_weights="$cropgan_dir/CropGAN/data/models/yolo/synthetic_pretrained_yolov3.pth"
yolo_b_weights="$cropgan_dir/CropGAN/data/models/yolo/synthetic_pretrained_yolov3.pth"
python -u train_cropgan.py --dataroot $dataroot \
             --num_threads 10\
             --name sythetic2bordenNight\
             --dataset_mode yolo_task_reverse\
             --checkpoints_dir  $checkpoints_dir \
             --no_flip\
             --preprocess aug\
             --model double_task_cycle_gan\
             --load_size 416\
             --crop_size 256\
             --lambda_yolo_b 0.1\
             --lambda_yolo_a 0.01\
             --batch_size 1\
             --cycle_gan_epoch 1\
             --yolo_eval_on_real_period 500\
             --yolo_epochs 0\
             --task_model_def $cropgan_dir/CropGAN/src/config/yolov3-tiny.cfg \
             --yolo_a_weights $yolo_a_weights \
             --yolo_b_weights $yolo_b_weights \
             --save_epoch_freq 1