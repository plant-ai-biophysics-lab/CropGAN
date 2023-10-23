"""
detector_b_weights can be selected from the weight you trained in STEP1
"""
dataroot="/home/michael/ucdavis/CropGANData/helios_synthetic_datasets/sytheticVis2bordenNight_n=1/"
checkpoints_dir="/home/michael/ucdavis/CropGANData/output/BordenNight/cropgan_main_branch_n=1_101223/"
# detector_a: obj detection on the synthetic source domain
# detector_b: obj detection on the target real domain
yolo_a_weights="/home/michael/ucdavis/CropGAN/data/models/yolo/synthetic_pretrained_yolov3.pth"
yolo_b_weights="/home/michael/ucdavis/CropGANData/output/BordenNight/train1_val1/checkpoints/best_mAp_yolov3_ckpt.pth"
python -u train_cropgan.py --dataroot $dataroot \
             --num_threads 4\
             --name cropgan_main_branch_n=1_101023\
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
             --task_model_def ../yolov3/config/yolov3-tiny.cfg \
             --yolo_a_weights $yolo_a_weights \
             --yolo_b_weights $yolo_b_weights \
             --save_epoch_freq 1