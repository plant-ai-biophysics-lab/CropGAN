"""
detector_b_weights can be selected from the weight you trained in STEP1
"""
dataroot="/home/michael/ucdavis/CropGANData/crop_gan_data/sytheticVis2bordenNight_n=1/"
checkpoints_dir="/home/michael/ucdavis/CropGANData/output/BordenNight/testtest/cropgan_checkpoints/"
# detector_a: obj detection on the synthetic source domain
# detector_b: obj detection on the target real domain
detector_a_weights="/home/michael/ucdavis/CropGAN/data/models/yolo/synthetic_pretrained_yolov3.pth"
# detector_b_weights="/home/michael/ucdavis/CropGANData/output/BordenNight/train1_val1/checkpoints/best_mAp_yolov3_ckpt.pth"
python -u train_cropgan.py --dataroot $dataroot \
             --num_threads 4\
             --name first_time\
             --dataset_mode detector_task_reverse\
             --checkpoints_dir  $checkpoints_dir \
             --no_flip\
             --preprocess aug\
             --model double_task_cycle_gan\
             --load_size 416\
             --crop_size 256\
             --lambda_detector_b 0.1\
             --lambda_detector_a 0.01\
             --batch_size 1\
             --cycle_gan_epoch 1\
             --detector_eval_on_real_period 500\
             --detector_epochs 0\
             --task_model_def /home/michael/ucdavis/CropGAN/yolov3/config/yolov3-tiny.cfg \
             --detector_a_weights $detector_a_weights \
             --detector_b_weights $detector_a_weights \
             --save_epoch_freq 25