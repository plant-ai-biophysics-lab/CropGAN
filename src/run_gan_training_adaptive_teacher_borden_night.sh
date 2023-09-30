"""
detector_b_weights can be selected from the weight you trained in STEP1
"""
task_model_def="/home/michael/ucdavis/CropGAN/adaptive_teacher/configs/faster_rcnn_VGG_borden_night_helios_raw.yaml"
dataroot="/home/michael/ucdavis/CropGANData/helios_synthetic_datasets/sytheticVis2bordenNight_n=0/"
checkpoints_dir="/home/michael/ucdavis/CropGANData/output/BordenNight/adaptive_teacher_091623/"
# detector_a: obj detection on the synthetic source domain
# detector_b: obj detection on the target real domain
detector_a_weights="/home/michael/ucdavis/CropGAN/adaptive_teacher/output/borden_night_helios_raw_n_0_091123/model_0021999.pth"
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
             --lambda_detector_b 2.0\
             --lambda_detector_a 0.2\
             --lambda_detector_a_rec_a 1.0\
             --batch_size 1\
             --cycle_gan_epoch 1\
             --detector_eval_on_real_period 500\
             --detector_epochs 0\
             --task_model_def $task_model_def \
             --detector_a_weights $detector_a_weights \
             --detector_b_weights $detector_a_weights \
             --save_epoch_freq 25