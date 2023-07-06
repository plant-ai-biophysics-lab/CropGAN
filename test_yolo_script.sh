"""
If not using synthetic pretrained
--model_def ./config/yolov3-tiny.cfg
--pretrained_weights /data2/zfei/data/cycleGAN/yolo/weights/yolov3-tiny.weights
--model_def ./config/yolov3.cfg
--pretrained_weights /data2/zfei/data/cycleGAN/yolo/weights/darknet53.conv.74
"""
# e.g ~/CropGANData/detection_datasets/BordenNight/
data_path="/home/michael/ucdavis/CropGANData/detection_datasets/BordenNight"
# save_dir="/home/michael/ucdavis/CropGANData/output/BordenNight"
# pretrained_weights="/home/michael/ucdavis/CropGAN/data/models/yolo/synthetic_pretrained_yolov3.pth"
pretrained_weights="/home/michael/ucdavis/CropGANData/output/BordenNight/train1_val1/checkpoints/best_mAp_yolov3_ckpt.pth"
# You can change to other data config by change dataname="traina_valb"
dataname="train1_val1"

python -u yolov3/test.py   --batch_size=1 \
                           --model_def yolov3/config/yolov3-tiny.cfg \
                           --data_config $data_path/data_configs/$dataname/data.data \
                           --weights_path $pretrained_weights \
                           --class_path $data_path/data_configs/$dataname/classes.names \
                           --iou_thres 0.5 \
                           --conf_thres 0.5 \
                           --nms_thres 0.5 \
                           --img_size=416 \
                           --adaptive_batchnorm \