# Enlisting 3D Crop Models and GANs for More Data Efficient and Generalizable Fruit Detection

We provide the pytorch implementation of a semantically constrained GAN to generate artificial realisitc fruit images for training to reduce the need of real image labeling in fruit detection.

**[Paper Link](https://arxiv.org/abs/2108.13344)**

<img src='imgs/fig-image-transfer-demo.png'>


## **Prepare Data**

### Source domain data  
The dataset used in this research:
1. Synthetic Grape Data

### Target domain data
The dataset used in this research:
1. Night Grape
2. Day Grape

## **Generate Semantic Consistent GAN Fruits**
Use the following script to load the pre-trained Semantic Consistent GAN model and generate target domain images from source synthetic image.
```bash

```







#### STEP1 Finetuning using N train image and K validation image
1. Run step1-finetuning.py script at yolov3 folder  
You can vary image size at 416 and 256
```bash
"""
If not using synthetic pretrained
--model_def ./config/yolov3-tiny.cfg
--pretrained_weights /data2/zfei/data/cycleGAN/yolo/weights/yolov3-tiny.weights
--model_def ./config/yolov3.cfg
--pretrained_weights /data2/zfei/data/cycleGAN/yolo/weights/darknet53.conv.74
"""
dataname="all"
dataname="train1_val1"
dataname="train10_val5"
dataname="train1_val14"
dataname="train4_val1"
dataname="train8_val2"
dataname="train12_val3"
dataname="train16_val4"
dataname="train24_val6"
dataname="train32_val8"
dataname="train40_val10"
dataname="train96_val24"
nohup python -u step1-finetuning.py --model_def ./config/yolov3-tiny.cfg \
                           --data_config /data2/zfei/data/cycleGAN/yolo/paper_dataset/BordenNight/data_configs/$dataname/data.data \
                           --pretrained_weights /data2/zfei/data/cycleGAN/yolo_model/syntheticSquare-all-vis-tiny-99.pth \
                           --batch_size=8 \
                           --img_size=416 \
                           --save_dir /data2/zfei/data/cycleGAN/yolo_model/paper_models/bordenNight/$dataname \
                           --checkpoint_interval 10\
                           --epochs=100 &

```
2. Run test.py script at yolov3 folder to get test results
```bash
"""
weights_path can be one of the following
/data2/zfei/data/cycleGAN/yolo_model/paper_models/bordenNight/imgsize256/*/checkpoints/best_mAp_yolov3_ckpt.pth
/data2/zfei/data/cycleGAN/yolo_model/paper_models/bordenNight/imgsize416/*/checkpoints/best_mAp_yolov3_ckpt.pth
img_size = 416/256
iou_thres = 0.3/0.5
"""
dataname="all"
dataname="train1_val1"
dataname="train4_val1"
dataname="train8_val2"
dataname="train12_val3"
dataname="train16_val4"
dataname="train24_val6"
dataname="train32_val8"
dataname="train40_val10"
dataname="train96_val24"
python test.py --model_def ./config/yolov3-tiny.cfg \
                --data_config /data2/zfei/data/cycleGAN/yolo/paper_dataset/BordenNight/data_configs/$dataname/data.data \
                --weights_path /data2/zfei/data/cycleGAN/yolo_model/paper_models/bordenNight/$dataname/checkpoints/best_mAp_yolov3_ckpt.pth \
                --batch_size=8 \
                --img_size=416 \
                --iou_thres=0.5

                
```

#### STEP2 Train Semantic Constrained Cycle GAM
1. Run train_double_constrain.py at src/
```bash
"""
Make sure lambda_yolo_a = 0 for not using labeled b in the data folder
crop_size = 256/416
yolo_b_weights can be selected from:
/data2/zfei/data/cycleGAN/yolo_model/paper_models/bordenNight/imgsize256/*/checkpoints/best_mAp_yolov3_ckpt.pth
/data2/zfei/data/cycleGAN/yolo_model/paper_models/bordenNight/imgsize416/*/checkpoints/best_mAp_yolov3_ckpt.pth
"""
# On Megatron
python -u train_double_constrain.py --dataroot /data2/zfei/data/cycleGAN/paired_data/sytheticVis2bordenNight \
             --num_threads 10\
             --name paperSythetic2bordenNightC256T1V1Human\
             --dataset_mode yolo_task_reverse\
             --checkpoints_dir  /data2/zfei/data/cycleGAN/domainAdaptationCheckpoints \
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
             --task_model_def /home/zfei/Documents/DeepDomainAdaptation/src/config/yolov3-tiny.cfg \
             --yolo_a_weights /data2/zfei/data/cycleGAN/yolo_model/syntheticSquare-all-vis-tiny-99.pth \
             --yolo_b_weights /data2/zfei/data/cycleGAN/yolo_model/paper_models/bordenNight/imgsize416/train1_val1/checkpoints/best_mAp_yolov3_ckpt.pth \
             --save_epoch_freq 1
# On Farm
dataname="all"
python -u train_double_constrain.py --dataroot /home/zfei/group/intermediate_data/zfei/cycleGAN/paired_data/sytheticVis2bordenNight \
             --num_threads 10\
             --name paperSythetic2bordenNight$dataname\
             --dataset_mode yolo_task_reverse\
             --checkpoints_dir  /home/zfei/group/intermediate_data/zfei/cycleGAN/domainAdaptationCheckpoints \
             --no_flip\
             --display_id 0\
             --preprocess aug\
             --model double_task_cycle_gan\
             --load_size 416\
             --crop_size 256\
             --lambda_yolo_b 0.1\
             --lambda_yolo_a 0.0\
             --batch_size 1\
             --cycle_gan_epoch 1\
             --yolo_eval_on_real_period 500\
             --yolo_epochs 0\
             --task_model_def /home/zfei/Documents/DeepDomainAdaptation/src/config/yolov3-tiny.cfg \
             --yolo_a_weights /home/zfei/group/intermediate_data/zfei/cycleGAN/yolo_model/syntheticSquare-all-vis-tiny-99.pth \
             --yolo_b_weights /home/zfei/group/intermediate_data/zfei/cycleGAN/yolo_model/paper_models/bordenNight/imgsize416/$dataname/checkpoints/best_mAp_yolov3_ckpt.pth \
             --save_epoch_freq 1
```

#### STEP3 Generate Fake Images With Labels
1. Run generateGANImages.ipynb # TODO make it into a python file  
```bash
"""
The generated dataset is at /data2/zfei/data/cycleGAN/yolo/paper_dataset/GANGenerated/FakeBordenNight/
"""
```
2. Python way
```bash
dataname="all"
dataname="train1_val1"
dataname="train4_val1"
dataname="train8_val2"
dataname="train12_val3"
dataname="train16_val4"
dataname="train24_val6"
dataname="train32_val8"
dataname="train40_val10"
dataname="train96_val24"
dataname="trainCycleGAN"

nohup python -u generate_fakedata.py --model double_task_cycle_gan\
             --dataroot /data2/zfei/data/cycleGAN/paired_data/sytheticVis2bordenDay/ \
             --name "NewaperSythetic2bordenNight$dataname"\
             --dataset_mode yolo_task_reverse\
             --checkpoints_dir /data2/zfei/data/cycleGAN/domainAdaptationCheckpoints/ \
             --no_flip\
             --preprocess resize_and_crop\
             --load_size 256\
             --crop_size 256\
             --batch_size 1\
             --yolo_img_size 416 \
             --task_model_def /home/zfei/Documents/DeepDomainAdaptation/src/config/yolov3-tiny.cfg \
             --yolo_a_weights /data2/zfei/data/cycleGAN/yolo_model/syntheticSquare-all-vis-tiny-99.pth \
             --yolo_b_weights /data2/zfei/data/cycleGAN/yolo_model/paper_models/bordenNight/$dataname/checkpoints/best_mAp_yolov3_ckpt.pth \
             --fake_dataset_save_dir /data2/zfei/data/cycleGAN/yolo/paper_dataset/GANGenerated/FakeBordenNight/$dataname/ \
             --validation_path /data2/zfei/data/cycleGAN/yolo/paper_dataset/BordenNight/data_configs/$dataname/valid.txt \
             --test_path /data2/zfei/data/cycleGAN/yolo/paper_dataset/BordenNight/data_configs/$dataname/test.txt &
```

3. Mix generated images into validation
```bash
python -u mix_fake_into_val.py --fake_dataset_save_dir /data2/zfei/data/cycleGAN/yolo/paper_dataset/GANGenerated/mix_val/FakeBordenNight/$dataname/ \
             --validation_path /data2/zfei/data/cycleGAN/yolo/paper_dataset/BordenNight/data_configs/$dataname/valid.txt \
             --test_path /data2/zfei/data/cycleGAN/yolo/paper_dataset/BordenNight/data_configs/$dataname/test.txt
```


#### STEP4 Further Train The YOLO Model using Fake Images and Validate On the Same Val
1. Further train
```bash
# At yolov3 folder
dataname="all"
dataname="train1_val1"
dataname="train4_val1"
dataname="train8_val2"
dataname="train12_val3"
dataname="train16_val4"
dataname="train24_val6"
dataname="train32_val8"
dataname="train40_val10"
dataname="train96_val24"
dataname="trainCycleGAN"

nohup python -u step1-finetuning.py --model_def ./config/yolov3-tiny.cfg \
                           --data_config /data2/zfei/data/cycleGAN/yolo/paper_dataset/GANGenerated/FakeBordenNight/$dataname/data.data \
                           --pretrained_weights /data2/zfei/data/cycleGAN/yolo_model/paper_models/bordenNight/$dataname/checkpoints/best_mAp_yolov3_ckpt.pth \
                           --batch_size=8 \
                           --img_size=416 \
                           --save_dir /data2/zfei/data/cycleGAN/yolo_model/paper_models/bordenNight/GANRefined/$dataname \
                           --checkpoint_interval 10\
                           --epochs=100 &
```

# If use mixed validation
nohup python -u step1-finetuning.py --model_def ./config/yolov3-tiny.cfg \
                           --data_config /data2/zfei/data/cycleGAN/yolo/paper_dataset/GANGenerated/mix_val/FakeBordenNight/$dataname/data.data \
                           --pretrained_weights /data2/zfei/data/cycleGAN/yolo_model/paper_models/bordenNight/$dataname/checkpoints/best_mAp_yolov3_ckpt.pth \
                           --batch_size=8 \
                           --img_size=416 \
                           --save_dir /data2/zfei/data/cycleGAN/yolo_model/paper_models/bordenNight/GANRefined/mix_val/$dataname \
                           --checkpoint_interval 10\
                           --epochs=100 &

3. Run test.py script at yolov3 folder to get test results
```bash
"""
weights_path can be one of the following
/data2/zfei/data/cycleGAN/yolo_model/paper_models/bordenNight/GANRefined/*/checkpoints/best_mAp_yolov3_ckpt.pth
img_size = 416/256
iou_thres = 0.3/0.5
"""
dataname="all"
dataname="train1_val1"
dataname="train4_val1"
dataname="train8_val2"
dataname="train12_val3"
dataname="train16_val4"
dataname="train24_val6"
dataname="train32_val8"
dataname="train40_val10"
dataname="train96_val24"

dataname="trainCycleGAN"
python test.py --model_def ./config/yolov3-tiny.cfg \
                --data_config /data2/zfei/data/cycleGAN/yolo/paper_dataset/BordenNight/data_configs/all/data.data \
                --weights_path /data2/zfei/data/cycleGAN/yolo_model/paper_models/bordenNight/GANRefined/$dataname/checkpoints/best_mAp_yolov3_ckpt.pth \
                --batch_size=8 \
                --img_size=416 \
                --iou_thres=0.5

# Test mixed val
python test.py --model_def ./config/yolov3-tiny.cfg \
                --data_config /data2/zfei/data/cycleGAN/yolo/paper_dataset/BordenNight/data_configs/all/data.data \
                --weights_path /data2/zfei/data/cycleGAN/yolo_model/paper_models/bordenNight/GANRefined/mix_val/$dataname/checkpoints/best_mAp_yolov3_ckpt.pth \
                --batch_size=8 \
                --img_size=416 \
                --iou_thres=0.5 

```

