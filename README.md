# Enlisting 3D Crop Models and GANs for More Data Efficient and Generalizable Fruit Detection

We provide the pytorch implementation of a semantically constrained GAN to generate artificial realisitc fruit images for training to reduce the need of real image labeling in fruit detection.

**[Paper Link](https://openaccess.thecvf.com/content/ICCV2021W/CVPPA/papers/Fei_Enlisting_3D_Crop_Models_and_GANs_for_More_Data_Efficient_ICCVW_2021_paper.pdf)**

<img src='imgs/fig-image-transfer-demo.png'>

## **1. Install Requirements**  
```
pip install -r requirements.txt
```
## **2. Using Trained CropGAN Model**
Use [generateGANImages.ipynb](notebook/generateGANImages.ipynb) notebook to load the pre-trained Semantic Consistent GAN model and generate target domain images from source synthetic image.
  
## **3. Prepare Data**

We provide our data in this [CropGANData](https://github.com/plant-ai-biophysics-lab/CropGANData) repo as an example, you can prepare your own dataset following this format. 

### **Source domain (domain A) data**
Crop images + Bounding box labels for each image  
The dataset used in this research (you can use any domain data even its not synthetically generated, as long as you have labels):
1. Synthetic Grape Data

### **Target domain (domain B) data**
Images (with a few of them labeled, 1 at least), source and target image do not need to be paird.
The dataset used in this research:
1. Night Grape
2. Day Grape

### **Data organization**
Data used to train CropGAN
```bash
crop_gan_data
└── sytheticVis2bordenNight
    ├── labelA # (labels for domain A images)
    ├── trainA # (domain A images)
    └── trainB # (domain B images)
```
Data used to train object detection model
```bash
detection_datasets
└── TargetDomainData
    ├── class.txt
    ├── data_configs
    ├── test
    ├── train
    └── valid
```
## **3. Train the model**
#### STEP1 Finetuning using N train image and K validation image
1. Run step1-finetuning.py script at yolov3 folder  
```bash
"""
If not using synthetic pretrained
--model_def ./config/yolov3-tiny.cfg
--pretrained_weights /data2/zfei/data/cycleGAN/yolo/weights/yolov3-tiny.weights
--model_def ./config/yolov3.cfg
--pretrained_weights /data2/zfei/data/cycleGAN/yolo/weights/darknet53.conv.74
"""
# e.g ~/CropGANData/detection_datasets/BordenNight/
data_path="/home/zhenghaofei/Documents/CropGANData/detection_datasets/BordenNight/"
save_dir="/home/zhenghaofei/Documents/CropGANData/output/BordenNight"
pretrained_weights="/home/zhenghaofei/Downloads/yolo_model/yolov3-tiny.weights"
# You can change to other data config by change dataname="traina_valb"
dataname="train1_val1"

cd yolov3
python -u step1-finetuning.py --model_def ./config/yolov3-tiny.cfg \
                           --data_config $data_path/data_configs/$dataname/data.data \
                           --pretrained_weights $pretrained_weights \
                           --batch_size=8 \
                           --img_size=416 \
                           --save_dir $save_dir/$dataname \
                           --checkpoint_interval 10\
                           --epochs=100
```

## Funding
This project was partly funded by the [National AI Institute for Food Systems (AIFS)](https://aifs.ucdavis.edu).





