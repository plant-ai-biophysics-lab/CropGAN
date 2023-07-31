# Stacking Data and Architecture Domain Adaptation to Train Agriculture Models on Synthetic Data

The original CropGAN work **[Paper Link](https://openaccess.thecvf.com/content/ICCV2021W/CVPPA/papers/Fei_Enlisting_3D_Crop_Models_and_GANs_for_More_Data_Efficient_ICCVW_2021_paper.pdf)** focused on data-centric domain adaptation (DA): adapt the synthetic training data to look more like the real test data.

This repo begins with that work, and stacks architecture-based DA on top of it. Architecture-based DA would include any changes to the model (loss function, adversarial training, gradient reversals) to facillitate model generalization from the training (source) domain to the test (target) domain.  Here, we will be using a recent DA technique: Adaptive Teacher.

Adaptive Teacher:  
**[Github Link](https://github.com/facebookresearch/adaptive_teacherhttps://github.com/facebookresearch/adaptive_teacher)**  
**[Paper Link](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Cross-Domain_Adaptive_Teacher_for_Object_Detection_CVPR_2022_paper.pdf)**

## Prerequisites
These probably do not need to be specific, but I am using:

- Python 3.8
- pytorch==1.13.1 
- torchvision==0.14.1
- Detectron2==v0.6 (built from source) - adaptive_teacher uses v0.3, but I am using v0.6 as the dependencies were simpler and it works fine.


## **1. Install Requirements**  
Unfortunately, adaptive_teacher does not have a setup.py, requirements.txt, etc.  So for now, clone the repo to your machine,
`git clone https://github.com/facebookresearch/adaptive_teacher.git` fot HTTPS or
`git clone git@github.com:facebookresearch/adaptive_teacher.git` for SSH.

Then add these lines to `train_net_cropgan.py` or any training .py file:
```
import sys
sys.path.append("/path/to/adaptive_teacher")
```

## **2. Training Adaptive Teacher Model**
### **i. Prepare Data**

Adaptive Teacher expects data in either COCO or VOC formats. To add new datasets to the adaptive teacher training, use `register_coco_instance` function, for example
```register_coco_instances("helios_raw_synthetic", {}, "path/to/annotations/_annotations.coco.json", "path/to/data/train/")```
where `helios_raw_synthetic` is the dataset name to be used in config.  

### **ii. Config**

Adaptive Teacher uses Detectron2's [YACS Config system](https://detectron2.readthedocs.io/en/latest/tutorials/configs.html) which allows configs to inherit from each other with `_BASE_`. An example config for our CropGAN experiments is `faster_rcnn_VGG_CROPGAN_borden_night.yaml`.

### **iii. Train**
Once the dataset is registered and config is written, training can be run by calling `train_net_cropgan.py` from the command line. An example:
```python train_net_cropgan.py --num-gpus 1 --config-file configs/faster_rcnn_VGG_CROPGAN_borden_night.yaml ```

## **2. Using Adaptive Teacher Model**
To run evaluation, make sure that the config file has a path at `MODEL.WEIGHTS` to the model weights, and pass the `--eval-only` flag when calling `train_net_cropgan.py`:
```python train_net_cropgan.py --eval-only --num-gpus 1 --config-file configs/faster_rcnn_VGG_CROPGAN_borden_night.yaml ```

