# Stacking Data and Architecture Domain Adaptation to Train Agriculture Models on Synthetic Data

We provide the pytorch implementation of a semantically constrained GAN to generate artificial realisitc fruit images for training to reduce the need of real image labeling in fruit detection.

**[Github](https://github.com/facebookresearch/adaptive_teacherhttps://github.com/facebookresearch/adaptive_teacher)**
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

## **2. Training Adaptive Teacher Model**
[Michael to add this]

## **2. Using Adaptive Teacher Model**
[Michael to add this]

## **3. Prepare Data**

Adaptive Teacher expects data in either COCO or VOC formats. [Add detail here]