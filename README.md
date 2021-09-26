# Enlisting 3D Crop Models and GANs for More Data Efficient and Generalizable Fruit Detection

We provide the pytorch implementation of a semantically constrained GAN to generate artificial realisitc fruit images for training to reduce the need of real image labeling in fruit detection.

**[Paper Link](https://arxiv.org/abs/2108.13344)**

<img src='imgs/fig-image-transfer-demo.png'>

## **1. Install Requirements**  
```
pip install -r requirements.txt
```
## **2. Generate Semantic Consistent GAN Fruits**
Use [generateGANImages.ipynb](notebook/generateGANImages.ipynb) notebook to load the pre-trained Semantic Consistent GAN model and generate target domain images from source synthetic image.

## **3. Prepare Data**

### Source domain data  
The dataset used in this research:
1. Synthetic Grape Data

### Target domain data
The dataset used in this research:
1. Night Grape
2. Day Grape









