# Metropolitan Project – VGG16 Image Classifier

## Overview
This is an image classification project using the **VGG16 deep learning model**.  
It classifies images into categories using transfer learning.

## Tech Stack
- Python 3.12.8  
- TensorFlow 2.18.0  
- Keras  
- Git LFS (for large model files)

## Dataset
Images are organized in `train`, `valid`, and `test` folders.  

## Model
- VGG16 pretrained on ImageNet  
- Custom layers added for classification  
- Model file: `vgg16_final_model.h5` (~107 MB, tracked via Git LFS)

## How to Use
1. Clone the repo:
```bash
git clone https://github.com/AhtishamIjaz/VGG16-Image-Classifier-.git
