# FER20E: Facial Expression Recognition Dataset and Models

Welcome to the FER20E repository, dedicated to research on facial expression recognition using deep learning models. This repository contains scripts and resources for training and testing various models for facial expression recognition.

## Contents

- **CNNvsVIT.py**: Script comparing CNN and Vision Transformer (ViT) models.
- **DCNN_mobileNet.py**: Deep Convolutional Neural Network (DCNN) using MobileNet architecture.
- **FER20E_Samples.zip**: Sample images from the FER20E dataset.
- **README.md**: You are currently reading this file!
- **Smaple_Test_landmarks.zip**: Sample test landmarks for evaluation.
- **ViT.py**: Proposed Vision Transformer model.
- **ViT21k.py**: Pretrained and standard ViT model.
- **ViT_Basic.py**: Lightweight basic transformer model.
- **model.m**: Proposed DCNN model.
- **for_single_image.m**: Script for testing a single image.
- **model_test.m**: Script for testing a set of test data.

## Models and Scripts

- **ViT.py**: This script implements our proposed Vision Transformer model tailored for facial expression recognition.
- **ViT21k.py**: Utilizes a pretrained ViT model on a standard dataset for comparison.
- **ViT_Basic.py**: A lightweight transformer model suitable for less computational resources.
- **model.m**: Our deep convolutional neural network (DCNN) model designed specifically for FER.
- **for_single_image.m**: Use this script to test individual images against the models.
- **model_test.m**: Script to evaluate a set of test data against all implemented models.

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/akstheme/FER20E.git
   cd FER20E
