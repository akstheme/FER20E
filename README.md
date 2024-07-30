# FER20E: Facial Expression Recognition Dataset and Models

Welcome to the FER20E repository, dedicated to advancing research on facial expression recognition using state-of-the-art deep learning models. This repository provides scripts and resources for training and testing various models for facial expression recognition. Below you will find a detailed overview of the contents and usage instructions.

## Contents

- **CNNvsVIT.py**: Script comparing the performance of Convolutional Neural Networks (CNN) and Vision Transformer (ViT) models.
- **DCNN_mobileNet.py**: Implementation of a Deep Convolutional Neural Network (DCNN) using the MobileNet architecture.
- **FER20E_Samples.zip**: A compressed file containing sample images from the FER20E dataset for preliminary testing and evaluation.
- **README.md**: You are currently reading this file! It contains detailed information about the repository.
- **Sample_Test_landmarks.zip**: Sample test landmarks used for evaluating model performance.
- **ViT.py**: Implementation of our proposed Vision Transformer model tailored for facial expression recognition.
- **ViT21k.py**: Utilizes a pretrained Vision Transformer model trained on a standard dataset for comparative analysis.
- **ViT_Basic.py**: A lightweight Vision Transformer model suitable for environments with limited computational resources.
- **model.m**: Our deep convolutional neural network (DCNN) model designed specifically for facial expression recognition.
- **for_single_image.m**: Script for testing a single image against the implemented models.
- **model_test.m**: Script for evaluating a set of test data against all implemented models.

## Models and Scripts

### Vision Transformer Models

- **ViT.py**: This script implements our proposed Vision Transformer (ViT) model. The model is designed specifically for facial expression recognition, leveraging the transformer architecture to capture complex patterns and features in facial images.
- **ViT21k.py**: This script uses a pretrained ViT model, initially trained on a large, standard dataset (21k images). It serves as a benchmark to compare the performance of our custom models.
- **ViT_Basic.py**: A basic and lightweight version of the Vision Transformer model. This version is optimized for environments with limited computational resources while still providing competitive performance.

### Deep Convolutional Neural Network Models

- **DCNN_mobileNet.py**: This script implements a Deep Convolutional Neural Network using the MobileNet architecture. MobileNet is known for its efficiency and speed, making it suitable for mobile and embedded applications.
- **model.m**: This MATLAB script contains our proposed DCNN model, meticulously designed to enhance facial expression recognition accuracy.

### Testing and Evaluation Scripts

- **for_single_image.m**: This MATLAB script allows users to test a single image against the implemented models. It is useful for quick evaluations and demonstrations.
- **model_test.m**: This MATLAB script is designed to evaluate a set of test data. It provides a comprehensive assessment of model performance across multiple images and metrics.

## Dataset

- **FER20E_Samples.zip**: This compressed file contains sample images from the FER20E dataset. The FER20E dataset is a comprehensive collection of over 100,000 facial images representing 20 discrete facial expressions. These samples provide a glimpse into the diversity and complexity of the dataset.
- **Sample_Test_landmarks.zip**: This compressed file contains sample test landmarks. These landmarks are used to evaluate the performance of the models on specific facial features and expressions.

## Usage

### Cloning the Repository

To get started with the FER20E repository, clone the repository to your local machine using the following command:

```bash
git clone https://github.com/akstheme/FER20E.git
cd FER20E
