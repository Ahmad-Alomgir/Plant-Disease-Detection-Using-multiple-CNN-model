# Plant-Disease-Detection-Using-multiple-CNN-model

## Overview

This project implements multiple **Convolutional Neural Network (CNN)** models for classifying plant diseases into 15 different categories. The models are trained using the **PlantVillage Dataset**, which contains images of plants suffering from various diseases. The goal is to compare different CNN architectures, including pre-trained models, custom CNNs with attention mechanisms, and CNNs with regularization techniques, to find the optimal configuration for plant disease detection.

## Dataset

The **PlantVillage dataset** is used for training and testing the models. The dataset contains labeled images of plants, each categorized by its disease or as healthy.

- **Classes**: 15 different plant disease categories (e.g., Tomato Bacterial Spot, Potato Early Blight, etc.)
- **Dataset Source**: [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Image Dimensions**: 128x128 pixels (resized for training)
- **Input Type**: RGB images (with the exception of the Simple CNN model, which uses grayscale images)

## Models Implemented

### 1. ResNet50V2
- **Description**: A **pre-trained ResNet50V2** model used for feature extraction, followed by custom dense layers for classification.
- **Training Accuracy**: **89%**
- **Key Features**: Transfer learning, feature extraction, efficient architecture.

### 2. Custom CNN (Scratch)
- **Description**: A custom CNN built from scratch with two **Conv2D** + **MaxPooling** layers, followed by dense layers for classification.
- **Training Accuracy**: **71%**
- **Key Features**: Lightweight, no pre-training, faster training.

### 3. Self-Attention CNN
- **Description**: A custom CNN with **multi-head self-attention** and **skip connections** to focus on important regions of the image.
- **Training Accuracy**: **91%**
- **Key Features**: Self-attention mechanism, skip connections, better feature learning.

### 4. Cross-Attention CNN
- **Description**: A custom CNN with a **cross-attention mechanism**, separate **query** and **key-value projections**, followed by **global average pooling** and dense layers.
- **Training Accuracy**: **88%**
- **Key Features**: Cross-attention mechanism, separate projections for query/key-value, improved performance.

### 5. Regularized CNN
- **Description**: A CNN with **three convolutional layers**, **batch normalization**, **dropout**, and **softmax output** for multi-class classification.
- **Training Accuracy**: **92%**
- **Key Features**: Regularization, batch normalization, dropout to prevent overfitting.

### 6. Simple CNN (Gray-Scale)
- **Description**: A simple CNN using **gray-scale images** as input, with three **Conv2D** layers.
- **Training Accuracy**: **66.57%**
- **Key Features**: Simplified architecture using gray-scale images, reduced input complexity.

## Model Comparison

| **Model**                  | **Accuracy (%)** | **F1-Score (Tomato)** | **F1-Score (Potato)** | **F1-Score (Pepper)** | **Notes**                       |
|----------------------------|------------------|-----------------------|-----------------------|-----------------------|---------------------------------|
| **ResNet50V2**              | 89               | 0.90                  | 0.88                  | 0.87                  | Pre-trained model, transfer learning |
| **Custom CNN (Scratch)**    | 71               | 0.65                  | 0.70                  | 0.60                  | Custom CNN built from scratch, faster training |
| **Self-Attention CNN**      | 91               | 0.92                  | 0.89                  | 0.90                  | Multi-head self-attention, skip connections |
| **Cross-Attention CNN**     | 88               | 0.85                  | 0.89                  | 0.83                  | Cross-attention mechanism, improved feature focus |
| **Regularized CNN**         | 92               | 0.93                  | 0.90                  | 0.91                  | Regularization, batch normalization, dropout |
| **Simple CNN (Gray-Scale)** | 66.57            | 0.60                  | 0.65                  | 0.58                  | Simplified model using gray-scale images |

### Accuracy Comparison:
- **Regularized CNN** achieved the highest accuracy (**92%**), followed by **Self-Attention CNN** (**91%**), and **ResNet50V2** (**89%**).
- The **Simple CNN (Gray-Scale)** performed the worst, with an accuracy of **66.57%**, highlighting the importance of using RGB images for better plant disease detection.

## Installation and Setup

### Prerequisites

Ensure you have the following installed:
- Python 3.7+
- TensorFlow 2.x
- Keras
- Matplotlib
- NumPy
- Pandas

### Installing Dependencies

You can install the required libraries by running:
```bash
pip install -r requirements.txt
