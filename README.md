Pneumonia Classification from Chest X-rays using CNN

https://img.shields.io/badge/Pneumonia-Detection-red
https://img.shields.io/badge/Deep-Learning-blue
https://img.shields.io/badge/CNN-Architecture-green

This project implements a Convolutional Neural Network (CNN) to classify chest X-ray images as either normal or showing signs of pneumonia. The model achieves approximately 90% accuracy in detecting pneumonia from radiographic images.
Table of Contents

    Project Overview

    Dataset

    Model Architecture

    Results

    Installation

    Usage

    Contributing

    License

Project Overview

This deep learning project:

    Automatically downloads the chest X-ray dataset from Kaggle using Kaggle API

    Implements a CNN model with data augmentation

    Evaluates model performance on test data

    Provides visualization of training metrics

Dataset

The dataset comes from Kaggle and contains:

    5,863 chest X-ray images (JPEG)

    Two classes: Normal and Pneumonia

    Split into training, validation and test sets

Dataset source: Chest X-Ray Images (Pneumonia) on Kaggle
Model Architecture

The CNN model consists of:

    4 convolutional layers with ReLU activation

    Max pooling layers for dimensionality reduction

    Fully connected layers for classification

    Sigmoid activation for binary classification

python

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

Results

The model achieves:

    Training accuracy: ~95%

    Validation accuracy: ~90%

    Test accuracy: ~90%

Training history visualization:
https://images/training_metrics.png
