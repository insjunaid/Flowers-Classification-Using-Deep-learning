# Flower Classification Using Deep Learning
![Screenshot 2024-12-15 173452](https://github.com/user-attachments/assets/11c6dc20-8b31-4a0f-b516-97537ad6e2d4)

This repository contains a PyTorch-based implementation for classifying flower images into 17 categories. The project utilizes a pre-trained EfficientNet-B0 model fine-tuned for flower classification, and it also provides a Flask web application where users can upload images to obtain predictions.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Model Architecture](#model-architecture)
- [Training](#training)
  - [Data Augmentation](#data-augmentation)
- [Evaluation](#evaluation)
  - [Confusion Matrix](#confusion-matrix)
  - [Classification Report](#classification-report)
- [Web Application](#web-application)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction
Flower classification is a common problem in computer vision, and this project aims to classify flower images into 17 distinct categories. By using deep learning, the model assists in identifying flowers based on their image features. The trained model is deployed in a Flask web application for ease of use by non-technical users.

## Dataset
The dataset used for training consists of flower images, categorized into the following 17 classes:
- Bluebell
- Buttercup
- Colts Foot
- Cowslip
- Crocus
- Daffodil
- Daisy
- Dandelion
- Fritillary
- Iris
- Lily Valley
- Pansy
- Snowdrop
- Sunflower
- Tigerlily
- Yellow Tulip
- Windflower

The dataset is split into training (70%), validation (15%), and testing (15%) sets.

## Requirements
Dependencies:
- Python 3.8+
- PyTorch
- Torchvision
- Flask
- Pillow
- scikit-learn
- Matplotlib
- Seaborn
- tqdm

Install dependencies using:
```bash
pip install -r requirements.txt

```
## Model Architecture
The model is based on EfficientNet-B0, pretrained on ImageNet. The classifier head is modified as follows:

```python
nn.Linear(in_features, 512)
ReLU()
Dropout(0.4)
nn.Linear(512, 17)
```
## Training

### Data Augmentation
Training data is augmented with transformations including:
- Random Resized Crop
- Horizontal Flip
- Random Rotation
- Color Jitter
- Normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Evaluation

### Confusion Matrix
A confusion matrix is plotted to analyze performance across the 17 flower classes.

### Classification Report
The classification report includes precision, recall, and F1-score for each class.

## Web Application
The Flask application allows users to upload flower images and receive predictions. Features include:
- Displaying the predicted flower class
- Returning the uploaded image alongside the prediction

Run the Flask app:
```bash
python app.py
```
Access the app at `http://127.0.0.1:5000/`.

## Results
- **Test Accuracy**: Achieved high accuracy on the test set.
- **Confusion Matrix**: Provides insights into misclassified categories.

## Conclusion
This project demonstrates the effectiveness of deep learning in classifying flower species from images. The model provides accurate predictions, and the Flask web application offers a user-friendly interface for deployment.

Contributors

Junaid Shariff

