# 🚗 Car Classification using SIFT + BoW + SVM

This project uses Bag-of-Words model with SIFT descriptors and an SVM classifier to detect whether an image contains a car or not. It is trained and tested on a custom dataset using OpenCV in Python.

## 🔧 Features
- Loads grayscale images from labeled folders (car / non-car)
- Extracts local features using SIFT
- Builds a visual vocabulary with KMeans
- Trains a linear SVM on BoW features
- Evaluates accuracy on test data


## 🔧 Methodology

1. **Preprocessing**  
   - Resize images to 128×128  
   - Convert to grayscale  
   - Filter out images with too few keypoints

2. **Feature Extraction**  
   - Detect keypoints using SIFT  
   - Cluster descriptors using KMeans to create BoW vocabulary  
   - Extract BoW histograms for training and testing

3. **Classification**  
   - Train a linear SVM classifier  
   - Predict and evaluate test samples

4. **Visualization**  
   - Overlay predictions on images  
   - Display random samples from the test set

---


## 📁 Dataset
Dataset is mounted from Google Drive:
- `/content/drive/MyDrive/car_dataset/Train/car`
- `/content/drive/MyDrive/car_dataset/Test/non-car`

> Note: The dataset is not included in this repository due to size limits.

## 🧠 Model Accuracy
## 📈 Experiments

We tested different numbers of visual words (KMeans clusters) to evaluate their impact on accuracy.

| Visual Words (k) | Accuracy on Test Set |
|------------------|----------------------|
| 100              | 74.00%               |
| 200              | 81.31%              |

## 📔 Notebook
[🔗 Open in Colab](https://colab.research.google.com/github/snz-mlcoder/image-processing-projects/blob/main/car-classification-bow-sift/car_classifier.ipynb)
