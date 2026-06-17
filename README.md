# Heartbeat to Heatmap: Heart Disease Prediction, Ensemble Learning & CNN-Based Digit Recognition

## Overview

This project implements a complete machine learning pipeline for healthcare analytics and handwritten digit recognition. The system combines unsupervised learning, ensemble methods, neural networks, computer vision, and interactive deployment techniques to support clinical decision-making and medical data digitization.

Developed as part of **DS-3002 Data Mining – Assignment #4 (Spring 2026)** at FAST-NUCES.

---

## Project Objectives

The project addresses two real-world healthcare AI challenges:

### 1. Heart Disease Prediction

Build interpretable and accurate models that assist cardiologists in identifying patients at risk of heart disease using structured clinical measurements.

### 2. Handwritten Digit Recognition

Develop a lightweight Convolutional Neural Network (CNN) capable of recognizing handwritten digits from patient intake forms.

---

## Datasets

### Dataset 1: UCI Heart Disease (Cleveland)

**Source:** UCI Machine Learning Repository

**Records:** 303 patients

**Features:** 13 clinical attributes + 1 target label

**Task:** Binary Classification

Target Classes:

* 0 → No Heart Disease
* 1 → Heart Disease Present

Clinical Features:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise-Induced Angina
* ST Depression
* Slope of ST Segment
* Number of Major Vessels
* Thalassemia

---

### Dataset 2: MNIST Handwritten Digits

**Source:** TensorFlow/Keras Built-in Dataset

**Training Subset:** 12,000 Images

**Testing Subset:** 2,000 Images

**Image Size:** 28 × 28 Grayscale

**Classes:** Digits 0–9

---


# Part A: Unsupervised Learning

## K-Means Clustering

Performed clustering on standardized patient records to discover hidden clinical subgroups.

### Tasks

* Elbow Method
* Silhouette Analysis
* PCA Visualization
* Cluster Profiling
* Adjusted Rand Index Evaluation

### Clinical Goal

Identify naturally occurring patient risk groups without using disease labels.

---

## Hierarchical Clustering

Ward linkage hierarchical clustering was used to compare cluster structures against K-Means.

### Outputs

* Dendrogram
* Cluster-Label Cross Tables
* Cluster Similarity Analysis

---

## Dimensionality Reduction

### PCA

Used for:

* Variance Analysis
* Feature Compression
* Cluster Visualization

### t-SNE

Used for:

* Non-linear Visualization
* Disease Class Separation Analysis

---

# Part B: Bagging & Boosting

## Random Forest

### Hyperparameter Tuning

* n_estimators = {50, 100, 200}
* max_depth = {None, 5, 10}

### Analysis

* OOB Error Curves
* Feature Importance
* Clinical Interpretation

---

## Gradient Boosting

Implemented using:

* XGBoost

or

* LightGBM

### Features

* Cross Validation
* Early Stopping
* SHAP Explainability
* ROC Analysis

---

## Ensemble Comparison

Compared:

1. Best Part-A Classifier
2. Random Forest
3. XGBoost / LightGBM

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* AUC-ROC

---

# Part C: Artificial Neural Networks

## Single Layer Perceptron (SLP)

Architecture:

```text
Input Layer
     ↓
Sigmoid Output Neuron
```

Purpose:

* Establish linear baseline performance
* Analyze learned feature weights

---

## Multi Layer Perceptron (MLP)

Evaluated multiple architectures:

### Small

```text
Input → Dense(32) → Output
```

### Medium

```text
Input → Dense(64) → Dense(32) → Output
```

### Large

```text
Input → Dense(128) → Dense(64) → Dense(32) → Output
```

### Techniques

* Early Stopping
* Dropout
* L2 Regularization
* Cross Validation

---

## Ablation Study

Evaluated contribution of:

* Dropout
* Activation Functions
* Early Stopping

Purpose:

Understand which architectural decisions most influence performance.

---

# Part D: CNN for Handwritten Digit Recognition

## Baseline MLP

Architecture:

```text
Flatten
 ↓
Dense(64)
 ↓
Softmax(10)
```

Used as a non-convolutional benchmark.

---

## Lightweight CNN

Architecture:

```text
Conv2D(16)
↓
MaxPooling
↓
Conv2D(32)
↓
MaxPooling
↓
Flatten
↓
Dense(64)
↓
Dropout(0.3)
↓
Softmax(10)
```

### Features

* Data Augmentation
* Adam Optimizer
* Confusion Matrix Analysis
* Feature Map Visualization

---

## CNN Interpretability

Visualized:

* Learned Filters
* Convolutional Feature Maps
* Edge and Shape Detection

Purpose:

Understand how CNNs learn visual representations.

---

# Part E: Local Clinical Dashboard

Built using Streamlit.

## Features

### Patient Input Form

* 13 Clinical Variables
* Input Validation
* Sample Patient Preloaded

### Prediction Panel

Displays:

* Disease Prediction
* Confidence Score
* Risk Level
* Top Contributing Features
* Clinical Explanation

### Local Execution

```bash
streamlit run app/app.py
```

---

# Evaluation Metrics

Heart Disease Models:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

CNN Models:

* Accuracy
* Macro F1
* Confusion Matrix

---

# Technologies Used

## Data Science

* Python
* Pandas
* NumPy

## Visualization

* Matplotlib
* Seaborn

## Machine Learning

* Scikit-Learn
* Imbalanced-Learn
* XGBoost / LightGBM
* SHAP

## Deep Learning

* TensorFlow / Keras

## Dashboard

* Streamlit

---

# Installation

```bash
git clone https://github.com/your-username/heartbeat-to-heatmap.git

cd heartbeat-to-heatmap

pip install -r requirements.txt
```

---

# Running the Dashboard

```bash
streamlit run app/app.py
```

---

# Expected Outcomes

The completed pipeline provides:

* Discovery of hidden patient groups
* Accurate heart disease prediction
* Explainable clinical decision support
* Efficient neural network training on CPU
* Automated handwritten digit recognition
* Interactive clinical dashboard for real-time use



FAST-NUCES
