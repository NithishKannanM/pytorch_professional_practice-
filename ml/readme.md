# Diabetes Prediction using PyTorch

A binary classification model built with PyTorch to predict diabetes outcomes using the diabetes dataset. The model uses a neural network with batch normalization and dropout for improved performance on imbalanced data.

## Project Overview

This project implements a deep learning classifier that predicts whether a patient has diabetes based on medical features including:
- Pregnancies
- Glucose levels
- Insulin levels
- BMI (Body Mass Index)
- Age

## Features

- **Data Preprocessing**: StandardScaler normalization for feature scaling
- **Class Imbalance Handling**: Weighted random sampling and pos_weight in loss function
- **Neural Network Architecture**: 
  - Input layer: 5 features
  - Hidden layers: 64 → 32 neurons with BatchNorm1d and Dropout(0.2)
  - Output layer: 1 neuron (binary classification)
- **Loss Function**: BCEWithLogitsLoss with positive weight for handling class imbalance
- **Optimizer**: Adam optimizer with learning rate 0.001

## Model Architecture

```
Input (5 features)
    ↓
Linear(5 → 64) + BatchNorm1d(64) + ReLU + Dropout(0.2)
    ↓
Linear(64 → 32) + BatchNorm1d(32) + ReLU + Dropout(0.2)
    ↓
Linear(32 → 1)
    ↓
Output (Binary classification)
```

## Requirements

- pandas
- numpy
- torch
- scikit-learn
- seaborn
- matplotlib

## Usage

1. Ensure the dataset is placed at `dataset/diabetes.csv`
2. Run the Jupyter notebook `linearregression.ipynb` cell by cell
3. The model trains for 500 epochs and prints loss every 100 epochs
4. Evaluation metrics (Accuracy, Precision, Recall, F1 Score) are displayed

## Training Process

- **Batch Size**: 32
- **Epochs**: 500
- **Loss Function**: BCEWithLogitsLoss with positive weight balancing
- **Data Split**: 80% training, 20% testing (stratified)

## Evaluation Metrics

The model evaluates performance using:
- **Accuracy**: Overall correctness
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1 Score**: Harmonic mean of precision and recall

## Visualization

The project includes:
- Correlation heatmap of features
- Class distribution plot
- Train vs Validation loss plot

## Results

Training and validation loss curves are plotted to visualize model convergence and detect overfitting.

## Author

Created for diabetes prediction classification task using deep learning.
