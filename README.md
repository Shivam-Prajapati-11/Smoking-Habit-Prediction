# Smoker Status Prediction using SVM

A machine learning practical that trains a **Support Vector Machine (SVM)** to
classify whether a person is a smoker (1) or non-smoker (0) based on health
bio-signals such as age, blood pressure, cholesterol, and liver enzymes.

## Dataset
- **Source:** Kaggle — Smoker Status Prediction (Playground Series S3E24)
- **Target:** `smoking` (0 = Non-Smoker, 1 = Smoker)
- **Features:** 22 bio-signal measurements per person

## Workflow
1. Load and split data into features (X) and target (y)
2. Train/test split (80/20, stratified)
3. **Feature scaling** with `StandardScaler` (required for distance-based SVM)
4. Train `SVC` with an **RBF kernel**
5. Evaluate using accuracy, confusion matrix, and classification report

## Tech Stack
Python · pandas · scikit-learn

## Key Concepts
- Support vectors & margin maximization
- RBF kernel for non-linear boundaries
- Hyperparameters: `C` (misclassification penalty)