# Diabetes-Prediction
This GitHub repository contains a machine learning project for predicting diabetes using the Pima Indians Diabetes dataset. It includes data preprocessing, model training with traditional and deep learning approaches, hyperparameter tuning, and feature importance analysis. The core implementation is in a Colab notebook, making it easy to replicate results.
Key Points

Overview: The project compares models like Decision Tree, SVM, LSTM, GRU, Random Forest, and XGBoost on a benchmark diabetes dataset, achieving ROC-AUC scores around 0.80-0.84 with ensembles performing best.
Dataset: Sourced from UCI, it features 768 samples of Pima Indian women, with 8 clinical attributes predicting diabetes onset—widely used but note potential biases in population-specific data.
Setup and Run: Use Google Colab for the notebook; install dependencies via pip for local runs. Results are reproducible with provided seeds.
Insights: Glucose emerges as the top predictor across models, followed by BMI and Age; deep learning shows promise but may overfit on small datasets.
Limitations: Models are exploratory—consult healthcare professionals for real applications; further validation on diverse datasets is recommended.

Installation
Clone the repo and install requirements:
    git clone https://github.com/yourusername/diabetes-prediction.git
    cd diabetes-prediction
    pip install -r requirements.txt

Requirements include pandas, numpy, scikit-learn, torch, xgboost, optuna, matplotlib.
Usage
Open diabetes_prediction.ipynb in Colab. Run cells sequentially:

Preprocess data (Cell 1).
Train baseline models (Cells 2-6).
Tune with Optuna (Cell 7).
Analyze features (Cells 4, 8).

Dataset: diabetes.csv is included; load directly in the notebook.
Results Summary
Ensembles like Random Forest (ROC-AUC ~0.82) and tuned versions (~0.84) outperform others, but all models hover around 70-80% accuracy due to dataset size and imbalance.

This repository hosts a comprehensive machine learning pipeline for diabetes prediction, leveraging the Pima Indians Diabetes dataset to evaluate a range of models from traditional classifiers to deep neural networks. Built primarily in a Jupyter/Colab notebook, it demonstrates end-to-end workflow: data loading, preprocessing (handling invalid zeros with KNN imputation and standardization), model training, evaluation, hyperparameter optimization via Optuna, and interpretability through feature importance visualizations. The project is designed for reproducibility, with random seeds and artifact saving for models and summaries.
Project Motivation and Structure
Motivated by the global rise in diabetes cases, this project aims to benchmark predictive models on a classic dataset, highlighting strengths of ensembles for small, imbalanced data while exploring deep learning's potential for pattern capture. It addresses common ML challenges like missing data and overfitting, drawing from best practices in GitHub ML repos (e.g., structured notebooks, artifact directories).
The repo structure:

/data/: Contains diabetes.csv (original dataset).
/artifacts/: Saved models (e.g., rf_baseline.joblib), plots (e.g., featimp_rf.png), and summaries (e.g., rf_xgb_baseline_summary.json).
diabetes_prediction.ipynb: The main Colab notebook with cells 1-8.
requirements.txt: List of dependencies.
README.md: This file.

Dataset Details
The Pima Indians Diabetes dataset originates from the National Institute of Diabetes and Digestive and Kidney Diseases, via UCI Machine Learning Repository. It includes 768 instances of female patients of Pima Indian heritage (aged 21+), with the goal of predicting diabetes onset within five years.
Features:

Pregnancies: Number of times pregnant.
Glucose: Plasma glucose concentration (2-hour oral glucose tolerance test).
BloodPressure: Diastolic blood pressure (mm Hg).
SkinThickness: Triceps skin fold thickness (mm).
Insulin: 2-Hour serum insulin (mu U/ml).
BMI: Body mass index (weight in kg/(height in m)^2).
DiabetesPedigreeFunction: Diabetes pedigree function (genetic score).
Age: Age in years.
Outcome: Class variable (0: non-diabetic, 1: diabetic).

Key notes: The dataset has class imbalance (~65% non-diabetic) and invalid zeros in features like Glucose and Insulin, handled via KNN imputation in the code. Source citation: Smith et al. (1988). Usage: Ideal for binary classification benchmarks, but generalize cautiously due to demographic specificity.
Dependencies and Environment
Run in Python 3.12+ (tested in Colab). Key libraries:

Data: pandas, numpy.
ML: scikit-learn (imputation, scaling, DT, SVM, RF), xgboost.
DL: torch (LSTM, GRU).
Tuning: optuna.
Viz: matplotlib.
Utils: joblib (model saving).

Install via pip install -r requirements.txt. No internet needed post-setup, except for initial pip. For GPU acceleration in DL models, ensure CUDA if local.
Step-by-Step Usage Guide

Setup: Clone repo or open in Colab via GitHub link. Upload diabetes.csv if not present.
Preprocessing (Cell 1): Loads data, replaces invalid zeros with NaN, imputes using KNN (n=5), scales features. Outputs: Imputed DF, scaled X, y.
Baseline Models:

Cell 2: DT and SVM on 70/30 split; reports classification metrics and matrices.
Cell 3: LSTM with PyTorch; trains with early stopping, saves best model.
Cell 5: GRU similar to LSTM; includes device check (CPU/GPU).
Cell 6: RF and XGBoost on 75/25 split; saves models and JSON summary.


Tuning and Analysis:

Cell 4: DT feature importance plot.
Cell 7: Optuna for RF/XGB (20 trials, cross-validated AUC); saves studies.
Cell 8: Multi-model importances (DT, RF, XGB, permutation for SVM/RF); JSON summary.


Reproducing Results: Use random_state=42 for consistency. Artifacts in /artifacts/ for loading pre-trained models.
Customization: Adjust splits, epochs, or add data augmentation for experiments.
