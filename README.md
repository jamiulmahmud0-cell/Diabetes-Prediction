# Diabetes-Prediction

This GitHub repository contains a machine learning project for predicting diabetes using the Pima Indians Diabetes dataset. It includes data preprocessing, model training with traditional and deep learning approaches, hyperparameter tuning, feature importance analysis, explainability with SHAP, and a deployable prototype using Streamlit. The core implementation is in a Colab notebook, making it easy to replicate results and extend experiments. This updated README reflects advancements from the interim report, incorporating optimized models (e.g., LightGBM), interpretability enhancements, and a functional deployment interface.

## Key Points

- **Overview**: The project compares models including Decision Tree, SVM, LSTM, GRU, Random Forest, XGBoost, and LightGBM on the Pima dataset, achieving ROC-AUC scores of 0.80-0.84, with tuned ensembles leading performance. New additions like LightGBM and SHAP enhance robustness and interpretability.
- **Dataset**: Sourced from UCI, it includes 768 samples of Pima Indian women with 8 clinical attributes predicting diabetes onset. Note potential biases due to population specificity and class imbalance (~65% non-diabetic).
- **Setup and Run**: Use Google Colab for the notebook; install dependencies via pip for local runs. Results are reproducible with provided seeds and saved artifacts.
- **Insights**: Glucose remains the top predictor across models, followed by BMI and Age. Ensembles outperform deep learning on small tabular data, though DL shows potential with optimization.
- **Limitations**: Models are exploratory—consult healthcare professionals for real-world use. Small dataset size risks overfitting; external validation on diverse cohorts is needed.
- **Improvements**: Added Optuna tuning, LightGBM, SHAP explainability, and Streamlit deployment since the interim report, addressing gaps in performance and practical applicability.

## Installation

Clone the repo and install requirements:

    git clone jamiulmahmud0-cell/Diabetes-Prediction
    
    cd diabetes-prediction
    
    pip install -r requirements.txt

**Requirements** include pandas, numpy, scikit-learn, torch, xgboost, optuna, lightgbm, matplotlib, shap, streamlit, and joblib.

## Usage

Open `diabetes_prediction.ipynb` in Colab. Run cells sequentially:

- **Preprocessing (Cell 1)**: Loads data, replaces invalid zeros with NaN, imputes with KNN (n=5), scales features.
- **Train Baseline Models (Cells 2-6)**: DT, SVM, LSTM, GRU, RF, XGBoost with default settings.
- **Tune with Optuna (Cell 7)**: Optimizes RF, XGBoost, and later LightGBM.
- **Analyze Features (Cells 4, 8, 16)**: Plots DT importances, multi-model summary, SHAP visualizations.
- **Prototype Deployment (Cell 15)**: Runs Streamlit app for real-time predictions.

**Dataset**: `diabetes.csv` is included; load directly in the notebook.

## Results Summary

Ensembles like Random Forest (baseline ROC-AUC ~0.82, tuned ~0.84) and LightGBM (tuned ~0.84) outperform others. Tuned models show 70-76% accuracy, with AUC gains of 0.02-0.04 over interim baselines (0.80-0.82), reflecting tuning and ensemble strengths on this small, imbalanced dataset.

## Project Motivation and Structure

Motivated by the global rise in diabetes cases, this project benchmarks predictive models on a classic dataset, emphasizing ensembles' advantages for small, imbalanced data while exploring deep learning's potential for pattern capture. It tackles ML challenges like missing data and overfitting, drawing from best practices in GitHub ML repos (e.g., structured notebooks, artifact directories). Updates from the interim include enhanced optimization, interpretability, and deployment.

### Repo Structure

- `/data/`: Contains `diabetes.csv` (original dataset).
- `diabetes_prediction.ipynb`: Main Colab notebook with cells 1-17.
- `requirements.txt`: List of dependencies.
- `/artifacts/`: Stores saved models, SHAP plots, and summaries.
- `README.md`: This file.

## Dataset Details

The Pima Indians Diabetes dataset, from the National Institute of Diabetes and Digestive and Kidney Diseases via UCI Machine Learning Repository, includes 768 instances of female patients of Pima Indian heritage (aged 21+), predicting diabetes onset within five years.

### Features

- **Pregnancies**: Number of times pregnant.
- **Glucose**: Plasma glucose concentration (2-hour oral glucose tolerance test).
- **BloodPressure**: Diastolic blood pressure (mm Hg).
- **SkinThickness**: Triceps skin fold thickness (mm).
- **Insulin**: 2-Hour serum insulin (mu U/ml).
- **BMI**: Body mass index (weight in kg/(height in m)^2).
- **DiabetesPedigreeFunction**: Diabetes pedigree function (genetic score).
- **Age**: Age in years.

### Outcome
- **Class variable**: 0 (non-diabetic), 1 (diabetic).

### Key Notes
- Class imbalance (~65% non-diabetic) and invalid zeros (e.g., Glucose, Insulin) are handled via KNN imputation.
- Source citation: Smith et al. (1988).
- Usage: Ideal for binary classification benchmarks, but generalize cautiously due to demographic specificity.

## Dependencies and Environment

Run in Python 3.12+ (tested in Colab). Key libraries:

- **Data**: pandas, numpy.
- **ML**: scikit-learn (imputation, scaling, DT, SVM, RF), xgboost, lightgbm.
- **DL**: torch (LSTM, GRU).
- **Tuning**: optuna.
- **Viz**: matplotlib, seaborn.
- **Explainability**: shap.
- **Deployment**: streamlit.
- **Utils**: joblib (model saving).

## Step-by-Step Usage Guide

### Setup
- Clone repo or open in Colab via GitHub link. Upload `diabetes.csv` if not present.

### Preprocessing (Cell 1)
- Loads data, replaces invalid zeros with NaN, imputes using KNN (n=5), scales features with StandardScaler. Outputs: Imputed DataFrame, scaled X, y.

### Baseline Models
- **Cell 2**: DT and SVM on 70/30 split; generates classification reports and confusion matrices.
- **Cell 3**: LSTM with PyTorch; trains with early stopping (20 epochs), saves best model.
- **Cell 5**: GRU similar to LSTM; includes device check (CPU/GPU).
- **Cell 6**: RF and XGBoost on 75/25 split; saves models and JSON summary.

### Tuning and Analysis
- **Cell 4**: DT feature importance bar plot.
- **Cell 7**: Optuna tunes RF/XGB (20 trials, 4-fold CV); saves studies.
- **Cell 8**: Multi-model importances (DT, RF, XGB, permutation for SVM/RF); JSON summary.
- **Cell 10**: Grid tunes LSTM/GRU; reports best Val AUC.
- **Cell 11**: Initial LightGBM training.
- **Cell 16**: SHAP explainability for XGB/LightGBM; generates beeswarm and bar plots.

### Visualization and Deployment
- **Cell 17**: Plots ROC-AUC bar comparison, XGB confusion heatmap, combined ROC curves.
- **Cell 15**: Streamlit app for real-time predictions; loads pre-trained model and preprocessors.

### Reproducing Results
- Use `random_state=42` for consistency. Artifacts in `/artifacts/` allow loading pre-trained models.

### Customization
- Adjust splits (e.g., 80/20), epochs, or add data augmentation for experiments. Modify `requirements.txt` for additional libraries.

## Extended Results and Insights

Since the interim report, the pipeline has evolved with hyperparameter optimization using Optuna, enhancing RF (tuned AUC 0.8289), XGB (0.8271), and introducing LightGBM (tuned AUC 0.8416). LSTM and GRU tuning yielded Val AUCs of 0.8101 and 0.8107, respectively, showing DL's limits on small data. SHAP analysis confirms Glucose as the dominant predictor, with positive impacts for high values, followed by BMI and Age, aligning with clinical expectations. The Streamlit prototype now supports user inputs (e.g., Glucose, BMI) and outputs probability scores, bridging research to practice.

### Final Performance Highlights
- **Tuned RF**: Accuracy 0.7552, F1 (class 1) 0.6179, AUC 0.8260.
- **Tuned XGB**: Accuracy 0.7552, F1 (class 1) 0.6240, AUC 0.8271.
- **Tuned GRU**: Accuracy 0.7135, F1 (class 1) 0.6154, AUC 0.8010.
- **Tuned LightGBM**: Accuracy 0.7604, F1 (class 1) 0.6515, AUC 0.8416.

### Visualization Enhancements
- **ROC-AUC Barplot**: LightGBM leads at ~0.84.
- **XGB Confusion Heatmap**: [[103,22],[24,43]] shows balanced detection.
- **Combined ROC Curves**: LightGBM AUC 0.8059 highlights its edge.

## Future Work

- Validate on larger, diverse datasets (e.g., NHANES) to reduce overfitting.
- Integrate real-time data feeds via APIs in the Streamlit app.
- Explore hybrid models (e.g., CNN-LSTM) for sequential feature patterns.

## Acknowledgments

Inspired by ML communities on GitHub and research from Kavakiotis et al. (2025) and others. Dataset credit: UCI Machine Learning Repository.
