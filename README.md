# Churn Analysis

Predicts customer churn (Yes/No) from account and usage attributes, and serves the trained model through a FastAPI endpoint.

## Project Overview

This project builds a binary classification model to predict whether a customer will churn, based on features like tenure, monthly/total charges, contract type, payment method, and other account attributes. Several models were compared and the best-performing one (by test AUC) was saved and wrapped in a FastAPI service for real-time predictions.

## Data Preprocessing
 
- Converted whitespace/blank strings to `NaN` and coerced numeric-looking text columns (e.g. `TotalCharges`) to proper numeric types
- Imputed remaining missing numeric values with the column median
- Encoded binary categorical columns (e.g. `gender`, `Churn`) as 0/1
- One-hot encoded nominal categorical columns with more than 2 categories (e.g. `Contract`, `PaymentMethod`) to avoid implying a false ordinal relationship
- Target (`Churn`) mapped as `Yes → 1`, `No → 0`

## Modeling Approach
 
Five model families were compared using `RandomizedSearchCV` (5-fold CV, scored on **ROC-AUC** rather than accuracy, since the target is imbalanced — ~73% no-churn / ~27% churn):
 
| Model | Notes |
|---|---|
| Logistic Regression | `class_weight='balanced'`, scaled via pipeline |
| Random Forest | `class_weight='balanced'`, no scaling needed |
| Gradient Boosting | no scaling needed |
| SVC | scaled via pipeline, reduced search space for speed |
| K-Nearest Neighbors | scaled via pipeline |
 
Feature scaling (`StandardScaler`) was applied only where needed (LogisticRegression, SVC, KNN), and wrapped in a `Pipeline` so scaling is refit within each CV fold — avoiding leakage from validation data into the scaler.
 
The best model was selected by **Test AUC** and saved with `joblib`, along with the training column order (`model_columns.pkl`) needed to correctly align incoming API requests after one-hot encoding.

## Running Locally
 
### 1. Install dependencies
```bash
pip install -r requirements.txt
```
 
### 2. Train the model (or use the pre-trained `churn_model.pkl`)
Run through `DataCleaningandTraining-Final.ipynb` top to bottom to reproduce preprocessing, model comparison, and saving of `churn_model.pkl` / `model_columns.pkl`.
 
### 3. Start the API
```bash
uvicorn main:app --reload
```
Interactive docs available at `http://127.0.0.1:8000/docs`.

## Requirements
 
```
fastapi
uvicorn
scikit-learn
pandas
numpy
joblib
scipy
streamlit
mlflow
```