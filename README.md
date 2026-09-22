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

## Model Comparison Results

Five models were evaluated using `RandomizedSearchCV` (5-fold CV, scored on ROC-AUC) to predict customer churn. Results are ranked by Test AUC.

| Rank | Model | Best CV AUC | Test AUC | Best Parameters |
|---|---|---|---|---|
| 1 | **Gradient Boosting** | 0.924 | **0.932** | `learning_rate=0.0119`, `max_depth=5`, `n_estimators=287` |
| 2 | Logistic Regression | 0.919 | 0.927 | `C=0.0746`, `penalty='l2'` |
| 3 | SVC | 0.918 | 0.926 | `C=21.37`, `kernel='linear'` |
| 4 | Random Forest | 0.918 | 0.923 | `max_depth=10`, `min_samples_split=9`, `n_estimators=288` |
| 5 | KNN | 0.886 | 0.887 | `n_neighbors=23`, `weights='uniform'` |

### Key observations

- **Gradient Boosting performed best** (Test AUC 0.932), with a low learning rate (0.012) compensated by a high estimator count (287) and moderate depth (5) — a "slow and steady" configuration that typically generalizes well rather than overfitting.
- **Logistic Regression, SVC, and Random Forest cluster tightly** (0.923–0.927 Test AUC). Notably, SVC's best kernel was **linear**, not RBF — reinforcing that the churn signal in this data is close to linearly separable, since a linear decision boundary (Logistic Regression, linear-kernel SVC) performs nearly as well as non-linear tree ensembles.
- **Logistic Regression's low C (0.0746)** indicates strong regularization was preferred — the search found that a simpler, more constrained linear model generalized better than a looser one, consistent with this being a fairly linearly-separable problem.
- **KNN trails clearly** (0.887), needing a large neighborhood (`n_neighbors=23`) with uniform weighting just to stabilize — a sign the raw feature space (particularly after one-hot encoding) isn't well-suited to distance-based classification.
- **CV and Test AUC stay close for every model** (largest gap ~0.008, Random Forest), indicating no meaningful overfitting in the final selected hyperparameters for any model.

### Selected model
**Gradient Boosting** was saved as the production model (`churn_model.pkl`) based on highest Test AUC (0.932).

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