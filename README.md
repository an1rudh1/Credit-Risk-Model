# Explainable Credit Risk Assessment

Loan default prediction with XGBoost, calibrated probabilities and SHAP explanations.

**Live demo:** <paste your Streamlit link here>

## What it does
- Takes applicant details and returns a calibrated default probability
- Applies a tuned decision threshold (chosen on out-of-fold predictions)
- Explains each decision with SHAP: which factors raised or lowered the risk

## Model
- Data: public Kaggle credit-risk dataset (~32K applicants, ~22% default)
- Pipeline: median/constant imputation, one-hot encoding, cost-sensitive XGBoost (scale_pos_weight)
- Tuning: RandomizedSearchCV, 150 candidates x 5-fold stratified CV, scored on PR-AUC
- Baseline: Logistic Regression (default-class F1 0.65) vs tuned XGBoost (0.81)
- Probability calibration: Platt scaling
- The dataset used here has loan amount ranging from 500 to 35,000. So this model can predict from 500 to 35,000 only.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

Notebook: Credit_Risk.ipynb
