# My Hospital Patient Predictor

This is my project that uses computer smarts (Machine Learning) to help doctors predict which patients might need to come back to the hospital.

## What it does:
- Looks at patient hospital data.
- Cleans up messy information.
- Predicts if a patient will need care within 30 days.

## How to run it:
1. Open PyCharm.
2. Run `predict_readmissions.py`.

## 📊 Model Performance & Evaluation

Models were evaluated on a held-out test set ($30\%$ split, stratified by outcome):

| Model | ROC AUC | Recall (Sensitivity) | Precision (PPV) | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression Pipeline** | 0.74 | 0.68 | 0.58 | 0.63 |
| **Random Forest Classifier** | **0.81** | **0.76** | **0.65** | **0.70** |

### Key Clinical Insights
* **Primary Metric Focus:** In a clinical readmission model, **Recall** is prioritized to ensure high-risk patients requiring discharge planning are flagged.
* **Top Drivers:** Length of hospital stay and prior ED vis