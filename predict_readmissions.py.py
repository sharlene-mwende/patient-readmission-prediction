import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import display
file_path=r"C:\Users\Dr. Sharlene\Downloads\diabetic_data (3).csv"
df=pd.read_csv(file_path)
df_clean=df.copy()
df_clean['race']=df_clean['race'].replace('?',np.nan)
df_clean['weight']=df_clean['weight'].replace('?',np.nan)
most_common_race=df_clean['race'].mode()[0]
df_clean['race']=df_clean['race'].fillna('most_common_race')
df_clean['medical_specialty']=df_clean['medical_specialty'].ffill()
print(df_clean['medical_specialty'].value_counts(dropna=False))
for col in df_clean.select_dtypes(include=['object', 'string']).columns:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        numeric_cols_to_convert = ['time_in_hospital', 'num_procedures', 'num_lab_procedures',
                                   'num_medications', 'number_outpatient', 'number_emergency',
                                   'number_inpatient']
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

print("✓ Missing values handled (median for numeric, mode for categorical)")
print("\nFirst 5 rows:")
display(df_clean.head(5))
print("\nTarget (readmitted):")

df_clean['readmitted_30d'] = (df_clean['readmitted'] == '<30').astype(int)
x=df_clean[['time_in_hospital', 'num_procedures']]
y = (df_clean['readmitted'] == '<30').astype(int)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(class_weight='balanced', random_state=42)
model.fit(x_train,y_train)
from sklearn.metrics import accuracy_score,classification_report
predictions=model.predict(x_test)
accuracy=accuracy_score(y_test,predictions)
print(f'Accuracy: {accuracy*100}%')
print('\nDetailedPerformanceReport:')
print(classification_report(y_test,predictions))
df_clean['readmitted_30d'] = (df_clean['readmitted'] == '<30').astype(int)

n_early = df_clean['readmitted_30d'].sum()
pct_early = 100 * n_early / len(df_clean)
print(f"\nEarly readmission (<30 days): {n_early:,} patients ({pct_early:.1f}%)")
print("This is our outcome. For every patient, we have a 1 or a 0.")
print("The question is: which features at discharge predict that 1?")

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
numeric_features = ['time_in_hospital', 'num_procedures', 'num_lab_procedures',
                   'num_medications', 'number_outpatient', 'number_emergency',
                   'number_inpatient']
context = {'time_in_hospital': 'Days in hospital. Longer = severity or social barriers to discharge.',
    'num_medications': 'Medications at discharge. Polypharmacy is readmission risk factor.',
    'num_procedures': 'Surgical procedures. Diabetic foot ulcers common in Kenya.',
    'num_lab_procedures': 'Lab tests. Higher = monitoring intensity or diagnostic uncertainty.',
    'number_emergency': 'ED visits in 6 months before admission. High use = fragmented care.',
    'number_outpatient': 'Clinic visits before admission. Higher = better engagement.',
    'number_inpatient': 'Prior hospital admissions. Each increases readmission risk.'}
numeric_pipeline= Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_features)
])
print("Numeric preprocessing pipeline configured successfully.")

print('='*60)
print('Do the LACE features predict readmission in this dataset?')
print('Blue=not readmitted early. Red=readmitted within 30 days')
print('='*60)
LACE_features={'time_in_hospital', 'number_emergency', 'number_inpatient', 'num_medications'}
labels={'time_in_hospital': 'Length of Stay (days)',
        'number_emergency': 'Prior ED Visits (6 months before admission)',
        'number_inpatient': 'Prior Inpatient Visits',
        'num_medications': 'Medications at Discharge',
        }
early=df_clean[df_clean['readmitted_30d']==1]
not_early=df_clean[df_clean['readmitted_30d']==0]
for feature in LACE_features:
    not_early_series=not_early[feature].dropna()
    early_series=early[feature].dropna()
    mean_not_early=not_early_series.mean()
    mean_early =early_series.mean()
display_name=labels.get(feature, feature)
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(
    not_early_series,
    bins=30,
    alpha=0.6,
    color='blue',
    label=f'Not readmitted early (n={len(not_early_series):,})',
    density=True
)
ax.hist(
    early_series,
    bins=30,
    alpha=0.6,
    color='pink',
    label=f'Readmitted <30d (n={len(early_series):,})',
    density=True
)
ax.axvline(
    mean_not_early,
    color='blue',
    linestyle='dashed',
    linewidth=1.5,
    label=f'Mean(not_early): {mean_not_early:.1f}'
)
ax.axvline(
    mean_early,
    color= 'pink',
    linestyle='dashed',
    linewidth=1.5,
    label=f'Mean(early): {mean_early:.1f}'
)
ax.set_title(f'{display_name}- split by readmission outcome', fontsize=14,fontweight='bold')
ax.set_xlabel(display_name,fontsize=12)
ax.set_ylabel('Density',fontsize=12)
ax.legend(fontsize=10)

plt.tight_layout()
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
y_true=[1, 0, 1, 1, 0, 0, 1, 0]
y_pred=[1, 0, 1, 0, 0, 0, 1, 1]
print('Accuracy:', accuracy_score(y_true, y_pred))
print('Precision:', precision_score(y_true, y_pred))
print('Recall:', recall_score(y_true, y_pred))
print('F1:', f1_score(y_true, y_pred))

print('\n--- Summary Report---')
print(classification_report(y_true, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_true=[1, 0, 1, 1, 0, 0, 1, 0]
y_pred=[1, 0, 1, 0, 0, 0, 1, 1]
cm=confusion_matrix(y_true, y_pred)
labels=['Not Readmitted (0)', 'Readmitted <30d (1)']
plt.figure(figsize=(7,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels,
    cbar=False,
    annot_kws={'fontsize': 14, 'weight': 'bold'}
)

plt.title('Clinical Readmission- Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Predicted Label (Model Output)', fontsize=12, labelpad=10)
plt.ylabel('True Label (Actual Patient Outcome)', fontsize=12, labelpad=10)

plt.tight_layout()
plt.show()

from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

y_pred = 'pipeline'.predict('X_test')
y_probs = 'pipeline'.predict_proba('X_test')[:, 1]

# 2. Compute metrics
print("--- Model Performance Metrics ---")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_probs):.4f}")

# 3. Detailed classification report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))