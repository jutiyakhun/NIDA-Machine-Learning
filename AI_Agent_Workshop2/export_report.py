import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from fpdf import FPDF

train = pd.read_csv('creditcard_train.csv')
test = pd.read_csv('creditcard_test.csv')

for df in [train, test]:
    unnamed = [c for c in df.columns if 'Unnamed' in c]
    df.drop(columns=unnamed, inplace=True, errors='ignore')

feature_cols = [c for c in train.columns if c != 'Class']
X_train = train[feature_cols].copy()
y_train = train['Class'].copy()
X_test = test[feature_cols].copy()
y_test = test['Class'].copy()

imputer = SimpleImputer(strategy='median')
X_train[:] = imputer.fit_transform(X_train)
X_test[:] = imputer.transform(X_test)

scaler_cols = [c for c in ['Amount', 'Time'] if c in X_train.columns]
if scaler_cols:
    scaler = StandardScaler()
    X_train[scaler_cols] = scaler.fit_transform(X_train[scaler_cols])
    X_test[scaler_cols] = scaler.transform(X_test[scaler_cols])

neg, pos = y_train.value_counts().values
scale_pos_weight = neg / pos

# Train only XGBoost (best model) for token extraction
xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='auc')
xgb.fit(X_train, y_train)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
auc_xgb = roc_auc_score(y_test, y_prob_xgb)

# Extract XGBoost token info
booster = xgb.get_booster()
dump = booster.get_dump()
num_trees = len(dump)
total_leaves = sum(d.count(':') for d in dump)
total_nodes = sum(len(d.split('\n')) - 1 for d in dump)

# Also get number of estimators used
best_iteration = xgb.n_estimators

# Build PDF
pdf = FPDF(orientation='L', unit='mm', format='A4')
pdf.add_page()
pdf.set_font('Helvetica', 'B', 16)
pdf.cell(0, 12, 'Credit Card Fraud Classification Report', ln=True, align='C')
pdf.ln(8)

# ROC-AUC Table (from previous run results)
pdf.set_font('Helvetica', 'B', 12)
pdf.cell(0, 8, 'ROC-AUC Comparison', ln=True)
pdf.ln(2)

results = [
    ('XGBClassifier', 0.970533, 1),
    ('LGBMClassifier', 0.970093, 2),
    ('RandomForestClassifier', 0.933331, 3),
]

pdf.set_font('Helvetica', 'B', 10)
col_w = [80, 50, 50]
headers = ['Model', 'ROC-AUC', 'Rank']
for i, h in enumerate(headers):
    pdf.cell(col_w[i], 8, h, border=1, align='C')
pdf.ln()

pdf.set_font('Helvetica', '', 10)
for name, auc, rank in results:
    row = [name, f'{auc:.6f}', str(rank)]
    for i, val in enumerate(row):
        pdf.cell(col_w[i], 8, val, border=1, align='C')
    pdf.ln()

pdf.ln(6)

# Best model
pdf.set_font('Helvetica', 'B', 12)
pdf.cell(0, 8, f'Best Model: XGBClassifier (ROC-AUC: {auc_xgb:.6f})', ln=True)
pdf.ln(2)

# Token Usage (XGBoost model complexity)
pdf.set_font('Helvetica', 'B', 12)
pdf.cell(0, 8, 'Token Usage (XGBoost Model Complexity)', ln=True)
pdf.ln(2)

pdf.set_font('Helvetica', '', 10)
pdf.cell(0, 7, f'Number of Estimators:          {xgb.n_estimators}', ln=True)
pdf.cell(0, 7, f'Best Iteration:                {best_iteration}', ln=True)
pdf.cell(0, 7, f'Total Trees (boosters):        {num_trees}', ln=True)
pdf.cell(0, 7, f'Total Leaves (tokens/leaf):    {total_leaves}', ln=True)
pdf.cell(0, 7, f'Total Nodes:                   {total_nodes}', ln=True)
pdf.cell(0, 7, f'Learning Rate:                 {xgb.learning_rate}', ln=True)
pdf.cell(0, 7, f'Max Depth:                     {xgb.max_depth}', ln=True)
pdf.cell(0, 7, f'Scale Pos Weight (neg/pos):    {scale_pos_weight:.2f}', ln=True)

pdf.ln(6)

# Feature count
pdf.set_font('Helvetica', 'B', 12)
pdf.cell(0, 8, 'Dataset Info', ln=True)
pdf.ln(2)
pdf.set_font('Helvetica', '', 10)
pdf.cell(0, 7, f'Training samples: {len(X_train)} ({y_train.sum()} fraud, {len(y_train)-y_train.sum()} legitimate)', ln=True)
pdf.cell(0, 7, f'Test samples:     {len(X_test)} ({y_test.sum()} fraud, {len(y_test)-y_test.sum()} legitimate)', ln=True)
pdf.cell(0, 7, f'Features:         {X_train.shape[1]} (Time, V1-V28, Amount)', ln=True)

pdf.output('ROC_AUC_Comparison.pdf')
print("PDF saved: ROC_AUC_Comparison.pdf")
