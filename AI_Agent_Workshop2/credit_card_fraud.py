import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

train = pd.read_csv('creditcard_train.csv')
test = pd.read_csv('creditcard_test.csv')

for df in [train, test]:
    unnamed = [c for c in df.columns if 'Unnamed' in c]
    df.drop(columns=unnamed, inplace=True, errors='ignore')

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Train NaN:\n{train.isnull().sum()}")
print(f"Test NaN:\n{test.isnull().sum()}")

feature_cols = [c for c in train.columns if c != 'Class']

# drop columns with >50% missing in either set
high_missing = set()
for col in feature_cols:
    if train[col].isnull().mean() > 0.5 or test[col].isnull().mean() > 0.5:
        high_missing.add(col)

if high_missing:
    print(f"Dropping columns with >50% missing: {high_missing}")
    train.drop(columns=high_missing, inplace=True)
    test.drop(columns=high_missing, inplace=True)
    feature_cols = [c for c in feature_cols if c not in high_missing]

X_train = train[feature_cols].copy()
y_train = train['Class'].copy()
X_test = test[feature_cols].copy()
y_test = test['Class'].copy()

# Impute missing values with median
imputer = SimpleImputer(strategy='median')
X_train[:] = imputer.fit_transform(X_train)
X_test[:] = imputer.transform(X_test)

# StandardScaler on Amount and Time only
scaler_cols = [c for c in ['Amount', 'Time'] if c in X_train.columns]
if scaler_cols:
    scaler = StandardScaler()
    X_train[scaler_cols] = scaler.fit_transform(X_train[scaler_cols])
    X_test[scaler_cols] = scaler.transform(X_test[scaler_cols])

# Compute scale_pos_weight for XGBoost
neg, pos = y_train.value_counts().values
scale_pos_weight = neg / pos

models = {
    'XGBClassifier': XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='auc',
        use_label_encoder=False
    ),
    'LGBMClassifier': LGBMClassifier(
        class_weight='balanced',
        random_state=42,
        verbose=-1
    ),
    'RandomForestClassifier': RandomForestClassifier(
        class_weight='balanced',
        n_estimators=200,
        random_state=42
    )
}

results = {}
best_model = None
best_score = -1

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    results[name] = auc
    print(f"{name} ROC-AUC: {auc:.6f}")
    if auc > best_score:
        best_score = auc
        best_model = (name, model, y_prob)

print("\n" + "=" * 50)
print("ROC-AUC Comparison:")
print("=" * 50)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {name:>25s}: {auc:.6f}")
print("=" * 50)
print(f"Best model: {best_model[0]} (ROC-AUC: {best_score:.6f})")

# Save predictions
best_name, _, best_probs = best_model
best_label = (best_probs >= 0.5).astype(int)

out = test.copy()
out['predicted_prob'] = best_probs
out['predicted_label'] = best_label
out.to_csv('predictions.csv', index=False)
print(f"\nSaved predictions.csv using {best_name}")
