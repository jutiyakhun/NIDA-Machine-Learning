# AGENTS2.md
## Credit Card Fraud Classification

### Data Files
- `creditcard_train.csv` — training data
- `creditcard_test.csv` — test data (**includes `Class` labels**)

### Data Schema
- **Data:** https://drive.google.com/drive/folders/1sc40lk4pcLszH9Be1pdFIMMXafiDpRGh?usp=sharing
- **Label:** `Class` (binary: 0 = legitimate, 1 = fraud)
- **Features (30):** `Time`, `V1`–`V28`, `Amount` (all numeric; V1–V28 are PCA-transformed)
- **Class imbalance:** ~0.17% fraud
- Missing values: may exist — handle before training

### Missing Value Handling
- Check missing values with `df.isnull().sum()`
- **Numeric features (V1–V28):** impute with **median** (robust to outliers/fraud skew)
- **`Amount` and `Time`:** impute with **median**
- Drop any column where missing > 50% of rows

### Models to Train
Train all 3 models below, then select the one with the highest ROC-AUC on test set:

| # | Model | Key Parameters |
|---|---|---|
| 1 | `XGBClassifier` | `scale_pos_weight=neg/pos, random_state=42, eval_metric='auc'` |
| 2 | `LGBMClassifier` | `class_weight='balanced', random_state=42, verbose=-1` |
| 3 | `RandomForestClassifier` | `class_weight='balanced', n_estimators=200, random_state=42` |

### Workflow
1. Load both CSVs; drop any unnamed index columns
2. Handle missing values per schema above using `SimpleImputer(strategy='median')`
3. Features = all columns except `Class`; apply `StandardScaler` to `Amount` and `Time` only
4. Train all 3 models on training set
5. Evaluate each model on test set using **ROC-AUC score**
6. Print a comparison table of all 3 AUC scores
7. Use the **best model** to generate final predictions
8. Save `predictions.csv` with test data + `predicted_prob` + `predicted_label` (threshold=0.5)

### Success Metric
Maximize **ROC-AUC** on `creditcard_test.csv`.
