"""
Credit Card Customer Segmentation
Following AGENTS.md procedure
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------
import sys
def save_fig(name, dpi=100):
    path = os.path.join(OUTPUT_DIR, name)
    print(f"  saving {name} (dpi={dpi})...")
    sys.stdout.flush()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  saved {name}")
    sys.stdout.flush()

# ------------------------------------------------------------
# Download dataset from Kaggle
# ------------------------------------------------------------
print("=" * 60)
print("Loading dataset from Kaggle...")
candidate = os.path.join(os.environ['USERPROFILE'], '.cache', 'kagglehub',
                         'datasets', 'arjunbhasin2013', 'ccdata', 'versions', '1')
if os.path.exists(candidate):
    path = candidate
    print(f"Using cached dataset at: {path}")
else:
    import kagglehub
    path = kagglehub.dataset_download("arjunbhasin2013/ccdata")
    print(f"Dataset downloaded to: {path}")
df = pd.read_csv(os.path.join(path, "CC GENERAL.csv"))
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head(3))
print()

# ------------------------------------------------------------
# 1. Explore Data
# ------------------------------------------------------------
print("=" * 60)
print("1. EXPLORE DATA")
print("=" * 60)

# 1.1. Ignore unnecessary column e.g. index column
print("\n1.1. Dropping CUST_ID...")
cust_ids = df['CUST_ID'].copy()
cust_index = np.arange(len(cust_ids))
df.drop(columns=['CUST_ID'], inplace=True)
print(f"Shape after drop: {df.shape}")

# ------------------------------------------------------------
# 1.2. Assign type of data
# ------------------------------------------------------------
print("\n1.2. Assigning types and handling missing values...")

# Auto-detect column types:
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Also treat int columns with few unique values as categorical
for c in num_cols[:]:
    if df[c].nunique() <= 10 and df[c].dtype in (np.int64, np.int32, int):
        cat_cols.append(c)
        num_cols.remove(c)

print(f"Numerical columns ({len(num_cols)}): {num_cols}")
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

df_num = df[num_cols].copy()
df_cat = df[cat_cols].copy()

# 1.2.a Numerical variables
# i. Fill missing value using mode value
print("\n1.2.a.i Filling numerical missing values with MODE...")
for c in df_num.columns:
    if df_num[c].isna().sum() > 0:
        mode_val = df_num[c].mode().iloc[0]
        df_num[c].fillna(mode_val, inplace=True)
        print(f"  {c}: filled {df[c].isna().sum()} missing values with mode={mode_val:.4f}")

# ii. Encode them using one-hot encoder
print("\n1.2.a.ii One-hot encoding numerical variables...")
# Group continuous values into bins to make them encodable
df_num_binned = pd.DataFrame()
for c in df_num.columns:
    n_unique = df_num[c].nunique()
    if n_unique > 20:
        n_bins = min(20, int(np.sqrt(len(df_num))))
        labels = [f"{c}_bin{i}" for i in range(n_bins)]
        df_num_binned[c] = pd.cut(df_num[c], bins=n_bins, labels=labels)
    else:
        df_num_binned[c] = df_num[c].astype(str)

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_arr = encoder.fit_transform(df_num_binned)
encoded_cols = encoder.get_feature_names_out(df_num.columns)
df_num_encoded = pd.DataFrame(encoded_arr, columns=encoded_cols, index=df_num.index)
print(f"  Numerical data shape after one-hot: {df_num_encoded.shape}")

# 1.2.b Categorical variables
# i. Fill missing value using median value
print("\n1.2.b.i Filling categorical missing values with MEDIAN...")
for c in df_cat.columns:
    if df_cat[c].isna().sum() > 0:
        # For categorical, convert to numeric if possible, compute median
        numeric_vals = pd.to_numeric(df_cat[c], errors='coerce')
        median_val = numeric_vals.median()
        df_cat[c].fillna(median_val, inplace=True)
        print(f"  {c}: filled {df[c].isna().sum()} missing values with median={median_val:.4f}")

# ------------------------------------------------------------
# 1.3. Visualization
# ------------------------------------------------------------
print("\n1.3. Creating visualizations...")

# 1.3.a.i: Relationship between each quantity variable and CUST_ID within a single chart
print("  1.3.a.i Plotting numerical variables vs CUST_ID...")
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()
for i, c in enumerate(num_cols):
    if i >= len(axes):
        break
    axes[i].scatter(cust_index, df_num[c], s=0.3, alpha=0.2)
    axes[i].set_title(f'{c} vs CUST_ID')
    axes[i].set_xlabel('Customer Index')
    axes[i].set_ylabel(c)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.subplots_adjust(wspace=0.3, hspace=0.3)
print("  saving...")
save_fig('01_numerical_vs_custid.png')
print("  Saved: 01_numerical_vs_custid.png")

# 1.3.a.ii: Box plots of each quantity variable
print("  1.3.a.ii Plotting box plots...")
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()
for i, c in enumerate(num_cols):
    if i >= len(axes):
        break
    sns.boxplot(y=df_num[c], ax=axes[i])
    axes[i].set_title(f'{c}')
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
save_fig('02_boxplots_numerical.png')
print("  Saved: 02_boxplots_numerical.png")

# 1.3.b.i: Bar chart for categorical variables
if len(cat_cols) > 0:
    print("  1.3.b.i Plotting bar charts for categorical variables...")
    fig, axes = plt.subplots(1, len(cat_cols), figsize=(6 * len(cat_cols), 5))
    if len(cat_cols) == 1:
        axes = [axes]
    for i, c in enumerate(cat_cols):
        df_cat[c].value_counts().sort_index().plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'Frequency of CUST_ID in {c}')
        axes[i].set_xlabel(c)
        axes[i].set_ylabel('Count')
    plt.tight_layout()
    save_fig('03_categorical_barcharts.png')
    print("  Saved: 03_categorical_barcharts.png")
else:
    print("  1.3.b.i No categorical variables to plot.")

# ------------------------------------------------------------
# 1.4. Normalize Numerical + combine with Categorical → NORM_DATA
# ------------------------------------------------------------
print("\n1.4. Normalizing numerical variables and combining...")
scaler = StandardScaler()
num_scaled = scaler.fit_transform(df_num)
df_num_scaled = pd.DataFrame(num_scaled, columns=num_cols, index=df_num.index)

if len(cat_cols) > 0:
    NORM_DATA = pd.concat([df_num_scaled, df_cat], axis=1)
else:
    NORM_DATA = df_num_scaled.copy()

print(f"NORM_DATA shape: {NORM_DATA.shape}")
print(f"NORM_DATA columns: {list(NORM_DATA.columns[:5])}...")

# ------------------------------------------------------------
# 2. Dimension Reduction
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("2. DIMENSION REDUCTION (PCA)")
print("=" * 60)

# 2.1. PCA
print("\n2.1. Applying PCA...")
pca = PCA(n_components=2, random_state=42)
PCA_DATA = pca.fit_transform(NORM_DATA)
print(f"PCA_DATA shape: {PCA_DATA.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# PCA visualization
plt.figure(figsize=(10, 8))
plt.scatter(PCA_DATA[:, 0], PCA_DATA[:, 1], s=5, alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection')
save_fig('04_pca_projection.png')
print("  Saved: 04_pca_projection.png")

# ------------------------------------------------------------
# 3. Clustering
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("3. CLUSTERING")
print("=" * 60)

# ------------------------------------------------------------
# 3.1. K-Means
# ------------------------------------------------------------
print("\n3.1. K-Means Clustering...")
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(PCA_DATA)
print(f"K-Means clusters (k=5): {np.unique(kmeans_labels)}")
print(f"Cluster sizes: {np.bincount(kmeans_labels)}")

# K-Means scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(PCA_DATA[:, 0], PCA_DATA[:, 1], c=kmeans_labels,
                       cmap='tab10', s=5, alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('K-Means Clustering (k=5) on PCA Data')
save_fig('05_kmeans_clusters.png')
print("  Saved: 05_kmeans_clusters.png")

# ------------------------------------------------------------
# 3.2. DBSCAN with tuning
# ------------------------------------------------------------
print("\n3.2. DBSCAN Clustering (with tuning)...")

def tune_dbscan(data, eps_values, min_samples_values):
    best_score = -1
    best_params = {'eps': None, 'min_samples': None}
    best_labels = None
    results = []
    for eps in eps_values:
        for min_samples in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(data)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            if n_clusters >= 2:
                score = silhouette_score(data, labels)
                results.append({
                    'eps': eps, 'min_samples': min_samples,
                    'n_clusters': n_clusters, 'n_noise': n_noise,
                    'silhouette': score
                })
                if score > best_score:
                    best_score = score
                    best_params = {'eps': eps, 'min_samples': min_samples}
                    best_labels = labels
                    best_n_clusters = n_clusters
                    best_n_noise = n_noise
    return best_params, best_score, best_labels, best_n_clusters, best_n_noise, results

eps_range = np.arange(0.3, 2.1, 0.2)
min_samples_range = range(3, 12, 2)
best_params, best_score, dbscan_labels, n_clusters, n_noise, dbscan_results = \
    tune_dbscan(PCA_DATA, eps_range, min_samples_range)

print(f"Best DBSCAN params: eps={best_params['eps']}, min_samples={best_params['min_samples']}")
print(f"Best silhouette score: {best_score:.4f}")
print(f"Number of clusters found: {n_clusters}")
print(f"Number of noise points: {n_noise}")

# DBSCAN results table
if dbscan_results:
    print("\nDBSCAN Tuning Results (top 5 by silhouette score):")
    results_df = pd.DataFrame(dbscan_results)
    results_df = results_df.sort_values('silhouette', ascending=False).head(5)
    print(results_df.to_string(index=False))

# DBSCAN scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(PCA_DATA[:, 0], PCA_DATA[:, 1], c=dbscan_labels,
                       cmap='tab10', s=5, alpha=0.6)
plt.colorbar(scatter, label='Cluster (-1 = noise)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'DBSCAN Clustering (eps={best_params["eps"]}, min_samples={best_params["min_samples"]})')
save_fig('06_dbscan_clusters.png')
print("  Saved: 06_dbscan_clusters.png")

# ------------------------------------------------------------
# 4. Select best K in [2, 10] for K-Means
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("4. SELECT BEST K FOR K-MEANS (K in [2, 10])")
print("=" * 60)

K_range = range(2, 11)
inertias = []
silhouettes = []
kmeans_models = {}

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(PCA_DATA)
    inertias.append(km.inertia_)
    sil = silhouette_score(PCA_DATA, labels)
    silhouettes.append(sil)
    kmeans_models[k] = (km, labels)
    print(f"  K={k}: inertia={km.inertia_:.2f}, silhouette={sil:.4f}")

# 4.1. Elbow Method
print("\n4.1. Elbow Method plot...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.plot(list(K_range), inertias, 'bo-', markersize=8)
ax1.set_xlabel('Number of clusters (K)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method for Optimal K')
ax1.set_xticks(list(K_range))

# Find elbow point
from kneed import KneeLocator
try:
    kl = KneeLocator(list(K_range), inertias, curve='convex', direction='decreasing')
    best_k_elbow = kl.elbow
    if best_k_elbow:
        ax1.axvline(x=best_k_elbow, color='r', linestyle='--', alpha=0.7,
                    label=f'Elbow at K={best_k_elbow}')
        ax1.legend()
    print(f"Elbow method suggests K={best_k_elbow}")
except Exception as e:
    print(f"KneeLocator not available ({e}), using manual elbow detection")
    # Simple second derivative approach
    diffs = np.diff(inertias, 2)
    best_k_elbow = list(K_range)[2:][np.argmin(diffs)]
    ax1.axvline(x=best_k_elbow, color='r', linestyle='--', alpha=0.7,
                label=f'Elbow at K={best_k_elbow}')
    ax1.legend()
    print(f"Manual elbow suggests K={best_k_elbow}")

# 4.2. Silhouette scores
print("\n4.2. Silhouette Score plot...")
ax2.plot(list(K_range), silhouettes, 'rs-', markersize=8)
best_k_sil = list(K_range)[np.argmax(silhouettes)]
ax2.plot(best_k_sil, silhouettes[list(K_range).index(best_k_sil)],
         'g*', markersize=20, label=f'Best K={best_k_sil}')
ax2.set_xlabel('Number of clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score for Optimal K')
ax2.set_xticks(list(K_range))
ax2.legend()

plt.tight_layout()
save_fig('07_elbow_silhouette.png')
print("  Saved: 07_elbow_silhouette.png")
print(f"Best K by elbow: {best_k_elbow}")
print(f"Best K by silhouette: {best_k_sil}")

# ------------------------------------------------------------
# 5. Demonstrate the best K and Silhouette scores
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("5. BEST K AND SILHOUETTE SCORES")
print("=" * 60)

best_k = best_k_sil  # Use silhouette-based best K
best_km_model, best_km_labels = kmeans_models[best_k]

print(f"\nBest K for K-Means (by Silhouette): K={best_k}")
print(f"K-Means Silhouette Score at K={best_k}: {silhouettes[list(K_range).index(best_k)]:.4f}")
print(f"K-Means cluster sizes: {np.bincount(best_km_labels)}")

print(f"\n--- Silhouette scores for all K ---")
for k, s in zip(K_range, silhouettes):
    print(f"  K={k}: {s:.4f}")

print(f"\n--- DBSCAN Results ---")
print(f"Best DBSCAN params: eps={best_params['eps']}, min_samples={best_params['min_samples']}")
print(f"DBSCAN Silhouette Score: {best_score:.4f}")
print(f"DBSCAN clusters: {n_clusters}")
print(f"DBSCAN noise points: {n_noise}")

# Comparison table
print(f"\n--- Summary Comparison ---")
print(f"{'Method':<15} {'Clusters':<10} {'Silhouette':<12} {'Notes':<20}")
print(f"{'-'*60}")
print(f"{'K-Means (best)':<15} {best_k:<10} {silhouettes[list(K_range).index(best_k)]:<12.4f} {'K=' + str(best_k):<20}")
eps_display = 'eps=' + str(round(best_params['eps'], 2))
print(f"{'DBSCAN':<15} {n_clusters:<10} {best_score:<12.4f} {eps_display:<20}")

# ------------------------------------------------------------
# 6. Scatter plots of data classified by group
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("6. SCATTER PLOTS OF DATA CLASSIFIED BY GROUP")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# K-Means at best K
ax = axes[0]
scatter = ax.scatter(PCA_DATA[:, 0], PCA_DATA[:, 1], c=best_km_labels,
                      cmap='tab10', s=10, alpha=0.6)
ax.set_title(f'K-Means (K={best_k}, Silhouette={silhouettes[list(K_range).index(best_k)]:.3f})')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.colorbar(scatter, ax=ax, label='Cluster')

# DBSCAN
ax = axes[1]
scatter = ax.scatter(PCA_DATA[:, 0], PCA_DATA[:, 1], c=dbscan_labels,
                      cmap='tab10', s=10, alpha=0.6)
ax.set_title(f'DBSCAN (eps={best_params["eps"]}, Silhouette={best_score:.3f})')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.colorbar(scatter, ax=ax, label='Cluster')

# Original data (no clusters)
ax = axes[2]
ax.scatter(PCA_DATA[:, 0], PCA_DATA[:, 1], c='steelblue', s=10, alpha=0.5)
ax.set_title('Original Data (no clustering)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

plt.tight_layout()
save_fig('08_clustering_comparison.png')
print("  Saved: 08_clustering_comparison.png")

# Additional: All K-Means results for K in [2,10]
print("\n  Generating scatter plots for all K values...")
fig, axes = plt.subplots(3, 3, figsize=(20, 18))
axes = axes.flatten()
for idx, k in enumerate(K_range):
    ax = axes[idx]
    _, labels = kmeans_models[k]
    scatter = ax.scatter(PCA_DATA[:, 0], PCA_DATA[:, 1], c=labels,
                          cmap='tab10', s=5, alpha=0.6)
    ax.set_title(f'K-Means K={k} (Sil={silhouettes[idx]:.3f})')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.colorbar(scatter, ax=ax, label='Cluster')
plt.tight_layout()
save_fig('09_kmeans_all_k.png')
print("  Saved: 09_kmeans_all_k.png")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nAll output files saved to: {OUTPUT_DIR}")
print("Generated files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith(('.py', '.md', '.png')):
        print(f"  - {f}")
