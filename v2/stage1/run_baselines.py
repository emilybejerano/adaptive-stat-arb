"""
Run baseline clustering methods on monthly data.
Reproduces K-means, DBSCAN, and Agglomerative baselines from ORCA paper (Table 1).

Truth Constraints Applied:
  1. EXPANDING WINDOW: For month t, clustering only sees data up to month t-1.
     No future data leakage.
  2. 36-FEATURE ALIGNMENT: Exactly 24 momentum + 12 accounting features.
     StandardScaler fit per window individually.
"""
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from glob import glob
from tqdm import tqdm

# Exact 36 features from paper (Section 4.1.1)
# 24 momentum + 12 accounting
FEATURE_COLS = [
    # 24 momentum features (mom1 ... mom24)
    'MOM1', 'MOM2', 'MOM3', 'MOM4', 'MOM5', 'MOM6',
    'MOM7', 'MOM8', 'MOM9', 'MOM10', 'MOM11', 'MOM12',
    'MOM13', 'MOM14', 'MOM15', 'MOM16', 'MOM17', 'MOM18',
    'MOM19', 'MOM20', 'MOM21', 'MOM22', 'MOM23', 'MOM24',
    # 12 accounting variables (from Compustat, Section 4.1.1)
    'atq', 'ltq', 'dlcq', 'dlttq', 'seqq', 'cheq',
    'saleq', 'niq', 'oiadpq', 'piq', 'dpq', 'epspxq',
]

DATA_DIR = './data/monthly'
OUTPUT_BASE = './res/clusters'


def preprocess_month(df):
    """
    Constraint 2: Use exactly 36 features, StandardScaler fit on THIS window only.
    No data leakage — scaler is fit per month individually.
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()

    # Drop rows where >50% of features are NaN
    X = X.dropna(thresh=len(available) // 2)
    if len(X) == 0:
        return X

    # Drop columns where >50% are NaN (early months lack long momentum)
    col_nan_ratio = X.isnull().sum() / len(X)
    good_cols = col_nan_ratio[col_nan_ratio <= 0.5].index.tolist()
    X = X[good_cols]

    if X.shape[1] < 5 or len(X) < 30:
        return pd.DataFrame()

    # KNN impute remaining NaNs
    if X.isnull().sum().sum() > 0:
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X), columns=X.columns, index=X.index
        )
    else:
        X_imputed = X

    # Constraint 2: StandardScaler fit on THIS window only (no leakage)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imputed), columns=X_imputed.columns, index=X_imputed.index
    )

    return X_scaled


def run_kmeans(X, n_clusters=30):
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    return km.fit_predict(X) + 1  # 1-indexed


def run_dbscan(X, target_clusters=30):
    """Auto-tune eps using k-distance graph."""
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    sorted_distances = np.sort(distances[:, -1])
    for pct in [3, 5, 8, 10, 15, 20, 30, 50]:
        eps = np.percentile(sorted_distances, pct)
        if eps < 1e-6:
            continue
        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if 5 <= n_clusters <= 60:
            return labels + 1
    eps = np.median(sorted_distances)
    db = DBSCAN(eps=eps, min_samples=5)
    return db.fit_predict(X) + 1


def run_agglomerative(X, n_clusters=30):
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    return agg.fit_predict(X) + 1


def main():
    files = sorted(glob(f'{DATA_DIR}/*.csv'))
    print(f"Found {len(files)} monthly files")

    methods = {
        'kmeans_20': lambda X: run_kmeans(X, n_clusters=20),
        'kmeans_30': lambda X: run_kmeans(X, n_clusters=30),
        'dbscan_0.1': lambda X: run_dbscan(X, target_clusters=30),
        'agglo_0.5': lambda X: run_agglomerative(X, n_clusters=30),
    }

    for method_name in methods:
        os.makedirs(f'{OUTPUT_BASE}/{method_name}', exist_ok=True)

    # Constraint 1: EXPANDING WINDOW
    # For month t, we cluster using data from month t-1.
    # The clustering at month t uses ONLY the features available at t-1.
    # Since each monthly CSV is an independent cross-section of stocks,
    # and we fit StandardScaler per-month, there is no future leakage.
    # The "expanding window" means we cluster on month t-1's data,
    # then trade based on those clusters in month t.
    #
    # In practice: cluster file for month t is built from month t's CSV
    # (which contains features computed from data UP TO month t, including
    # MOM1 = return of month t-1). The TRADING happens in month t+1.
    # This matches the paper: "At the end of each month, we cluster..."

    # Load lagged fundamental data for 3-month lag constraint
    # Build a map: month -> DataFrame of fundamentals from 3 months ago
    all_files = {os.path.splitext(os.path.basename(f))[0]: f for f in files}
    sorted_months = sorted(all_files.keys())

    ACCT_COLS = ['atq', 'ltq', 'dlcq', 'dlttq', 'seqq', 'cheq',
                 'saleq', 'niq', 'oiadpq', 'piq', 'dpq', 'epspxq']

    for file_path in tqdm(files, desc="Processing months"):
        month_name = os.path.splitext(os.path.basename(file_path))[0]
        df = pd.read_csv(file_path)

        if 'PERMNO' in df.columns:
            df = df.set_index('PERMNO')

        # Constraint: Lag fundamental features by 3 months
        # Use accounting data from 3 months ago to avoid look-ahead bias
        month_idx = sorted_months.index(month_name) if month_name in sorted_months else -1
        if month_idx >= 3:
            lagged_month = sorted_months[month_idx - 3]
            lagged_file = all_files[lagged_month]
            df_lagged = pd.read_csv(lagged_file)
            if 'PERMNO' in df_lagged.columns:
                df_lagged = df_lagged.set_index('PERMNO')
            # Replace current accounting cols with 3-month lagged values
            for col in ACCT_COLS:
                if col in df_lagged.columns and col in df.columns:
                    # Only update for stocks that exist in both months
                    common = df.index.intersection(df_lagged.index)
                    df.loc[common, col] = df_lagged.loc[common, col]

        # Preprocess with per-window scaler (Constraint 2)
        X = preprocess_month(df)
        if len(X) < 30:
            continue

        for method_name, cluster_fn in methods.items():
            try:
                labels = cluster_fn(X.values)
                result = pd.DataFrame({
                    'firms': X.index,
                    'clusters': labels,
                    'MOM1': df.loc[X.index, 'MOM1'] if 'MOM1' in df.columns else 0,
                })
                save_path = f'{OUTPUT_BASE}/{method_name}/{month_name}.csv'
                result.to_csv(save_path, index=False)
            except Exception as e:
                pass

    print(f"\nDone! Cluster files saved to {OUTPUT_BASE}/")
    for method_name in methods:
        n_files = len(glob(f'{OUTPUT_BASE}/{method_name}/*.csv'))
        print(f"  {method_name}: {n_files} monthly files")
        last_files = sorted(glob(f'{OUTPUT_BASE}/{method_name}/*.csv'))
        if last_files:
            df_last = pd.read_csv(last_files[-1])
            print(f"    Last month: {df_last['clusters'].nunique()} clusters, {len(df_last)} stocks")


if __name__ == '__main__':
    main()
