"""
Build spread/OU/feature data for expanded universe (154 cointegrated pairs).
Uses existing price data (46 tickers from yfinance).

Output: datasets/ with all parquet files + artifacts.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import norm
import pickle, os
import warnings
from zscore_utils import ZSCORE_METHOD, ZSCORE_WINDOW, compute_spread_zscore
warnings.filterwarnings('ignore')
np.random.seed(42)

os.makedirs('datasets', exist_ok=True)

# Load existing price + macro data
df_prices = pd.read_parquet('datasets/pair_prices.parquet')
df_macro = pd.read_parquet('datasets/macro.parquet')
tickers = [c for c in df_prices.columns if c != 'VIX']
print(f"Loaded {len(tickers)} tickers, {df_prices.shape[0]} days")
print(f"Z-score method: {ZSCORE_METHOD} (window={ZSCORE_WINDOW})")

TRAIN_END = '2018-12-31'  # train 2010-2018, val 2019, test 2020-2023


# ============================================================
# 1. FIND ALL COINTEGRATED PAIRS
# ============================================================
print("\n1. Screening all pairs for cointegration...")
train_prices = df_prices[df_prices.index <= TRAIN_END].dropna(axis=1)
valid_tickers = [t for t in tickers if t in train_prices.columns]
print(f"   {len(valid_tickers)} tickers with full training data")

cointegrated = []
total = len(valid_tickers) * (len(valid_tickers) - 1) // 2
checked = 0

for a, b in combinations(valid_tickers, 2):
    checked += 1
    if checked % 200 == 0:
        print(f"   Checked {checked}/{total}...", flush=True)

    pa = train_prices[a]; pb = train_prices[b]
    common = pa.dropna().index.intersection(pb.dropna().index)
    if len(common) < 500:
        continue

    # Correlation filter
    corr = pa.loc[common].pct_change().corr(pb.loc[common].pct_change())
    if abs(corr) < 0.3:
        continue

    # Spread + ADF on last 2 years of training
    pa_2y = pa.loc[common].iloc[-504:]
    pb_2y = pb.loc[common].iloc[-504:]
    X = sm.add_constant(pa_2y.values)
    try:
        beta = sm.OLS(pb_2y.values, X).fit().params[1]
        spread = pb_2y - beta * pa_2y
        adf_stat, pvalue = adfuller(spread.dropna(), maxlag=20)[:2]
    except:
        continue

    if pvalue <= 0.10:
        cointegrated.append((a, b, pvalue, corr))

cointegrated.sort(key=lambda x: x[2])
print(f"   Found {len(cointegrated)} cointegrated pairs")


# ============================================================
# 2. COMPUTE SPREADS (rolling 252-day OLS hedge ratio)
# ============================================================
print(f"\n2. Computing spreads for {len(cointegrated)} pairs...")

def compute_rolling_spread(prices_a, prices_b, window=252):
    spread = pd.Series(index=prices_a.index, dtype=float)
    for i in range(window, len(prices_a)):
        y = prices_b.iloc[i-window:i].values
        x = prices_a.iloc[i-window:i].values
        X = sm.add_constant(x)
        try:
            beta = sm.OLS(y, X).fit().params[1]
        except:
            beta = np.nan
        spread.iloc[i] = prices_b.iloc[i] - beta * prices_a.iloc[i]
    return spread.dropna()

all_spreads = {}
pair_names = []

for idx, (a, b, pval, corr) in enumerate(cointegrated):
    if (idx + 1) % 20 == 0:
        print(f"   {idx+1}/{len(cointegrated)}...", flush=True)

    pa = df_prices[a].dropna()
    pb = df_prices[b].dropna()
    common = pa.index.intersection(pb.index)
    if len(common) < 500:
        continue

    spread = compute_rolling_spread(pa.loc[common], pb.loc[common])
    if len(spread) < 500:
        continue

    name = f"{a}/{b}"
    all_spreads[name] = spread
    pair_names.append(name)

spreads_df = pd.DataFrame(all_spreads)
spreads_df.to_parquet('datasets/spreads.parquet')
print(f"   Saved {len(pair_names)} pairs")


# ============================================================
# 3. OU PARAMETER ESTIMATION (rolling 60-day AR(1))
# ============================================================
print(f"\n3. Estimating OU parameters...")

def estimate_ou(spread, dt=1/252):
    spread = spread.dropna()
    if len(spread) < 10:
        return {'theta': 0, 'mu': 0, 'sigma': 0, 'half_life': np.inf,
                'is_stationary': False}
    y = spread.values[1:]; x = spread.values[:-1]
    X = sm.add_constant(x)
    try:
        model = sm.OLS(y, X).fit()
        a_hat, b_hat = model.params[0], model.params[1]
        resid_var = np.var(model.resid, ddof=2)
    except:
        return {'theta': 0, 'mu': spread.mean(), 'sigma': spread.std(),
                'half_life': np.inf, 'is_stationary': False}
    if b_hat >= 1.0 or b_hat <= 0:
        return {'theta': 0, 'mu': spread.mean(), 'sigma': spread.std(),
                'half_life': np.inf, 'is_stationary': False}
    theta = -np.log(b_hat) / dt
    mu = a_hat / (1 - b_hat)
    sigma = np.sqrt(2 * theta * resid_var / (1 - b_hat**2))
    theta = np.clip(theta, 0.01, 20)
    half_life = np.log(2) / theta * 252
    return {'theta': theta, 'mu': mu, 'sigma': sigma, 'half_life': half_life,
            'is_stationary': True}

all_ou = {}
for idx, name in enumerate(pair_names):
    if (idx + 1) % 20 == 0:
        print(f"   {idx+1}/{len(pair_names)}...", flush=True)

    spread = all_spreads[name]
    results = []
    for i in range(60, len(spread)):
        window = spread.iloc[i-60:i]
        params = estimate_ou(window)
        params['date'] = spread.index[i]
        results.append(params)
    ou_df = pd.DataFrame(results).set_index('date')
    all_ou[name] = ou_df

    safe = name.replace('/', '_')
    ou_df.to_parquet(f'datasets/ou_params_{safe}.parquet')

print(f"   Saved OU params for {len(all_ou)} pairs")


# ============================================================
# 4. BUILD FEATURES
# ============================================================
print(f"\n4. Building features...")

def compute_kelly(theta, mu, sigma, current, horizon_days=20, dt=1/252):
    if theta <= 0.01 or sigma <= 0: return 0.0
    H = horizon_days * dt
    dev = current - mu
    if abs(dev) < 1e-10: return 0.0
    exp_future = mu + dev * np.exp(-theta * H)
    future_var = (sigma**2 / (2*theta)) * (1 - np.exp(-2*theta*H))
    future_std = np.sqrt(max(future_var, 1e-10))
    target = current - 0.5 * dev
    p = norm.cdf(target, loc=exp_future, scale=future_std) if dev > 0 else \
        1 - norm.cdf(target, loc=exp_future, scale=future_std)
    p = np.clip(p, 0.01, 0.99); q = 1 - p
    exp_profit = 0.5 * abs(dev)
    exp_loss = max(future_std, abs(dev) * 0.1)
    K = exp_profit / exp_loss
    edge = p * K - q
    return float(np.clip(edge / K, 0, 1)) if edge > 0 else 0.0

FEATURE_COLS = [
    'theta', 'mu', 'sigma', 'z_score', 'half_life', 'ou_resid_var',
    'VIX', 'VIX_5d_change', 'yield_10y', 'hy_spread',
    'pair_corr_20d', 'spread_vol_20d', 'kelly_fraction', 'kelly_edge'
]

all_features = {}
for idx, name in enumerate(pair_names):
    if (idx + 1) % 20 == 0:
        print(f"   {idx+1}/{len(pair_names)}...", flush=True)

    spread = all_spreads[name]
    ou_df = all_ou[name]
    common = spread.index.intersection(ou_df.index).intersection(df_macro.index)
    common = common[common.isin(df_prices.index)]

    feat = pd.DataFrame(index=common)
    feat['theta'] = ou_df.reindex(common)['theta']
    feat['mu'] = ou_df.reindex(common)['mu']
    feat['sigma'] = ou_df.reindex(common)['sigma']

    s_aligned = spread.reindex(common)
    z_score = compute_spread_zscore(s_aligned, window=ZSCORE_WINDOW, method=ZSCORE_METHOD)
    feat['z_score'] = z_score
    feat['half_life'] = ou_df.reindex(common)['half_life']
    feat['ou_resid_var'] = s_aligned.rolling(60).std() ** 2

    feat['VIX'] = df_prices['VIX'].reindex(common) if 'VIX' in df_prices.columns else np.nan
    feat['VIX_5d_change'] = feat['VIX'].pct_change(5)
    feat['yield_10y'] = df_macro['10Y_Yield'].reindex(common)
    feat['hy_spread'] = df_macro['HY_Spread'].reindex(common)

    a, b = name.split('/')
    if a in df_prices.columns and b in df_prices.columns:
        ra = df_prices[a].pct_change(); rb = df_prices[b].pct_change()
        feat['pair_corr_20d'] = ra.rolling(20).corr(rb).reindex(common)
    else:
        feat['pair_corr_20d'] = np.nan
    feat['spread_vol_20d'] = s_aligned.rolling(20).std()

    kelly_fracs = []
    for date in common:
        th = ou_df.loc[date, 'theta'] if date in ou_df.index else 0
        mu = ou_df.loc[date, 'mu'] if date in ou_df.index else 0
        sig = ou_df.loc[date, 'sigma'] if date in ou_df.index else 0
        sv = spread.loc[date] if date in spread.index else 0
        kelly_fracs.append(compute_kelly(th, mu, sig, sv))

    feat['kelly_fraction'] = kelly_fracs
    feat['kelly_edge'] = [kf * 1.0 if kf > 0 else 0 for kf in kelly_fracs]
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()

    all_features[name] = feat
    safe = name.replace('/', '_')
    feat.to_parquet(f'datasets/features_{safe}.parquet')

print(f"   Saved features for {len(all_features)} pairs")


# ============================================================
# 5. FIT SCALER + PCA ON TRAINING DATA
# ============================================================
print(f"\n5. Fitting scaler and PCA...")

train_all = []
for name, feat in all_features.items():
    train_all.append(feat[feat.index <= TRAIN_END][FEATURE_COLS])
train_df = pd.concat(train_all).replace([np.inf, -np.inf], np.nan).dropna()
print(f"   Training feature matrix: {train_df.shape}")

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df)

pca = PCA(n_components=0.95, random_state=42)
pca.fit(train_scaled)
print(f"   PCA: {len(FEATURE_COLS)} -> {pca.n_components_} components ({pca.explained_variance_ratio_.sum():.1%})")


# ============================================================
# 6. SAVE ARTIFACTS
# ============================================================
artifacts = {
    'scaler': scaler,
    'pca': pca,
    'cointegrated_pairs': pair_names,
    'STATIC_FEATURE_COLS': FEATURE_COLS,
    'state_dim': pca.n_components_ + 3,
    'n_pairs': len(pair_names),
}

with open('datasets/artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print(f"\n{'='*60}")
print(f"DONE: {len(pair_names)} pairs")
print(f"Artifacts saved to datasets/")
print(f"State dim: {pca.n_components_ + 3}")
print(f"{'='*60}")
