"""
ORCA (Ornstein-Uhlenbeck Reversion and Contrastive Arbitrage)
Adapted for yfinance data.

Paper: Kim, Na & Song (ICAIF 2025)
"Deep Mean-Reversion: A Physics-Informed Contrastive Approach to Pairs Trading"

Original uses CRSP/Compustat (3000+ stocks, 36 features: 24 momentum + 12 fundamentals).
This adaptation uses yfinance (~46 NYSE tickers) with:
  - 12 momentum features (mom1..mom12, shortened from 24 due to fewer tickers)
  - 6 proxy fundamental features derived from price/volume data
  Total: 18 features per stock per month

Pipeline:
  1. Download prices via yfinance
  2. Compute monthly returns + features
  3. Train ORCA (contrastive + PINN) to cluster stocks
  4. Run the paper's Algorithm 1 (cluster-based mean-reversion strategy)
  5. Compare: K-means baseline vs ORCA

Usage:
  python orca_yfinance.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
import os
import pickle

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ============================================================
# 1. DATA: Download + Feature Engineering
# ============================================================

# Expanded universe: ~120 liquid large-cap names across 11 GICS sectors
# Target: ~8-12 stocks per cluster with K=15, giving PINN enough mass
TICKERS = [
    # Energy (10)
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'VLO', 'PSX', 'OXY',
    # Financials (15)
    'JPM', 'BAC', 'GS', 'MS', 'C', 'WFC', 'BLK', 'SCHW', 'AXP', 'USB',
    'PNC', 'TFC', 'BK', 'CME', 'ICE',
    # Healthcare (12)
    'JNJ', 'PFE', 'UNH', 'MRK', 'ABT', 'LLY', 'BMY', 'AMGN', 'GILD',
    'ISRG', 'TMO', 'MDT',
    # Tech (15)
    'AAPL', 'MSFT', 'GOOG', 'META', 'NVDA', 'AVGO', 'INTC', 'AMD', 'QCOM',
    'TXN', 'ORCL', 'CRM', 'ADBE', 'IBM', 'NOW',
    # Consumer Staples (10)
    'PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'GIS', 'K', 'MO', 'PM',
    # Consumer Discretionary (10)
    'AMZN', 'TSLA', 'HD', 'LOW', 'MCD', 'NKE', 'SBUX', 'TGT', 'TJX', 'ROST',
    # Industrials (12)
    'CAT', 'DE', 'GE', 'HON', 'MMM', 'UPS', 'FDX', 'LMT', 'RTX', 'BA',
    'WM', 'ETN',
    # Utilities (8)
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL',
    # Real Estate (8)
    'AMT', 'PLD', 'CCI', 'SPG', 'EQIX', 'O', 'WELL', 'DLR',
    # Materials (8)
    'LIN', 'APD', 'SHW', 'DD', 'FCX', 'NEM', 'NUE', 'ECL',
    # Communication (7)
    'T', 'VZ', 'DIS', 'CMCSA', 'NFLX', 'TMUS', 'CHTR',
]


def download_data(tickers, start='2005-01-01', end='2023-12-31', cache_path='datasets/orca_prices.parquet'):
    """Download daily price + volume data, cache to parquet."""
    if os.path.exists(cache_path):
        print(f"Loading cached prices from {cache_path}")
        return pd.read_parquet(cache_path)

    import yfinance as yf
    print(f"Downloading {len(tickers)} tickers from yfinance...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=True)

    # Build a clean dataframe: columns = (ticker, field)
    close = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data[['Close']]
    volume = data['Volume'] if isinstance(data.columns, pd.MultiIndex) else data[['Volume']]

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Save close and volume separately within one parquet
    result = pd.DataFrame()
    for t in tickers:
        if t in close.columns:
            result[f'{t}_close'] = close[t]
            result[f'{t}_volume'] = volume[t]
    result.to_parquet(cache_path)
    print(f"Saved to {cache_path}")
    return result


def compute_monthly_features(raw_data, tickers, n_mom=12):
    """
    Compute per-stock monthly features following the paper:
      - Momentum features: mom1 = r_{t-1}, mom_i = cumulative return over i months
      - Proxy fundamentals from price/volume (since we lack Compustat)

    Returns:
      features: dict of {month_end_date: DataFrame(n_stocks × n_features)}
      returns: dict of {month_end_date: Series(n_stocks)}  — next month's return
    """
    # Extract close prices per ticker
    close_cols = {t: f'{t}_close' for t in tickers if f'{t}_close' in raw_data.columns}
    vol_cols = {t: f'{t}_volume' for t in tickers if f'{t}_volume' in raw_data.columns}
    valid_tickers = sorted(set(close_cols.keys()) & set(vol_cols.keys()))

    # Compute monthly returns
    monthly_close = pd.DataFrame()
    monthly_volume = pd.DataFrame()
    for t in valid_tickers:
        daily_close = raw_data[close_cols[t]].dropna()
        daily_vol = raw_data[vol_cols[t]].dropna()
        monthly_close[t] = daily_close.resample('ME').last()
        monthly_volume[t] = daily_vol.resample('ME').mean()  # avg daily volume

    monthly_returns = monthly_close.pct_change()

    # Build feature snapshots for each month
    features = {}
    next_returns = {}
    dates = monthly_returns.index[n_mom + 1:]  # need n_mom months of lookback + 1 for next return

    for i, date in enumerate(dates):
        idx = monthly_returns.index.get_loc(date)
        if idx + 1 >= len(monthly_returns.index):
            break  # no next-month return available

        feat_rows = []
        ret_dict = {}
        next_date = monthly_returns.index[idx + 1]

        for t in valid_tickers:
            ret_history = monthly_returns[t].iloc[:idx + 1]
            if ret_history.isna().sum() > n_mom // 2:
                continue
            ret_history = ret_history.fillna(0)

            # --- Momentum features (paper Section 4.1.1) ---
            # mom1 = r_{t-1}
            # mom_i = cumulative return over past i months, for i in 2..n_mom
            row = {}
            row['mom1'] = ret_history.iloc[-1]
            for m in range(2, n_mom + 1):
                cum = (1 + ret_history.iloc[-m:]).prod() - 1
                row[f'mom{m}'] = cum

            # --- Proxy fundamental features ---
            # We lack Compustat, so derive proxies from price/volume:
            price = monthly_close[t].iloc[idx]
            vol = monthly_volume[t].iloc[idx]
            ret_12m = ret_history.iloc[-12:]

            row['log_price'] = np.log(max(price, 0.01))
            row['log_volume'] = np.log(max(vol, 1))
            row['volatility_12m'] = ret_12m.std()
            row['skewness_12m'] = ret_12m.skew() if len(ret_12m) >= 3 else 0
            row['max_return_12m'] = ret_12m.max()
            row['min_return_12m'] = ret_12m.min()

            row['ticker'] = t
            feat_rows.append(row)

            # Next month's return
            next_ret = monthly_returns[t].iloc[idx + 1] if not pd.isna(monthly_returns[t].iloc[idx + 1]) else 0
            ret_dict[t] = next_ret

        if len(feat_rows) < 10:  # need enough stocks
            continue

        feat_df = pd.DataFrame(feat_rows).set_index('ticker')
        features[date] = feat_df
        next_returns[date] = pd.Series(ret_dict)

    return features, next_returns, valid_tickers


# ============================================================
# 2. MODEL: ORCA Architecture (faithful to paper Section 3)
# ============================================================

class PiecewiseLinearEncoding(nn.Module):
    """
    Paper Section 3.1.1: PLE tokenizer.
    Bins each feature into T bins, produces a piecewise-linear encoding,
    then projects to d-dimensional embedding via a linear layer.
    """
    def __init__(self, n_features, n_bins=32, d_embed=64):
        super().__init__()
        self.n_features = n_features
        self.n_bins = n_bins
        self.d_embed = d_embed

        # Bin boundaries will be set from training data (quantile-based)
        # Shape: (n_features, n_bins + 1)
        self.register_buffer('boundaries', torch.zeros(n_features, n_bins + 1))

        # One linear projection per feature: T -> d
        self.projections = nn.ModuleList([
            nn.Linear(n_bins, d_embed) for _ in range(n_features)
        ])

    def set_boundaries(self, X_train):
        """Set bin boundaries from training data quantiles."""
        for k in range(self.n_features):
            col = X_train[:, k]
            col = col[~torch.isnan(col)]
            if len(col) < self.n_bins:
                self.boundaries[k] = torch.linspace(col.min(), col.max(), self.n_bins + 1, device=col.device)
            else:
                quantiles = torch.linspace(0, 1, self.n_bins + 1, device=col.device)
                self.boundaries[k] = torch.quantile(col, quantiles)
            # Ensure first and last are -inf, +inf
            self.boundaries[k, 0] = -1e6
            self.boundaries[k, -1] = 1e6

    def forward(self, x):
        """
        x: (batch, n_features) — already standardized
        Returns: (batch, n_features, d_embed)
        """
        batch_size = x.shape[0]
        embeddings = []

        for k in range(self.n_features):
            xk = x[:, k]  # (batch,)
            bounds = self.boundaries[k]  # (n_bins + 1,)

            # PLE encoding: Equation (1) from paper
            ple = torch.zeros(batch_size, self.n_bins, device=x.device)
            for t in range(self.n_bins):
                b_low = bounds[t]
                b_high = bounds[t + 1]
                width = b_high - b_low
                width = torch.clamp(width, min=1e-8)

                below = (xk < b_low).float()
                above = (xk >= b_high).float()
                between = 1.0 - below - above

                frac = (xk - b_low) / width
                frac = torch.clamp(frac, 0, 1)

                # If t > 0 and x < b_low: 0
                # If t < T-1 and x >= b_high: 1
                # Otherwise: fractional
                if t == 0:
                    ple[:, t] = above + between * frac
                elif t == self.n_bins - 1:
                    ple[:, t] = between * frac
                else:
                    ple[:, t] = above + between * frac

            # Project: (batch, n_bins) -> (batch, d_embed)
            emb = self.projections[k](ple)
            embeddings.append(emb)

        return torch.stack(embeddings, dim=1)  # (batch, n_features, d_embed)


class TransformerEncoder(nn.Module):
    """
    Paper Section 3.1.2: Bidirectional transformer with [CLS] token.
    """
    def __init__(self, d_embed=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.d_embed = d_embed
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_embed) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_embed, nhead=n_heads,
            dim_feedforward=d_embed * 4, dropout=dropout,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_embed)

    def forward(self, x):
        """
        x: (batch, n_tokens, d_embed)
        Returns: h = CLS output, (batch, d_embed)
        """
        batch_size = x.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)  # prepend [CLS]
        x = self.transformer(x)
        h = self.norm(x[:, 0, :])  # [CLS] output
        return h


class ORCA(nn.Module):
    """
    Full ORCA model: PLE + Transformer + Contrastive heads + PINN head.

    Paper Section 3:
      - Backbone: PLE encoder + Transformer -> h (representation)
      - Instance projection head g_ins(h) -> z (for instance contrastive loss)
      - Cluster projection head g_clu(h) -> y (soft cluster assignments)
      - OU parameter head g_ou(h_bar_k) -> (theta_k, mu_k, sigma_k) per cluster

    Loss = L_ins + alpha * L_clu + beta * L_pinn
    """
    def __init__(self, n_features, n_clusters=10, n_bins=32, d_embed=64,
                 n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.d_embed = d_embed

        # Backbone
        self.ple = PiecewiseLinearEncoding(n_features, n_bins, d_embed)
        self.transformer = TransformerEncoder(d_embed, n_heads, n_layers, dropout)

        # Instance projection head: h -> z (paper uses 2-layer MLP)
        self.g_ins = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, d_embed)
        )

        # Cluster projection head: h -> y (soft labels, M clusters)
        self.g_clu = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, n_clusters),
        )

        # OU parameter head: h_bar_k -> (theta, mu, sigma) per cluster
        # Paper Equation (10)
        self.g_ou = nn.Sequential(
            nn.Linear(d_embed, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # outputs: [theta, mu, sigma]
        )

    def forward(self, x):
        """
        x: (batch, n_features) — standardized input
        Returns dict with all intermediate representations.
        """
        tokens = self.ple(x)           # (batch, n_features, d_embed)
        h = self.transformer(tokens)    # (batch, d_embed)
        z = self.g_ins(h)              # (batch, d_embed)
        y_logits = self.g_clu(h)       # (batch, n_clusters)
        y = F.softmax(y_logits, dim=-1)  # soft cluster assignments

        return {'h': h, 'z': z, 'y': y, 'y_logits': y_logits}

    def get_ou_params(self, h, y):
        """
        Compute per-cluster OU parameters via weighted average of representations.
        Paper Equations (9-10).

        h: (batch, d_embed)
        y: (batch, n_clusters) — soft assignments
        Returns: theta (K,), mu (K,), sigma (K,)
        """
        # h_bar_k = sum(P_ik * h_i) / (sum(P_ik) + eps)
        # y.T @ h gives (n_clusters, d_embed)
        cluster_weights = y.sum(dim=0) + 1e-8  # (K,)
        h_bar = (y.T @ h) / cluster_weights.unsqueeze(-1)  # (K, d_embed)

        # Predict OU params per cluster
        ou_raw = self.g_ou(h_bar)  # (K, 3)

        theta = F.softplus(ou_raw[:, 0]) + 0.01  # theta > 0 (mean-reversion speed)
        mu = ou_raw[:, 1]                          # mu can be any real number
        sigma = F.softplus(ou_raw[:, 2]) + 0.01   # sigma > 0

        return theta, mu, sigma

    def augment_mask(self, x, mask_ratio=0.1):
        """Paper augmentation: random masking (set fraction to zero)."""
        mask = torch.bernoulli(torch.full_like(x, 1 - mask_ratio))
        return x * mask

    def augment_noise(self, x, std=0.1):
        """Paper augmentation: Gaussian noise."""
        return x + torch.randn_like(x) * std


# ============================================================
# 3. LOSS FUNCTIONS
# ============================================================

def instance_contrastive_loss(z_a, z_b, temperature=0.5):
    """
    Paper Equations (4-5): NT-Xent (SimCLR-style) instance loss.
    z_a, z_b: (N, d) — projections from two augmented views.
    """
    N = z_a.shape[0]
    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)

    # Similarity matrix between all pairs across views
    # z = [z_a; z_b], shape (2N, d)
    z = torch.cat([z_a, z_b], dim=0)
    sim = torch.mm(z, z.T) / temperature  # (2N, 2N)

    # Mask out self-similarity
    mask = torch.eye(2 * N, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    # Positive pairs: (i, i+N) and (i+N, i)
    labels = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)

    loss = F.cross_entropy(sim, labels)
    return loss


def cluster_contrastive_loss(y_a, y_b, temperature=1.0):
    """
    Paper Equations (6-7): Cluster-level contrastive loss + entropy regularization.
    y_a, y_b: (N, M) — soft cluster assignment probabilities.
    """
    N = y_a.shape[0]
    y_a_norm = F.normalize(y_a, dim=1)
    y_b_norm = F.normalize(y_b, dim=1)

    y = torch.cat([y_a_norm, y_b_norm], dim=0)
    sim = torch.mm(y, y.T) / temperature

    mask = torch.eye(2 * N, device=y.device).bool()
    sim.masked_fill_(mask, -1e9)

    labels = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(y.device)
    contrastive = F.cross_entropy(sim, labels)

    # Entropy regularization: maximize H(Y) to prevent cluster collapse
    # H(Y) = -sum(p_k * log(p_k)) where p_k = mean assignment to cluster k
    avg_y = (y_a.mean(dim=0) + y_b.mean(dim=0)) / 2 + 1e-8
    entropy = -(avg_y * torch.log(avg_y)).sum()

    return contrastive - entropy  # minimize contrastive, maximize entropy


def pinn_loss(returns, y, theta, mu, sigma, dt=1/12):
    """
    Paper Equation (11-12): Physics-informed OU regularization.

    returns: (N, T) — time series of returns for each asset (monthly)
    y: (N, K) — soft cluster assignments
    theta, mu, sigma: (K,) — OU params per cluster
    dt: time step (1/12 for monthly)
    """
    N, T = returns.shape
    K = theta.shape[0]

    total_loss = 0.0
    count = 0

    for k in range(K):
        P_k = y[:, k]  # (N,) soft weights for this cluster
        if P_k.sum() < 1e-6:
            continue

        # Residuals: R_ik(t) = (r(t) - r(t-1)) - theta_k * (mu_k - r(t-1)) * dt
        # Paper Equation (11)
        r_t = returns[:, 1:]     # (N, T-1)
        r_prev = returns[:, :-1]  # (N, T-1)

        predicted_drift = theta[k] * (mu[k] - r_prev) * dt  # (N, T-1)
        residuals = (r_t - r_prev) - predicted_drift          # (N, T-1)

        # E_t[R^2] per asset
        mean_sq_residuals = (residuals ** 2).mean(dim=1)  # (N,)

        # Negative log-likelihood: log(sigma) + E[R^2] / (2 * sigma^2 * dt)
        # Paper Equation (12)
        nll = torch.log(sigma[k]) + mean_sq_residuals / (2 * sigma[k] ** 2 * dt)  # (N,)

        # Weight by soft assignment
        weighted_nll = (P_k * nll).sum()
        total_loss += weighted_nll
        count += 1

    return total_loss / max(count, 1)


# ============================================================
# 4. TRAINING
# ============================================================

class MonthlyDataset(Dataset):
    """Dataset for one month: features + return history."""
    def __init__(self, features_tensor, returns_history_tensor):
        self.features = features_tensor
        self.returns_history = returns_history_tensor

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.returns_history[idx]


def train_orca(features_dict, returns_dict, valid_tickers,
               n_clusters=10, n_epochs=50, lr=0.002, batch_size=256,
               alpha=1.0, beta=1.0, n_bins=32, d_embed=64,
               mask_ratio=0.1, noise_std=0.1,
               train_end='2018-12-31', val_end='2019-12-31'):
    """
    Train ORCA following paper Section 4.1.5.

    Walk-forward: train on months <= train_end, validate on train_end..val_end.
    """
    # Split dates
    all_dates = sorted(features_dict.keys())
    train_dates = [d for d in all_dates if d <= pd.Timestamp(train_end)]
    val_dates = [d for d in all_dates if pd.Timestamp(train_end) < d <= pd.Timestamp(val_end)]
    test_dates = [d for d in all_dates if d > pd.Timestamp(val_end)]

    print(f"Dates: {len(train_dates)} train, {len(val_dates)} val, {len(test_dates)} test")

    if len(train_dates) < 24:
        print("WARNING: fewer than 24 training months. Results may be unreliable.")

    # Determine feature columns (from first available month)
    sample_df = features_dict[train_dates[0]]
    feature_cols = [c for c in sample_df.columns]
    n_features = len(feature_cols)
    print(f"Features: {n_features} columns: {feature_cols[:5]}...")

    # Build pooled training tensors
    # For each month, we have a (n_stocks, n_features) matrix
    # We also need return history for PINN loss
    # Strategy: pool all month-stock observations for training the encoder,
    # but keep monthly structure for PINN and trading

    train_features_list = []
    train_returns_hist_list = []

    # For PINN loss, we need per-asset return histories.
    # Collect 12-month rolling return windows per asset per month.
    for date in train_dates:
        df = features_dict[date]
        date_idx = all_dates.index(date)

        for ticker in df.index:
            row = df.loc[ticker, feature_cols].values.astype(np.float32)
            if np.isnan(row).sum() > 0:
                row = np.nan_to_num(row, 0)
            train_features_list.append(row)

            # Collect last 12 months of returns for this ticker
            ret_hist = []
            for lookback in range(min(12, date_idx + 1)):
                past_date = all_dates[date_idx - lookback]
                if past_date in returns_dict and ticker in returns_dict[past_date]:
                    ret_hist.append(returns_dict[past_date][ticker])
                else:
                    ret_hist.append(0.0)
            # Pad to 12 if needed
            while len(ret_hist) < 12:
                ret_hist.append(0.0)
            ret_hist = ret_hist[::-1]  # oldest first
            train_returns_hist_list.append(ret_hist)

    X_train = torch.tensor(np.array(train_features_list), dtype=torch.float32)
    R_train = torch.tensor(np.array(train_returns_hist_list), dtype=torch.float32)
    print(f"Training data: {X_train.shape[0]} stock-month observations, {n_features} features")

    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train.numpy())
    X_train = torch.tensor(X_train_np, dtype=torch.float32)

    # Initialize model
    model = ORCA(n_features, n_clusters=n_clusters, n_bins=n_bins,
                 d_embed=d_embed, n_heads=4, n_layers=2, dropout=0.1).to(device)

    # Set PLE boundaries from training data
    model.ple.set_boundaries(X_train.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    dataset = MonthlyDataset(X_train, R_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Training loop
    print(f"\nTraining ORCA: {n_epochs} epochs, {n_clusters} clusters")
    print(f"Loss = L_ins + {alpha}*L_clu + {beta}*L_pinn")

    best_loss = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        epoch_l_ins = 0
        epoch_l_clu = 0
        epoch_l_pinn = 0
        n_batches = 0

        for features_batch, returns_batch in loader:
            features_batch = features_batch.to(device)
            returns_batch = returns_batch.to(device)

            # Data augmentation (paper Section 3.2.1)
            x_a = model.augment_mask(features_batch, mask_ratio)
            x_b = model.augment_noise(features_batch, noise_std)

            # Forward pass on both views
            out_a = model(x_a)
            out_b = model(x_b)

            # Instance contrastive loss (Eq 4-5)
            L_ins = instance_contrastive_loss(out_a['z'], out_b['z'], temperature=0.5)

            # Cluster contrastive loss (Eq 6-7)
            L_clu = cluster_contrastive_loss(out_a['y'], out_b['y'], temperature=1.0)

            # PINN loss (Eq 11-12)
            # Use the first view's assignments for OU params
            y_avg = (out_a['y'] + out_b['y']) / 2
            h_avg = (out_a['h'] + out_b['h']) / 2
            theta, mu, sigma = model.get_ou_params(h_avg, y_avg)
            L_pinn = pinn_loss(returns_batch, y_avg, theta, mu, sigma, dt=1/12)

            # Total loss (Eq 13)
            loss = L_ins + alpha * L_clu + beta * L_pinn

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_l_ins += L_ins.item()
            epoch_l_clu += L_clu.item()
            epoch_l_pinn += L_pinn.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f} "
                  f"(ins={epoch_l_ins/n_batches:.4f} "
                  f"clu={epoch_l_clu/n_batches:.4f} "
                  f"pinn={epoch_l_pinn/n_batches:.4f})")

    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    print(f"Best training loss: {best_loss:.4f}")

    return model, scaler, feature_cols


# ============================================================
# 5. CLUSTER ASSIGNMENT (inference)
# ============================================================

def assign_clusters(model, scaler, features_df, feature_cols, device='cpu'):
    """
    Run trained ORCA on one month's features to get cluster assignments.
    Returns: dict {ticker: cluster_id}
    """
    tickers = features_df.index.tolist()
    X = features_df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, 0)
    X = scaler.transform(X)
    X_t = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        out = model(X_t)
        y = out['y']  # (N, K) soft assignments
        cluster_ids = y.argmax(dim=1).cpu().numpy()

    return {t: int(c) for t, c in zip(tickers, cluster_ids)}


def assign_clusters_kmeans(features_df, feature_cols, scaler, n_clusters=10):
    """
    K-means baseline (paper Section 4.1.3).
    """
    tickers = features_df.index.tolist()
    X = features_df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, 0)
    X = scaler.transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    return {t: int(c) for t, c in zip(tickers, labels)}


# ============================================================
# 6. TRADING STRATEGY (Paper Algorithm 1)
# ============================================================

def run_trading_strategy(cluster_assignments, returns_this_month, prev_month_returns, gamma=1.0):
    """
    Paper Algorithm 1: Cluster-Based Mean-Reversion Strategy.

    1. For each cluster, sort assets by prior month return (mom1)
    2. Compute momentum spread for each asset
    3. Long if spread < -gamma * sigma, short if spread > gamma * sigma
    4. Equal-weight portfolio of all signaled assets
    5. Return: log(1 + s(x) * R_{t+1}(x)) averaged over active assets

    Args:
        cluster_assignments: {ticker: cluster_id}
        returns_this_month: {ticker: float} — actual returns realized this month
        prev_month_returns: {ticker: float} — last month's returns (for ranking)
        gamma: threshold multiplier (paper: 1.0)

    Returns: portfolio log return for this month
    """
    # Group by cluster
    clusters = {}
    for ticker, cid in cluster_assignments.items():
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(ticker)

    # Compute momentum spread per asset
    all_spreads = {}
    for cid, members in clusters.items():
        if len(members) < 3:
            continue

        # Sort by prior month return
        member_rets = {t: prev_month_returns.get(t, 0) for t in members}
        sorted_members = sorted(member_rets.items(), key=lambda x: x[1])

        n = len(sorted_members)
        cluster_mean_ret = np.mean([r for _, r in sorted_members])

        for ticker, ret in sorted_members:
            # Spread = deviation from cluster mean
            spread = ret - cluster_mean_ret
            all_spreads[ticker] = spread

    if len(all_spreads) == 0:
        return 0.0

    # Threshold: gamma * std of all spreads
    spreads_array = np.array(list(all_spreads.values()))
    sigma_delta = spreads_array.std()
    if sigma_delta < 1e-8:
        return 0.0
    threshold = gamma * sigma_delta

    # Generate signals
    signals = {}
    for ticker, spread in all_spreads.items():
        if spread < -threshold:
            signals[ticker] = +1  # long (underperformed = buy)
        elif spread > threshold:
            signals[ticker] = -1  # short (overperformed = sell)

    if len(signals) == 0:
        return 0.0

    # Portfolio return
    log_returns = []
    for ticker, signal in signals.items():
        actual_ret = returns_this_month.get(ticker, 0)
        # Apply 10% stop-loss per position (paper Section 4.1.2)
        position_ret = signal * actual_ret
        position_ret = max(position_ret, -0.10)
        log_ret = np.log(1 + position_ret) if position_ret > -1 else -10
        log_returns.append(log_ret)

    return np.mean(log_returns)


# ============================================================
# 7. BACKTEST
# ============================================================

def run_backtest(model, scaler, feature_cols, features_dict, returns_dict,
                 start_date, end_date, n_clusters=10, gamma=1.0, method='orca'):
    """
    Monthly-rebalanced backtest (paper Section 4.1.2).
    """
    all_dates = sorted(features_dict.keys())
    test_dates = [d for d in all_dates if pd.Timestamp(start_date) <= d <= pd.Timestamp(end_date)]

    monthly_returns = []
    n_signals_total = 0

    for i, date in enumerate(test_dates):
        date_idx = all_dates.index(date)
        if date_idx < 1:
            continue

        prev_date = all_dates[date_idx - 1]
        feat_df = features_dict[date]

        # Cluster assignment
        if method == 'orca':
            assignments = assign_clusters(model, scaler, feat_df, feature_cols, device)
        elif method == 'kmeans':
            assignments = assign_clusters_kmeans(feat_df, feature_cols, scaler, n_clusters)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Previous month returns (for ranking within clusters)
        prev_rets = returns_dict.get(prev_date, {})

        # This month's actual returns (for P&L)
        this_rets = returns_dict.get(date, {})

        # Run Algorithm 1
        port_ret = run_trading_strategy(assignments, this_rets, prev_rets, gamma)
        monthly_returns.append({'date': date, 'return': port_ret})

    return pd.DataFrame(monthly_returns)


def compute_backtest_metrics(results_df):
    """Compute standard financial metrics from monthly returns."""
    rets = results_df['return'].values
    n = len(rets)
    if n < 12:
        return None

    cum_ret = np.exp(np.cumsum(rets))
    total_ret = cum_ret[-1] - 1
    ann_ret = (1 + total_ret) ** (12 / n) - 1

    monthly_std = np.std(rets)
    ann_vol = monthly_std * np.sqrt(12)

    sharpe = (ann_ret - 0.04) / ann_vol if ann_vol > 1e-8 else 0

    # Max drawdown
    peak = np.maximum.accumulate(cum_ret)
    dd = (cum_ret - peak) / peak
    mdd = abs(dd.min())

    # Sortino
    downside = rets[rets < 0]
    downside_std = np.std(downside) * np.sqrt(12) if len(downside) > 0 else ann_vol
    sortino = (ann_ret - 0.04) / downside_std if downside_std > 1e-8 else 0

    # Calmar
    calmar = ann_ret / mdd if mdd > 1e-8 else 0

    return {
        'AR': ann_ret, 'Vol': ann_vol, 'MDD': mdd,
        'Sharpe': sharpe, 'Sortino': sortino, 'Calmar': calmar,
        'Months': n
    }


# ============================================================
# 8. MAIN
# ============================================================

def main():
    print("=" * 70)
    print("  ORCA (yfinance adaptation)")
    print("  Kim, Na & Song (ICAIF 2025)")
    print("=" * 70)

    # --- Data ---
    os.makedirs('datasets', exist_ok=True)
    raw_data = download_data(TICKERS, start='2005-01-01', end='2023-12-31',
                             cache_path='datasets/orca_prices_expanded.parquet')
    features_dict, returns_dict, valid_tickers = compute_monthly_features(raw_data, TICKERS, n_mom=12)
    print(f"\nBuilt features for {len(features_dict)} months, {len(valid_tickers)} tickers")

    # Determine feature columns
    sample_df = features_dict[sorted(features_dict.keys())[0]]
    feature_cols = list(sample_df.columns)

    # --- Train ORCA ---
    # Paper: K=30 clusters for 3000+ stocks. We use K=15 for ~115 stocks (~8 per cluster).
    N_CLUSTERS = 15

    model, scaler, feature_cols = train_orca(
        features_dict, returns_dict, valid_tickers,
        n_clusters=N_CLUSTERS, n_epochs=80, lr=0.002, batch_size=512,
        alpha=1.0, beta=1.0, n_bins=32, d_embed=64,
        mask_ratio=0.1, noise_std=0.1,
        train_end='2018-12-31', val_end='2019-12-31'
    )

    # --- Backtest: ORCA vs K-means (paper Table 1 comparison) ---
    TEST_START = '2020-01-01'
    TEST_END = '2023-12-31'

    print(f"\n{'='*70}")
    print(f"  BACKTEST: {TEST_START} to {TEST_END}")
    print(f"{'='*70}")

    results = {}
    for method in ['orca', 'kmeans']:
        res = run_backtest(model, scaler, feature_cols, features_dict, returns_dict,
                           TEST_START, TEST_END, n_clusters=N_CLUSTERS, gamma=1.0,
                           method=method)
        metrics = compute_backtest_metrics(res)
        results[method] = metrics
        print(f"\n  {method.upper()}:")
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
        else:
            print("    Insufficient data")

    # --- Summary table ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY (paper Table 1 format)")
    print(f"{'='*70}")
    print(f"{'Method':<12} {'AR':>8} {'Vol':>8} {'MDD':>8} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8}")
    print("-" * 68)
    for method, m in results.items():
        if m:
            print(f"{method.upper():<12} {m['AR']:>+8.4f} {m['Vol']:>8.4f} {m['MDD']:>8.4f} "
                  f"{m['Sharpe']:>8.4f} {m['Sortino']:>8.4f} {m['Calmar']:>8.4f}")

    # --- Save artifacts ---
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_features': len(feature_cols),
        'n_clusters': N_CLUSTERS,
        'feature_cols': feature_cols,
    }, 'datasets/orca_model.pt')

    with open('datasets/orca_artifacts.pkl', 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'feature_cols': feature_cols,
            'valid_tickers': valid_tickers,
            'n_clusters': N_CLUSTERS,
        }, f)

    print(f"\nSaved model to datasets/orca_model.pt")
    print(f"Saved artifacts to datasets/orca_artifacts.pkl")

    # --- Connection to your DQN pipeline ---
    print(f"\n{'='*70}")
    print(f"  NEXT STEP: Connect to DQN threshold agent")
    print(f"{'='*70}")
    print(f"""
  To use ORCA clusters with your DQN adaptive threshold:

  1. Run this script first to generate clusters
  2. In train_adaptive_threshold.py, replace the ADF-based pair selection
     with ORCA cluster assignments:

       # Instead of: all cointegrated pairs from ADF test
       # Use: within-cluster pairs from ORCA
       clusters = assign_clusters(model, scaler, features_df, feature_cols)
       for cluster_id in set(clusters.values()):
           members = [t for t, c in clusters.items() if c == cluster_id]
           # form pairs within this cluster
           for a, b in combinations(members, 2):
               ...  # compute spread, OU params, etc.

  This gives Emily's 2x2 ablation:
    - K-means + static threshold (baseline)
    - K-means + DQN threshold  (does DQN help with simple clusters?)
    - ORCA + static threshold  (does ORCA help with static execution?)
    - ORCA + DQN threshold     (full system)
    """)


if __name__ == '__main__':
    main()
