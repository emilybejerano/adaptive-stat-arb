"""
2×2 Ablation: Clustering Method × Threshold Method

  Rows:    ORCA clusters vs K-means clusters
  Columns: Static 1.0σ threshold vs DQN adaptive threshold

This script:
  1. Loads the trained ORCA model + price data
  2. Gets cluster assignments (ORCA + K-means) at end of training period
  3. Forms within-cluster pairs, filters by ADF cointegration
  4. Computes spreads (vectorized rolling OLS) + OU parameters
  5. Trains a DQN threshold agent on each set of pairs
  6. Backtests all 4 cells: {ORCA, K-means} × {static, DQN}
  7. Prints comparison table with Wilcoxon tests

Prerequisites:
  - Run orca_yfinance.py first (produces datasets/orca_model.pt, orca_artifacts.pkl,
    orca_prices_expanded.parquet)

Usage:
  python ablation_2x2.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from itertools import combinations
from collections import deque, namedtuple
from scipy import stats
import pickle, os, random, warnings

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ============================================================
# TRADING CONSTANTS (same as train_adaptive_threshold.py)
# ============================================================
THRESHOLDS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
K_ACTIONS = len(THRESHOLDS)
CAPITAL = 100000.0
POSITION_FRAC = 0.10
TC = 0.001
MAX_DAYS = 60
EXIT_Z = 0.25
WEEKS_PER_EPISODE = 4
TRADING_DAYS_PER_WEEK = 5
STATE_DIM = 7

TRAIN_END = '2018-12-31'
VAL_START = '2019-01-01'; VAL_END = '2019-12-31'
TEST_START = '2020-01-01'; TEST_END = '2023-12-31'


# ============================================================
# 1. LOAD DATA + ORCA MODEL
# ============================================================

def load_orca_model():
    """Load trained ORCA model from orca_yfinance.py output."""
    # Import ORCA class
    from orca_yfinance import (ORCA, assign_clusters, assign_clusters_kmeans,
                                compute_monthly_features, TICKERS)

    # Load artifacts
    with open('datasets/orca_artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)

    scaler = artifacts['scaler']
    feature_cols = artifacts['feature_cols']
    n_clusters = artifacts['n_clusters']
    valid_tickers = artifacts['valid_tickers']
    n_features = len(feature_cols)

    # Load model
    ckpt = torch.load('datasets/orca_model.pt', map_location=device, weights_only=False)
    model = ORCA(n_features, n_clusters=n_clusters, n_bins=32, d_embed=64,
                 n_heads=4, n_layers=2, dropout=0.1).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load price data
    raw_data = pd.read_parquet('datasets/orca_prices_expanded.parquet')

    # Compute monthly features (needed for cluster assignment)
    features_dict, returns_dict, valid_tickers = compute_monthly_features(
        raw_data, TICKERS, n_mom=12)

    return model, scaler, feature_cols, features_dict, returns_dict, valid_tickers, n_clusters


def get_daily_prices(valid_tickers):
    """Extract daily close prices from the cached parquet."""
    raw = pd.read_parquet('datasets/orca_prices_expanded.parquet')
    prices = pd.DataFrame()
    for t in valid_tickers:
        col = f'{t}_close'
        if col in raw.columns:
            prices[t] = raw[col]

    # Also need VIX — download separately if not in cache
    vix_path = 'datasets/vix_daily.parquet'
    if os.path.exists(vix_path):
        vix = pd.read_parquet(vix_path)['VIX']
    else:
        try:
            import yfinance as yf
            vix_data = yf.download('^VIX', start='2005-01-01', end='2023-12-31',
                                   auto_adjust=True, progress=False)
            vix = vix_data['Close']
            vix.name = 'VIX'
            vix.to_frame().to_parquet(vix_path)
        except Exception:
            print("WARNING: Could not download VIX. Using placeholder.")
            vix = pd.Series(20.0, index=prices.index, name='VIX')

    prices['VIX'] = vix.reindex(prices.index).ffill().bfill()
    return prices


# ============================================================
# 2. CLUSTER -> PAIRS (with ADF filter)
# ============================================================

def clusters_to_pairs(cluster_assignments, daily_prices, train_end, min_common=500, adf_pvalue=0.10):
    """
    Form within-cluster pairs, filter by ADF cointegration test.

    Args:
        cluster_assignments: {ticker: cluster_id}
        daily_prices: DataFrame of daily close prices
        train_end: only use data up to this date for ADF testing
        min_common: minimum overlapping trading days
        adf_pvalue: ADF test significance level

    Returns: list of pair name strings "A/B"
    """
    # Group tickers by cluster
    clusters = {}
    for t, cid in cluster_assignments.items():
        clusters.setdefault(cid, []).append(t)

    train_prices = daily_prices[daily_prices.index <= train_end]

    cointegrated = []
    total_tested = 0

    for cid, members in clusters.items():
        if len(members) < 2:
            continue
        for a, b in combinations(sorted(members), 2):
            if a not in train_prices.columns or b not in train_prices.columns:
                continue

            pa = train_prices[a].dropna()
            pb = train_prices[b].dropna()
            common = pa.index.intersection(pb.index)
            if len(common) < min_common:
                continue

            total_tested += 1

            # ADF test on last 2 years of training data
            pa_2y = pa.loc[common].iloc[-504:]
            pb_2y = pb.loc[common].iloc[-504:]

            try:
                X = sm.add_constant(pa_2y.values)
                beta = sm.OLS(pb_2y.values, X).fit().params[1]
                spread = pb_2y - beta * pa_2y
                _, pvalue = adfuller(spread.dropna(), maxlag=20)[:2]
            except Exception:
                continue

            if pvalue <= adf_pvalue:
                cointegrated.append(f"{a}/{b}")

    print(f"    Tested {total_tested} within-cluster pairs, {len(cointegrated)} pass ADF (p<={adf_pvalue})")
    return cointegrated


# ============================================================
# 3. COMPUTE SPREADS + OU PARAMS (vectorized)
# ============================================================

def compute_spreads(pair_names, daily_prices, window=252):
    """
    Compute rolling-OLS hedge ratio spreads for all pairs.
    Vectorized: beta = rolling_cov(x,y) / rolling_var(x)
    """
    all_spreads = {}
    for i, name in enumerate(pair_names):
        a, b = name.split('/')
        if a not in daily_prices.columns or b not in daily_prices.columns:
            continue

        pa = daily_prices[a].dropna()
        pb = daily_prices[b].dropna()
        common = pa.index.intersection(pb.index)
        if len(common) < window + 100:
            continue

        pa = pa.loc[common]
        pb = pb.loc[common]

        # Vectorized rolling beta
        rolling_cov = pa.rolling(window).cov(pb)
        rolling_var = pa.rolling(window).var()
        beta = (rolling_cov / rolling_var.replace(0, np.nan))

        spread = pb - beta * pa
        spread = spread.dropna()

        if len(spread) >= 500:
            all_spreads[name] = spread

        if (i + 1) % 50 == 0:
            print(f"    Spreads: {i+1}/{len(pair_names)}...")

    print(f"    Computed spreads for {len(all_spreads)} pairs")
    return all_spreads


def compute_ou_params(all_spreads, dt=1/252):
    """
    Rolling 60-day AR(1) OU parameter estimation.
    Vectorized inner loop using pandas rolling.
    """
    all_ou = {}
    names = list(all_spreads.keys())

    for i, name in enumerate(names):
        spread = all_spreads[name]

        # Rolling AR(1): y_t = a + b * y_{t-1} + eps
        y = spread.iloc[1:]
        x = spread.iloc[:-1]
        x.index = y.index  # align

        # Rolling stats (60-day window)
        w = 60
        roll_mean_y = y.rolling(w).mean()
        roll_mean_x = x.rolling(w).mean()
        roll_cov = y.rolling(w).cov(x)
        roll_var_x = x.rolling(w).var()

        b_hat = (roll_cov / roll_var_x.replace(0, np.nan)).dropna()
        a_hat = (roll_mean_y - b_hat * roll_mean_x).dropna()

        # OU parameters
        common = b_hat.dropna().index.intersection(a_hat.dropna().index)
        b_hat = b_hat.loc[common]
        a_hat = a_hat.loc[common]

        # Filter valid: 0 < b < 1
        valid = (b_hat > 0) & (b_hat < 1)
        b_hat = b_hat[valid]
        a_hat = a_hat[valid]

        if len(b_hat) < 100:
            continue

        theta = -np.log(b_hat) / dt
        theta = theta.clip(0.01, 20)
        mu = a_hat / (1 - b_hat)
        half_life = np.log(2) / theta * 252

        # Residual variance estimate
        resid_var = roll_var_x.loc[b_hat.index] * (1 - b_hat ** 2)
        sigma = np.sqrt(2 * theta * resid_var.abs() / (1 - b_hat ** 2)).replace([np.inf, -np.inf], np.nan)

        ou_df = pd.DataFrame({
            'theta': theta, 'mu': mu, 'sigma': sigma, 'half_life': half_life,
        }).dropna()

        if len(ou_df) >= 100:
            all_ou[name] = ou_df

        if (i + 1) % 50 == 0:
            print(f"    OU params: {i+1}/{len(names)}...")

    print(f"    Computed OU params for {len(all_ou)} pairs")
    return all_ou


# ============================================================
# 4. DQN THRESHOLD AGENT (from train_adaptive_threshold.py)
# ============================================================

class QNetwork(nn.Module):
    """Paper: 3-layer MLP with {20, 10, K} neurons."""
    def __init__(self, state_dim=STATE_DIM, action_dim=K_ACTIONS):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 20), nn.ReLU(),
            nn.Linear(20, 10), nn.ReLU(),
            nn.Linear(10, action_dim),
        )
    def forward(self, x):
        return self.network(x)


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DQNAgent:
    def __init__(self, gamma=0.9, lr=0.0001, batch_size=1024,
                 eps_start=0.5, eps_end=0.1, eps_decay_rate=0.005):
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay_rate = eps_decay_rate

        self.q_network = QNetwork().to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=160000)

    def select_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(K_ACTIONS)
        with torch.no_grad():
            self.q_network.eval()
            q = self.q_network(torch.FloatTensor(state).unsqueeze(0).to(device))
            self.q_network.train()
            return q.argmax(dim=1).item()

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*batch))
        states = torch.FloatTensor(np.array(batch.state)).to(device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(batch.reward).to(device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(device)
        dones = torch.FloatTensor(batch.done).to(device)

        current_q = self.q_network(states).gather(1, actions).squeeze()
        with torch.no_grad():
            max_next_q = self.q_network(next_states).max(dim=1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon - self.eps_decay_rate)

    def push_transition(self, *args):
        self.memory.append(Transition(*args))


# ============================================================
# 5. TRAINING + BACKTESTING FUNCTIONS
# ============================================================

def get_market_context(vix_series, ou_df, date):
    """Get normalized VIX and theta for state."""
    vix = vix_series.loc[date] if date in vix_series.index else 20.0
    vix_norm = min(vix / 80.0, 1.0)
    theta = ou_df.loc[date, 'theta'] if date in ou_df.index else 5.0
    theta_norm = min(theta / 20.0, 1.0)
    return vix_norm, theta_norm


def generate_monthly_periods(t0, t1):
    """Generate (month_start, month_end) tuples."""
    periods = []
    current = pd.Timestamp(t0)
    end = pd.Timestamp(t1)
    while current < end:
        month_end = current + pd.offsets.MonthEnd(1)
        if month_end > end:
            month_end = end
        periods.append((str(current.date()), str(month_end.date())))
        current = month_end + pd.Timedelta(days=1)
    return periods


def run_episode(agent, spread, z_full, vix, ou_df, month_start, month_end, training=True):
    """One episode = one month of trading. Agent picks threshold each week."""
    s = spread[(spread.index >= month_start) & (spread.index <= month_end)]
    z = z_full[(z_full.index >= month_start) & (z_full.index <= month_end)]
    if len(z) < 5:
        return 0, []

    common = s.index.intersection(z.index)
    s = s.loc[common]; z = z.loc[common]
    dates = z.index
    n_days = len(dates)
    week_size = max(n_days // WEEKS_PER_EPISODE, 1)

    cum_wins = 0; cum_losses = 0; n_trades = 0
    portfolio = CAPITAL; pos = 0; ep_price = 0; ps = 0; dh = 0
    max_trades = 10
    episode_reward = 0

    for week in range(WEEKS_PER_EPISODE):
        w_start = week * week_size
        w_end = min((week + 1) * week_size, n_days)
        if w_start >= n_days:
            break

        # Build state
        w_norm = week / WEEKS_PER_EPISODE
        s_norm = cum_wins / max(CAPITAL * 0.1, 1)
        l_norm = cum_losses / max(CAPITAL * 0.1, 1)
        cc_norm = n_trades / max(max_trades, 1)
        t_norm = 0.5
        mid = dates[min((w_start + w_end) // 2, n_days - 1)]
        vix_n, theta_n = get_market_context(vix, ou_df, mid)
        state = np.array([w_norm, s_norm, l_norm, cc_norm, t_norm, vix_n, theta_n], dtype=np.float32)

        action = agent.select_action(state, eval_mode=not training)
        threshold = THRESHOLDS[action]

        # Simulate week
        week_wins = 0; week_losses = 0; week_trades = 0
        for i in range(w_start, min(w_end, n_days)):
            sv = s.iloc[i]; zv = z.iloc[i]
            if pos != 0: dh += 1
            if pos != 0 and (abs(zv) < EXIT_Z or dh >= MAX_DAYS):
                pnl = pos * (sv - ep_price) * ps
                cost = TC * abs(ps) * abs(sv)
                net = pnl - cost; portfolio += net
                if net > 0: week_wins += net
                else: week_losses += abs(net)
                week_trades += 1; pos = 0; ps = 0; dh = 0
            if pos == 0 and abs(zv) > threshold:
                pos = -1 if zv > threshold else 1
                ep_price = sv
                ps = CAPITAL * POSITION_FRAC / max(abs(sv), 0.01)
                portfolio -= TC * abs(ps) * abs(sv); dh = 0

        cum_wins += week_wins; cum_losses += week_losses; n_trades += week_trades

        reward = (week_wins - week_losses) / (CAPITAL * 0.01) * (week + 1)
        if n_trades > max_trades:
            reward -= 1.0 * (n_trades - max_trades)

        # Next state
        next_w = (week + 1) / WEEKS_PER_EPISODE
        next_state = np.array([next_w, cum_wins / max(CAPITAL * 0.1, 1),
                                cum_losses / max(CAPITAL * 0.1, 1),
                                n_trades / max(max_trades, 1),
                                action / (K_ACTIONS - 1),
                                vix_n, theta_n], dtype=np.float32)

        done = (week == WEEKS_PER_EPISODE - 1) or (w_end >= n_days)
        if training:
            agent.push_transition(state, action, reward, next_state, float(done))
            agent.optimize()

        episode_reward += reward

    return episode_reward, []


def train_dqn_agent(pair_names, all_spreads, all_ou, df_prices, n_iterations=200):
    """Train a DQN threshold agent on the given pairs."""
    agent = DQNAgent()
    train_months = generate_monthly_periods('2010-01-01', TRAIN_END)
    val_months = generate_monthly_periods(VAL_START, VAL_END)

    # Precompute z-scores for all pairs
    all_z = {}
    vix = df_prices['VIX'] if 'VIX' in df_prices.columns else pd.Series(20, index=df_prices.index)
    for name in pair_names:
        spread = all_spreads[name]
        rm = spread.rolling(60).mean()
        rs = spread.rolling(60).std()
        all_z[name] = ((spread - rm) / rs.replace(0, np.nan)).dropna()

    best_val_reward = -np.inf
    best_state = None

    for iteration in range(n_iterations):
        iter_reward = 0; n_eps = 0

        sampled_months = random.sample(train_months, min(12, len(train_months)))
        sampled_pairs = random.sample(pair_names, min(8, len(pair_names)))

        for ms, me in sampled_months:
            for pair in sampled_pairs:
                if pair not in all_z or pair not in all_ou:
                    continue
                ep_r, _ = run_episode(agent, all_spreads[pair], all_z[pair],
                                       vix, all_ou[pair], ms, me, training=True)
                iter_reward += ep_r; n_eps += 1

        agent.decay_epsilon()

        if (iteration + 1) % 50 == 0:
            val_reward = 0
            for ms, me in val_months:
                for pair in pair_names[:min(10, len(pair_names))]:
                    if pair not in all_z or pair not in all_ou:
                        continue
                    ep_r, _ = run_episode(agent, all_spreads[pair], all_z[pair],
                                           vix, all_ou[pair], ms, me, training=False)
                    val_reward += ep_r

            if val_reward > best_val_reward:
                best_val_reward = val_reward
                best_state = {k: v.clone() for k, v in agent.q_network.state_dict().items()}

            avg = iter_reward / max(n_eps, 1)
            print(f"      Iter {iteration+1:3d}: train={avg:.3f} val={val_reward:.3f} "
                  f"best_val={best_val_reward:.3f} eps={agent.epsilon:.3f}")

    if best_state:
        agent.q_network.load_state_dict(best_state)
    return agent


def backtest_pair(spread, z_full, vix, ou_df, t0, t1, threshold=None, agent=None):
    """
    Backtest one pair with either static threshold or DQN agent.
    Returns {values: [...], trades: [...]}
    """
    s = spread[(spread.index >= t0) & (spread.index <= t1)]
    z = z_full[(z_full.index >= t0) & (z_full.index <= t1)]
    if len(z) < 20:
        return None

    common = s.index.intersection(z.index)
    s = s.loc[common]; z = z.loc[common]
    dates = z.index
    n_days = len(dates)

    port = CAPITAL; pos = 0; ep2 = 0; ps = 0; dh = 0
    dv = []; trades = []

    if agent is not None:
        # DQN: pick threshold each week, reset state each month
        week_size = TRADING_DAYS_PER_WEEK
        cum_wins = 0; cum_losses = 0; n_trades_month = 0
        current_month = None

        for week_start in range(0, n_days, week_size):
            week_end = min(week_start + week_size, n_days)
            week_num = week_start // week_size

            # Reset cumulative state each month (fix for the bug I flagged)
            this_month = dates[week_start].month
            if current_month is not None and this_month != current_month:
                cum_wins = 0; cum_losses = 0; n_trades_month = 0
            current_month = this_month

            w_norm = (week_num % WEEKS_PER_EPISODE) / WEEKS_PER_EPISODE
            s_norm = cum_wins / max(CAPITAL * 0.1, 1)
            l_norm = cum_losses / max(CAPITAL * 0.1, 1)
            cc_norm = n_trades_month / 10.0
            t_norm = 0.5
            mid = dates[min((week_start + week_end) // 2, n_days - 1)]
            vix_n, theta_n = get_market_context(vix, ou_df, mid)
            state = np.array([w_norm, s_norm, l_norm, cc_norm, t_norm,
                              vix_n, theta_n], dtype=np.float32)
            with torch.no_grad():
                q = agent.q_network(torch.FloatTensor(state).unsqueeze(0).to(device))
                action = q.argmax(dim=1).item()
            thresh = THRESHOLDS[action]

            for i in range(week_start, week_end):
                sv = s.iloc[i]; zv = z.iloc[i]
                unr = pos * (sv - ep2) * ps if pos != 0 else 0
                if pos != 0: dh += 1
                dv.append(port + unr)
                if pos != 0 and (abs(zv) < EXIT_Z or dh >= MAX_DAYS):
                    pnl = pos * (sv - ep2) * ps; cost = TC * abs(ps) * abs(sv)
                    net = pnl - cost; port += net
                    trades.append({'pnl': net, 'days_held': dh})
                    if net > 0: cum_wins += net
                    else: cum_losses += abs(net)
                    n_trades_month += 1; pos = 0; dh = 0
                if pos == 0 and abs(zv) > thresh:
                    pos = -1 if zv > thresh else 1; ep2 = sv
                    ps = CAPITAL * POSITION_FRAC / max(abs(sv), 0.01)
                    port -= TC * abs(ps) * abs(sv); dh = 0
    else:
        # Static threshold
        for i in range(n_days):
            sv = s.iloc[i]; zv = z.iloc[i]
            unr = pos * (sv - ep2) * ps if pos != 0 else 0
            if pos != 0: dh += 1
            dv.append(port + unr)
            if pos != 0 and (abs(zv) < EXIT_Z or dh >= MAX_DAYS):
                pnl = pos * (sv - ep2) * ps; cost = TC * abs(ps) * abs(sv)
                port += pnl - cost; trades.append({'pnl': pnl - cost, 'days_held': dh}); pos = 0; dh = 0
            if pos == 0 and abs(zv) > threshold:
                pos = -1 if zv > threshold else 1; ep2 = sv
                ps = CAPITAL * POSITION_FRAC / max(abs(sv), 0.01); port -= TC * abs(ps) * abs(sv); dh = 0

    if pos != 0:
        pnl = pos * (s.iloc[-1] - ep2) * ps; port += pnl
        trades.append({'pnl': pnl, 'days_held': dh})

    return {'values': dv, 'trades': trades}


def calc_metrics(result):
    """Compute Sharpe, trap rate, etc."""
    if result is None or len(result['values']) < 20:
        return None
    vals = np.maximum(np.array(result['values']), 1.0)
    dr = np.diff(vals) / vals[:-1]; dr = dr[np.isfinite(dr)]
    if len(dr) < 20: return None
    nd = len(dr); tr = vals[-1] / vals[0] - 1
    ar = (1 + tr) ** (252 / nd) - 1
    av = np.std(dr) * np.sqrt(252)
    sh = (ar - 0.04) / av if av > 1e-8 else -10
    nt = len(result['trades'])
    if nt > 0:
        pnls = [t['pnl'] for t in result['trades']]
        trap = sum(1 for p in pnls if p < 0) / nt
    else:
        trap = 0
    return {'Sharpe': sh, 'Trap': trap, 'Trades': nt, 'AR': ar}


def run_full_backtest(pair_names, all_spreads, all_ou, df_prices, agent=None, threshold=1.0):
    """Backtest all pairs with either static threshold or DQN agent."""
    vix = df_prices['VIX'] if 'VIX' in df_prices.columns else pd.Series(20, index=df_prices.index)

    sharpes = {}  # keyed by pair for proper Wilcoxon pairing
    traps = {}
    trades_count = {}

    for pair in pair_names:
        if pair not in all_spreads or pair not in all_ou:
            continue

        spread = all_spreads[pair]
        rm = spread.rolling(60).mean()
        rs = spread.rolling(60).std()
        z_full = ((spread - rm) / rs.replace(0, np.nan)).dropna()

        result = backtest_pair(spread, z_full, vix, all_ou[pair],
                                TEST_START, TEST_END,
                                threshold=threshold if agent is None else None,
                                agent=agent)
        m = calc_metrics(result)
        if m:
            sharpes[pair] = m['Sharpe']
            traps[pair] = m['Trap']
            trades_count[pair] = m['Trades']

    return sharpes, traps, trades_count


# ============================================================
# 6. MAIN: RUN THE 2×2 ABLATION
# ============================================================

def main():
    print("=" * 70)
    print("  2×2 ABLATION: {ORCA, K-means} × {Static, DQN}")
    print("=" * 70)

    # --- Load ORCA model + data ---
    print("\n[1/7] Loading ORCA model and data...")
    model, scaler, feature_cols, features_dict, returns_dict, valid_tickers, n_clusters = load_orca_model()
    df_prices = get_daily_prices(valid_tickers)
    print(f"  {len(valid_tickers)} tickers, prices shape: {df_prices.shape}")

    # --- Get cluster assignments at end of training period ---
    print("\n[2/7] Getting cluster assignments...")
    # Find the last training-period month
    train_dates = sorted([d for d in features_dict.keys() if d <= pd.Timestamp(TRAIN_END)])
    last_train_date = train_dates[-1]
    train_features_df = features_dict[last_train_date]
    print(f"  Using features from {last_train_date.date()} ({len(train_features_df)} stocks)")

    from orca_yfinance import assign_clusters, assign_clusters_kmeans

    orca_clusters = assign_clusters(model, scaler, train_features_df, feature_cols, device)
    kmeans_clusters = assign_clusters_kmeans(train_features_df, feature_cols, scaler, n_clusters)

    # Print cluster sizes
    for name, clust in [('ORCA', orca_clusters), ('K-means', kmeans_clusters)]:
        sizes = {}
        for t, c in clust.items():
            sizes[c] = sizes.get(c, 0) + 1
        print(f"  {name}: {len(sizes)} clusters, sizes: {sorted(sizes.values(), reverse=True)}")

    # --- Form pairs from clusters ---
    print("\n[3/7] Forming within-cluster pairs (ADF filtered)...")
    print("  ORCA pairs:")
    orca_pairs = clusters_to_pairs(orca_clusters, df_prices, TRAIN_END)
    print("  K-means pairs:")
    kmeans_pairs = clusters_to_pairs(kmeans_clusters, df_prices, TRAIN_END)

    # --- Compute spreads ---
    print("\n[4/7] Computing spreads...")
    all_pair_names = list(set(orca_pairs + kmeans_pairs))
    print(f"  Total unique pairs: {len(all_pair_names)}")
    all_spreads = compute_spreads(all_pair_names, df_prices)

    # --- Compute OU params ---
    print("\n[5/7] Computing OU parameters...")
    all_ou = compute_ou_params(all_spreads)

    # Filter pair lists to only those with valid spreads + OU
    orca_pairs = [p for p in orca_pairs if p in all_spreads and p in all_ou]
    kmeans_pairs = [p for p in kmeans_pairs if p in all_spreads and p in all_ou]
    print(f"  ORCA pairs with valid data: {len(orca_pairs)}")
    print(f"  K-means pairs with valid data: {len(kmeans_pairs)}")

    if len(orca_pairs) < 5 or len(kmeans_pairs) < 5:
        print("ERROR: Not enough valid pairs. Try relaxing ADF threshold or adding more tickers.")
        return

    # --- Train DQN agents ---
    print("\n[6/7] Training DQN agents...")

    print(f"\n  Training DQN on ORCA pairs ({len(orca_pairs)} pairs)...")
    orca_agent = train_dqn_agent(orca_pairs, all_spreads, all_ou, df_prices, n_iterations=200)

    print(f"\n  Training DQN on K-means pairs ({len(kmeans_pairs)} pairs)...")
    kmeans_agent = train_dqn_agent(kmeans_pairs, all_spreads, all_ou, df_prices, n_iterations=200)

    # --- Run all 4 backtests ---
    print(f"\n[7/7] Backtesting all 4 cells (test: {TEST_START} to {TEST_END})...")

    cells = {}

    print("  ORCA + Static 1.0σ...")
    cells['ORCA+Static'] = run_full_backtest(orca_pairs, all_spreads, all_ou, df_prices, threshold=1.0)

    print("  ORCA + DQN...")
    cells['ORCA+DQN'] = run_full_backtest(orca_pairs, all_spreads, all_ou, df_prices, agent=orca_agent)

    print("  K-means + Static 1.0σ...")
    cells['KM+Static'] = run_full_backtest(kmeans_pairs, all_spreads, all_ou, df_prices, threshold=1.0)

    print("  K-means + DQN...")
    cells['KM+DQN'] = run_full_backtest(kmeans_pairs, all_spreads, all_ou, df_prices, agent=kmeans_agent)

    # --- Results table ---
    print(f"\n{'='*75}")
    print(f"  2×2 ABLATION RESULTS (test 2020-2023)")
    print(f"{'='*75}")
    print(f"\n{'Cell':<20} {'n pairs':>8} {'Sharpe':>8} {'Trap%':>8} {'Trades':>8}")
    print("-" * 56)

    for name, (sharpes, traps, trades) in cells.items():
        n = len(sharpes)
        if n == 0:
            print(f"{name:<20} {'N/A':>8}")
            continue
        mean_sh = np.mean(list(sharpes.values()))
        mean_tr = np.mean(list(traps.values())) * 100
        total_trades = sum(trades.values())
        print(f"{name:<20} {n:>8d} {mean_sh:>+8.3f} {mean_tr:>7.1f}% {total_trades:>8d}")

    # --- Formatted 2×2 matrix ---
    print(f"\n{'='*75}")
    print(f"  2×2 MATRIX")
    print(f"{'='*75}")
    print(f"\n{'':>20} {'Static 1.0σ':>15} {'DQN Adaptive':>15}")
    print("-" * 50)

    for cluster_name, static_key, dqn_key in [
        ('ORCA', 'ORCA+Static', 'ORCA+DQN'),
        ('K-means', 'KM+Static', 'KM+DQN')
    ]:
        static_sh = np.mean(list(cells[static_key][0].values())) if cells[static_key][0] else float('nan')
        dqn_sh = np.mean(list(cells[dqn_key][0].values())) if cells[dqn_key][0] else float('nan')
        static_tr = np.mean(list(cells[static_key][1].values())) * 100 if cells[static_key][1] else float('nan')
        dqn_tr = np.mean(list(cells[dqn_key][1].values())) * 100 if cells[dqn_key][1] else float('nan')

        print(f"{'':>20} {'Sharpe':>7} {'Trap':>7} {'Sharpe':>7} {'Trap':>7}")
        print(f"{cluster_name:<20} {static_sh:>+7.3f} {static_tr:>6.1f}% {dqn_sh:>+7.3f} {dqn_tr:>6.1f}%")

    # --- Statistical tests ---
    print(f"\n{'='*75}")
    print(f"  STATISTICAL TESTS (Wilcoxon signed-rank)")
    print(f"{'='*75}")

    # Test 1: Does DQN help within ORCA clusters?
    orca_static_sh = cells['ORCA+Static'][0]
    orca_dqn_sh = cells['ORCA+DQN'][0]
    orca_static_tr = cells['ORCA+Static'][1]
    orca_dqn_tr = cells['ORCA+DQN'][1]

    common_orca = sorted(set(orca_static_sh.keys()) & set(orca_dqn_sh.keys()))
    if len(common_orca) >= 5:
        sh_s = [orca_static_sh[p] for p in common_orca]
        sh_d = [orca_dqn_sh[p] for p in common_orca]
        tr_s = [orca_static_tr[p] for p in common_orca]
        tr_d = [orca_dqn_tr[p] for p in common_orca]

        _, p_sh = stats.wilcoxon(sh_d, sh_s)
        _, p_tr = stats.wilcoxon(tr_d, tr_s)
        print(f"\n  DQN vs Static (within ORCA clusters, n={len(common_orca)}):")
        print(f"    Sharpe: {np.mean(sh_d)-np.mean(sh_s):+.4f} (p={p_sh:.6f})")
        print(f"    Trap:   {(np.mean(tr_d)-np.mean(tr_s))*100:+.1f}pp (p={p_tr:.6f})")

    # Test 2: Does DQN help within K-means clusters?
    km_static_sh = cells['KM+Static'][0]
    km_dqn_sh = cells['KM+DQN'][0]
    km_static_tr = cells['KM+Static'][1]
    km_dqn_tr = cells['KM+DQN'][1]

    common_km = sorted(set(km_static_sh.keys()) & set(km_dqn_sh.keys()))
    if len(common_km) >= 5:
        sh_s = [km_static_sh[p] for p in common_km]
        sh_d = [km_dqn_sh[p] for p in common_km]
        tr_s = [km_static_tr[p] for p in common_km]
        tr_d = [km_dqn_tr[p] for p in common_km]

        _, p_sh = stats.wilcoxon(sh_d, sh_s)
        _, p_tr = stats.wilcoxon(tr_d, tr_s)
        print(f"\n  DQN vs Static (within K-means clusters, n={len(common_km)}):")
        print(f"    Sharpe: {np.mean(sh_d)-np.mean(sh_s):+.4f} (p={p_sh:.6f})")
        print(f"    Trap:   {(np.mean(tr_d)-np.mean(tr_s))*100:+.1f}pp (p={p_tr:.6f})")

    # Test 3: Does ORCA help vs K-means (with static threshold)?
    # Need pairs present in both — use Sharpe averaged per-pair
    print(f"\n  ORCA vs K-means (with static threshold):")
    print(f"    ORCA:    {len(orca_static_sh)} pairs, mean Sharpe {np.mean(list(orca_static_sh.values())):+.3f}")
    print(f"    K-means: {len(km_static_sh)} pairs, mean Sharpe {np.mean(list(km_static_sh.values())):+.3f}")

    print(f"\n{'='*75}")
    print(f"  INTERPRETATION GUIDE")
    print(f"{'='*75}")
    print(f"""
  The 2×2 answers three questions:

  1. Does DQN improve execution? (compare columns within each row)
     → If DQN column has lower trap rate, the adaptive threshold helps.

  2. Does ORCA improve pair selection? (compare rows within each column)
     → If ORCA row has better Sharpe, physics-informed clustering helps.

  3. Do they compound? (compare ORCA+DQN vs K-means+Static)
     → If the diagonal dominates, the full system is greater than its parts.
    """)


if __name__ == '__main__':
    main()
