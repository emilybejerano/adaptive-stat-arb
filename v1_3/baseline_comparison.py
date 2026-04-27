"""
Baseline Comparison: Simple Rules vs DQN Adaptive Threshold

Tests whether the DQN adds value over simple heuristic rules that
use the same information (theta, VIX) to pick thresholds.

From the DQN behavior analysis:
  - Agent uses 2 thresholds: 0.50σ (21%) and 1.25σ (79%)
  - Driven by theta (r=0.50), not VIX (r=0.06)
  - Aggressive when theta < ~5, cautious when theta > ~5

Baselines tested:
  1. Static 1.0σ  — ORCA default
  2. Static 1.25σ — the DQN's most common choice
  3. Theta rule    — mimic the DQN: 0.50σ when theta<5, else 1.25σ
  4. VIX rule      — 1.25σ when VIX>25, else 1.0σ
  5. DQN agent     — trained model
"""
import numpy as np
import pandas as pd
import torch
import pickle
import os
from scipy import stats
import warnings
from zscore_utils import ZSCORE_METHOD, ZSCORE_WINDOW, compute_spread_zscore, method_suffix
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load data ---
with open('datasets/artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)
cointegrated_pairs = artifacts['cointegrated_pairs']
spreads_df = pd.read_parquet('datasets/spreads.parquet')
all_ou = {}
for p in cointegrated_pairs:
    all_ou[p] = pd.read_parquet(f'datasets/ou_params_{p.replace("/","_")}.parquet')
df_prices = pd.read_parquet('datasets/pair_prices.parquet')
print(f"Loaded {len(cointegrated_pairs)} pairs")
CHECKPOINT_PATH = os.getenv('CHECKPOINT_PATH', f'adaptive_threshold_dqn_{method_suffix()}.pt')
RESULTS_PATH = f'datasets/baseline_results_{method_suffix()}.pkl'

# --- Constants (same as train_adaptive_threshold.py) ---
THRESHOLDS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
K = len(THRESHOLDS)
CAPITAL = 100000.0
POSITION_FRAC = 0.10
TC = 0.001
MAX_DAYS = 60
EXIT_Z = 0.25
WEEKS_PER_EPISODE = 4
TRADING_DAYS_PER_WEEK = 5
STATE_DIM = 7

# --- Load trained DQN ---
import torch.nn as nn
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 20), nn.ReLU(),
            nn.Linear(20, 10), nn.ReLU(),
            nn.Linear(10, action_dim),
        )
    def forward(self, x):
        return self.network(x)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
q_net = QNetwork(STATE_DIM, K).to(device)
q_net.load_state_dict(checkpoint['model_state_dict'])
q_net.eval()


def get_market_context(vix_series, ou_df, date_idx, dates):
    date = dates[min(date_idx, len(dates)-1)]
    vix = vix_series.loc[date] if date in vix_series.index else 0.0
    vix_norm = min(vix / 80.0, 1.0)
    theta = ou_df.loc[date, 'theta'] if date in ou_df.index else 0.0
    theta_norm = min(theta / 20.0, 1.0)
    return vix_norm, theta_norm, vix, theta


def backtest_pair_with_rule(pair_name, t0, t1, rule_fn):
    """
    Backtest a pair with a rule function that returns a threshold each week.
    rule_fn(vix_raw, theta_raw, week_num) -> threshold
    """
    spread = spreads_df[pair_name].dropna()
    ou_df = all_ou[pair_name]
    z_full = compute_spread_zscore(spread, window=ZSCORE_WINDOW, method=ZSCORE_METHOD).dropna()

    s = spread[(spread.index >= t0) & (spread.index <= t1)]
    z = z_full[(z_full.index >= t0) & (z_full.index <= t1)]
    if len(z) < 20:
        return None
    common = s.index.intersection(z.index)
    s = s.loc[common]; z = z.loc[common]
    vix = df_prices['VIX'].reindex(z.index).ffill()
    dates = z.index
    n_days = len(dates)

    week_size = TRADING_DAYS_PER_WEEK
    port = CAPITAL; pos = 0; ep2 = 0; ps = 0; dh = 0
    dv = []; trades = []; thresholds_used = []

    for week_start in range(0, n_days, week_size):
        week_end = min(week_start + week_size, n_days)
        week_num = week_start // week_size
        mid = (week_start + week_end) // 2
        date = dates[min(mid, len(dates)-1)]
        vix_raw = vix.loc[date] if date in vix.index else 20.0
        theta_raw = ou_df.loc[date, 'theta'] if date in ou_df.index else 5.0

        thresh = rule_fn(vix_raw, theta_raw, week_num)
        thresholds_used.append(thresh)

        for i in range(week_start, week_end):
            sv = s.iloc[i]; zv = z.iloc[i]
            unr = pos * (sv - ep2) * ps if pos != 0 else 0
            if pos != 0: dh += 1
            dv.append(port + unr)
            if pos != 0 and (abs(zv) < EXIT_Z or dh >= MAX_DAYS):
                pnl = pos * (sv - ep2) * ps; cost = TC*abs(ps)*abs(sv)
                net = pnl - cost; port += net
                trades.append({'pnl': net, 'days_held': dh})
                pos = 0; dh = 0
            if pos == 0 and abs(zv) > thresh:
                pos = -1 if zv > thresh else 1; ep2 = sv
                ps = CAPITAL * POSITION_FRAC / max(abs(sv), 0.01)
                port -= TC*abs(ps)*abs(sv); dh = 0

    if pos != 0:
        pnl = pos*(s.iloc[-1]-ep2)*ps; port += pnl
        trades.append({'pnl': pnl, 'days_held': dh})

    return {'values': dv, 'trades': trades, 'thresholds': thresholds_used}


def backtest_pair_dqn(pair_name, t0, t1):
    """Backtest with trained DQN agent."""
    spread = spreads_df[pair_name].dropna()
    ou_df = all_ou[pair_name]
    z_full = compute_spread_zscore(spread, window=ZSCORE_WINDOW, method=ZSCORE_METHOD).dropna()

    s = spread[(spread.index >= t0) & (spread.index <= t1)]
    z = z_full[(z_full.index >= t0) & (z_full.index <= t1)]
    if len(z) < 20:
        return None
    common = s.index.intersection(z.index)
    s = s.loc[common]; z = z.loc[common]
    vix = df_prices['VIX'].reindex(z.index).ffill()
    dates = z.index
    n_days = len(dates)

    week_size = TRADING_DAYS_PER_WEEK
    port = CAPITAL; pos = 0; ep2 = 0; ps = 0; dh = 0
    dv = []; trades = []; thresholds_used = []
    cum_wins = 0; cum_losses = 0; n_trades = 0

    for week_start in range(0, n_days, week_size):
        week_end = min(week_start + week_size, n_days)
        week_num = week_start // week_size
        w_norm = (week_num % WEEKS_PER_EPISODE) / WEEKS_PER_EPISODE
        s_norm = cum_wins / max(CAPITAL * 0.1, 1)
        l_norm = cum_losses / max(CAPITAL * 0.1, 1)
        cc_norm = n_trades / 10.0
        t_norm = 0.5
        mid = (week_start + week_end) // 2
        vix_n, theta_n, _, _ = get_market_context(vix, ou_df, mid, dates)
        state = np.array([w_norm, s_norm, l_norm, cc_norm, t_norm,
                          vix_n, theta_n], dtype=np.float32)
        with torch.no_grad():
            q = q_net(torch.FloatTensor(state).unsqueeze(0).to(device))
            action = q.argmax(dim=1).item()
        thresh = THRESHOLDS[action]
        thresholds_used.append(thresh)

        for i in range(week_start, week_end):
            sv = s.iloc[i]; zv = z.iloc[i]
            unr = pos * (sv - ep2) * ps if pos != 0 else 0
            if pos != 0: dh += 1
            dv.append(port + unr)
            if pos != 0 and (abs(zv) < EXIT_Z or dh >= MAX_DAYS):
                pnl = pos * (sv - ep2) * ps; cost = TC*abs(ps)*abs(sv)
                net = pnl - cost; port += net
                trades.append({'pnl': net, 'days_held': dh})
                if net > 0: cum_wins += net
                else: cum_losses += abs(net)
                n_trades += 1; pos = 0; dh = 0
            if pos == 0 and abs(zv) > thresh:
                pos = -1 if zv > thresh else 1; ep2 = sv
                ps = CAPITAL * POSITION_FRAC / max(abs(sv), 0.01)
                port -= TC*abs(ps)*abs(sv); dh = 0

    if pos != 0:
        pnl = pos*(s.iloc[-1]-ep2)*ps; port += pnl
        trades.append({'pnl': pnl, 'days_held': dh})

    return {'values': dv, 'trades': trades, 'thresholds': thresholds_used}


def calc_metrics(result):
    if result is None or len(result['values']) < 20:
        return None
    vals = np.maximum(np.array(result['values']), 1.0)
    dr = np.diff(vals)/vals[:-1]; dr = dr[np.isfinite(dr)]
    if len(dr) < 20: return None
    nd = len(dr); tr = vals[-1]/vals[0]-1
    ar = (1+tr)**(252/nd)-1; av = np.std(dr)*np.sqrt(252)
    sh = (ar-0.04)/av if av>1e-8 else -10
    nt = len(result['trades'])
    if nt > 0:
        pnls = [t['pnl'] for t in result['trades']]
        trap = sum(1 for p in pnls if p<0)/nt
    else: trap = 0
    return {'Sharpe': sh, 'AR': ar, 'Trap': trap, 'Trades': nt}


# ============================================================
# Define rule-based strategies
# ============================================================
rules = {
    'Static 1.0σ':  lambda vix, theta, w: 1.0,
    'Static 1.25σ': lambda vix, theta, w: 1.25,
    'Theta rule':   lambda vix, theta, w: 0.50 if theta < 5 else 1.25,
    'VIX rule':     lambda vix, theta, w: 1.25 if vix > 25 else 1.0,
    'Combined':     lambda vix, theta, w: 0.50 if (theta < 5 and vix > 25) else (1.25 if theta > 5 else 1.0),
}

T0, T1 = '2020-01-01', '2023-12-31'

# ============================================================
# Run all strategies on all pairs
# ============================================================
print(f"\nBacktesting {len(cointegrated_pairs)} pairs, 2020-2023...")
print(f"Strategies: {list(rules.keys())} + DQN\n")
print(f"Z-score method: {ZSCORE_METHOD} (window={ZSCORE_WINDOW})")
print(f"Checkpoint: {CHECKPOINT_PATH}\n")

results = {name: {'sharpe': [], 'trap': [], 'trades': []} for name in list(rules.keys()) + ['DQN']}

for i, pair in enumerate(sorted(cointegrated_pairs)):
    if (i+1) % 25 == 0:
        print(f"  Processing pair {i+1}/{len(cointegrated_pairs)}...")

    # Rule-based strategies
    for name, rule_fn in rules.items():
        r = backtest_pair_with_rule(pair, T0, T1, rule_fn)
        m = calc_metrics(r)
        if m:
            results[name]['sharpe'].append(m['Sharpe'])
            results[name]['trap'].append(m['Trap'])
            results[name]['trades'].append(m['Trades'])

    # DQN
    r = backtest_pair_dqn(pair, T0, T1)
    m = calc_metrics(r)
    if m:
        results['DQN']['sharpe'].append(m['Sharpe'])
        results['DQN']['trap'].append(m['Trap'])
        results['DQN']['trades'].append(m['Trades'])


# ============================================================
# Results
# ============================================================
print(f"\n{'='*75}")
print(f"  BASELINE COMPARISON: Simple Rules vs DQN (154 pairs, 2020-2023)")
print(f"{'='*75}")

print(f"\n{'Strategy':<16} {'Sharpe':>8} {'Trap%':>8} {'Trades':>8} {'n':>5}")
print("-" * 50)
for name in list(rules.keys()) + ['DQN']:
    r = results[name]
    n = len(r['sharpe'])
    if n == 0:
        continue
    print(f"{name:<16} {np.mean(r['sharpe']):>+8.3f} {np.mean(r['trap'])*100:>7.1f}% {np.mean(r['trades']):>8.1f} {n:>5}")

# ============================================================
# Statistical tests: each strategy vs Static 1.0σ
# ============================================================
print(f"\n{'='*75}")
print(f"  STATISTICAL TESTS (Wilcoxon signed-rank vs Static 1.0σ)")
print(f"{'='*75}")

ref = results['Static 1.0σ']
n_ref = len(ref['sharpe'])

for name in list(rules.keys())[1:] + ['DQN']:
    r = results[name]
    n = min(len(r['sharpe']), n_ref)
    if n < 5:
        continue

    # Sharpe
    _, p_sh = stats.wilcoxon(r['sharpe'][:n], ref['sharpe'][:n])
    diff_sh = np.mean(r['sharpe'][:n]) - np.mean(ref['sharpe'][:n])

    # Trap
    _, p_tr = stats.wilcoxon(r['trap'][:n], ref['trap'][:n])
    diff_tr = (np.mean(r['trap'][:n]) - np.mean(ref['trap'][:n])) * 100

    sig_s = "***" if p_sh<0.001 else ("**" if p_sh<0.01 else ("*" if p_sh<0.05 else "ns"))
    sig_t = "***" if p_tr<0.001 else ("**" if p_tr<0.01 else ("*" if p_tr<0.05 else "ns"))

    print(f"\n{name} vs Static 1.0σ (n={n}):")
    print(f"  Sharpe: {diff_sh:+.4f} (p={p_sh:.6f}) {sig_s}")
    print(f"  Trap:   {diff_tr:+.1f}pp (p={p_tr:.6f}) {sig_t}")

# ============================================================
# Head-to-head: DQN vs Theta rule (the critical comparison)
# ============================================================
print(f"\n{'='*75}")
print(f"  DQN vs THETA RULE (head-to-head)")
print(f"{'='*75}")

n = min(len(results['DQN']['sharpe']), len(results['Theta rule']['sharpe']))
if n >= 5:
    dqn_sh = results['DQN']['sharpe'][:n]
    theta_sh = results['Theta rule']['sharpe'][:n]
    dqn_tr = results['DQN']['trap'][:n]
    theta_tr = results['Theta rule']['trap'][:n]

    _, p_sh = stats.wilcoxon(dqn_sh, theta_sh)
    _, p_tr = stats.wilcoxon(dqn_tr, theta_tr)

    wins_sh = sum(1 for i in range(n) if dqn_sh[i] > theta_sh[i])
    wins_tr = sum(1 for i in range(n) if dqn_tr[i] < theta_tr[i])

    print(f"  DQN Sharpe wins: {wins_sh}/{n} ({wins_sh/n*100:.0f}%)")
    print(f"  DQN Trap wins:   {wins_tr}/{n} ({wins_tr/n*100:.0f}%)")
    print(f"  Sharpe diff: {np.mean(dqn_sh)-np.mean(theta_sh):+.4f} (p={p_sh:.6f})")
    print(f"  Trap diff:   {(np.mean(dqn_tr)-np.mean(theta_tr))*100:+.1f}pp (p={p_tr:.6f})")

    if p_tr < 0.05:
        print(f"\n  >> DQN significantly outperforms theta rule on trap rate")
    elif p_tr > 0.05 and abs(np.mean(dqn_tr)-np.mean(theta_tr)) < 0.01:
        print(f"\n  >> DQN and theta rule are statistically indistinguishable")
        print(f"     The DQN learned approximately the same policy as the simple rule.")
    else:
        print(f"\n  >> No significant difference between DQN and theta rule")

print(f"\nDone.")

with open(RESULTS_PATH, 'wb') as f:
    pickle.dump({
        'zscore_method': ZSCORE_METHOD,
        'zscore_window': ZSCORE_WINDOW,
        'results': results,
    }, f)
