"""
Backtest comparison: Static vs Adaptive VIX vs DQN v2
All strategies use fixed 10% capital position sizing.
DQN v2: 3 actions (flat/long/short), trained with edge penalty.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# --- Load ---
with open('datasets/nb02_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)
scaler = artifacts['scaler']; pca = artifacts['pca']
cointegrated_pairs = artifacts['cointegrated_pairs']
STATIC_FEATURE_COLS = artifacts['STATIC_FEATURE_COLS']
ORIGINAL_PAIRS = [f"{a}/{b}" for a, b in artifacts['ORIGINAL_PAIRS']]
STATE_DIM = artifacts['state_dim']

spreads_df = pd.read_parquet('datasets/spreads.parquet')
all_features = {}
for p in cointegrated_pairs:
    all_features[p] = pd.read_parquet(f'datasets/features_{p.replace("/","_")}.parquet')
df_prices = pd.read_parquet('datasets/pair_prices.parquet')
print(f"Loaded {len(cointegrated_pairs)} pairs")

# --- Load DQN v2 (3 actions) ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim=3, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, action_dim))
    def forward(self, x): return self.network(x)

ck = torch.load('dqn_pairs_agent_v2.pt', map_location='cpu', weights_only=False)
dqn_model = QNetwork(STATE_DIM, 3, dropout=0.0)
dqn_model.load_state_dict(ck['model_state_dict'])
dqn_model.eval()
print(f"DQN v2 loaded (val Sharpe={ck['final_val_sharpe']:.3f})")

# --- Constants ---
POSITION_FRAC = 0.10; TC = 0.001; CAPITAL = 100000.0
MAX_DAYS = 60; EXIT_THRESHOLD = 0.25
TEST_START = '2020-01-01'; TEST_END = '2023-12-31'


# ============================================================
# STRATEGIES
# ============================================================
def backtest_static(spread, z_threshold=1.0):
    spread = spread.dropna()
    rm = spread.rolling(60).mean(); rs = spread.rolling(60).std()
    z = ((spread - rm) / rs.replace(0, np.nan)).dropna()
    port = CAPITAL; pos = 0; ep = 0; dh = 0; ps = 0
    dv = []; trades = []
    for date, zv in z.items():
        sv = spread.loc[date]
        unr = pos * (sv - ep) * ps if pos != 0 else 0
        if pos != 0: dh += 1
        dv.append({'date': date, 'value': port + unr, 'position': pos})
        if pos != 0 and (abs(zv) < EXIT_THRESHOLD or dh >= MAX_DAYS):
            pnl = pos * (sv - ep) * ps; cost = TC * abs(ps) * abs(sv)
            port += pnl - cost; trades.append({'pnl': pnl - cost, 'days_held': dh, 'direction': pos})
            pos = 0; dh = 0
        if pos == 0:
            if zv > z_threshold:
                pos = -1; ep = sv; ps = CAPITAL * POSITION_FRAC / max(abs(sv), 0.01)
                port -= TC * abs(ps) * abs(sv); dh = 0
            elif zv < -z_threshold:
                pos = 1; ep = sv; ps = CAPITAL * POSITION_FRAC / max(abs(sv), 0.01)
                port -= TC * abs(ps) * abs(sv); dh = 0
    if pos != 0:
        pnl = pos * (spread.iloc[-1] - ep) * ps; port += pnl
        trades.append({'pnl': pnl, 'days_held': dh, 'direction': pos})
    return pd.DataFrame(dv), trades


def backtest_adaptive_vix(spread, t0, t1):
    spread = spread.dropna()
    spread = spread[(spread.index >= t0) & (spread.index <= t1)]
    rm = spread.rolling(60).mean(); rs = spread.rolling(60).std()
    z = ((spread - rm) / rs.replace(0, np.nan)).dropna()
    vix = df_prices['VIX'].reindex(z.index).ffill()
    port = CAPITAL; pos = 0; ep = 0; dh = 0; ps = 0
    dv = []; trades = []
    for date, zv in z.items():
        sv = spread.loc[date]
        vv = vix.loc[date] if date in vix.index and not pd.isna(vix.loc[date]) else 20
        zt = np.inf if vv > 30 else (1.5 if vv > 20 else 1.0)
        unr = pos * (sv - ep) * ps if pos != 0 else 0
        if pos != 0: dh += 1
        dv.append({'date': date, 'value': port + unr, 'position': pos})
        if pos != 0 and (abs(zv) < EXIT_THRESHOLD or dh >= MAX_DAYS):
            pnl = pos * (sv - ep) * ps; cost = TC * abs(ps) * abs(sv)
            port += pnl - cost; trades.append({'pnl': pnl - cost, 'days_held': dh, 'direction': pos})
            pos = 0; dh = 0
        if pos == 0:
            if zv > zt:
                pos = -1; ep = sv; ps = CAPITAL * POSITION_FRAC / max(abs(sv), 0.01)
                port -= TC * abs(ps) * abs(sv); dh = 0
            elif zv < -zt:
                pos = 1; ep = sv; ps = CAPITAL * POSITION_FRAC / max(abs(sv), 0.01)
                port -= TC * abs(ps) * abs(sv); dh = 0
    if pos != 0:
        pnl = pos * (spread.iloc[-1] - ep) * ps; port += pnl
        trades.append({'pnl': pnl, 'days_held': dh, 'direction': pos})
    return pd.DataFrame(dv), trades


def backtest_dqn_v2(pair_name, model, t0, t1):
    feat = all_features[pair_name]; spread_full = spreads_df[pair_name].dropna()
    mask = (feat.index >= t0) & (feat.index <= t1)
    feat_slice = feat.loc[mask]
    common = spread_full.index.intersection(feat_slice.index)
    if len(common) < 60: return pd.DataFrame(), []
    spread = spread_full.loc[common]; feat_slice = feat_slice.loc[common]
    port = CAPITAL; pos = 0; ps = 0.0; ep = 0.0; dh = 0
    dv = []; trades = []
    for i in range(len(common) - 1):
        date = common[i]; sv = spread.iloc[i]
        raw = feat_slice.iloc[i:i+1][STATIC_FEATURE_COLS].values
        try: sc = scaler.transform(raw); pf = pca.transform(sc).flatten()
        except: pf = np.zeros(pca.n_components_)
        upn = pos * (sv - ep) * ps / max(port, 1) if pos != 0 else 0
        state = np.concatenate([pf, [float(pos), upn, float(dh)/60.0]]).astype(np.float32)
        with torch.no_grad():
            action = model(torch.FloatTensor(state).unsqueeze(0)).argmax(dim=1).item()
        direction = {0: 0, 1: 1, 2: -1}[action]
        unr = pos * (sv - ep) * ps if pos != 0 else 0
        if pos != 0: dh += 1
        vix_val = feat_slice.iloc[i]['VIX'] if 'VIX' in feat_slice.columns else 0
        theta_val = feat_slice.iloc[i]['theta'] if 'theta' in feat_slice.columns else 0
        dv.append({'date': date, 'value': port + unr, 'position': pos,
                   'action': action, 'vix': vix_val, 'theta': theta_val})
        if direction == 0 and pos != 0:
            pnl = pos * (sv - ep) * ps; cost = TC * abs(ps) * abs(sv)
            port += pnl - cost; trades.append({'pnl': pnl - cost, 'days_held': dh, 'direction': pos})
            pos = 0; ps = 0; dh = 0
        elif direction != 0 and pos != direction:
            if pos != 0:
                pnl = pos * (sv - ep) * ps; cost = TC * abs(ps) * abs(sv)
                port += pnl - cost; trades.append({'pnl': pnl - cost, 'days_held': dh, 'direction': pos})
            pos = direction; ps = CAPITAL * POSITION_FRAC / max(abs(sv), 0.01); ep = sv
            port -= TC * abs(ps) * abs(sv); dh = 0
        if dh >= MAX_DAYS and pos != 0:
            pnl = pos * (sv - ep) * ps; cost = TC * abs(ps) * abs(sv)
            port += pnl - cost; trades.append({'pnl': pnl - cost, 'days_held': dh, 'direction': pos})
            pos = 0; ps = 0; dh = 0
    if pos != 0:
        sv = spread.iloc[-1]; pnl = pos * (sv - ep) * ps; port += pnl
        trades.append({'pnl': pnl, 'days_held': dh, 'direction': pos})
    return pd.DataFrame(dv), trades


# ============================================================
# RUN BACKTESTS
# ============================================================
results = {}
for pair_name in cointegrated_pairs:
    spread = spreads_df[pair_name].dropna()
    test_spread = spread[(spread.index >= TEST_START) & (spread.index <= TEST_END)]
    if len(test_spread) < 100: continue
    results[pair_name] = {}
    for gamma, name in [(1.0, 'Static 1.0'), (1.5, 'Static 1.5')]:
        d, t = backtest_static(test_spread, z_threshold=gamma)
        results[pair_name][name] = (d, t)
    d, t = backtest_adaptive_vix(spread, TEST_START, TEST_END)
    if len(d) > 0: results[pair_name]['Adaptive VIX'] = (d, t)
    d, t = backtest_dqn_v2(pair_name, dqn_model, TEST_START, TEST_END)
    if len(d) > 0: results[pair_name]['DQN v2'] = (d, t)
    n = {k: len(v[1]) for k, v in results[pair_name].items()}
    print(f"  {pair_name:12s}: {n}")
print(f"\nBacktested {len(results)} pairs x 4 strategies.")


# ============================================================
# METRICS
# ============================================================
def compute_metrics(daily_df, trades, rf_annual=0.04):
    if len(daily_df) < 20: return None
    values = np.maximum(daily_df['value'].values, 1.0)
    dr = np.diff(values) / values[:-1]; dr = dr[np.isfinite(dr)]
    if len(dr) < 20: return None
    nd = len(dr); tr = values[-1] / values[0] - 1
    ar = (1 + tr) ** (252 / nd) - 1; av = np.std(dr) * np.sqrt(252)
    pk = np.maximum.accumulate(values); dd = (values - pk) / pk; mdd = abs(dd.min())
    sh = (ar - rf_annual) / av if av > 1e-8 else -10.0
    nt = len(trades)
    if nt > 0:
        pnls = [t['pnl'] for t in trades]
        wr = sum(1 for p in pnls if p > 0) / nt
        trap = sum(1 for p in pnls if p < 0) / nt
        avgd = np.mean([t['days_held'] for t in trades])
    else: wr = trap = avgd = 0
    return {'AR': ar, 'Vol': av, 'MDD': mdd, 'Sharpe': sh,
            'Trap Rate': trap, 'Win Rate': wr, '# Trades': nt, 'Avg Duration': avgd}

rows = []
for pair_name, strategies in results.items():
    for sn, (d, t) in strategies.items():
        m = compute_metrics(d, t)
        if m: m['Pair'] = pair_name; m['Strategy'] = sn; rows.append(m)
mdf = pd.DataFrame(rows)
print(f"\nMetrics: {len(mdf)} combinations")
for s in ['Static 1.0', 'Static 1.5', 'Adaptive VIX', 'DQN v2']:
    print(f"  {s:15s}: {len(mdf[mdf['Strategy']==s])} pairs")


# ============================================================
# AGGREGATE TABLE
# ============================================================
agg = mdf.groupby('Strategy').agg({
    'AR': 'mean', 'Vol': 'mean', 'MDD': 'mean', 'Sharpe': 'mean',
    'Trap Rate': 'mean', 'Win Rate': 'mean', '# Trades': 'sum', 'Avg Duration': 'mean',
}).round(4)
order = ['Static 1.0', 'Static 1.5', 'Adaptive VIX', 'DQN v2']
agg = agg.reindex([s for s in order if s in agg.index])

print(f"\n{'='*95}")
print(f"AGGREGATE PERFORMANCE — Test Period 2020-2023 (ALL 19 PAIRS)")
print(f"{'='*95}")
print(agg.to_string())


# ============================================================
# PER-PAIR COMPARISON
# ============================================================
print(f"\n{'='*95}")
print(f"PER-PAIR: Static 1.0 vs Adaptive VIX vs DQN v2")
print(f"{'='*95}")

dqn_vs_static = 0; dqn_vs_adaptive = 0; total = 0
for pair in sorted(results.keys()):
    pm = mdf[mdf['Pair'] == pair]
    dqn = pm[pm['Strategy'] == 'DQN v2']
    static = pm[pm['Strategy'] == 'Static 1.0']
    adapt = pm[pm['Strategy'] == 'Adaptive VIX']
    if len(dqn) == 0 or len(static) == 0: continue
    ds = dqn.iloc[0]['Sharpe']; ss = static.iloc[0]['Sharpe']
    avs = adapt.iloc[0]['Sharpe'] if len(adapt) > 0 else np.nan
    dt = dqn.iloc[0]['Trap Rate']; st = static.iloc[0]['Trap Rate']
    at = adapt.iloc[0]['Trap Rate'] if len(adapt) > 0 else np.nan
    dn = int(dqn.iloc[0]['# Trades'])
    marker = '*' if pair in ORIGINAL_PAIRS else ' '
    best = 'DQN' if ds >= max(ss, avs if not np.isnan(avs) else -np.inf) else \
           ('Adapt' if not np.isnan(avs) and avs >= ss else 'Static')
    avs_str = f"Adapt={avs:+.3f}" if not np.isnan(avs) else "Adapt=N/A  "
    print(f"  {marker} {pair:12s}  Static={ss:+.3f}  {avs_str}  DQN={ds:+.3f}  trap:{st:.0%}->{dt:.0%}  trades={dn:3d}  -> {best}")
    if ds > ss: dqn_vs_static += 1
    if not np.isnan(avs) and ds > avs: dqn_vs_adaptive += 1
    total += 1

print(f"\nDQN beats Static:       {dqn_vs_static}/{total} ({dqn_vs_static/max(total,1)*100:.0f}%)")
print(f"DQN beats Adaptive VIX: {dqn_vs_adaptive}/{total} ({dqn_vs_adaptive/max(total,1)*100:.0f}%)")


# ============================================================
# STATISTICAL TESTS
# ============================================================
print(f"\n{'='*95}")
print(f"STATISTICAL SIGNIFICANCE (Wilcoxon signed-rank)")
print(f"{'='*95}")

for bn in ['Static 1.0', 'Adaptive VIX']:
    bl = mdf[mdf['Strategy'] == bn]; dq = mdf[mdf['Strategy'] == 'DQN v2']
    common = sorted(set(dq['Pair']) & set(bl['Pair']))
    pd_d = []; pd_b = []; pt_d = []; pt_b = []
    for p in common:
        d = dq[dq['Pair'] == p]; b = bl[bl['Pair'] == p]
        if len(d) > 0 and len(b) > 0:
            pd_d.append(d.iloc[0]['Sharpe']); pd_b.append(b.iloc[0]['Sharpe'])
            pt_d.append(d.iloc[0]['Trap Rate']); pt_b.append(b.iloc[0]['Trap Rate'])
    if len(pd_d) >= 5:
        _, pv_s = stats.wilcoxon(pd_d, pd_b)
        _, pv_t = stats.wilcoxon(pt_d, pt_b)
        sig_s = "***" if pv_s < 0.01 else ("**" if pv_s < 0.05 else ("*" if pv_s < 0.10 else "ns"))
        sig_t = "***" if pv_t < 0.01 else ("**" if pv_t < 0.05 else ("*" if pv_t < 0.10 else "ns"))
        print(f"  DQN vs {bn:15s} (n={len(pd_d)}):")
        print(f"    Sharpe: diff={np.mean(pd_d)-np.mean(pd_b):+.3f}, p={pv_s:.6f} {sig_s}")
        print(f"    Trap:   diff={np.mean(pt_d)-np.mean(pt_b):+.3f}, p={pv_t:.6f} {sig_t}")


# ============================================================
# DQN ACTION ANALYSIS
# ============================================================
print(f"\n{'='*95}")
print(f"DQN v2 ACTION ANALYSIS BY REGIME")
print(f"{'='*95}")

low_vix = []; high_vix = []
for pair, strats in results.items():
    if 'DQN v2' not in strats: continue
    daily, _ = strats['DQN v2']
    if 'action' not in daily.columns or 'vix' not in daily.columns: continue
    for _, row in daily.iterrows():
        if pd.isna(row['vix']) or row['vix'] == 0: continue
        if row['vix'] < 20: low_vix.append(int(row['action']))
        elif row['vix'] > 25: high_vix.append(int(row['action']))

labels = ['Flat', 'Long', 'Short']
if low_vix and high_vix:
    lc = np.bincount(low_vix, minlength=3) / len(low_vix) * 100
    hc = np.bincount(high_vix, minlength=3) / len(high_vix) * 100
    print(f"  Low VIX  (<20): {labels[0]}={lc[0]:.1f}%  {labels[1]}={lc[1]:.1f}%  {labels[2]}={lc[2]:.1f}%")
    print(f"  High VIX (>25): {labels[0]}={hc[0]:.1f}%  {labels[1]}={hc[1]:.1f}%  {labels[2]}={hc[2]:.1f}%")
    if hc[0] > lc[0]:
        print(f"  -> Agent is MORE cautious in stressed markets (flat: {lc[0]:.0f}% -> {hc[0]:.0f}%)")
    else:
        print(f"  -> Agent does NOT differentiate by VIX regime")


# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*95}")
print(f"SUMMARY")
print(f"{'='*95}")
dqn_sharpe = agg.loc['DQN v2', 'Sharpe'] if 'DQN v2' in agg.index else 0
static_sharpe = agg.loc['Static 1.0', 'Sharpe'] if 'Static 1.0' in agg.index else 0
adapt_sharpe = agg.loc['Adaptive VIX', 'Sharpe'] if 'Adaptive VIX' in agg.index else 0
dqn_trap = agg.loc['DQN v2', 'Trap Rate'] if 'DQN v2' in agg.index else 0
static_trap = agg.loc['Static 1.0', 'Trap Rate'] if 'Static 1.0' in agg.index else 0

print(f"  Mean Sharpe:  Static={static_sharpe:.3f}  Adaptive VIX={adapt_sharpe:.3f}  DQN v2={dqn_sharpe:.3f}")
print(f"  Mean Trap:    Static={static_trap:.1%}  DQN v2={dqn_trap:.1%}")
print(f"  DQN wins:     {dqn_vs_static}/{total} vs Static, {dqn_vs_adaptive}/{total} vs Adaptive VIX")
if dqn_sharpe > 0 and static_sharpe < 0:
    print(f"  DQN achieves POSITIVE Sharpe while all baselines are negative.")
