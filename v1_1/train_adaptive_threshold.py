"""
Adaptive Threshold Selection for Pairs Trading via Deep Q-Network

Based on: Shen & Kurshan (ICAIF 2020) — "Deep Q-Network-based Adaptive
Alert Threshold Selection Policy for Payment Fraud Systems"

Key mapping from fraud → pairs trading:
  Fraud scoring threshold  →  Z-score entry threshold
  Hourly threshold update  →  Weekly threshold update
  Alert processing capacity → Capital/risk budget
  Fraud savings (S)        →  Cumulative PnL from reverting trades
  Fraud losses (L)         →  Cumulative losses from traps
  Hour of day (H)          →  Week of period

Architecture follows paper exactly:
  - 3-layer MLP: {20, 10, K} neurons
  - Experience replay (160K buffer, 1024 batch)
  - Epsilon-greedy: 0.5 → 0.1
  - Gamma: 0.9, Adam lr=0.0001
  - MSE loss
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import pickle
import os
from scipy import stats
import warnings
from pca_state_utils import (
    BASE_STATE_DIM,
    TRAIN_END,
    build_agent_state,
    build_feature_store,
    build_pca_store,
    load_or_fit_pca,
)
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# --- Load data (expanded universe: 154 pairs) ---
with open('datasets/artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)
cointegrated_pairs = artifacts['cointegrated_pairs']
spreads_df = pd.read_parquet('datasets/spreads.parquet')
all_ou = {}
for p in cointegrated_pairs:
    all_ou[p] = pd.read_parquet(f'datasets/ou_params_{p.replace("/","_")}.parquet')
df_prices = pd.read_parquet('datasets/pair_prices.parquet')
print(f"Loaded {len(cointegrated_pairs)} pairs")

scaler_path = 'datasets/pca_scaler.pkl'
pca_path = 'datasets/pca_model.pkl'
feature_store = build_feature_store(cointegrated_pairs, spreads_df, all_ou, df_prices)
scaler, pca = load_or_fit_pca(feature_store, scaler_path, pca_path, train_end=TRAIN_END)
pca_store = build_pca_store(feature_store, scaler, pca)
PCA_DIM = pca.n_components_
STATE_DIM = BASE_STATE_DIM + PCA_DIM
print(f"PCA state: base={BASE_STATE_DIM}, components={PCA_DIM}, total={STATE_DIM}")


# ============================================================
# CONSTANTS (following paper's approach)
# ============================================================
# Action space: K discrete z-score thresholds
# Action space: K discrete z-score thresholds
THRESHOLDS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
K = len(THRESHOLDS)  # 6 actions

# Trading params
CAPITAL = 100000.0
POSITION_FRAC = 0.10
TC = 0.001
MAX_DAYS = 60
EXIT_Z = 0.25

# Episode = 1 month, agent picks threshold each WEEK (4 decisions per episode)
WEEKS_PER_EPISODE = 4
TRADING_DAYS_PER_WEEK = 5


# ============================================================
# DQN (following paper Section 5.3 exactly)
# 3-layer MLP: {20, 10, K}
# RELU on first 2 layers, linear output (Q-values)
# Adam, lr=0.0001, MSE loss
# ============================================================
class QNetwork(nn.Module):
    """Paper: 3-layer MLP with {20, 10, K} neurons."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, action_dim),  # linear output = Q-values
        )

    def forward(self, x):
        return self.network(x)


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    """Paper: replay memory capacity N=160,000."""
    def __init__(self, capacity=160000):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class AdaptiveThresholdAgent:
    """
    Paper: epsilon starts 0.5, decreases 5% per iteration until 0.1.
    Gamma=0.9, Adam lr=0.0001, mini-batch=1024.
    """
    def __init__(self, state_dim=STATE_DIM, action_dim=K,
                 gamma=0.9, lr=0.0001, batch_size=1024,
                 eps_start=0.5, eps_end=0.1, eps_decay_rate=0.005):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay_rate = eps_decay_rate

        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayMemory(160000)

    def select_action(self, state, eval_mode=False):
        """Paper: epsilon-greedy. Random with prob epsilon, else argmax Q."""
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            self.q_network.eval()
            q = self.q_network(torch.FloatTensor(state).unsqueeze(0).to(device))
            self.q_network.train()
            return q.argmax(dim=1).item()

    def optimize(self):
        """Paper: sample mini-batch from replay, gradient descent on MSE."""
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*batch))

        states = torch.FloatTensor(np.array(batch.state)).to(device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(batch.reward).to(device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(device)
        dones = torch.FloatTensor(batch.done).to(device)

        # Current Q(s, a)
        current_q = self.q_network(states).gather(1, actions).squeeze()

        # Target: r + gamma * max_a' Q(s', a')
        with torch.no_grad():
            max_next_q = self.q_network(next_states).max(dim=1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def decay_epsilon(self):
        """Paper: epsilon decreases by 5% per iteration until 0.1."""
        self.epsilon = max(self.eps_end, self.epsilon - self.eps_decay_rate)


# ============================================================
# SIMULATION: run static strategy with given threshold for a week
# ============================================================
def simulate_week(spread, z_series, vix_series, ou_df,
                  start_idx, end_idx, threshold,
                  portfolio, position_state):
    """
    Run the static pairs strategy for one week with the given threshold.
    Returns: (pnl_wins, pnl_losses, n_trades, new_portfolio, new_position_state)

    position_state = (position, entry_price, position_size, days_held)
    """
    pos, ep, ps, dh = position_state
    wins_pnl = 0.0
    losses_pnl = 0.0
    n_trades = 0
    port = portfolio

    for i in range(start_idx, min(end_idx, len(z_series))):
        sv = spread.iloc[i]
        zv = z_series.iloc[i]

        # Track unrealized
        if pos != 0:
            dh += 1

        # Exit
        if pos != 0 and (abs(zv) < EXIT_Z or dh >= MAX_DAYS):
            pnl = pos * (sv - ep) * ps
            cost = TC * abs(ps) * abs(sv)
            net = pnl - cost
            port += net
            if net > 0:
                wins_pnl += net
            else:
                losses_pnl += abs(net)
            n_trades += 1
            pos = 0; ps = 0; dh = 0

        # Entry with current threshold
        if pos == 0 and abs(zv) > threshold:
            pos = -1 if zv > threshold else 1
            ep = sv
            ps = CAPITAL * POSITION_FRAC / max(abs(sv), 0.01)
            port -= TC * abs(ps) * abs(sv)
            dh = 0

    return wins_pnl, losses_pnl, n_trades, port, (pos, ep, ps, dh)


# ============================================================
# EPISODE: one month of trading across one pair
# Agent picks threshold each week (4 decisions per episode)
# ============================================================
def run_episode(agent, pair_name, month_start, month_end, training=True):
    """
    Paper: one episode = one day (24 hourly decisions).
    Ours: one episode = one month (4 weekly decisions).
    """
    spread = spreads_df[pair_name].dropna()
    ou_df = all_ou[pair_name]

    # Compute z-score on FULL spread (need 60-day lookback), then slice to month
    rm_full = spread.rolling(60).mean(); rs_full = spread.rolling(60).std()
    z_full = ((spread - rm_full) / rs_full.replace(0, np.nan)).dropna()

    # Slice to month
    s = spread[(spread.index >= month_start) & (spread.index <= month_end)]
    z = z_full[(z_full.index >= month_start) & (z_full.index <= month_end)]
    if len(z) < 5:
        return 0, []

    # Align
    common = s.index.intersection(z.index)
    s = s.loc[common]; z = z.loc[common]

    # Split into weeks
    dates = z.index
    n_days = len(dates)
    week_size = max(n_days // WEEKS_PER_EPISODE, 1)

    # Reset episode state (paper: reset CC=0, H=1 each day)
    cum_wins = 0.0    # S: cumulative savings
    cum_losses = 0.0  # L: cumulative losses
    n_trades_total = 0  # CC: consumed capacity
    portfolio = CAPITAL
    pos_state = (0, 0, 0, 0)  # (position, entry_price, size, days_held)

    max_trades_per_month = 10  # capacity constraint
    episode_reward = 0
    transitions = []

    for week in range(WEEKS_PER_EPISODE):
        w_start = week * week_size
        w_end = min((week + 1) * week_size, n_days)
        if w_start >= n_days:
            break

        # Build state (paper Section 4.2)
        w_norm = week / WEEKS_PER_EPISODE  # W: week of month
        s_norm = cum_wins / max(CAPITAL * 0.1, 1)  # S: normalized savings
        l_norm = cum_losses / max(CAPITAL * 0.1, 1)  # L: normalized losses
        cc_norm = n_trades_total / max(max_trades_per_month, 1)  # CC: capacity used
        t_norm = 0.5  # T: will be set by action

        mid_idx = min((w_start + w_end) // 2, n_days - 1)
        state = build_agent_state(
            pair_name,
            dates[mid_idx],
            w_norm,
            s_norm,
            l_norm,
            cc_norm,
            t_norm,
            pca_store,
            pca,
        )

        # Agent selects threshold (paper: select action)
        action = agent.select_action(state, eval_mode=not training)
        threshold = THRESHOLDS[action]

        # Simulate week with chosen threshold
        week_wins, week_losses, week_trades, portfolio, pos_state = simulate_week(
            s, z, None, ou_df, w_start, w_end, threshold, portfolio, pos_state)

        cum_wins += week_wins
        cum_losses += week_losses
        n_trades_total += week_trades

        # Paper reward: (S - L) * H
        # Scale by 100 so gradient is meaningful
        reward = (week_wins - week_losses) / (CAPITAL * 0.01) * (week + 1)

        # Capacity penalty (paper: alerts dropped when CC > Cmax)
        if n_trades_total > max_trades_per_month:
            reward -= 1.0 * (n_trades_total - max_trades_per_month)

        # Next state
        next_w_norm = (week + 1) / WEEKS_PER_EPISODE
        next_cc_norm = n_trades_total / max(max_trades_per_month, 1)
        next_s_norm = cum_wins / max(CAPITAL * 0.1, 1)
        next_l_norm = cum_losses / max(CAPITAL * 0.1, 1)
        t_norm_next = action / (K - 1)

        next_idx = min(w_end, n_days - 1)
        next_state = build_agent_state(
            pair_name,
            dates[next_idx],
            next_w_norm,
            next_s_norm,
            next_l_norm,
            next_cc_norm,
            t_norm_next,
            pca_store,
            pca,
        )

        done = (week == WEEKS_PER_EPISODE - 1) or (w_end >= n_days)

        if training:
            agent.memory.push(state, action, reward, next_state, float(done))
            agent.optimize()

        episode_reward += reward
        transitions.append({
            'week': week, 'threshold': threshold, 'action': action,
            'wins': week_wins, 'losses': week_losses, 'trades': week_trades,
            'reward': reward,
        })

    return episode_reward, transitions


# ============================================================
# FULL BACKTEST: run static strategy with a fixed or adaptive threshold
# ============================================================
def backtest_pair(pair_name, t0, t1, threshold=None, agent=None):
    """
    If threshold given: static strategy.
    If agent given: adaptive threshold (agent picks each week).
    """
    spread = spreads_df[pair_name].dropna()
    ou_df = all_ou[pair_name]

    # Compute z on full spread, then slice
    rm_full = spread.rolling(60).mean(); rs_full = spread.rolling(60).std()
    z_full = ((spread - rm_full) / rs_full.replace(0, np.nan)).dropna()

    s = spread[(spread.index >= t0) & (spread.index <= t1)]
    z = z_full[(z_full.index >= t0) & (z_full.index <= t1)]
    if len(z) < 20:
        return None

    common = s.index.intersection(z.index)
    s = s.loc[common]; z = z.loc[common]

    dates = z.index
    n_days = len(dates)

    if agent is not None:
        # Adaptive: pick threshold each week
        week_size = TRADING_DAYS_PER_WEEK
        port = CAPITAL; pos = 0; ep2 = 0; ps = 0; dh = 0
        dv = []; trades = []; thresholds_used = []
        cum_wins = 0; cum_losses = 0; n_trades = 0

        for week_start in range(0, n_days, week_size):
            week_end = min(week_start + week_size, n_days)
            week_num = week_start // week_size

            # Build state
            w_norm = (week_num % WEEKS_PER_EPISODE) / WEEKS_PER_EPISODE
            s_norm = cum_wins / max(CAPITAL * 0.1, 1)
            l_norm = cum_losses / max(CAPITAL * 0.1, 1)
            cc_norm = n_trades / 10.0
            t_norm = 0.5
            mid = min((week_start + week_end) // 2, n_days - 1)
            state = build_agent_state(
                pair_name,
                dates[mid],
                w_norm,
                s_norm,
                l_norm,
                cc_norm,
                t_norm,
                pca_store,
                pca,
            )

            action = agent.select_action(state, eval_mode=True)
            thresh = THRESHOLDS[action]
            thresholds_used.append(thresh)

            # Run week
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

    else:
        # Static
        port = CAPITAL; pos = 0; ep2 = 0; ps = 0; dh = 0
        dv = []; trades = []
        for i in range(n_days):
            sv = s.iloc[i]; zv = z.iloc[i]
            unr = pos*(sv-ep2)*ps if pos!=0 else 0
            if pos!=0: dh+=1
            dv.append(port+unr)
            if pos!=0 and(abs(zv)<EXIT_Z or dh>=MAX_DAYS):
                pnl=pos*(sv-ep2)*ps; cost=TC*abs(ps)*abs(sv)
                port+=pnl-cost; trades.append({'pnl':pnl-cost,'days_held':dh}); pos=0; dh=0
            if pos==0 and abs(zv)>threshold:
                pos=-1 if zv>threshold else 1; ep2=sv
                ps=CAPITAL*POSITION_FRAC/max(abs(sv),0.01); port-=TC*abs(ps)*abs(sv); dh=0
        if pos!=0:
            pnl=pos*(s.iloc[-1]-ep2)*ps; port+=pnl; trades.append({'pnl':pnl,'days_held':dh})
        return {'values': dv, 'trades': trades}


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
        wr = sum(1 for p in pnls if p>0)/nt
        trap = sum(1 for p in pnls if p<0)/nt
    else: wr=trap=0
    return {'Sharpe':sh, 'AR':ar, 'Trap':trap, 'WR':wr, 'Trades':nt}


# ============================================================
# TRAINING (following paper Section 5.1)
# Paper: 100 iterations, train Mar-Sep, test Oct-Dec
# Ours: 200 iterations, train 2010-2018, val 2019, test 2020-2023
# ============================================================
def generate_monthly_periods(t0, t1):
    """Generate list of (month_start, month_end) date strings."""
    periods = []
    start = pd.Timestamp(t0)
    end = pd.Timestamp(t1)
    current = start
    while current < end:
        month_end = current + pd.offsets.MonthEnd(1)
        if month_end > end: month_end = end
        periods.append((str(current.date()), str(month_end.date())))
        current = month_end + pd.Timedelta(days=1)
    return periods


print(f"\nArchitecture (following ICAIF paper):")
print(f"  Network: {STATE_DIM} -> 20 -> 10 -> {K}")
print(f"  Actions: {K} thresholds {THRESHOLDS}")
print(f"  State: [week, cum_wins, cum_losses, capacity_used, threshold] + PCA({PCA_DIM})")
print(f"  PCA artifacts: {scaler_path}, {pca_path}")
print(f"  Replay: 160K buffer, batch=1024")
print(f"  Epsilon: 0.5 -> 0.1 (5% decay per iteration)")
print(f"  Gamma: 0.9, lr: 0.0001, MSE loss")

# Training periods
train_months = generate_monthly_periods('2010-01-01', '2018-12-31')
val_months = generate_monthly_periods('2019-01-01', '2019-12-31')
test_months = generate_monthly_periods('2020-01-01', '2023-12-31')

print(f"\n  Train: {len(train_months)} months (2010-2018)")
print(f"  Val:   {len(val_months)} months (2019)")
print(f"  Test:  {len(test_months)} months (2020-2023)")


# ============================================================
# TRAIN
# ============================================================
N_ITERATIONS = 300  # Paper: 100. We use 300 for slower epsilon decay.

print(f"\n{'='*70}")
print(f"  Training: {N_ITERATIONS} iterations")
print(f"{'='*70}")

agent = AdaptiveThresholdAgent()
best_val_reward = -np.inf
best_state = None

for iteration in range(N_ITERATIONS):
    # Paper: each iteration loops over all days
    # Ours: each iteration loops over all months × all pairs
    iter_reward = 0
    n_episodes = 0

    # Sample a subset of months and pairs each iteration
    sampled_months = random.sample(train_months, min(12, len(train_months)))
    sampled_pairs = random.sample(cointegrated_pairs, min(5, len(cointegrated_pairs)))

    for month_start, month_end in sampled_months:
        for pair in sampled_pairs:
            ep_reward, _ = run_episode(agent, pair, month_start, month_end, training=True)
            iter_reward += ep_reward
            n_episodes += 1

    # Paper: decay epsilon 5% per iteration
    agent.decay_epsilon()

    # Validate every 10 iterations
    if (iteration + 1) % 10 == 0:
        val_reward = 0
        for month_start, month_end in val_months:
            for pair in cointegrated_pairs[:5]:
                ep_r, _ = run_episode(agent, pair, month_start, month_end, training=False)
                val_reward += ep_r

        if val_reward > best_val_reward:
            best_val_reward = val_reward
            best_state = {k: v.clone() for k, v in agent.q_network.state_dict().items()}

        avg = iter_reward / max(n_episodes, 1)
        print(f"  Iter {iteration+1:3d}: train_reward={avg:.4f} val_reward={val_reward:.4f} "
              f"best_val={best_val_reward:.4f} eps={agent.epsilon:.3f}")

if best_state:
    agent.q_network.load_state_dict(best_state)
print(f"\nBest val reward: {best_val_reward:.4f}")


# ============================================================
# BACKTEST
# ============================================================
T0 = '2020-01-01'; T1 = '2023-12-31'
print(f"\n{'='*70}")
print(f"  BACKTEST 2020-2023: Static thresholds vs Adaptive DQN")
print(f"{'='*70}")

print(f"\n{'Pair':12s}", end="")
for th in [1.0, 1.5]:
    print(f"  {'S'+str(th):>6s}", end="")
print(f"  {'DQN':>6s}  {'S.Trap':>6s}  {'D.Trap':>6s}  {'D.#':>4s}  {'AvgThr':>6s}  Best")
print("-" * 80)

ss = {th: [] for th in [1.0, 1.5]}
ds = []; st_trap = []; dt_trap = []

for pair in sorted(cointegrated_pairs):
    # Static baselines
    static_results = {}
    for th in [1.0, 1.5]:
        r = backtest_pair(pair, T0, T1, threshold=th)
        static_results[th] = calc_metrics(r)

    # Adaptive DQN
    r_dqn = backtest_pair(pair, T0, T1, agent=agent)
    m_dqn = calc_metrics(r_dqn)

    if static_results[1.0] is None:
        continue

    marker = ' '
    line = f"{marker} {pair:12s}"
    for th in [1.0, 1.5]:
        m = static_results[th]
        sh = m['Sharpe'] if m else -10
        line += f"  {sh:+.3f}"
        ss[th].append(sh)

    if m_dqn:
        avg_thr = np.mean(r_dqn['thresholds']) if 'thresholds' in r_dqn else 1.0
        line += f"  {m_dqn['Sharpe']:+.3f}  {static_results[1.0]['Trap']:.0%}    {m_dqn['Trap']:.0%}    {m_dqn['Trades']:3d}   {avg_thr:.2f}"
        ds.append(m_dqn['Sharpe']); dt_trap.append(m_dqn['Trap'])
        st_trap.append(static_results[1.0]['Trap'])

        vals = [('S1.0', static_results[1.0]['Sharpe']),
                ('S1.5', static_results[1.5]['Sharpe'] if static_results[1.5] else -10),
                ('DQN', m_dqn['Sharpe'])]
        best = max(vals, key=lambda x: x[1])[0]
        line += f"    {best}"
    else:
        line += "    N/A"

    print(line)


# Summary
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
n = min(len(ss[1.0]), len(ds))
if n > 0:
    wins = sum(1 for i in range(n) if ds[i] > ss[1.0][i])
    print(f"DQN wins vs Static 1.0: {wins}/{n} ({wins/n*100:.0f}%)")
    print(f"Mean Sharpe: Static 1.0={np.mean(ss[1.0]):.3f}  Static 1.5={np.mean(ss[1.5]):.3f}  DQN={np.mean(ds):.3f}")
    print(f"Mean Trap:   Static={np.mean(st_trap):.1%}  DQN={np.mean(dt_trap):.1%}")

    if n >= 5:
        _, pv_s = stats.wilcoxon(ds[:n], ss[1.0][:n])
        _, pv_t = stats.wilcoxon(dt_trap[:n], st_trap[:n])
        sig_s = "***" if pv_s<0.01 else ("**" if pv_s<0.05 else ("*" if pv_s<0.10 else "ns"))
        sig_t = "***" if pv_t<0.01 else ("**" if pv_t<0.05 else ("*" if pv_t<0.10 else "ns"))
        print(f"\nWilcoxon (n={n}):")
        print(f"  Sharpe: diff={np.mean(ds[:n])-np.mean(ss[1.0][:n]):+.3f}, p={pv_s:.6f} {sig_s}")
        print(f"  Trap:   diff={np.mean(dt_trap[:n])-np.mean(st_trap[:n]):+.3f}, p={pv_t:.6f} {sig_t}")

# Threshold analysis
print(f"\nThreshold Selection Analysis:")
all_thresholds = []
for pair in cointegrated_pairs:
    r = backtest_pair(pair, T0, T1, agent=agent)
    if r and 'thresholds' in r:
        all_thresholds.extend(r['thresholds'])
if all_thresholds:
    for th in THRESHOLDS:
        pct = sum(1 for t in all_thresholds if t == th) / len(all_thresholds) * 100
        print(f"  {th:.2f}σ: {pct:.1f}%")

# Save
torch.save({
    'model_state_dict': agent.q_network.state_dict(),
    'state_dim': STATE_DIM, 'action_dim': K,
    'thresholds': THRESHOLDS,
    'best_val_reward': best_val_reward,
    'pca_dim': PCA_DIM,
}, 'adaptive_threshold_dqn.pt')
print(f"\nSaved to adaptive_threshold_dqn.pt")
print(f"\nApproach follows Shen & Kurshan (ICAIF 2020):")
print(f"  DQN selects z-score threshold each week based on market state.")
print(f"  Direction always from z-score. No long bias possible.")
