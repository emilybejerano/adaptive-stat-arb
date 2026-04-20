"""
DQN v2 Training Script — Adaptive Execution for Pairs Trading

Trains a Deep Q-Network to decide flat/long/short on pairs trading spreads.
Uses fixed 10% capital position sizing so every action has real consequences.
Includes a penalty for trading when the OU model shows no edge (kelly=0).

Walk-forward CV:
  Fold 1: Train 2010-2014, Val 2015-2016
  Fold 2: Train 2010-2016, Val 2017-2018
  Final:  Train 2010-2018, Val 2019, Test 2020-2023

Usage:
  conda activate elen4904
  python train_dqn_v2.py

Output: dqn_pairs_agent_v2.pt
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
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# LOAD DATA
# ============================================================
with open('datasets/nb02_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)
scaler = artifacts['scaler']; pca = artifacts['pca']
cointegrated_pairs = artifacts['cointegrated_pairs']
STATIC_FEATURE_COLS = artifacts['STATIC_FEATURE_COLS']
STATE_DIM = artifacts['state_dim']; ACTION_DIM = 3

spreads_df = pd.read_parquet('datasets/spreads.parquet')
all_features = {}
for p in cointegrated_pairs:
    all_features[p] = pd.read_parquet(f'datasets/features_{p.replace("/","_")}.parquet')
print(f"Loaded {len(cointegrated_pairs)} pairs, state_dim={STATE_DIM}")


# ============================================================
# ENVIRONMENT — Fixed position sizing, 3 actions
# ============================================================
class PairsTradingEnvV2(gym.Env):
    """
    Pairs trading environment with fixed position sizing.

    Actions: 0=flat, 1=long, 2=short
    Position size: always 10% of initial capital
    Penalty for trading when kelly_fraction=0 (no OU edge)
    """
    ACTION_MAP = {0: 0, 1: 1, 2: -1}

    def __init__(self, spread_series, features_df, kelly_series,
                 scaler, pca_model, tc=0.001, risk_lambda=0.5,
                 max_position_days=60, initial_capital=100000.0,
                 position_frac=0.10, no_edge_penalty=0.001):
        super().__init__()
        self.spread = spread_series.values
        self.features_raw = features_df[STATIC_FEATURE_COLS].values
        self.kelly_fractions = kelly_series.values
        self.scaler = scaler; self.pca_model = pca_model
        self.tc = tc; self.risk_lambda = risk_lambda
        self.max_days = max_position_days
        self.initial_capital = initial_capital
        self.position_frac = position_frac
        self.no_edge_penalty = no_edge_penalty
        self.n_pca = pca_model.n_components_
        self.state_dim = self.n_pca + 3
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.state_dim,), np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0; self.position = 0; self.position_size = 0.0
        self.entry_price = 0.0; self.days_in_trade = 0
        self.capital = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.returns_history = []
        return self._get_obs(), {}

    def _get_obs(self):
        if self.step_idx >= len(self.features_raw):
            return np.zeros(self.state_dim, np.float32)
        raw = self.features_raw[self.step_idx:self.step_idx+1]
        raw = np.nan_to_num(raw, 0, 0, 0)
        try:
            scaled = self.scaler.transform(raw)
            pca_feat = self.pca_model.transform(scaled).flatten()
        except:
            pca_feat = np.zeros(self.n_pca)
        unr = 0.0
        if self.position != 0 and self.step_idx < len(self.spread):
            unr = self.position * (self.spread[self.step_idx] - self.entry_price) * self.position_size
            unr /= max(self.portfolio_value, 1)
        return np.concatenate([pca_feat, [float(self.position), unr,
                               float(self.days_in_trade)/self.max_days]]).astype(np.float32)

    def step(self, action):
        if self.step_idx >= len(self.spread) - 1:
            return self._get_obs(), 0.0, True, False, {}
        direction = self.ACTION_MAP[action]
        sv = self.spread[self.step_idx]
        kf = self.kelly_fractions[self.step_idx] if self.step_idx < len(self.kelly_fractions) else 0
        old_value = self.portfolio_value
        penalty = 0.0

        if direction == 0:
            if self.position != 0:
                pnl = self.position * (sv - self.entry_price) * self.position_size
                cost = self.tc * abs(self.position_size) * abs(sv)
                self.capital += pnl - cost
                self.position = 0; self.position_size = 0; self.days_in_trade = 0
        else:
            if self.position != direction:
                if self.position != 0:
                    pnl = self.position * (sv - self.entry_price) * self.position_size
                    cost = self.tc * abs(self.position_size) * abs(sv)
                    self.capital += pnl - cost
                self.position = direction
                self.position_size = self.initial_capital * self.position_frac / max(abs(sv), 0.01)
                self.entry_price = sv
                self.capital -= self.tc * abs(self.position_size) * abs(sv)
                self.days_in_trade = 0
                if kf <= 0: penalty = self.no_edge_penalty

        self.step_idx += 1
        if self.position != 0: self.days_in_trade += 1

        if self.days_in_trade >= self.max_days and self.position != 0:
            sv2 = self.spread[self.step_idx]
            pnl = self.position * (sv2 - self.entry_price) * self.position_size
            cost = self.tc * abs(self.position_size) * abs(sv2)
            self.capital += pnl - cost
            self.position = 0; self.position_size = 0; self.days_in_trade = 0

        unrealized = 0.0
        if self.position != 0:
            unrealized = self.position * (self.spread[self.step_idx] - self.entry_price) * self.position_size
        self.portfolio_value = self.capital + unrealized

        if old_value > 0 and self.portfolio_value > 0:
            log_ret = np.log(self.portfolio_value / old_value)
        else:
            log_ret = -1.0
        self.returns_history.append(log_ret)
        roll_var = np.var(self.returns_history[-20:]) if len(self.returns_history) >= 20 else 0
        reward = log_ret - self.risk_lambda * roll_var - penalty

        done = self.step_idx >= len(self.spread) - 1
        if self.portfolio_value < 0.5 * self.initial_capital:
            done = True; reward -= 1.0

        return self._get_obs(), reward, done, False, {
            'portfolio_value': self.portfolio_value, 'log_return': log_ret}


# ============================================================
# DQN AGENT
# ============================================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim=3, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, action_dim))
    def forward(self, x): return self.network(x)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args): self.buffer.append(Transition(*args))
    def sample(self, n): return random.sample(self.buffer, n)
    def __len__(self): return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim=3, lr=1e-4, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.995,
                 batch_size=64, tau=0.005, weight_decay=1e-4, dropout=0.2):
        self.action_dim = action_dim; self.gamma = gamma
        self.epsilon = eps_start; self.eps_end = eps_end; self.eps_decay = eps_decay
        self.batch_size = batch_size; self.tau = tau
        self.policy_net = QNetwork(state_dim, action_dim, dropout).to(device)
        self.target_net = QNetwork(state_dim, action_dim, dropout).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=weight_decay)
        self.memory = ReplayBuffer(100000)

    def select_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            self.policy_net.eval()
            q = self.policy_net(torch.FloatTensor(state).unsqueeze(0).to(device))
            self.policy_net.train()
            return q.argmax(dim=1).item()

    def optimize(self):
        if len(self.memory) < self.batch_size: return 0.0
        batch = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*batch))
        s = torch.FloatTensor(np.array(batch.state)).to(device)
        a = torch.LongTensor(batch.action).unsqueeze(1).to(device)
        r = torch.FloatTensor(batch.reward).to(device)
        ns = torch.FloatTensor(np.array(batch.next_state)).to(device)
        d = torch.FloatTensor(batch.done).to(device)
        cur_q = self.policy_net(s).gather(1, a).squeeze()
        with torch.no_grad():
            na = self.policy_net(ns).argmax(dim=1, keepdim=True)
            nq = self.target_net(ns).gather(1, na).squeeze()
            tgt = r + (1 - d) * self.gamma * nq
        loss = nn.SmoothL1Loss()(cur_q, tgt)
        self.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(self.tau * pp.data + (1 - self.tau) * tp.data)
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)


# ============================================================
# HELPERS
# ============================================================
def make_env(pair_name, t0, t1):
    feat = all_features[pair_name]; spread = spreads_df[pair_name].dropna()
    mask = (feat.index >= t0) & (feat.index <= t1)
    feat_slice = feat.loc[mask]
    common = spread.index.intersection(feat_slice.index)
    if len(common) < 60: return None
    return PairsTradingEnvV2(
        spread_series=spread.loc[common], features_df=feat_slice.loc[common],
        kelly_series=feat_slice.loc[common, 'kelly_fraction'],
        scaler=scaler, pca_model=pca)

def evaluate_agent(agent, pairs, t0, t1):
    sharpes = []
    for pair in pairs:
        env = make_env(pair, t0, t1)
        if env is None: continue
        obs, _ = env.reset(); done = False; rets = []
        while not done:
            action = agent.select_action(obs, eval_mode=True)
            obs, _, done, _, info = env.step(action)
            rets.append(info['log_return'])
        if len(rets) > 20:
            r = np.array(rets)
            sharpes.append(np.mean(r) / (np.std(r) + 1e-10) * np.sqrt(252))
    return np.mean(sharpes) if sharpes else 0.0

def run_episode(agent, env):
    obs, _ = env.reset(); total_reward = 0; done = False
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, done, _, _ = env.step(action)
        agent.memory.push(obs, action, reward, next_obs, float(done))
        agent.optimize(); obs = next_obs; total_reward += reward
    agent.decay_epsilon()
    return total_reward


# ============================================================
# TRAINING CONFIG
# ============================================================
MAX_EPISODES = 600
EARLY_STOP_PATIENCE = 40
EVAL_EVERY = 10

FOLDS = [
    {'train_end': '2014-12-31', 'val_start': '2015-01-01', 'val_end': '2016-12-31'},
    {'train_end': '2016-12-31', 'val_start': '2017-01-01', 'val_end': '2018-12-31'},
]
FINAL_TRAIN_END = '2018-12-31'
FINAL_VAL_START = '2019-01-01'
FINAL_VAL_END = '2019-12-31'

print(f"\nConfig: {MAX_EPISODES} episodes, patience={EARLY_STOP_PATIENCE}, "
      f"3 actions, fixed 10% sizing")


# ============================================================
# WALK-FORWARD CV
# ============================================================
fold_results = []
for fi, fold in enumerate(FOLDS):
    te = fold['train_end']; vs = fold['val_start']; ve = fold['val_end']
    print(f"\n{'='*70}")
    print(f"  Fold {fi+1}: Train 2010-{te[:4]}, Val {vs[:4]}-{ve[:4]}")
    print(f"{'='*70}")
    agent = DQNAgent(STATE_DIM, action_dim=3)
    train_envs = [(p, make_env(p, '2010-01-01', te)) for p in cointegrated_pairs]
    train_envs = [(p, e) for p, e in train_envs if e is not None]
    print(f"  {len(train_envs)} training pairs")

    best_val = -np.inf; best_ep = 0; best_state = None; patience = 0
    for ep in range(MAX_EPISODES):
        pair, _ = random.choice(train_envs)
        env = make_env(pair, '2010-01-01', te)
        if env is None: continue
        run_episode(agent, env)
        if (ep + 1) % EVAL_EVERY == 0:
            vs_val = evaluate_agent(agent, cointegrated_pairs, vs, ve)
            if vs_val > best_val:
                best_val = vs_val; best_ep = ep
                best_state = {k: v.clone() for k, v in agent.policy_net.state_dict().items()}
                patience = 0
            else:
                patience += EVAL_EVERY
            if (ep + 1) % 50 == 0:
                print(f"  Ep {ep+1:3d}: val_sharpe={vs_val:.3f} best={best_val:.3f} eps={agent.epsilon:.3f}")
            if patience >= EARLY_STOP_PATIENCE * EVAL_EVERY:
                print(f"  Early stop at ep {ep+1} (best {best_val:.3f} at ep {best_ep+1})")
                break
    if best_state: agent.policy_net.load_state_dict(best_state)
    print(f"  Best val Sharpe: {best_val:.3f} at ep {best_ep+1}")
    fold_results.append({'fold': fi+1, 'val_sharpe': best_val, 'best_ep': best_ep})


# ============================================================
# FINAL MODEL
# ============================================================
print(f"\n{'='*70}")
print(f"  Final Model: Train 2010-2018, Val 2019")
print(f"{'='*70}")
final_agent = DQNAgent(STATE_DIM, action_dim=3)
train_envs = [(p, make_env(p, '2010-01-01', FINAL_TRAIN_END)) for p in cointegrated_pairs]
train_envs = [(p, e) for p, e in train_envs if e is not None]
print(f"  {len(train_envs)} training pairs")

best_val = -np.inf; best_ep = 0; best_state = None; patience = 0
for ep in range(MAX_EPISODES):
    pair, _ = random.choice(train_envs)
    env = make_env(pair, '2010-01-01', FINAL_TRAIN_END)
    if env is None: continue
    run_episode(final_agent, env)
    if (ep + 1) % EVAL_EVERY == 0:
        vs_val = evaluate_agent(final_agent, cointegrated_pairs, FINAL_VAL_START, FINAL_VAL_END)
        if vs_val > best_val:
            best_val = vs_val; best_ep = ep
            best_state = {k: v.clone() for k, v in final_agent.policy_net.state_dict().items()}
            patience = 0
        else:
            patience += EVAL_EVERY
        if (ep + 1) % 50 == 0:
            print(f"  Ep {ep+1:3d}: val_sharpe={vs_val:.3f} best={best_val:.3f} eps={final_agent.epsilon:.3f}")
        if patience >= EARLY_STOP_PATIENCE * EVAL_EVERY:
            print(f"  Early stop at ep {ep+1} (best {best_val:.3f} at ep {best_ep+1})")
            break

if best_state: final_agent.policy_net.load_state_dict(best_state)
print(f"\n  Best val Sharpe: {best_val:.3f} at ep {best_ep+1}")

# Save
torch.save({
    'model_state_dict': final_agent.policy_net.state_dict(),
    'state_dim': STATE_DIM, 'action_dim': 3,
    'fold_results': fold_results, 'final_val_sharpe': best_val,
}, 'dqn_pairs_agent_v2.pt')
print(f"  Saved to dqn_pairs_agent_v2.pt ({os.path.getsize('dqn_pairs_agent_v2.pt')/1024:.1f} KB)")
print(f"\nCV Summary: {', '.join(f'Fold {r[\"fold\"]}={r[\"val_sharpe\"]:.3f}' for r in fold_results)}")
print(f"Final val Sharpe: {best_val:.3f}")
print("Done. Run run_backtest_v2.py for evaluation.")
