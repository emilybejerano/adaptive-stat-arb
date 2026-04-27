"""
Stage 2 DQN Environment v2: Adaptive Threshold with Sit-Out Action.

Key change from v1: Action 0 = "sit out" (don't trade this month).
This lets the agent skip bad months, which should improve Sharpe by
avoiding negative returns rather than just picking the right γ.

State space (9 features): same as v1
Action space: Discrete(5) → {SIT_OUT, γ=0.5, γ=1.0, γ=1.25, γ=2.0}

Reward:
  - If trading: r_t - λ·r_t²  (risk-adjusted return)
  - If sitting out: small negative penalty (-0.001) to discourage
    always sitting out. This is ~half of monthly TC, so the agent
    only sits out when it expects worse than -0.001 from trading.
"""
import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from glob import glob

TC_PER_SIDE = 0.0010

# Action 0 = sit out, Actions 1-4 = trade with these γ values
GAMMA_CHOICES_V2 = [None, 0.5, 1.0, 1.25, 2.0]
SIT_OUT_PENALTY = -0.001  # small cost for doing nothing


class ThresholdEnvV2(gym.Env):
    """Gym environment with sit-out action for adaptive threshold selection."""

    metadata = {"render_modes": []}

    def __init__(self, cluster_dir, log_returns_df,
                 start_month='2000-01', end_month='2023-12',
                 reward_lambda=0.5, stoploss=-0.10):
        super().__init__()

        self.cluster_dir = cluster_dir
        self.log_returns_df = log_returns_df
        self.reward_lambda = reward_lambda
        self.stoploss = stoploss

        # Build month sequence
        cluster_files = sorted(glob(f'{cluster_dir}/*.csv'))
        available_months = [os.path.splitext(os.path.basename(f))[0]
                            for f in cluster_files]
        return_months = sorted([str(c) for c in log_returns_df.columns])

        self.trading_months = [
            m for m in available_months
            if start_month <= m <= end_month and m in return_months
        ]
        self.return_months = return_months

        # Pre-load all cluster data for speed
        self.cluster_data = {}
        for month in self.trading_months:
            fpath = os.path.join(cluster_dir, f'{month}.csv')
            df = pd.read_csv(fpath)
            if 'Unnamed: 0' in df.columns:
                if 'firms' in df.columns:
                    df = df.drop(columns=['Unnamed: 0'])
                else:
                    df.rename(columns={'Unnamed: 0': 'firms'}, inplace=True)
            if 'firms' in df.columns:
                df['firms'] = df['firms'].astype(str)
                df = df.set_index('firms')
            self.cluster_data[month] = df

        # Precompute cluster membership for stability metric
        self._precompute_membership()

        # Spaces — 5 actions: sit_out + 4 gamma levels
        self.action_space = spaces.Discrete(len(GAMMA_CHOICES_V2))
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(9,), dtype=np.float32
        )

        # Episode state
        self.current_step = 0
        self.prev_spread_sigma = None
        self.recent_returns = []

        # Running stats for z-score normalization
        self._obs_mean = np.zeros(9)
        self._obs_var = np.ones(9)
        self._obs_count = 0

    def _precompute_membership(self):
        self.membership = {}
        for month in self.trading_months:
            df = self.cluster_data[month]
            if 'clusters' in df.columns:
                self.membership[month] = df['clusters'].to_dict()
            else:
                self.membership[month] = {}

    def _compute_cluster_stability(self, current_month, prev_month):
        if prev_month is None or prev_month not in self.membership:
            return 0.5

        curr = self.membership.get(current_month, {})
        prev = self.membership.get(prev_month, {})

        common = set(curr.keys()) & set(prev.keys())
        if len(common) == 0:
            return 0.5

        same_cluster = sum(1 for s in common if curr[s] == prev[s])
        return same_cluster / len(common)

    def _get_observation(self, month):
        df = self.cluster_data[month]

        if 'MOM1' not in df.columns or 'clusters' not in df.columns:
            return np.zeros(9, dtype=np.float32)

        df_clean = df.dropna(subset=['MOM1'])

        all_spreads = []
        cluster_count = 0
        tradeable_count = 0

        for cid in df_clean['clusters'].unique():
            cdf = df_clean[df_clean['clusters'] == cid]
            if len(cdf) < 5:
                continue
            cluster_count += 1
            spread = cdf['MOM1'] - cdf['MOM1'].median()
            all_spreads.extend(spread.values)

            sigma = np.std(spread.values)
            if sigma > 1e-8:
                threshold = 1.0 * sigma
                has_long = (spread < -threshold).any()
                has_short = (spread > threshold).any()
                if has_long and has_short:
                    tradeable_count += 1

        global_sigma = np.std(all_spreads) if len(all_spreads) > 0 else 0.0

        if self.prev_spread_sigma is not None and self.prev_spread_sigma > 1e-8:
            spread_velocity = (global_sigma - self.prev_spread_sigma) / self.prev_spread_sigma
        else:
            spread_velocity = 0.0
        self.prev_spread_sigma = global_sigma

        step_idx = self.trading_months.index(month)
        prev_month = self.trading_months[step_idx - 1] if step_idx > 0 else None
        stability = self._compute_cluster_stability(month, prev_month)

        frac_tradeable = tradeable_count / cluster_count if cluster_count > 0 else 0.0
        cs_vol = df_clean['MOM1'].std() if len(df_clean) > 0 else 0.0
        market_ret = df_clean['MOM1'].mean() if len(df_clean) > 0 else 0.0

        r1 = self.recent_returns[-1] if len(self.recent_returns) >= 1 else 0.0
        r3 = np.mean(self.recent_returns[-3:]) if len(self.recent_returns) >= 1 else 0.0
        r6 = np.mean(self.recent_returns[-6:]) if len(self.recent_returns) >= 1 else 0.0

        raw = np.array([
            global_sigma, spread_velocity, stability, frac_tradeable,
            cs_vol, market_ret, r1, r3, r6
        ], dtype=np.float32)

        self._obs_count += 1
        old_mean = self._obs_mean.copy()
        self._obs_mean += (raw - self._obs_mean) / self._obs_count
        self._obs_var += (raw - old_mean) * (raw - self._obs_mean)

        if self._obs_count > 1:
            std = np.sqrt(self._obs_var / (self._obs_count - 1))
            std = np.clip(std, 1e-6, None)
            obs = (raw - self._obs_mean) / std
        else:
            obs = np.zeros(9, dtype=np.float32)

        return np.clip(obs, -5.0, 5.0).astype(np.float32)

    def _execute_trade(self, month, gamma):
        """Run Algorithm 1 for one month with given gamma. Returns log return."""
        df = self.cluster_data[month]

        if 'MOM1' not in df.columns or 'clusters' not in df.columns:
            return 0.0

        df_clean = df.dropna(subset=['MOM1'])
        if len(df_clean) < 10:
            return 0.0

        month_idx = self.return_months.index(month) if month in self.return_months else -1
        if month_idx < 0 or month_idx + 1 >= len(self.return_months):
            return 0.0
        next_month = self.return_months[month_idx + 1]
        next_returns = self.log_returns_df[next_month]
        next_returns.index = next_returns.index.astype(str)

        cluster_spreads = {}
        all_spread_values = []

        for cid in df_clean['clusters'].unique():
            cdf = df_clean[df_clean['clusters'] == cid]
            if len(cdf) < 5:
                continue
            spread = cdf['MOM1'] - cdf['MOM1'].median()
            cluster_spreads[cid] = spread
            all_spread_values.extend(spread.values)

        if len(all_spread_values) < 10:
            return 0.0

        sigma_delta = np.std(all_spread_values)
        if sigma_delta < 1e-8:
            return 0.0

        threshold = gamma * sigma_delta

        cluster_returns = []
        for cid, spread in cluster_spreads.items():
            long_idx = spread[spread < -threshold].index.astype(str)
            short_idx = spread[spread > threshold].index.astype(str)

            if len(long_idx) == 0 or len(short_idx) == 0:
                continue

            long_rets = next_returns.reindex(long_idx).dropna()
            short_rets = next_returns.reindex(short_idx).dropna()

            if len(long_rets) == 0 or len(short_rets) == 0:
                continue

            if self.stoploss is not None:
                sl_log = np.log1p(self.stoploss)
                long_rets = long_rets.clip(lower=sl_log)

            cluster_ret = long_rets.mean() - short_rets.mean()
            cluster_ret -= 2 * TC_PER_SIDE
            cluster_returns.append(cluster_ret)

        if len(cluster_returns) == 0:
            return 0.0

        return np.mean(cluster_returns)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.prev_spread_sigma = None
        self.recent_returns = []
        obs = self._get_observation(self.trading_months[0])
        return obs, {}

    def step(self, action):
        month = self.trading_months[self.current_step]
        gamma = GAMMA_CHOICES_V2[action]

        if gamma is None:
            # Sit out — no trade this month
            ret = 0.0
            reward = SIT_OUT_PENALTY
            action_label = 'sit_out'
        else:
            ret = self._execute_trade(month, gamma)
            reward = ret - self.reward_lambda * (ret ** 2)
            action_label = gamma

        self.recent_returns.append(ret)

        self.current_step += 1
        terminated = self.current_step >= len(self.trading_months)
        truncated = False

        if not terminated:
            obs = self._get_observation(self.trading_months[self.current_step])
        else:
            obs = np.zeros(9, dtype=np.float32)

        info = {'month': month, 'gamma': action_label, 'return': ret}
        return obs, reward, terminated, truncated, info
