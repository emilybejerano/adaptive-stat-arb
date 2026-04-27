"""
Stage 2 DQN Environment with OU Parameters: ORCA-only.

Extends v3 by adding 3 OU-derived state features from ORCA's trading_data:
  9.  median_theta    — median mean-reversion speed across clusters
  10. median_sigma_ou — median OU volatility across clusters
  11. theta_dispersion — std of theta across clusters (are clusters similar?)

These directly measure mean-reversion quality, which determines whether
aggressive (high γ) or conservative (low γ) thresholds work better.

Action space: Discrete(5) → {sit_out, γ=0.5, γ=1.0, γ=1.25, γ=2.0}
Reward: Differential Sharpe (same as v3)
Discount: γ_RL=0 (bandit)
"""
import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from glob import glob

TC_PER_SIDE = 0.0010
GAMMA_CHOICES_OU = [None, 0.5, 1.0, 1.25, 2.0]
SIT_OUT_PENALTY = -0.002
N_FEATURES = 12  # 9 original + 3 OU


class ThresholdEnvOU(gym.Env):
    """DQN environment with OU parameters in state space."""

    metadata = {"render_modes": []}

    def __init__(self, cluster_dir, log_returns_df,
                 trading_data_dir=None,
                 start_month='2000-01', end_month='2023-12',
                 stoploss=-0.10, training=True):
        super().__init__()

        self.cluster_dir = cluster_dir
        self.log_returns_df = log_returns_df
        self.trading_data_dir = trading_data_dir
        self.stoploss = stoploss
        self.training = training

        cluster_files = sorted(glob(f'{cluster_dir}/*.csv'))
        available_months = [os.path.splitext(os.path.basename(f))[0]
                            for f in cluster_files]
        return_months = sorted([str(c) for c in log_returns_df.columns])

        self.trading_months = [
            m for m in available_months
            if start_month <= m <= end_month and m in return_months
        ]
        self.return_months = return_months

        # Pre-load cluster data
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

        # Pre-load OU parameters from trading_data
        self.ou_data = {}
        if trading_data_dir:
            for month in self.trading_months:
                td_path = os.path.join(trading_data_dir, f'{month}.csv')
                if os.path.exists(td_path):
                    td = pd.read_csv(td_path)
                    # Extract per-cluster OU stats
                    if 'cluster_theta' in td.columns and 'cluster_sigma' in td.columns:
                        thetas = td.groupby('clusters')['cluster_theta'].first().values
                        sigmas = td.groupby('clusters')['cluster_sigma'].first().values
                        thetas = thetas[thetas > 0.01]  # filter degenerate
                        sigmas = sigmas[sigmas > 1e-6]
                        self.ou_data[month] = {
                            'median_theta': np.median(thetas) if len(thetas) > 0 else 0,
                            'median_sigma': np.median(sigmas) if len(sigmas) > 0 else 0,
                            'theta_disp': np.std(thetas) if len(thetas) > 1 else 0,
                        }
                    else:
                        self.ou_data[month] = {'median_theta': 0, 'median_sigma': 0, 'theta_disp': 0}
                else:
                    self.ou_data[month] = {'median_theta': 0, 'median_sigma': 0, 'theta_disp': 0}

        self._precompute_membership()
        self._precompute_returns()

        self.action_space = spaces.Discrete(len(GAMMA_CHOICES_OU))
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(N_FEATURES,), dtype=np.float32
        )

        self.current_step = 0
        self.prev_spread_sigma = None
        self.recent_returns = []
        self._A = 0.0
        self._B = 0.0
        self._eta = 0.1

        self._obs_mean = np.zeros(N_FEATURES)
        self._obs_var = np.ones(N_FEATURES)
        self._obs_count = 0

    def _precompute_membership(self):
        self.membership = {}
        for month in self.trading_months:
            df = self.cluster_data[month]
            if 'clusters' in df.columns:
                self.membership[month] = df['clusters'].to_dict()
            else:
                self.membership[month] = {}

    def _precompute_returns(self):
        self.cached_returns = {}
        self.cached_spread_sigma = {}

        for month in self.trading_months:
            df = self.cluster_data[month]
            month_idx = self.return_months.index(month) if month in self.return_months else -1
            if month_idx < 0 or month_idx + 1 >= len(self.return_months):
                for g in [0.5, 1.0, 1.25, 2.0]:
                    self.cached_returns[(month, g)] = 0.0
                self.cached_spread_sigma[month] = 0.0
                continue

            next_month = self.return_months[month_idx + 1]
            next_returns = self.log_returns_df[next_month]
            next_returns.index = next_returns.index.astype(str)

            if 'MOM1' not in df.columns or 'clusters' not in df.columns:
                for g in [0.5, 1.0, 1.25, 2.0]:
                    self.cached_returns[(month, g)] = 0.0
                self.cached_spread_sigma[month] = 0.0
                continue

            df_clean = df.dropna(subset=['MOM1'])
            cluster_spreads = {}
            all_spread_values = []

            for cid in df_clean['clusters'].unique():
                cdf = df_clean[df_clean['clusters'] == cid]
                if len(cdf) < 5:
                    continue
                spread = cdf['MOM1'] - cdf['MOM1'].median()
                cluster_spreads[cid] = spread
                all_spread_values.extend(spread.values)

            sigma_delta = np.std(all_spread_values) if len(all_spread_values) > 10 else 0.0
            self.cached_spread_sigma[month] = sigma_delta

            for gamma in [0.5, 1.0, 1.25, 2.0]:
                if sigma_delta < 1e-8:
                    self.cached_returns[(month, gamma)] = 0.0
                    continue
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
                        long_rets = long_rets.clip(lower=np.log1p(self.stoploss))
                    cr = long_rets.mean() - short_rets.mean() - 2 * TC_PER_SIDE
                    cluster_returns.append(cr)
                self.cached_returns[(month, gamma)] = np.mean(cluster_returns) if cluster_returns else 0.0

    def _compute_cluster_stability(self, current_month, prev_month):
        if prev_month is None or prev_month not in self.membership:
            return 0.5
        curr = self.membership.get(current_month, {})
        prev = self.membership.get(prev_month, {})
        common = set(curr.keys()) & set(prev.keys())
        if len(common) == 0:
            return 0.5
        return sum(1 for s in common if curr[s] == prev[s]) / len(common)

    def _get_observation(self, month):
        df = self.cluster_data[month]
        if 'MOM1' not in df.columns or 'clusters' not in df.columns:
            return np.zeros(N_FEATURES, dtype=np.float32)

        df_clean = df.dropna(subset=['MOM1'])
        global_sigma = self.cached_spread_sigma.get(month, 0.0)

        if self.prev_spread_sigma is not None and self.prev_spread_sigma > 1e-8:
            spread_velocity = (global_sigma - self.prev_spread_sigma) / self.prev_spread_sigma
        else:
            spread_velocity = 0.0
        self.prev_spread_sigma = global_sigma

        step_idx = self.trading_months.index(month)
        prev_month = self.trading_months[step_idx - 1] if step_idx > 0 else None
        stability = self._compute_cluster_stability(month, prev_month)

        cluster_count = 0
        tradeable_count = 0
        for cid in df_clean['clusters'].unique():
            cdf = df_clean[df_clean['clusters'] == cid]
            if len(cdf) < 5:
                continue
            cluster_count += 1
            spread = cdf['MOM1'] - cdf['MOM1'].median()
            sigma = np.std(spread.values)
            if sigma > 1e-8:
                if (spread < -sigma).any() and (spread > sigma).any():
                    tradeable_count += 1

        frac_tradeable = tradeable_count / cluster_count if cluster_count > 0 else 0.0
        cs_vol = df_clean['MOM1'].std() if len(df_clean) > 0 else 0.0
        market_ret = df_clean['MOM1'].mean() if len(df_clean) > 0 else 0.0

        r1 = self.recent_returns[-1] if len(self.recent_returns) >= 1 else 0.0
        r3 = np.mean(self.recent_returns[-3:]) if len(self.recent_returns) >= 1 else 0.0
        r6 = np.mean(self.recent_returns[-6:]) if len(self.recent_returns) >= 1 else 0.0

        # OU features (new)
        ou = self.ou_data.get(month, {'median_theta': 0, 'median_sigma': 0, 'theta_disp': 0})
        median_theta = ou['median_theta']
        median_sigma = ou['median_sigma']
        theta_disp = ou['theta_disp']

        raw = np.array([
            global_sigma, spread_velocity, stability, frac_tradeable,
            cs_vol, market_ret, r1, r3, r6,
            median_theta, median_sigma, theta_disp
        ], dtype=np.float32)

        if self.training:
            raw += np.random.normal(0, 0.01, size=raw.shape).astype(np.float32)

        self._obs_count += 1
        old_mean = self._obs_mean.copy()
        self._obs_mean += (raw - self._obs_mean) / self._obs_count
        self._obs_var += (raw - old_mean) * (raw - self._obs_mean)

        if self._obs_count > 1:
            std = np.sqrt(self._obs_var / (self._obs_count - 1))
            std = np.clip(std, 1e-6, None)
            obs = (raw - self._obs_mean) / std
        else:
            obs = np.zeros(N_FEATURES, dtype=np.float32)

        return np.clip(obs, -5.0, 5.0).astype(np.float32)

    def _differential_sharpe(self, ret):
        delta_A = ret - self._A
        delta_B = ret**2 - self._B
        denom = (self._B - self._A**2)
        if denom > 1e-10:
            ds = (self._B * delta_A - 0.5 * self._A * delta_B) / (denom ** 1.5)
        else:
            ds = ret * 10.0
        self._A += self._eta * delta_A
        self._B += self._eta * delta_B
        return float(np.clip(ds, -2.0, 2.0))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.prev_spread_sigma = None
        self.recent_returns = []
        self._A = 0.0
        self._B = 0.0
        obs = self._get_observation(self.trading_months[0])
        return obs, {}

    def step(self, action):
        month = self.trading_months[self.current_step]
        gamma = GAMMA_CHOICES_OU[action]

        if gamma is None:
            ret = 0.0
            reward = SIT_OUT_PENALTY
            action_label = 'sit_out'
            self._differential_sharpe(0.0)
        else:
            ret = self.cached_returns.get((month, gamma), 0.0)
            reward = self._differential_sharpe(ret)
            action_label = gamma

        self.recent_returns.append(ret)
        self.current_step += 1
        terminated = self.current_step >= len(self.trading_months)

        if not terminated:
            obs = self._get_observation(self.trading_months[self.current_step])
        else:
            obs = np.zeros(N_FEATURES, dtype=np.float32)

        info = {'month': month, 'gamma': action_label, 'return': ret}
        return obs, reward, terminated, False, info
