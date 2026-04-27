import os
from collections import deque

import numpy as np


REWARD_MODE = os.getenv("REWARD_MODE", "original").strip().lower()
LAMBDA_SWITCH = float(os.getenv("LAMBDA_SWITCH", "0.01"))
LAMBDA_TRAP = float(os.getenv("LAMBDA_TRAP", "0.05"))
LAMBDA_VOL = float(os.getenv("LAMBDA_VOL", "0.05"))
ROLLING_WINDOW = int(os.getenv("REWARD_ROLLING_WINDOW", "20"))
EPS = float(os.getenv("REWARD_EPS", "1e-8"))

VALID_REWARD_MODES = {
    "original",
    "switch_penalty",
    "trap_penalty",
    "vol_penalty",
    "sharpe_proxy",
}

if REWARD_MODE not in VALID_REWARD_MODES:
    raise ValueError(
        f"Unsupported REWARD_MODE={REWARD_MODE!r}. "
        f"Expected one of {sorted(VALID_REWARD_MODES)}."
    )


def reward_formula_text():
    if REWARD_MODE == "original":
        return "reward = ((week_wins - week_losses) / (capital * 0.01)) * (week + 1) - capacity_penalty"
    if REWARD_MODE == "switch_penalty":
        return "reward = original_reward - lambda_switch * abs(action_t - action_{t-1})"
    if REWARD_MODE == "trap_penalty":
        return "reward = original_reward - lambda_trap * trap_indicator"
    if REWARD_MODE == "vol_penalty":
        return "reward = original_reward - lambda_vol * rolling_return_volatility"
    if REWARD_MODE == "sharpe_proxy":
        return "reward = rolling_mean_return / (rolling_std_return + eps)"
    return REWARD_MODE


def mode_suffix():
    return REWARD_MODE


class RewardTracker:
    def __init__(self, window=ROLLING_WINDOW):
        self.window = window
        self.return_history = deque(maxlen=window)

    def update(self, weekly_return):
        self.return_history.append(float(weekly_return))

    def rolling_vol(self):
        if len(self.return_history) < 2:
            return 0.0
        return float(np.std(self.return_history))

    def sharpe_proxy(self):
        if len(self.return_history) < 2:
            return 0.0
        arr = np.asarray(self.return_history, dtype=float)
        return float(np.mean(arr) / (np.std(arr) + EPS))


def compute_original_reward(week_wins, week_losses, capital, week_idx, capacity_excess):
    reward = (week_wins - week_losses) / (capital * 0.01) * (week_idx + 1)
    if capacity_excess > 0:
        reward -= 1.0 * capacity_excess
    return float(reward)


def apply_reward_mode(original_reward, action, prev_action, trap_indicator, reward_tracker):
    if REWARD_MODE == "original":
        return float(original_reward), {}

    if REWARD_MODE == "switch_penalty":
        switch_cost = 0.0 if prev_action is None else LAMBDA_SWITCH * abs(action - prev_action)
        return float(original_reward - switch_cost), {"switch_cost": float(switch_cost)}

    if REWARD_MODE == "trap_penalty":
        trap_cost = LAMBDA_TRAP * float(trap_indicator)
        return float(original_reward - trap_cost), {"trap_cost": float(trap_cost)}

    if REWARD_MODE == "vol_penalty":
        vol_cost = LAMBDA_VOL * reward_tracker.rolling_vol()
        return float(original_reward - vol_cost), {"vol_cost": float(vol_cost)}

    if REWARD_MODE == "sharpe_proxy":
        return float(reward_tracker.sharpe_proxy()), {}

    return float(original_reward), {}
