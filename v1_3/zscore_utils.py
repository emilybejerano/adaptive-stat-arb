import os

import numpy as np
import pandas as pd


ZSCORE_METHOD = os.getenv("ZSCORE_METHOD", "mad").strip().lower()
ZSCORE_WINDOW = int(os.getenv("ZSCORE_WINDOW", "60"))
EPS = float(os.getenv("ZSCORE_EPS", "1e-8"))

VALID_METHODS = {"standard", "mad", "ewm"}
if ZSCORE_METHOD not in VALID_METHODS:
    raise ValueError(
        f"Unsupported ZSCORE_METHOD={ZSCORE_METHOD!r}. "
        f"Expected one of {sorted(VALID_METHODS)}."
    )


def compute_spread_zscore(spread, window=ZSCORE_WINDOW, method=ZSCORE_METHOD, eps=EPS):
    spread = pd.Series(spread).astype(float)

    if method == "standard":
        rolling_mean = spread.rolling(window).mean()
        rolling_std = spread.rolling(window).std()
        z = (spread - rolling_mean) / (rolling_std + eps)
    elif method == "mad":
        rolling_med = spread.rolling(window).median()
        rolling_mad = (spread - rolling_med).abs().rolling(window).median()
        z = 0.6745 * (spread - rolling_med) / (rolling_mad + eps)
    elif method == "ewm":
        mu = spread.ewm(span=window, adjust=False).mean()
        sigma = spread.ewm(span=window, adjust=False).std()
        z = (spread - mu) / (sigma + eps)
    else:
        raise ValueError(f"Unhandled z-score method: {method}")

    return z.replace([np.inf, -np.inf], np.nan)


def method_suffix(method=ZSCORE_METHOD):
    return method.lower()
