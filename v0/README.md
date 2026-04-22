# Adaptive Execution in Statistical Arbitrage via Deep Reinforcement Learning


---

## Overview

Statistical arbitrage strategies rely on mean-reversion: when a spread between related assets deviates from equilibrium, positions are taken expecting it to revert. In practice, this is implemented with **static thresholds** — trade when the z-score exceeds ±1σ.

The problem: **~60% of these signals are structural traps** where the spread continues diverging instead of reverting, particularly during regime shifts (COVID crash, rate hikes).

This project builds a **Deep Q-Network (DQN)** execution layer that learns when market conditions support mean-reversion and when they don't, using Ornstein-Uhlenbeck dynamics and macroeconomic indicators as state features.∑

---

## Results

**Test period: 2020-2023 (out-of-sample, never seen during training)**

| Strategy | Mean Sharpe | Trap Rate | Wins |
|---|---|---|---|
| Static 1.0σ | -1.31 | 60.0% | baseline |
| Adaptive VIX | -1.05 | 64.3% | — |
| **DQN v2** | **+0.72** | **49.6%** | **19/19 pairs** |

- Sharpe improvement: p < 0.001 (Wilcoxon signed-rank)
- Trap rate reduction: p = 0.002

**Important caveat:** Analysis of the DQN's action distribution reveals a directional (long) bias that contributes significantly to performance during this bull market period. See Analysis section below.

---

## Method

### Data
- **Prices:** 46 NYSE tickers, daily close, 2010-2023 (yfinance)
- **Macro:** VIX, 10Y Treasury yield, HY credit spread (FRED)
- **Pairs:** 24 candidates → 19 pass ADF cointegration test
- **Spreads:** Rolling 252-day OLS hedge ratio

### State Representation (14 dimensions)
- **OU dynamics (6):** θ (reversion speed), μ, σ, z-score, half-life, residual variance
- **Market context (6):** VIX, VIX 5d change, 10Y yield, HY spread, pair correlation, spread volatility
- **Kelly (2):** fraction, edge
- Compressed via PCA (14 → 11 components, 95% variance) + 3 position features

### DQN Architecture
```
Input (14) → Linear(128) → ReLU → Dropout(0.2)
           → Linear(64)  → ReLU → Dropout(0.2)
           → Linear(32)  → ReLU
           → Linear(3)   → Q-values [flat, long, short]
```
- 12,487 parameters
- Double DQN with soft target updates
- AdamW optimizer with L2 weight decay
- Fixed 10% capital position sizing (consistent across all strategies)

### Training
- **Walk-forward CV:** Fold 1 (train 2010-2014, val 2015-2016), Fold 2 (train 2010-2016, val 2017-2018)
- **Final model:** Train 2010-2018, validate 2019, test 2020-2023
- **Early stopping** on validation Sharpe
- **Reward:** log(return) - 0.5 × rolling_variance - penalty_if_no_edge

### Baselines
- **Static 1.0σ / 1.5σ:** ORCA's approach — fixed entry threshold
- **Adaptive VIX:** VIX < 20 → 1.0σ, VIX 20-30 → 1.5σ, VIX > 30 → no trade

---

## Analysis and Limitations

### Directional Bias
The DQN selects long positions ~80% of the time across all VIX regimes. During the 2020-2023 bull market, this bias is profitable. To test whether the DQN learned genuine regime awareness vs. directional bias, we ran **ablation experiments**:

1. **Direction-constrained DQN** (z-score dictates direction, agent only chooses take/skip): Model skips 95%+ of signals — effectively learns "don't trade"
2. **PPO (Stable Baselines3)** with same constraint: Similar result — skips 98%+ of signals
3. **Supervised signal gate:** AUC 0.75 but doesn't translate to profitable filtering

**Conclusion:** When directional freedom is removed, no model architecture improves on static execution. The features (OU parameters, macro indicators) capture regime-level information but are insufficient for individual trade-level prediction.

### Other Limitations
- **yfinance data:** No bid-ask, volume, or execution quality data
- **25 pairs:** Small universe vs. ORCA's 3,000+ stocks
- **10 bps transaction costs:** May be optimistic for less liquid pairs
- **Single test period:** 2020-2023 is one market regime; longer evaluation needed

---

## Reproducing Results

### Setup
```bash
conda env create -f environment.yml
conda activate elen4904
```

### Run Pipeline
```bash
cd v0/

# 1. Data + OU estimation + features (downloads from yfinance/FRED)
jupyter nbconvert --execute 01_environment_and_ou.ipynb

# 2. Train DQN (uses GPU if available)
python train_dqn_v2.py

# 3. Backtest all strategies
python run_backtest_v2.py

# 4. Or run notebooks interactively
jupyter notebook
```

### File Structure
```
v0/
├── 00_understand_data.ipynb    # EDA and data exploration
├── 01_environment_and_ou.ipynb # Data pipeline, OU estimation, RL environment
├── 02_dqn_training.ipynb       # DQN training (reference; use train_dqn_v2.py)
├── 03_backtest_evaluation.ipynb# Backtest and results
├── train_dqn_v2.py             # Training script (v2: fixed sizing, 3 actions)
├── run_backtest_v2.py          # Backtest comparison script
├── dqn_pairs_agent_v2.pt       # Trained model weights
├── datasets/                   # Cached data (parquet files)
│   ├── pair_prices.parquet     # Raw price data
│   ├── spreads.parquet         # Computed spreads
│   ├── macro.parquet           # VIX, yields, credit spreads
│   ├── features_*.parquet      # Per-pair feature matrices
│   ├── ou_params_*.parquet     # Per-pair OU parameters
│   └── nb02_artifacts.pkl      # Scaler, PCA, pair lists
└── README.md
```

---

## Course Connections

| Concept | Application |
|---|---|
| Lecture 9, Slides 36-37: Kelly Criterion | Position sizing from OU transition density |
| Slide 39: Fractional Kelly | Edge penalty when kelly_fraction = 0 |
| Slides 41-42: Mean-Variance Optimization | Reward = log_return - λ × variance |
| Slide 44: Factor Models / PCA | 14 features → 11 principal components |
| Slides 20-24: Cointegration | ADF testing, spread construction |
| LTCM context | Drawdown penalty, regime awareness motivation |

---

## References

1. Elliott, R. J., Van Der Hoek, J., & Malcolm, W. P. (2005). Pairs trading. *Quantitative Finance*, 5(3), 271-276.
2. Kim, N., Na, Y., & Song, J. W. (2025). Deep Mean-Reversion: A Physics-Informed Contrastive Approach to Pairs Trading. *ICAIF 2025*, 405-412.
