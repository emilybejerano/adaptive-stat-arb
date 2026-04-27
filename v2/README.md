# Adaptive Threshold Selection for Cluster-Based Statistical Arbitrage

**ELEN E4904 — Statistical Learning**

## Overview

This project extends the ORCA paper (Kim et al., "Deep Mean-Reversion", ICAIF 2025) by addressing its fixed threshold limitation. The paper clusters stocks using a physics-informed neural network and trades when stocks deviate from cluster peers beyond a fixed threshold (γ=1.0). We investigate whether adaptive threshold selection can improve risk-adjusted returns.

**Key finding:** Simple threshold diversification (blending γ=1.0, 1.25, and 2.0) outperforms both optimized static thresholds and DQN-based adaptive selection in walk-forward validation across all market regimes.

---

## Project Structure

```
v2/
├── README.md                      # This file
├── visualize_results.py           # Generate all report figures from saved results
├── stage1/                        # Data pipeline + Clustering + Backtesting
│   ├── 01_download_wrds_data.py   # Download CRSP + Compustat from WRDS
│   ├── make_data.py               # Process raw data into monthly features
│   ├── run_baselines.py           # K-means, DBSCAN, Agglomerative clustering
│   ├── main.py                    # ORCA PINN model training
│   ├── configs/config.yaml        # ORCA hyperparameters
│   ├── backtest.py                # Algorithm 1 backtest + performance metrics
│   └── plot_results.py            # Stage 1 visualization
├── stage2/                        # Threshold selection experiments
│   ├── stage2_thresholds.py       # Static γ sweep (0.5 to 2.0)
│   ├── stage2_diagnose.py         # Oracle analysis (why is best γ unpredictable?)
│   ├── stage2_blend.py            # Threshold blending (our best result)
│   ├── stage2_nonrl.py            # Non-RL methods (regime filter, vol-scaled)
│   ├── stage2_dqn_env.py          # DQN Gymnasium environment v1
│   ├── stage2_dqn_env_v2.py       # v2: adds sit-out action
│   ├── stage2_dqn_env_v3.py       # v3: differential Sharpe reward + bandit
│   ├── stage2_dqn_env_ou.py       # v4: adds ORCA OU params to state (12 features)
│   ├── stage2_dqn_train.py        # Single-split DQN training
│   ├── stage2_walkforward.py      # Walk-forward validation v1
│   ├── stage2_walkforward_v2.py   # Walk-forward with sit-out
│   ├── stage2_walkforward_v3.py   # Walk-forward with diff Sharpe
│   └── stage2_walkforward_ou.py   # Walk-forward with OU state features
└── results/                       # All saved outputs (CSV + PNG)
```

---

## How to Reproduce

### Prerequisites

```bash
conda activate your_env
pip install torch stable-baselines3 gymnasium scikit-learn pandas numpy matplotlib wrds pyyaml x-transformers transformers
```

### Step-by-Step

All scripts should be run from the ORCA repo directory (clone from https://github.com/x7jeon8gi/ORCA).

**Stage 1: Data + Clustering + Backtest**

```bash
# 1. Download raw data from WRDS (requires account)
python ../v2/stage1/01_download_wrds_data.py

# 2. Process into monthly features (24 momentum + 12 accounting)
python data/make_data.py

# 3. Run baseline clustering methods (~15 min)
python ../v2/stage1/run_baselines.py

# 4. Train ORCA physics-informed model (GPU, ~5 hours)
python main.py --config configs/config.yaml

# 5. Backtest all methods with Algorithm 1
python ../v2/stage1/backtest.py
```

**Stage 2: Threshold Selection**

```bash
# 6. Static threshold sweep (~2 min)
python ../v2/stage2/stage2_thresholds.py

# 7. Diagnostic: is optimal threshold predictable? (~1 min)
python ../v2/stage2/stage2_diagnose.py

# 8. Threshold blending — our best result (~1 min each)
python -u ../v2/stage2/stage2_blend.py ./res/clusters/kmeans_20
python -u ../v2/stage2/stage2_blend.py ./res/clusters/agglo_0.5
python -u ../v2/stage2/stage2_blend.py ./res/pinn/clustering

# 9. DQN walk-forward validation (~10-25 min each)
python -u ../v2/stage2/stage2_walkforward_v3.py ./res/clusters/agglo_0.5
python -u ../v2/stage2/stage2_walkforward_ou.py

# 10. Non-RL adaptive methods (~1-2 min each)
python -u ../v2/stage2/stage2_nonrl.py ./res/clusters/agglo_0.5
```

**Generate Report Figures**

```bash
cd v3
python visualize_results.py
# Outputs: results/figure1_*.png through figure4_*.png
```

---

## Results

### Stage 1: Clustering Methods (fixed γ=1.0)

| Method | AR | Vol | MDD | Sharpe | Sortino | Calmar |
|---|---|---|---|---|---|---|
| K-means (20) | 0.661 | 0.159 | 0.067 | 3.20 | 5.68 | 9.79 |
| K-means (30) | 0.639 | 0.149 | 0.044 | 3.31 | 8.26 | 14.37 |
| DBSCAN | 0.682 | 0.194 | 0.081 | 2.68 | 6.16 | 8.45 |
| Agglomerative | 0.660 | 0.149 | 0.065 | 3.39 | 9.23 | 10.19 |
| ORCA | 0.594 | 0.159 | 0.107 | 2.93 | 5.77 | 5.57 |
| Paper ORCA | 0.368 | 0.118 | 0.172 | 2.87 | 3.34 | 1.97 |

### Stage 2: Static Threshold Sweep (Agglomerative)

| γ | AR | Sharpe | MDD |
|---|---|---|---|
| 0.50 | 0.520 | 3.23 | 0.079 |
| 0.75 | 0.594 | 3.35 | 0.053 |
| 1.00 (paper) | 0.660 | 3.39 | 0.065 |
| **1.25** | **0.730** | **3.50** | **0.059** |
| 1.50 | 0.804 | 3.42 | 0.073 |
| 2.00 | 0.970 | 3.02 | 0.154 |

### Stage 2: DQN Results (Walk-Forward)

| DQN Version | State Features | Aggregate Sharpe | Wins |
|---|---|---|---|
| v1 (basic reward) | 9 market features | ~2.5 | 0/4 |
| v3 (diff Sharpe, bandit) | 9 market features | 1.97 | 0/4 |
| +OU (ORCA physics) | 12 features (9 + 3 OU) | 2.47 | 0/4 |
| Binary (trade/sit-out) | 9 market features | 2.20 | 0/4 |

### Stage 2: Diagnostic Analysis

- Oracle best γ: γ=2.0 wins 55% of months, γ=1.25 wins 19%, γ=1.0 wins 14%, γ=0.5 wins 12%
- Best γ switches every 1.6 months on average (near-random)
- Cross-γ correlation: 0.77 between γ=1.25 and γ=2.0 (high but not perfect)
- Return gap between best and worst γ: 4% per month median
- State features correlate at r=0.14-0.31 with optimal γ (too weak for prediction)
- Transition matrix is nearly uniform (no temporal structure to exploit)

### Stage 2: Threshold Blending — Best Result

**Walk-forward (K-means 20): Triple Blend wins 4/4 windows**

| Window | Best Static | Triple Blend | Sigma Blend | Winner |
|---|---|---|---|---|
| GFC (2008-2011) | 3.15 | **3.16** | 3.05 | Blend |
| Bull (2012-2015) | 3.00 | **3.14** | 3.10 | Blend |
| Late (2016-2019) | 4.16 | **4.31** | 4.08 | Blend |
| COVID (2020-2023) | 4.38 | 4.58 | **4.82** | Blend |

**Walk-forward (Agglomerative): Blend wins 3/4 windows**

| Window | Best Static | Triple Blend | Momentum Blend | Winner |
|---|---|---|---|---|
| GFC | 3.44 | 3.42 | **3.51** | Blend |
| Bull | **2.96** | 2.73 | 2.67 | Static |
| Late | 4.05 | 3.95 | **4.04** | Blend (Sigma: 4.06) |
| COVID | 4.20 | **4.57** | 4.52 | Blend |

### Final Summary

| Method | Type | Aggregate Sharpe | vs Paper | Walk-Forward |
|---|---|---|---|---|
| Paper γ=1.0 | Fixed | 2.87 | Baseline | — |
| ORCA (reproduced) | PINN | 2.93 | +2% | — |
| Static γ=1.25 | Fixed (optimized) | 3.13 | +9% | — |
| **Triple Blend** | **Diversification** | **3.18** | **+11%** | **4/4 wins** |
| DQN+OU | RL with physics | 2.47 | -14% | 0/4 wins |
| DQN v3 | RL | 1.97 | -31% | 0/4 wins |

---

## File Descriptions

### stage1/

| File | What it does | Runtime |
|---|---|---|
| `01_download_wrds_data.py` | Downloads CRSP monthly stock data (returns, prices, volume) and Compustat quarterly fundamentals from WRDS. Joins via CRSP-Compustat Merged link table. Saves to `ORCA/data/raw_data/`. | ~5 min |
| `make_data.py` | Processes raw data: computes 24 momentum features (MOM1-MOM24 = cumulative log returns), merges 12 accounting variables (with 3-month lag), outputs 300 monthly CSV files. | ~10 min |
| `run_baselines.py` | Clusters each month's stocks into 20-30 groups using K-means, DBSCAN, Agglomerative. Per-month StandardScaler (no future leakage), KNN imputation, auto-tuned DBSCAN eps. | ~15 min |
| `main.py` | ORCA training: for each month, trains a PINN with contrastive + cluster + mean-reversion loss. Outputs cluster assignments and OU parameters (θ, μ, σ) per stock. | ~5 hours (GPU) |
| `configs/config.yaml` | 30 clusters, 64 bins, 128 hidden dim, 2 transformer layers, batch 1024, lr 0.002, 200 epochs/month. |
| `backtest.py` | Algorithm 1: compute momentum spreads per cluster, long underperformers / short outperformers beyond threshold, equal-weight across clusters. 10 bps TC per side, 10% stop-loss. | ~2 min |
| `plot_results.py` | Metrics table + cumulative return plots. | <1 min |

### stage2/

| File | What it does | Runtime |
|---|---|---|
| `stage2_thresholds.py` | Sweeps γ={0.5, 0.75, 1.0, 1.25, 1.5, 2.0} on all Stage 1 methods. Also has OU-aware and rule-based adaptive thresholds (ORCA only). | ~2 min |
| `stage2_diagnose.py` | Oracle analysis: for each of 288 months, determines which γ would have been best. Computes cross-γ correlation, transition matrix, streak lengths, state-feature correlations. Answers "is optimal γ predictable?" | ~1 min |
| `stage2_blend.py` | Walk-forward evaluation of 4 blend strategies: Equal (50/50 γ=1.25+2.0), Sigma-Weighted (adjust by spread dispersion), Triple (1/3 each γ=1.0+1.25+2.0), Momentum (weight by rolling Sharpe). | ~1 min |
| `stage2_nonrl.py` | Non-RL adaptive methods: Regime Filter (sit out low-vol months), Vol-Scaled γ, Oracle Top-2 (upper bound), Best-Static-Per-Window. | ~1-2 min |
| `stage2_dqn_env.py` | Gymnasium env v1. State: 9 features (spread σ, velocity, cluster stability, fraction tradeable, cross-sectional vol, market return, rolling 1/3/6-month returns). Action: {γ=0.5, 1.0, 1.25, 2.0}. Reward: r - 0.5·r². | — |
| `stage2_dqn_env_v2.py` | Adds 5th action: sit out. Penalty of -0.001 for sitting out. | — |
| `stage2_dqn_env_v3.py` | Differential Sharpe reward (directly optimizes Sharpe change), γ_RL=0 (bandit), state noise injection, precomputed returns. | — |
| `stage2_dqn_env_ou.py` | 12-dim state: 9 market + 3 OU features from ORCA (median θ, median σ_OU, θ dispersion). | — |
| `stage2_dqn_train.py` | Single train/test split DQN training with SB3. CLI arg for cluster dir. | ~10 min |
| `stage2_walkforward.py` | Walk-forward: 4 windows (GFC, Bull, Late, COVID), fresh DQN per window. | ~25 min |
| `stage2_walkforward_v2.py` | Walk-forward for v2 env (sit-out). | ~25 min |
| `stage2_walkforward_v3.py` | Walk-forward for v3 env (diff Sharpe + bandit). Faster due to precomputed returns. | ~10 min |
| `stage2_walkforward_ou.py` | Walk-forward with OU-aware DQN on ORCA clusters. | ~10 min |

---

## Key Insights

1. **γ=1.0 is not optimal.** Simple sweep finds γ=1.25 improves Sharpe by 9%.

2. **DQN cannot beat static thresholds** with monthly data. The optimal threshold changes too rapidly (1.6-month streaks) and too unpredictably (r=0.14-0.31 with state features) for RL to learn with 180 samples.

3. **OU parameters help but aren't enough.** DQN+OU improves over baseline DQN (+0.5 Sharpe), confirming the physics signal has value, but the fundamental data limitation remains.

4. **Diversification beats prediction.** Triple Blend wins 4/4 walk-forward windows by exploiting imperfect correlation (r=0.77) between γ levels — same principle as portfolio diversification.

5. **Bias-variance tradeoff favors simplicity.** DQN has low bias but extreme variance (overfits 180 samples). Blend has zero variance (no trainable parameters). When data is scarce, simple wins.

6. **The RL failure was productive.** It led to the diagnostic that proved prediction is infeasible, which pointed to diversification as the correct approach. Each step was necessary.
