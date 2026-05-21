# Adaptive Threshold Selection for Statistical Arbitrage

Final project for **Statistical Learning with Applications in Quantitative Trading** (ELEN 4904) at Columbia University.

## About

Statistical arbitrage strategies trade when a stock deviates from its peer group beyond some threshold. Most approaches, including the [ORCA framework](https://github.com/x7jeon8gi/ORCA) (Kim et al., ICAIF 2025), use a fixed threshold regardless of market conditions. This project explores whether adaptive threshold selection can improve performance.

We investigate several approaches -- static optimization, deep reinforcement learning (DQN), rule-based methods, and threshold blending -- and evaluate them with walk-forward validation across different market regimes.

## Structure

The project went through three iterations:

- **v0/** -- Pair-level DQN on daily data (19 pairs, yfinance)
- **v1/** -- Scaled to 154 cointegrated pairs with baseline comparisons
- **v2/** -- Cluster-level analysis on CRSP/Compustat data via WRDS (final version)

Each directory has its own README with details and reproduction steps.

## Setup

```bash
conda env create -f environment.yml
conda activate elen4904
pip install torch stable-baselines3 gymnasium wrds pyyaml x-transformers transformers
```

v0 and v1 use yfinance. v2 requires a [WRDS](https://wrds-www.wharton.upenn.edu/) account.

## References

- Kim, N., Na, Y., & Song, J. W. (2025). Deep Mean-Reversion: A Physics-Informed Contrastive Approach to Pairs Trading. *ICAIF 2025*.
- Shen, D. & Kurshan, E. (2020). Deep Q-Network-based Adaptive Alert Threshold Selection Policy for Payment Fraud. *ICAIF 2020*.
