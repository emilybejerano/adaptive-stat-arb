# Literature Motivation: The Static Threshold Problem in Statistical Arbitrage

## The Universal Pattern

Across decades of pairs trading and statistical arbitrage research, there is a consistent architectural pattern: sophisticated methods for **finding** tradable relationships, followed by naive fixed rules for **executing** on them. The pair selection side of the literature has evolved dramatically — from simple correlation to cointegration to machine learning to deep learning. The execution side has barely moved. Nearly every approach uses some variant of "enter when the spread exceeds X standard deviations from the mean."

This creates a fundamental mismatch: the pair selection methods implicitly acknowledge that markets are complex, non-linear, and regime-dependent, but the execution methods assume the trading environment is stationary.

---

## The Evolution of Pair Selection (and the Stagnation of Execution)

### Classical Statistical Methods (1980s–2000s)

**Gatev, Goetzmann & Rouwenhorst (2006)** — the foundational empirical study. They proposed the "distance method": find pairs whose normalized prices have the smallest sum of squared distances over a formation period, then trade when prices diverge by more than 2 historical standard deviations. The pair selection is data-driven. The execution is a fixed 2σ rule.

**Engle & Granger (1987), Johansen (1991)** — cointegration testing. Rather than correlation (which can be spurious for non-stationary series), cointegration identifies long-run equilibrium relationships. You compute the spread, run an ADF test, and if the spread is stationary, you trade it. With a fixed Z-score threshold.

**Vidyamurthy (2004)** — formalized the OU process model for pairs trading. The spread follows dS = θ(μ - S)dt + σdW. This gives you the theoretical framework to compute mean-reversion speed, half-life, and stationary distribution. But the trading rule is still: enter at ±1-2σ, exit at 0. The OU parameters are estimated but not used to adapt the entry decision.

The pattern is already visible: increasingly rigorous models of the spread dynamics, same fixed execution rule.

### Machine Learning Methods (2010s)

**Huck (2010, 2015)** — applied SVMs and random forests to predict whether a spread would converge after crossing a threshold. This is one of the earliest attempts to use ML on the execution side, but it's a binary classifier (trade/don't trade) applied on top of a fixed threshold, not a replacement for it.

**Sarmento & Horta (2020)** — used OPTICS clustering on a feature space of returns, volatility, and sector characteristics to find pairs. Significantly more sophisticated than correlation or cointegration for pair selection. Execution? Static Z-score bands.

**Krauss, Do & Huck (2017)** — applied deep learning (LSTMs, autoencoders) to predict relative returns in a large cross-section. The prediction model is learned, but the trading rule is still a fixed percentile threshold on the predicted spread.

**Han, He & Toh (2023)** — the paper ORCA directly builds on. They used unsupervised learning (K-means and variants) on momentum features to cluster assets, then applied mean-reversion trading within clusters. Again: learned clustering, static trading rule. ORCA's innovation was adding physics-informed regularization to the clustering. The execution stayed the same.

### Deep Learning and Physics-Informed Methods (2020s)

**Kim, Na & Song (2025) — ORCA** — the immediate predecessor to our work. ORCA's contribution is integrating contrastive learning with a PINN module that enforces Ornstein-Uhlenbeck dynamics during clustering. This ensures clusters have genuine mean-reversion properties, not just feature similarity. The PINN module estimates θ, μ, σ per cluster during training. But at execution time, all of this is discarded in favor of a fixed γ=1.0 threshold on momentum spreads (Algorithm 1, Section 3.3). Their own ablation study (Table 2) shows that removing the PINN increases maximum drawdown by 8.7%, confirming that the OU dynamics matter — but they only use them for clustering, not execution.

**Shen & Kurshan (2020) — ICAIF DQN** — showed that adaptive threshold selection via Deep Q-Network outperforms static thresholds in fraud detection alert systems. They formulated threshold selection as a sequential decision problem and used a 3-layer MLP to learn hourly threshold adjustments based on system state. Their approach reduced fraud losses by 6% over the best static threshold. This paper provides the methodological foundation for our approach — we transfer their framework from fraud alerting to pairs trading execution.

**Brim (2020)** — applied Double DQN to pairs trading, but with a fixed pair (no clustering), no OU modeling, no regime conditioning, and fixed position sizing. The RL agent learns entry/exit but is blind to mean-reversion dynamics and market regime.

---

## The Gap

Every method above falls into one of two categories:

1. **Sophisticated selection, static execution:** ORCA, Sarmento & Horta, Han et al., Gatev et al., cointegration methods. They invest heavily in finding the right pairs/clusters but trade them with a fixed rule.

2. **RL execution, but without regime or dynamics conditioning:** Brim, Kim et al. (2022), Xu & Luo (2023). They use RL to learn entry/exit but don't give the agent information about the spread's current mean-reversion state or the macro environment.

Nobody combines the ICAIF DQN's adaptive threshold framework with domain-specific state features (OU dynamics, macro regime) for pairs trading execution.

---

## Why ORCA + ICAIF DQN Together

We chose these two papers as our starting points because they are complementary:

- **ORCA** gives us the **domain**: pairs trading with OU-modeled spreads, where static thresholds are an acknowledged limitation
- **Shen & Kurshan** gives us the **method**: DQN for adaptive threshold selection, validated in a different domain (fraud detection)

Our contribution is the **transfer**: applying the ICAIF DQN framework to the execution layer of OU-based pairs trading, with state features adapted to the financial domain (VIX, OU theta).

---

## Generalization Beyond Pairs Trading

The static threshold problem appears anywhere you have a scoring model and a fixed cutoff:

- **ETF arbitrage:** enter when tracking error exceeds X basis points
- **Futures basis trades:** enter when basis spread exceeds X standard deviations
- **Volatility arbitrage:** enter when implied-realized vol spread exceeds X
- **Fixed income relative value:** enter when yield spread exceeds X
- **Credit (CDS-bond basis):** enter when basis exceeds X — relevant to our project proposal

The ICAIF DQN framework applies to all of these. The state features change (VIX might become swap spreads for fixed income), but the architecture and training procedure are identical.
