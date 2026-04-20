# Adaptive Threshold Selection for Statistical Arbitrage via Deep Q-Network

**Course:** ELEN 4904 — Statistical Learning, Columbia University  
**Based on:** Shen & Kurshan (ICAIF 2020) + Kim et al. / ORCA (ICAIF 2025)

---

## What This Does

Traditional pairs trading uses a **fixed Z-score threshold** (e.g., ±1σ) to decide when to enter trades. This threshold stays the same regardless of market conditions. When mean-reversion is strong, it works. When it breaks down (COVID, rate hikes), you walk into structural traps — the spread keeps diverging instead of reverting.

This project replaces the fixed threshold with a **DQN agent that adapts the threshold weekly** based on market state. The agent selects from 6 discrete thresholds [0.5, 0.75, 1.0, 1.25, 1.5, 2.0] each week, learning to be cautious in stressed markets and aggressive in calm ones.

The approach follows Shen & Kurshan (ICAIF 2020), who used the same DQN framework for adaptive alert thresholds in fraud detection. We transfer their method to statistical arbitrage.

---

## Results (154 pairs, test 2020-2023)

| Metric | Static 1.0σ | Adaptive DQN | |
|---|---|---|---|
| **Trap Rate** | **57.3%** | **54.5%** | **p < 10⁻⁶** |
| Mean Sharpe | -1.137 | -1.129 | p = 0.46 (ns) |
| DQN wins (by pair) | — | 84/154 (55%) | |
| Mean trades/pair | ~22 | ~25 | |

### Baseline comparison (simple rules vs DQN)

We tested whether simple heuristic rules using the same information (theta, VIX) could replicate the DQN's performance. **No simple rule reduces trap rate — the DQN is the only strategy that improves over the static baseline.**

| Strategy | Sharpe | Trap Rate | vs Static 1.0σ |
|---|---|---|---|
| Static 1.0σ | -1.137 | 57.3% | baseline |
| Static 1.25σ | -1.112 | 60.4% | +3.1pp worse |
| Theta rule (θ<5→0.5σ, else 1.25σ) | -1.175 | 59.5% | +2.2pp worse |
| VIX rule (VIX>25→1.25σ, else 1.0σ) | -1.188 | 57.9% | +0.6pp worse |
| Combined (theta + VIX) | -1.157 | 59.9% | +2.6pp worse |
| **DQN** | **-1.129** | **54.5%** | **-2.8pp better** |

- DQN beats the theta rule by **5.0pp** on trap rate (p ≈ 0), winning on 75% of pairs
- Tighter static thresholds (1.25σ) *increase* trap rate — fewer trades, but the remaining ones are more likely to be extreme divergences that don't revert
- The DQN's nonlinear policy over all 7 state features threads a needle that no simple rule can replicate

Run `python baseline_comparison.py` to reproduce (requires trained model from step 2).

### What the DQN learned

The agent uses only **2 of 6 available thresholds**: 0.50σ (21% of weeks, aggressive) and 1.25σ (79%, cautious).

- **Primary driver: OU theta** (r = 0.50 with threshold). When mean-reversion is weak (low θ), the agent goes aggressive. When reversion is strong (high θ), it's selective.
- **VIX has minimal influence** (r = 0.06). The agent responds to spread dynamics, not macro fear.
- Aggressive weeks: mean θ = 1.24, mean VIX = 22.0
- Cautious weeks: mean θ = 10.27, mean VIX = 23.1

This is counterintuitive — going aggressive when reversion is weak seems risky. But the baseline comparison proves the DQN's *timing* of when to be aggressive matters. The simple theta rule that mimics the same logic (aggressive when θ < 5) actually increases trap rates. The DQN uses the full 7-dim state to pick the right moments.

### What this means

- **Trap rate reduction is highly statistically significant** (p < 10⁻⁶) across 154 cointegrated pairs. The agent learned to avoid some structural traps.
- **No simple heuristic replicates this.** Tested 4 rule-based strategies — all fail to reduce traps. The DQN adds genuine value beyond basic regime filtering.
- **Sharpe improvement is directionally positive but not significant.** Both strategies lose money in 2020-2023 (hostile period for mean-reversion). The DQN loses slightly less.
- **Effect size is modest** — 2.8pp trap reduction. Meaningful at scale but not dramatic.

### Strengths
- Statistically significant trap reduction across 154 pairs, 7 sectors (p < 10⁻⁶)
- **DQN outperforms all simple baselines** — theta rules, VIX rules, and combined heuristics all fail where the DQN succeeds
- Cross-domain transfer from fraud detection (ICAIF 2020) to stat arb validated — same architecture, same framing, different domain
- Simple, reproducible architecture — 3-layer MLP with ~500 parameters, trains in 20 min on CPU
- Strict temporal separation: train 2010-2018, validate 2019, test 2020-2023 (test touched once)
- The ICAIF framing is principled: threshold selection IS a sequential decision problem where today's choice affects tomorrow's capacity

### Limitations
- **Sharpe improvement not significant** (p=0.46). The agent avoids some bad trades but doesn't find better entry points. Both strategies lose money in 2020-2023.
- **Small effect size.** 2.8pp trap reduction (57.3% → 54.5%). Economically meaningful only at scale.
- **Richer state features didn't help.** We tested adding OU sigma, half-life, Kelly fraction, VIX change (12-dim state). Results were worse than the 7-dim state. The threshold decision is simple enough that extra features add noise.
- **Test period is hostile.** 2020-2023 includes COVID and fastest rate hike cycle in decades. Mean-reversion strategies broadly underperformed. The DQN correctly learned caution, but "don't trade" isn't an exciting result.
- **yfinance as proxy** for CRSP/Compustat. Limited to ~46 liquid NYSE tickers vs ORCA's full universe.

### Open questions
1. Would v0's action space (direction + Kelly sizing) work on 154 pairs?
2. Can contrastive/representation learning produce better regime features than hand-crafted ones?
3. Does this generalize to other asset classes (CDS-bond basis, futures)?

---

## How to Replicate

### Setup

```bash
conda activate elen4904
pip install torch gymnasium pyarrow  # if not already installed
```

### Run

```bash
cd v1/

# Step 1: Build the 154-pair universe (~5 min)
# Downloads prices from yfinance, macro from FRED
# Computes spreads, OU parameters, features
# Filters pairs via ADF cointegration test
python build_expanded_universe.py

# Step 2: Train adaptive threshold DQN + backtest (~20-25 min)
# Trains on 2010-2018, validates on 2019 (early stopping)
# Tests on 2020-2023 and prints all results
python train_adaptive_threshold.py
```

Results are printed at the end of `train_adaptive_threshold.py` — Sharpe, trap rate, p-values, per-pair comparison.

### Data Sources
- **yfinance:** stock prices for 46 tickers + VIX (2010-2023)
- **FRED:** 10Y Treasury yield (DGS10), high-yield credit spread (BAMLC0A0CM)
- All data cached to `datasets/` after first download

---

## Architecture

### ICAIF Paper Mapping (Fraud → Pairs Trading)

| Fraud (Shen & Kurshan) | Pairs Trading (ours) |
|---|---|
| Fraud score threshold | Z-score entry threshold |
| Hourly threshold update | Weekly threshold update |
| Alert processing capacity | Trade capacity (10/month) |
| Fraud savings (S) | Cumulative PnL from winning trades |
| Fraud losses (L) | Cumulative losses from traps |
| Hour of day (H) | Week of month (W) |

### State (7 features)

| Feature | Description | Source |
|---|---|---|
| W | Week of month (normalized) | ICAIF paper |
| S | Cumulative wins this month (normalized) | ICAIF paper |
| L | Cumulative losses this month (normalized) | ICAIF paper |
| CC | Trades taken / max capacity | ICAIF paper |
| T | Current threshold index (normalized) | ICAIF paper |
| VIX | Current VIX level (normalized) | Domain extension |
| θ | Mean OU theta across pairs (normalized) | Domain extension |

### Action Space (6 discrete)

| Action | Threshold |
|---|---|
| 0 | 0.50σ (very aggressive) |
| 1 | 0.75σ |
| 2 | 1.00σ (ORCA's static default) |
| 3 | 1.25σ |
| 4 | 1.50σ |
| 5 | 2.00σ (very cautious) |

### Network

```
Input (7) → Linear(20) → ReLU → Linear(10) → ReLU → Linear(6) → Q-values
```

- ~500 parameters (deliberately tiny)
- Experience replay: 160K buffer, batch size 1024
- Epsilon-greedy: 0.5 → 0.1
- Gamma: 0.9, Adam lr=0.0001, MSE loss
- Follows ICAIF paper Section 5.3 exactly

### Reward

```
reward = (week_wins - week_losses) / (capital * 0.01) * (week + 1)
```

- Time-weighted (later weeks count more, prevents front-loading)
- Capacity penalty if trades exceed 10/month
- Matches ICAIF paper: (S - L) * H

---

## Files

```
v1/
├── README.md                        # This file
├── team_status.md                   # Strengths, limitations, next steps for team
├── literature_motivation.md         # Literature survey motivating the gap
├── train_adaptive_threshold.py      # DQN training + backtest (main script)
├── baseline_comparison.py           # Simple rules vs DQN comparison
├── build_expanded_universe.py       # Data pipeline: 154 pairs from 46 tickers
├── adaptive_threshold_dqn.pt        # Trained model checkpoint
├── threshold_vs_vix.png             # DQN threshold choices over time vs VIX
├── vix_by_threshold.png             # VIX distribution by threshold choice
├── theta_by_threshold.png           # Theta distribution by threshold choice
├── datasets/                        # Cached data
│   ├── pair_prices.parquet          # 46 ticker prices
│   ├── macro.parquet                # VIX, 10Y yield, HY spread
│   ├── spreads.parquet              # 154 pair spreads
│   ├── artifacts.pkl                # Scaler, PCA, pair list
│   ├── features_*.parquet           # Per-pair feature matrices
│   └── ou_params_*.parquet          # Per-pair OU parameters
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| 154 pairs (not 5-19) | Statistical power — p < 10⁻⁶ vs p = 0.06 with 5 pairs |
| Weekly threshold (not daily) | Regimes don't change daily; matches ICAIF's hourly-for-fraud timescale |
| Direction from z-score only | Prevents directional bias — agent only controls aggressiveness |
| ICAIF architecture exactly | Published, validated approach; reproducible |
| 7-dim state | Lean state outperforms richer alternatives |
| Train 2010-2018, val 2019, test 2020-2023 | Strict temporal separation, test touched once |

---

## Next Steps

- [x] ~~Test simple baselines (VIX-rule, theta-rule) to isolate RL's contribution~~ — Done. DQN beats all simple rules.
- [x] ~~Feature importance analysis (what drives the agent's threshold choices?)~~ — Done. Agent keys off theta, uses 2 of 6 thresholds.
- [ ] Try v0-style action space (direction + Kelly sizing) on 154 pairs
- [ ] Contrastive/representation learning for regime features
- [ ] Cross-asset extension (CDS-bond basis, futures)
