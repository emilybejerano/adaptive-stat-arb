# v4: Reward Function Ablation

## Purpose

`v4` diagnoses and modifies the DQN reward function after `v2` and `v3` failed to improve performance.

This version keeps:
- the `v1` standard rolling z-score
- the same DQN architecture
- the same evaluation logic

and changes only the reward shaping during training.

## Original Reward

The original reward comes from `train_adaptive_threshold.py` and is factored into `reward_utils.py`:

```python
reward = ((week_wins - week_losses) / (capital * 0.01)) * (week + 1) - capacity_penalty
```

What it optimizes:
- weekly net PnL asymmetry
- later weeks more heavily than earlier weeks
- a small penalty only if monthly trade capacity is exceeded

What it does not optimize directly:
- Sharpe
- drawdown
- trap rate
- switching stability
- volatility control

## Reward Modes

Configured in `reward_utils.py`:

- `original`
- `switch_penalty`
- `trap_penalty`
- `vol_penalty`
- `sharpe_proxy`

Default conservative parameters:

- `lambda_switch = 0.01`
- `lambda_trap = 0.05`
- `lambda_vol = 0.05`
- `rolling_window = 20`

## Main Result

| Reward mode | Wins vs static | Mean Sharpe | Mean trap % |
|---|---:|---:|---:|
| `original` | 84/154 | -1.129 | 54.5% |
| `switch_penalty` | 77/154 | -1.147 | 54.0% |
| `trap_penalty` | 84/154 | -1.129 | 54.5% |
| `vol_penalty` | 82/154 | -1.149 | 53.9% |
| `sharpe_proxy` | 84/154 | -1.156 | 58.1% |

Additional p-values vs static 1.0sigma:

| Reward mode | Sharpe p | Trap p |
|---|---:|---:|
| `original` | 0.4569 | 0.0000004 |
| `switch_penalty` | 0.7848 | 0.0000054 |
| `trap_penalty` | 0.4501 | 0.0000004 |
| `vol_penalty` | 0.6144 | 0.00000005 |
| `sharpe_proxy` | 0.5547 | 0.3380 |

## Interpretation

- `original` remained the best overall mode
- `vol_penalty` produced the best trap rate, but not enough to offset weaker Sharpe and fewer wins
- `trap_penalty` had almost no effect, which suggests the current trap indicator is too weak as a learning signal
- `sharpe_proxy` did not improve Sharpe in practice and made trap behavior worse

## Main Files

- `reward_utils.py`: reward-mode definitions and helper logic
- `train_adaptive_threshold.py`: training with reward diagnostics
- `baseline_comparison.py`: reused evaluation script
- `checkpoints/`: mode-specific trained checkpoints

## How To Run

Original:

```bash
cd v4
REWARD_MODE=original python train_adaptive_threshold.py
REWARD_MODE=original python baseline_comparison.py
```

Switch penalty:

```bash
REWARD_MODE=switch_penalty python train_adaptive_threshold.py
REWARD_MODE=switch_penalty python baseline_comparison.py
```

Trap penalty:

```bash
REWARD_MODE=trap_penalty python train_adaptive_threshold.py
REWARD_MODE=trap_penalty python baseline_comparison.py
```

Volatility penalty:

```bash
REWARD_MODE=vol_penalty python train_adaptive_threshold.py
REWARD_MODE=vol_penalty python baseline_comparison.py
```

Sharpe proxy:

```bash
REWARD_MODE=sharpe_proxy python train_adaptive_threshold.py
REWARD_MODE=sharpe_proxy python baseline_comparison.py
```

## Takeaway

`v4` shows that reward shaping alone did not beat the original `v1` setup. The next useful step is probably improving the state signal or the trap definition rather than adding more reward penalties.
