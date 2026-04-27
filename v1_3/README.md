# v3: Robust Z-Score Ablation

## Purpose

`v3` tests whether changing spread normalization helps more than changing the RL model.

Instead of altering the DQN architecture, this version keeps the `v1` setup and swaps the z-score calculation used to generate trading signals.

## Z-Score Modes

Configured in `zscore_utils.py`:

- `standard`
- `mad`
- `ewm`

Default:

```python
ZSCORE_METHOD = "mad"
```

## What We Changed

- copied the working `v1` pipeline into `v3`
- added `zscore_utils.py`
- made the z-score method configurable
- saved separate checkpoints and result files for each normalization method

## Main Result

Comparison against the `v1` baseline:

| Method | Wins vs static | Mean Sharpe | Mean trap % |
|---|---:|---:|---:|
| `standard` | 84/154 | -1.129 | 54.5% |
| `mad` | 78/154 | -1.712 | 69.2% |
| `ewm` | 71/154 | -1.776 | 64.2% |

## Interpretation

- `standard` exactly matched the working `v1` behavior
- `mad` was clearly worse
- `ewm` was also worse overall

So neither robust alternative improved the project:
- MAD did not improve Sharpe or trap rate
- EWMA did not improve Sharpe or trap rate

## Main Files

- `zscore_utils.py`: configurable z-score methods
- `train_adaptive_threshold.py`: z-score-aware training
- `baseline_comparison.py`: reused evaluation path

## How To Run

MAD:

```bash
cd v3
python train_adaptive_threshold.py
python baseline_comparison.py
```

Standard:

```bash
ZSCORE_METHOD=standard python train_adaptive_threshold.py
ZSCORE_METHOD=standard python baseline_comparison.py
```

EWMA:

```bash
ZSCORE_METHOD=ewm python train_adaptive_threshold.py
ZSCORE_METHOD=ewm python baseline_comparison.py
```

## Takeaway

`v3` suggests the main bottleneck is not the z-score formula. The original standard rolling z-score remained the strongest option.
