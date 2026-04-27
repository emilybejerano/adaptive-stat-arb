"""Diagnose why DQN can't beat static thresholds."""
import pandas as pd
import numpy as np
from collections import Counter
from stage2_dqn_env import ThresholdEnv, GAMMA_CHOICES

log_returns_df = pd.read_pickle('data/log_returns_by_month.pkl')

# Run all 4 gammas on full period
print('Computing returns for each gamma on agglo_0.5...')
all_rets = {}
months = None
for g in GAMMA_CHOICES:
    env = ThresholdEnv('./res/clusters/agglo_0.5', log_returns_df,
                       start_month='2000-01', end_month='2023-12')
    obs, _ = env.reset()
    rets, ms = [], []
    done = False
    while not done:
        obs, _, t, tr, info = env.step(GAMMA_CHOICES.index(g))
        rets.append(info['return'])
        ms.append(info['month'])
        done = t or tr
    all_rets[g] = np.array(rets)
    if months is None:
        months = ms

n = len(months)

# === 1: Oracle best gamma ===
print(f'\n=== 1. WHICH GAMMA IS BEST EACH MONTH? ({n} months) ===')
oracle_best = [max(GAMMA_CHOICES, key=lambda g: all_rets[g][i]) for i in range(n)]
oracle_counts = pd.Series(oracle_best).value_counts().sort_index()
for g, c in oracle_counts.items():
    pct = 100 * c / n
    print(f'  g={g}: best in {c}/{n} months ({pct:.0f}%)')

# === 2: Return gap ===
print(f'\n=== 2. DOES GAMMA CHOICE EVEN MATTER? ===')
gaps = []
for i in range(n):
    best_ret = max(all_rets[g][i] for g in GAMMA_CHOICES)
    worst_ret = min(all_rets[g][i] for g in GAMMA_CHOICES)
    gaps.append(best_ret - worst_ret)
gaps = np.array(gaps)
print(f'  Monthly return gap (best - worst gamma):')
print(f'    Mean:   {gaps.mean():.4f}')
print(f'    Median: {np.median(gaps):.4f}')
print(f'    P75:    {np.percentile(gaps, 75):.4f}')
print(f'    Max:    {gaps.max():.4f}')
small = np.sum(gaps < 0.005)
print(f'  Months where gap < 0.5%: {small}/{n} ({100*small/n:.0f}%)')
print(f'  ** If most gaps are tiny, picking the right gamma barely matters **')

# === 3: Correlation between gammas ===
print(f'\n=== 3. ARE GAMMA RETURNS CORRELATED? ===')
corr = np.corrcoef([all_rets[g] for g in GAMMA_CHOICES])
header = '         ' + '  '.join(f'g={g:<4}' for g in GAMMA_CHOICES)
print(f'  {header}')
for i, g in enumerate(GAMMA_CHOICES):
    row = '  '.join(f'{corr[i,j]:>6.3f}' for j in range(4))
    print(f'  g={g:<4}  {row}')
print(f'  ** High correlation = all gammas move together **')

# === 4: Transition matrix ===
print(f'\n=== 4. IS BEST GAMMA STICKY OR RANDOM? ===')
transitions = Counter()
for i in range(1, n):
    transitions[(oracle_best[i-1], oracle_best[i])] += 1

print(f'  If best gamma this month is X, probability of best next month:')
header2 = '         ' + '  '.join(f'g={g:<4}' for g in GAMMA_CHOICES)
print(f'  {header2}')
for g1 in GAMMA_CHOICES:
    row = [transitions.get((g1, g2), 0) for g2 in GAMMA_CHOICES]
    total = sum(row)
    if total > 0:
        pcts = '  '.join(f'{100*r/total:5.1f}%' for r in row)
        print(f'  g={g1:<4}  {pcts}')

# === 5: Streak length ===
print(f'\n=== 5. STREAK LENGTH ===')
streaks = []
current = 1
for i in range(1, n):
    if oracle_best[i] == oracle_best[i-1]:
        current += 1
    else:
        streaks.append(current)
        current = 1
streaks.append(current)
print(f'  Mean streak: {np.mean(streaks):.1f} months')
print(f'  Max streak:  {max(streaks)} months')
print(f'  Streak distribution: 1mo={sum(1 for s in streaks if s==1)}, '
      f'2-3mo={sum(1 for s in streaks if 2<=s<=3)}, '
      f'4+mo={sum(1 for s in streaks if s>=4)}')
print(f'  ** Short streaks = best gamma changes rapidly = unpredictable **')

# === 6: Per-gamma stats ===
print(f'\n=== 6. RETURN STATS BY GAMMA ===')
print(f'  gamma    mean      std     neg%   Sharpe')
for g in GAMMA_CHOICES:
    r = all_rets[g]
    mean_r = np.mean(r)
    std_r = np.std(r, ddof=1)
    neg_pct = 100 * np.sum(r < 0) / n
    sharpe = (mean_r / std_r) * np.sqrt(12) if std_r > 1e-8 else 0
    print(f'  g={g:<4} {mean_r:>8.4f} {std_r:>8.4f} {neg_pct:>5.0f}% {sharpe:>7.2f}')

# === 7: VERDICT ===
print(f'\n=== VERDICT ===')
median_gap = np.median(gaps)
mean_corr = np.mean([corr[i,j] for i in range(4) for j in range(4) if i != j])
mean_streak = np.mean(streaks)
print(f'  Median return gap:     {median_gap:.4f} ({"TINY" if median_gap < 0.01 else "meaningful"})')
print(f'  Mean cross-gamma corr: {mean_corr:.3f} ({"HIGH" if mean_corr > 0.8 else "moderate"})')
print(f'  Mean streak length:    {mean_streak:.1f} months ({"SHORT" if mean_streak < 2 else "sticky"})')

if median_gap < 0.01 and mean_corr > 0.8:
    print(f'\n  >> ROOT CAUSE: Gamma choice barely matters most months.')
    print(f'     All gammas are highly correlated ({mean_corr:.2f}).')
    print(f'     The return gap is only {median_gap:.4f}/month at the median.')
    print(f'     DQN cant learn because the signal-to-noise ratio is too low.')
    print(f'     This is not a DQN bug — its a problem structure issue.')
elif mean_streak < 1.5:
    print(f'\n  >> ROOT CAUSE: Best gamma changes too rapidly to predict.')
    print(f'     Average streak is {mean_streak:.1f} months — essentially random.')
else:
    print(f'\n  >> There may be learnable structure. Check state features.')
