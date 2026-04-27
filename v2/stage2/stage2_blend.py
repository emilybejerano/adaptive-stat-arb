"""
Stage 2 Blend Strategy: Walk-forward evaluation.

Based on diagnosis findings:
  - γ=2.0 is best 55% of months but has highest vol
  - γ=1.25 has best Sharpe but lower AR
  - Blending diversifies across thresholds without needing prediction
  - Spread sigma predicts when aggressive γ works better

Methods tested:
  1. EQUAL BLEND: 50/50 average of γ=1.25 and γ=2.0 returns
  2. SIGMA-WEIGHTED BLEND: More γ=2.0 when spread_sigma is high
  3. TRIPLE BLEND: 1/3 each of γ=1.0, γ=1.25, γ=2.0
  4. MOMENTUM BLEND: Weight toward whichever γ did better recently
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob

TC_PER_SIDE = 0.0010


def load_cluster_data(cluster_dir, month):
    fpath = os.path.join(cluster_dir, f'{month}.csv')
    if not os.path.exists(fpath):
        return None
    df = pd.read_csv(fpath)
    if 'Unnamed: 0' in df.columns:
        if 'firms' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        else:
            df.rename(columns={'Unnamed: 0': 'firms'}, inplace=True)
    if 'firms' in df.columns:
        df['firms'] = df['firms'].astype(str)
        df = df.set_index('firms')
    return df


def execute_month(df, next_returns, gamma, stoploss=-0.10):
    if df is None or 'MOM1' not in df.columns or 'clusters' not in df.columns:
        return 0.0, 0.0
    df_clean = df.dropna(subset=['MOM1'])
    if len(df_clean) < 10:
        return 0.0, 0.0
    next_returns.index = next_returns.index.astype(str)

    cluster_spreads = {}
    all_spread_values = []
    for cid in df_clean['clusters'].unique():
        cdf = df_clean[df_clean['clusters'] == cid]
        if len(cdf) < 5:
            continue
        spread = cdf['MOM1'] - cdf['MOM1'].median()
        cluster_spreads[cid] = spread
        all_spread_values.extend(spread.values)

    if len(all_spread_values) < 10:
        return 0.0, np.std(all_spread_values) if all_spread_values else 0.0

    sigma_delta = np.std(all_spread_values)
    if sigma_delta < 1e-8:
        return 0.0, sigma_delta

    threshold = gamma * sigma_delta
    cluster_returns = []
    for cid, spread in cluster_spreads.items():
        long_idx = spread[spread < -threshold].index.astype(str)
        short_idx = spread[spread > threshold].index.astype(str)
        if len(long_idx) == 0 or len(short_idx) == 0:
            continue
        long_rets = next_returns.reindex(long_idx).dropna()
        short_rets = next_returns.reindex(short_idx).dropna()
        if len(long_rets) == 0 or len(short_rets) == 0:
            continue
        if stoploss is not None:
            long_rets = long_rets.clip(lower=np.log1p(stoploss))
        cluster_ret = long_rets.mean() - short_rets.mean()
        cluster_ret -= 2 * TC_PER_SIDE
        cluster_returns.append(cluster_ret)

    if len(cluster_returns) == 0:
        return 0.0, sigma_delta
    return np.mean(cluster_returns), sigma_delta


def get_trading_months(cluster_dir, log_returns_df, start, end):
    cluster_files = sorted(glob(f'{cluster_dir}/*.csv'))
    available = [os.path.splitext(os.path.basename(f))[0] for f in cluster_files]
    return_months = sorted([str(c) for c in log_returns_df.columns])
    return [m for m in available if start <= m <= end and m in return_months]


def compute_metrics(returns_array):
    r = np.array(returns_array, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 6:
        return {'AR': 0, 'Vol': 0, 'MDD': 0, 'Sharpe': 0}
    n_years = len(r) / 12.0
    ar = np.expm1(np.sum(r) / n_years)
    vol = np.std(r, ddof=1) * np.sqrt(12)
    wealth = np.exp(np.cumsum(r))
    running_max = np.maximum.accumulate(wealth)
    mdd = abs(np.min((wealth - running_max) / running_max))
    mean_m = np.mean(r)
    std_m = np.std(r, ddof=1)
    sharpe = (mean_m / std_m) * np.sqrt(12) if std_m > 1e-8 else 0
    return {'AR': ar, 'Vol': vol, 'MDD': mdd, 'Sharpe': sharpe}


def run_all_gammas(cluster_dir, log_returns_df, months):
    """Get returns + spread sigma for all gammas in one pass."""
    return_months = sorted([str(c) for c in log_returns_df.columns])
    gamma_list = [0.5, 1.0, 1.25, 2.0]
    results = {g: [] for g in gamma_list}
    sigmas = []

    for month in months:
        df = load_cluster_data(cluster_dir, month)
        idx = return_months.index(month) if month in return_months else -1
        if idx < 0 or idx + 1 >= len(return_months):
            for g in gamma_list:
                results[g].append(0.0)
            sigmas.append(0.0)
            continue

        next_rets = log_returns_df[return_months[idx + 1]]
        for g in gamma_list:
            ret, sigma = execute_month(df, next_rets, g)
            results[g].append(ret)
            if g == gamma_list[0]:
                sigmas.append(sigma)

    return {g: np.array(v) for g, v in results.items()}, np.array(sigmas)


def main():
    cluster_dir = sys.argv[1] if len(sys.argv) > 1 else './res/clusters/agglo_0.5'
    method_name = os.path.basename(cluster_dir)

    print(f"Loading data...")
    log_returns_df = pd.read_pickle('data/log_returns_by_month.pkl')
    print(f"Stage 1: {cluster_dir} ({method_name})")
    print(f"Blend strategies walk-forward\n")

    windows = [
        ('2000-01', '2007-12', '2008-01', '2011-12', 'GFC+recovery'),
        ('2000-01', '2011-12', '2012-01', '2015-12', 'Bull market'),
        ('2000-01', '2015-12', '2016-01', '2019-12', 'Late cycle'),
        ('2000-01', '2019-12', '2020-01', '2023-12', 'COVID+post'),
    ]

    all_returns = {
        'Static g=1.0': [], 'Static g=1.25': [], 'Static g=2.0': [],
        'Equal blend': [], 'Sigma blend': [], 'Triple blend': [],
        'Momentum blend': [],
    }
    window_results = []

    for train_start, train_end, test_start, test_end, label in windows:
        print(f"{'='*60}")
        print(f"  {label}: Train {train_start}-{train_end}, Test {test_start}-{test_end}")
        print(f"{'='*60}")

        train_months = get_trading_months(cluster_dir, log_returns_df, train_start, train_end)
        test_months = get_trading_months(cluster_dir, log_returns_df, test_start, test_end)

        # Get all gamma returns for train and test
        train_rets, train_sigmas = run_all_gammas(cluster_dir, log_returns_df, train_months)
        test_rets, test_sigmas = run_all_gammas(cluster_dir, log_returns_df, test_months)

        # Compute training sigma median for sigma-weighted blend
        sigma_median = np.median(train_sigmas[train_sigmas > 0])

        row = {'window': label}
        n_test = len(test_months)

        # Static baselines
        for g in [1.0, 1.25, 2.0]:
            m = compute_metrics(test_rets[g])
            all_returns[f'Static g={g}'].extend(test_rets[g])
            row[f'Static_{g}'] = m['Sharpe']
            print(f"  Static g={g:<4}: Sharpe {m['Sharpe']:>5.2f} | AR {m['AR']:>.3f} | Vol {m['Vol']:>.3f} | MDD {m['MDD']:>.3f}")

        # Method 1: Equal blend (50/50 of g=1.25 and g=2.0)
        equal_rets = 0.5 * test_rets[1.25] + 0.5 * test_rets[2.0]
        m = compute_metrics(equal_rets)
        all_returns['Equal blend'].extend(equal_rets)
        row['Equal'] = m['Sharpe']
        print(f"  Equal blend:  Sharpe {m['Sharpe']:>5.2f} | AR {m['AR']:>.3f} | Vol {m['Vol']:>.3f} | MDD {m['MDD']:>.3f}")

        # Method 2: Sigma-weighted blend
        # When spread sigma > median → weight more toward g=2.0 (aggressive)
        # When spread sigma < median → weight more toward g=1.25 (conservative)
        sigma_rets = []
        for i in range(n_test):
            if sigma_median > 0:
                ratio = test_sigmas[i] / sigma_median
                ratio = np.clip(ratio, 0.5, 2.0)
                # w2 = weight on g=2.0, from 0.25 (low sigma) to 0.75 (high sigma)
                w2 = 0.25 + 0.25 * (ratio - 0.5) / 1.5
                w2 = np.clip(w2, 0.2, 0.8)
            else:
                w2 = 0.5
            sigma_rets.append((1 - w2) * test_rets[1.25][i] + w2 * test_rets[2.0][i])
        sigma_rets = np.array(sigma_rets)
        m = compute_metrics(sigma_rets)
        all_returns['Sigma blend'].extend(sigma_rets)
        row['Sigma'] = m['Sharpe']
        print(f"  Sigma blend:  Sharpe {m['Sharpe']:>5.2f} | AR {m['AR']:>.3f} | Vol {m['Vol']:>.3f} | MDD {m['MDD']:>.3f}")

        # Method 3: Triple blend (1/3 each of g=1.0, 1.25, 2.0)
        triple_rets = (test_rets[1.0] + test_rets[1.25] + test_rets[2.0]) / 3.0
        m = compute_metrics(triple_rets)
        all_returns['Triple blend'].extend(triple_rets)
        row['Triple'] = m['Sharpe']
        print(f"  Triple blend: Sharpe {m['Sharpe']:>5.2f} | AR {m['AR']:>.3f} | Vol {m['Vol']:>.3f} | MDD {m['MDD']:>.3f}")

        # Method 4: Momentum blend
        # Track rolling 3-month Sharpe of each gamma, weight proportionally
        # Seed with last 6 months of training data
        rolling_rets = {g: list(train_rets[g][-6:]) for g in [1.25, 2.0]}
        mom_rets = []
        for i in range(n_test):
            # Compute rolling Sharpe for each gamma
            weights = {}
            for g in [1.25, 2.0]:
                recent = np.array(rolling_rets[g][-6:])
                if len(recent) > 1 and np.std(recent) > 1e-8:
                    s = np.mean(recent) / np.std(recent)
                else:
                    s = 0
                weights[g] = max(s, 0)  # only positive Sharpe gets weight

            total_w = sum(weights.values())
            if total_w > 0:
                w2 = weights[2.0] / total_w
            else:
                w2 = 0.5
            w2 = np.clip(w2, 0.2, 0.8)

            ret = (1 - w2) * test_rets[1.25][i] + w2 * test_rets[2.0][i]
            mom_rets.append(ret)

            # Update rolling windows
            for g in [1.25, 2.0]:
                rolling_rets[g].append(test_rets[g][i])

        mom_rets = np.array(mom_rets)
        m = compute_metrics(mom_rets)
        all_returns['Momentum blend'].extend(mom_rets)
        row['Momentum'] = m['Sharpe']
        print(f"  Momentum bld: Sharpe {m['Sharpe']:>5.2f} | AR {m['AR']:>.3f} | Vol {m['Vol']:>.3f} | MDD {m['MDD']:>.3f}")

        window_results.append(row)
        print()

    # === Aggregate ===
    print(f"{'='*60}")
    print(f"  AGGREGATE — {method_name}")
    print(f"{'='*60}\n")

    agg = []
    for name, rets in all_returns.items():
        m = compute_metrics(rets)
        agg.append((name, m))
    agg.sort(key=lambda x: x[1]['Sharpe'], reverse=True)

    print(f"  {'Method':<20} {'Sharpe':>7} {'AR':>7} {'Vol':>7} {'MDD':>7}")
    print(f"  {'-'*55}")
    for name, m in agg:
        print(f"  {name:<20} {m['Sharpe']:>7.2f} {m['AR']:>7.3f} {m['Vol']:>7.3f} {m['MDD']:>7.3f}")

    # Per-window
    print(f"\n  Per-window Sharpe:")
    cols = ['Static_1.0', 'Static_1.25', 'Static_2.0', 'Equal', 'Sigma', 'Triple', 'Momentum']
    labels = ['g=1.0', 'g=1.25', 'g=2.0', 'Equal', 'Sigma', 'Triple', 'Mom']
    print(f"  {'Window':<15}", end='')
    for l in labels:
        print(f" {l:>7}", end='')
    print()
    print(f"  {'-'*75}")

    for row in window_results:
        print(f"  {row['window']:<15}", end='')
        for col in cols:
            val = row.get(col, 0)
            print(f" {val:>7.2f}", end='')
        print()

    # Best method per window
    blend_methods = ['Equal', 'Sigma', 'Triple', 'Momentum']
    static_methods = ['Static_1.0', 'Static_1.25', 'Static_2.0']

    blend_wins = 0
    for row in window_results:
        best_static = max(row.get(k, 0) for k in static_methods)
        best_blend = max(row.get(k, 0) for k in blend_methods)
        if best_blend > best_static:
            blend_wins += 1

    print(f"\n  Blend beats best static in {blend_wins}/{len(window_results)} windows")

    # Save
    os.makedirs('./res/stage2', exist_ok=True)
    pd.DataFrame(window_results).to_csv(f'./res/stage2/blend_{method_name}.csv', index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    ax = axes[0]
    names = [n for n, _ in agg]
    sharpes = [m['Sharpe'] for _, m in agg]
    colors_bar = ['steelblue' if 'blend' in n.lower() else '#cccccc' for n in names]
    ax.barh(range(len(names)), sharpes, color=colors_bar)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Aggregate Sharpe')
    ax.set_title('Blend vs Static — Walk-Forward')
    ax.grid(True, alpha=0.3, axis='x')

    # Cumulative wealth
    ax = axes[1]
    key_methods = ['Static g=1.25', 'Static g=2.0', 'Equal blend', 'Sigma blend', 'Momentum blend']
    colors_line = ['gray', 'lightgray', 'steelblue', 'green', 'orange']
    for name, color in zip(key_methods, colors_line):
        rets = all_returns.get(name, [])
        if rets:
            wealth = np.exp(np.cumsum(rets))
            lw = 2 if 'blend' in name.lower() else 1
            ax.plot(wealth, label=name, color=color, linewidth=lw)
    ax.set_xlabel('Month (all test windows)')
    ax.set_ylabel('Wealth')
    ax.set_title('Cumulative Returns')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Blend Strategies — {method_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'./res/stage2/blend_{method_name}.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved ./res/stage2/blend_{method_name}.png")


if __name__ == '__main__':
    main()
