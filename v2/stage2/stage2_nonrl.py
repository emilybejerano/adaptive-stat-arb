"""
Stage 2 Non-RL Adaptive Thresholds.

Simple rule-based methods that don't require training.
Walk-forward evaluation on same 4 windows for fair comparison.

Methods:
  1. REGIME FILTER: Sit out low-volatility months, trade γ=1.25 otherwise.
     Uses rolling 6-month cross-sectional vol as regime signal.
  2. VOL-SCALED γ: γ = 1.25 × (current_spread_σ / historical_median_spread_σ).
     Wider spreads → higher threshold proportionally.
  3. ORACLE TOP-2: Each month, run all γ values, average the best 2.
     Look-ahead cheating — establishes upper bound on any adaptive method.
  4. BEST-STATIC-PER-WINDOW: Use best in-sample γ for each test window.
     Realistic — pick γ from training data, apply to test.
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
GAMMA_OPTIONS = [0.5, 1.0, 1.25, 2.0]


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
    """Algorithm 1 for one month. Returns log return."""
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
            sl_log = np.log1p(stoploss)
            long_rets = long_rets.clip(lower=sl_log)

        cluster_ret = long_rets.mean() - short_rets.mean()
        cluster_ret -= 2 * TC_PER_SIDE
        cluster_returns.append(cluster_ret)

    if len(cluster_returns) == 0:
        return 0.0, sigma_delta

    return np.mean(cluster_returns), sigma_delta


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


def get_trading_months(cluster_dir, log_returns_df, start, end):
    cluster_files = sorted(glob(f'{cluster_dir}/*.csv'))
    available = [os.path.splitext(os.path.basename(f))[0] for f in cluster_files]
    return_months = sorted([str(c) for c in log_returns_df.columns])
    return [m for m in available if start <= m <= end and m in return_months]


def run_static(cluster_dir, log_returns_df, months, gamma):
    """Run fixed γ, return list of monthly returns."""
    return_months = sorted([str(c) for c in log_returns_df.columns])
    returns = []
    for month in months:
        idx = return_months.index(month) if month in return_months else -1
        if idx < 0 or idx + 1 >= len(return_months):
            returns.append(0.0)
            continue
        next_month = return_months[idx + 1]
        df = load_cluster_data(cluster_dir, month)
        ret, _ = execute_month(df, log_returns_df[next_month], gamma)
        returns.append(ret)
    return returns


def run_regime_filter(cluster_dir, log_returns_df, months, train_months):
    """
    Method 1: Regime Filter.
    Compute rolling 6-month cross-sectional vol of MOM1.
    If below training median → sit out. If above → trade with γ=1.25.
    """
    return_months = sorted([str(c) for c in log_returns_df.columns])

    # Compute cross-sectional vol for training months to get median
    train_vols = []
    for month in train_months:
        df = load_cluster_data(cluster_dir, month)
        if df is not None and 'MOM1' in df.columns:
            train_vols.append(df['MOM1'].dropna().std())
    vol_median = np.median(train_vols) if train_vols else 0.05

    # Rolling window of recent vols
    recent_vols = list(train_vols[-6:])  # seed with last 6 training months
    returns = []
    actions = []

    for month in months:
        df = load_cluster_data(cluster_dir, month)
        if df is not None and 'MOM1' in df.columns:
            current_vol = df['MOM1'].dropna().std()
        else:
            current_vol = 0.0

        recent_vols.append(current_vol)
        rolling_vol = np.mean(recent_vols[-6:])

        idx = return_months.index(month) if month in return_months else -1
        if idx < 0 or idx + 1 >= len(return_months):
            returns.append(0.0)
            actions.append('sit_out')
            continue

        if rolling_vol < vol_median:
            # Low vol regime — sit out
            returns.append(0.0)
            actions.append('sit_out')
        else:
            # High vol regime — trade
            ret, _ = execute_month(df, log_returns_df[return_months[idx + 1]], gamma=1.25)
            returns.append(ret)
            actions.append(1.25)

    return returns, actions


def run_vol_scaled(cluster_dir, log_returns_df, months, train_months):
    """
    Method 2: Volatility-Scaled γ.
    γ = 1.25 × (current_spread_σ / historical_median_spread_σ)
    Clamp to [0.5, 2.5].
    """
    return_months = sorted([str(c) for c in log_returns_df.columns])

    # Compute spread sigma for training months to get median
    train_sigmas = []
    for month in train_months:
        df = load_cluster_data(cluster_dir, month)
        if df is None or 'MOM1' not in df.columns or 'clusters' not in df.columns:
            continue
        df_clean = df.dropna(subset=['MOM1'])
        all_spreads = []
        for cid in df_clean['clusters'].unique():
            cdf = df_clean[df_clean['clusters'] == cid]
            if len(cdf) < 5:
                continue
            spread = cdf['MOM1'] - cdf['MOM1'].median()
            all_spreads.extend(spread.values)
        if all_spreads:
            train_sigmas.append(np.std(all_spreads))

    sigma_median = np.median(train_sigmas) if train_sigmas else 0.05

    returns = []
    gammas_used = []

    for month in months:
        df = load_cluster_data(cluster_dir, month)
        idx = return_months.index(month) if month in return_months else -1
        if idx < 0 or idx + 1 >= len(return_months):
            returns.append(0.0)
            gammas_used.append(1.25)
            continue

        # Compute current spread sigma
        current_sigma = sigma_median  # fallback
        if df is not None and 'MOM1' in df.columns and 'clusters' in df.columns:
            df_clean = df.dropna(subset=['MOM1'])
            all_spreads = []
            for cid in df_clean['clusters'].unique():
                cdf = df_clean[df_clean['clusters'] == cid]
                if len(cdf) < 5:
                    continue
                spread = cdf['MOM1'] - cdf['MOM1'].median()
                all_spreads.extend(spread.values)
            if all_spreads:
                current_sigma = np.std(all_spreads)

        # Scale γ proportionally
        gamma = 1.25 * (current_sigma / sigma_median) if sigma_median > 1e-8 else 1.25
        gamma = np.clip(gamma, 0.5, 2.5)

        ret, _ = execute_month(df, log_returns_df[return_months[idx + 1]], gamma)
        returns.append(ret)
        gammas_used.append(round(gamma, 2))

    return returns, gammas_used


def run_oracle_top2(cluster_dir, log_returns_df, months):
    """
    Method 3: Oracle Top-2 Blend (CHEATING — upper bound).
    Each month, run all γ values, average the best 2 returns.
    """
    return_months = sorted([str(c) for c in log_returns_df.columns])
    returns = []

    for month in months:
        df = load_cluster_data(cluster_dir, month)
        idx = return_months.index(month) if month in return_months else -1
        if idx < 0 or idx + 1 >= len(return_months):
            returns.append(0.0)
            continue

        next_rets = log_returns_df[return_months[idx + 1]]
        month_returns = []
        for gamma in GAMMA_OPTIONS:
            ret, _ = execute_month(df, next_rets, gamma)
            month_returns.append(ret)

        # Average top 2
        sorted_rets = sorted(month_returns, reverse=True)
        returns.append(np.mean(sorted_rets[:2]))

    return returns


def run_best_static_per_window(cluster_dir, log_returns_df, train_months, test_months):
    """
    Method 4: Best Static Per Window.
    Pick the γ that had the best Sharpe on training data, apply to test.
    """
    # Find best γ on training data
    best_gamma = 1.25
    best_sharpe = -999

    for gamma in GAMMA_OPTIONS:
        train_rets = run_static(cluster_dir, log_returns_df, train_months, gamma)
        m = compute_metrics(train_rets)
        if m['Sharpe'] > best_sharpe:
            best_sharpe = m['Sharpe']
            best_gamma = gamma

    # Apply to test
    test_rets = run_static(cluster_dir, log_returns_df, test_months, best_gamma)
    return test_rets, best_gamma


def main():
    cluster_dir = sys.argv[1] if len(sys.argv) > 1 else './res/clusters/agglo_0.5'
    method_name = os.path.basename(cluster_dir)

    print(f"Loading data...")
    log_returns_df = pd.read_pickle('data/log_returns_by_month.pkl')
    print(f"Stage 1: {cluster_dir} ({method_name})")
    print(f"Non-RL adaptive threshold methods\n")

    windows = [
        ('2000-01', '2007-12', '2008-01', '2011-12', 'GFC+recovery'),
        ('2000-01', '2011-12', '2012-01', '2015-12', 'Bull market'),
        ('2000-01', '2015-12', '2016-01', '2019-12', 'Late cycle'),
        ('2000-01', '2019-12', '2020-01', '2023-12', 'COVID+post'),
    ]

    # Collect all test returns per method
    all_returns = {
        'Regime filter': [], 'Vol-scaled': [], 'Oracle top-2': [],
        'Best-static/window': [],
    }
    for g in GAMMA_OPTIONS:
        all_returns[f'Static g={g}'] = []

    window_results = []

    for train_start, train_end, test_start, test_end, label in windows:
        print(f"{'='*60}")
        print(f"  Window: Train {train_start}-{train_end}, Test {test_start}-{test_end} ({label})")
        print(f"{'='*60}")

        train_months = get_trading_months(cluster_dir, log_returns_df, train_start, train_end)
        test_months = get_trading_months(cluster_dir, log_returns_df, test_start, test_end)
        print(f"  Train: {len(train_months)} months, Test: {len(test_months)} months")

        row = {'window': label}

        # Static baselines
        for gamma in GAMMA_OPTIONS:
            rets = run_static(cluster_dir, log_returns_df, test_months, gamma)
            m = compute_metrics(rets)
            all_returns[f'Static g={gamma}'].extend(rets)
            row[f'Static_{gamma}_Sharpe'] = m['Sharpe']
            print(f"  Static g={gamma:<4}: Sharpe {m['Sharpe']:>6.2f} | AR {m['AR']:>.3f} | MDD {m['MDD']:>.3f}")

        # Method 1: Regime filter
        rets, actions = run_regime_filter(cluster_dir, log_returns_df, test_months, train_months)
        m = compute_metrics(rets)
        sit_count = sum(1 for a in actions if a == 'sit_out')
        all_returns['Regime filter'].extend(rets)
        row['Regime_Sharpe'] = m['Sharpe']
        row['Regime_sit_out'] = sit_count
        print(f"  Regime filt:  Sharpe {m['Sharpe']:>6.2f} | AR {m['AR']:>.3f} | MDD {m['MDD']:>.3f} | sat out {sit_count}/{len(test_months)}")

        # Method 2: Vol-scaled
        rets, gammas_used = run_vol_scaled(cluster_dir, log_returns_df, test_months, train_months)
        m = compute_metrics(rets)
        all_returns['Vol-scaled'].extend(rets)
        row['VolScaled_Sharpe'] = m['Sharpe']
        avg_gamma = np.mean([g for g in gammas_used if isinstance(g, (int, float))])
        print(f"  Vol-scaled:   Sharpe {m['Sharpe']:>6.2f} | AR {m['AR']:>.3f} | MDD {m['MDD']:>.3f} | avg g={avg_gamma:.2f}")

        # Method 3: Oracle top-2
        rets = run_oracle_top2(cluster_dir, log_returns_df, test_months)
        m = compute_metrics(rets)
        all_returns['Oracle top-2'].extend(rets)
        row['Oracle_Sharpe'] = m['Sharpe']
        print(f"  Oracle top2:  Sharpe {m['Sharpe']:>6.2f} | AR {m['AR']:>.3f} | MDD {m['MDD']:>.3f}  ** upper bound **")

        # Method 4: Best static per window
        rets, chosen_gamma = run_best_static_per_window(cluster_dir, log_returns_df, train_months, test_months)
        m = compute_metrics(rets)
        all_returns['Best-static/window'].extend(rets)
        row['BestStatic_Sharpe'] = m['Sharpe']
        row['BestStatic_gamma'] = chosen_gamma
        print(f"  Best-static:  Sharpe {m['Sharpe']:>6.2f} | AR {m['AR']:>.3f} | MDD {m['MDD']:>.3f} | chose g={chosen_gamma}")

        window_results.append(row)
        print()

    # --- Aggregate ---
    print(f"{'='*70}")
    print(f"  AGGREGATE — All test windows combined ({method_name})")
    print(f"{'='*70}\n")

    print(f"  {'Method':<25} {'Sharpe':>7} {'AR':>7} {'MDD':>7}")
    print(f"  {'-'*50}")

    agg_sorted = []
    for name, rets in all_returns.items():
        m = compute_metrics(rets)
        agg_sorted.append((name, m))

    agg_sorted.sort(key=lambda x: x[1]['Sharpe'], reverse=True)
    for name, m in agg_sorted:
        marker = ' **' if 'Oracle' in name else ''
        print(f"  {name:<25} {m['Sharpe']:>7.2f} {m['AR']:>7.3f} {m['MDD']:>7.3f}{marker}")

    # Per-window table
    print(f"\n  Per-window Sharpe comparison:")
    methods = ['Regime_Sharpe', 'VolScaled_Sharpe', 'Oracle_Sharpe', 'BestStatic_Sharpe']
    method_labels = ['Regime', 'VolScl', 'Oracle', 'BestSt']

    print(f"  {'Window':<15}", end='')
    for g in GAMMA_OPTIONS:
        print(f" {'g='+str(g):>7}", end='')
    for ml in method_labels:
        print(f" {ml:>7}", end='')
    print()
    print(f"  {'-'*85}")

    for row in window_results:
        print(f"  {row['window']:<15}", end='')
        for g in GAMMA_OPTIONS:
            print(f" {row.get(f'Static_{g}_Sharpe', 0):>7.2f}", end='')
        for mk in methods:
            print(f" {row.get(mk, 0):>7.2f}", end='')
        print()

    # --- Save ---
    os.makedirs('./res/stage2', exist_ok=True)
    pd.DataFrame(window_results).to_csv(f'./res/stage2/nonrl_{method_name}.csv', index=False)

    # --- Plot ---
    _plot_results(window_results, all_returns, method_name)
    print(f"\nSaved to ./res/stage2/")


def _plot_results(window_results, all_returns, method_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Aggregate Sharpe bar chart
    ax = axes[0]
    names, sharpes = [], []
    for name, rets in all_returns.items():
        m = compute_metrics(rets)
        names.append(name.replace('Static g=', 'g='))
        sharpes.append(m['Sharpe'])

    colors = ['#1f77b4' if 'Static' not in n and 'Oracle' not in n else
              '#ff7f0e' if 'Oracle' in n else '#cccccc' for n in names]

    bars = ax.barh(range(len(names)), sharpes, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Aggregate Sharpe')
    ax.set_title('All Methods — Aggregate Walk-Forward Sharpe')
    ax.grid(True, alpha=0.3, axis='x')

    # Panel 2: Cumulative wealth for key methods
    ax = axes[1]
    key_methods = ['Static g=1.25', 'Regime filter', 'Vol-scaled', 'Oracle top-2', 'Best-static/window']
    colors_line = ['gray', 'steelblue', 'green', 'orange', 'red']

    for (name, color) in zip(key_methods, colors_line):
        rets = all_returns.get(name, [])
        if rets:
            wealth = np.exp(np.cumsum(rets))
            ls = '--' if 'Oracle' in name else '-'
            ax.plot(wealth, label=name, color=color, linewidth=1.5, linestyle=ls)

    ax.set_xlabel('Month (all test windows)')
    ax.set_ylabel('Wealth')
    ax.set_title('Cumulative Returns — Key Methods')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Non-RL Adaptive Thresholds — {method_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'./res/stage2/nonrl_{method_name}.png', dpi=150, bbox_inches='tight')
    print(f"Saved ./res/stage2/nonrl_{method_name}.png")


if __name__ == '__main__':
    main()
