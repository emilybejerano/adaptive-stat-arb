"""
Backtest clustering methods using ORCA paper's Algorithm 1.

Truth Constraints Applied:
  3. ALGORITHM 1 TRADING LOGIC: Momentum spread Δ(x) computed within each cluster.
     Positions only if spread exceeds γ=1.0 std of GLOBAL spread distribution.
     Not every asset is traded — only the tails.
  4. STANDARDIZED METRICS: AR from log-growth (compounding), MDD from cumulative
     wealth curve to capture 2008/2020 drawdowns.
  5. TRANSACTION COSTS: 10 bps per side (20 bps round-trip) on every trade.
  6. SURVIVORSHIP BIAS CHECK: Count delisted/missing stocks per decade.

Reference: Kim et al., "Deep Mean-Reversion" (ICAIF 2025), Section 3.3, Algorithm 1
"""
import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

# Transaction cost: 10 bps per side
TC_PER_SIDE = 0.0010  # 0.10%


def compute_monthly_return(cluster_file, next_month_returns, gamma=1.0, stoploss=-0.10):
    """
    Algorithm 1 from ORCA paper, per-cluster long-short.

    Constraint 3:
      - Compute momentum spread Δ(x) = MOM1(x) - median(MOM1 in cluster)
      - σ_Δ = std of ALL spreads across ALL clusters (global, not per-cluster)
      - Long if Δ(x) < -γ·σ_Δ, Short if Δ(x) > γ·σ_Δ
      - Equal-weight across CLUSTERS (not across individual positions)

    Returns: (portfolio_log_return, n_long, n_short, n_clusters_traded)
    """
    df = pd.read_csv(cluster_file)
    if 'Unnamed: 0' in df.columns:
        if 'firms' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        else:
            df.rename(columns={'Unnamed: 0': 'firms'}, inplace=True)
    if 'firms' not in df.columns:
        return 0.0, 0, 0, 0

    df['firms'] = df['firms'].astype(str)
    df = df.set_index('firms')

    if 'MOM1' not in df.columns or 'clusters' not in df.columns:
        return 0.0, 0, 0, 0

    df = df.dropna(subset=['MOM1'])
    if len(df) < 10:
        return 0.0, 0, 0, 0

    next_month_returns.index = next_month_returns.index.astype(str)

    # --- Step 1: Compute spreads within each cluster ---
    cluster_spreads = {}  # cluster_id -> Series of spreads
    all_spread_values = []

    for cluster_id in df['clusters'].unique():
        cluster_df = df[df['clusters'] == cluster_id]
        if len(cluster_df) < 5:
            continue
        # Δ(x) = MOM1(x) - median(MOM1 in cluster)
        median_mom = cluster_df['MOM1'].median()
        spread = cluster_df['MOM1'] - median_mom
        cluster_spreads[cluster_id] = spread
        all_spread_values.extend(spread.values)

    if len(all_spread_values) < 10:
        return 0.0, 0, 0, 0

    # --- Step 2: Global threshold (Constraint 3) ---
    # σ_Δ is the std of ALL spreads across ALL clusters for THIS month
    sigma_delta = np.std(all_spread_values)
    if sigma_delta < 1e-8:
        return 0.0, 0, 0, 0

    threshold = gamma * sigma_delta

    # --- Step 3: Per-cluster long-short returns ---
    # Each cluster contributes ONE return = mean(long) - mean(short)
    # Portfolio = equal-weighted average across clusters
    cluster_returns = []
    total_long = 0
    total_short = 0

    for cluster_id, spread in cluster_spreads.items():
        # Only trade assets in the tails (Constraint 3)
        long_idx = spread[spread < -threshold].index.astype(str)
        short_idx = spread[spread > threshold].index.astype(str)

        # Must have BOTH long and short to form a spread trade
        if len(long_idx) == 0 or len(short_idx) == 0:
            continue

        long_rets = next_month_returns.reindex(long_idx).dropna()
        short_rets = next_month_returns.reindex(short_idx).dropna()

        if len(long_rets) == 0 or len(short_rets) == 0:
            continue

        # Apply stop-loss per position (paper: 10%)
        if stoploss is not None:
            sl_log = np.log1p(stoploss)  # -0.1054 for -10%
            long_rets = long_rets.clip(lower=sl_log)

        # Per-cluster return: long basket - short basket
        cluster_ret = long_rets.mean() - short_rets.mean()

        # Constraint 5: Transaction costs — 10 bps per side
        # Full turnover each month (all positions are new), so cost = 2 * TC_PER_SIDE
        # Applied per cluster since each cluster is a separate long-short trade
        cluster_ret -= 2 * TC_PER_SIDE

        cluster_returns.append(cluster_ret)
        total_long += len(long_rets)
        total_short += len(short_rets)

    if len(cluster_returns) == 0:
        return 0.0, 0, 0, 0

    # Equal-weighted across clusters
    portfolio_return = np.mean(cluster_returns)
    return portfolio_return, total_long, total_short, len(cluster_returns)


def compute_metrics(monthly_returns):
    """
    Constraint 4: Standardized performance metrics.
    AR from log-growth (compounding), MDD from cumulative wealth curve.
    """
    returns = np.array(monthly_returns, dtype=float)
    returns = returns[np.isfinite(returns)]

    if len(returns) < 12:
        return {k: np.nan for k in ['AR', 'Vol', 'MDD', 'Sharpe', 'Sortino', 'Calmar']}

    # --- AR: Annualized Return from log-growth (Constraint 4) ---
    # Total log return, then convert to annualized simple return
    total_log_return = np.sum(returns)
    n_years = len(returns) / 12.0
    # Annualized log return
    annual_log_return = total_log_return / n_years
    # Convert to simple return: AR = exp(annual_log_return) - 1
    ar = np.expm1(annual_log_return)

    # --- Vol: Annualized Volatility ---
    vol = np.std(returns, ddof=1) * np.sqrt(12)

    # --- MDD: Maximum Drawdown from cumulative WEALTH curve (Constraint 4) ---
    # This ensures 2008 and 2020 regimes are fully captured
    cum_log = np.cumsum(returns)
    wealth = np.exp(cum_log)  # cumulative wealth
    running_max = np.maximum.accumulate(wealth)
    drawdowns = (wealth - running_max) / running_max
    mdd = abs(np.min(drawdowns))

    # --- Sharpe: Annualized, 0 risk-free ---
    mean_monthly = np.mean(returns)
    std_monthly = np.std(returns, ddof=1)
    sharpe = (mean_monthly / std_monthly) * np.sqrt(12) if std_monthly > 1e-8 else 0

    # --- Sortino: Using downside deviation ---
    downside = returns[returns < 0]
    if len(downside) > 1:
        downside_dev = np.sqrt(np.mean(downside**2)) * np.sqrt(12)
    else:
        downside_dev = vol * 0.5  # fallback if almost no negative months
    sortino = (mean_monthly * 12) / downside_dev if downside_dev > 1e-8 else 0

    # --- Calmar: AR / MDD ---
    calmar = ar / mdd if mdd > 1e-4 else 0

    return {
        'AR': ar,
        'Vol': vol,
        'MDD': mdd,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
    }


def backtest_method(cluster_dir, log_returns_df, start_month='2000-01', gamma=1.0):
    """Run Algorithm 1 backtest for all months."""
    cluster_files = sorted(glob(f'{cluster_dir}/*.csv'))
    available_months = [os.path.splitext(os.path.basename(f))[0] for f in cluster_files]
    return_months = sorted([str(c) for c in log_returns_df.columns])
    trading_months = [m for m in available_months if m >= start_month and m in return_months]

    monthly_returns = []
    months_traded = []
    total_clusters_traded = 0

    for i, month in enumerate(tqdm(trading_months, desc=os.path.basename(cluster_dir))):
        month_idx = return_months.index(month) if month in return_months else -1
        if month_idx < 0 or month_idx + 1 >= len(return_months):
            continue
        next_month = return_months[month_idx + 1]
        next_returns = log_returns_df[next_month]
        cluster_file = os.path.join(cluster_dir, f'{month}.csv')
        if not os.path.exists(cluster_file):
            continue

        ret, n_long, n_short, n_clusters = compute_monthly_return(
            cluster_file, next_returns, gamma=gamma
        )
        monthly_returns.append(ret)
        months_traded.append(next_month)
        total_clusters_traded += n_clusters

    if monthly_returns:
        neg_months = sum(1 for r in monthly_returns if r < 0)
        avg_clusters = total_clusters_traded / len(monthly_returns)
        print(f"    {len(monthly_returns)} months, {neg_months} negative ({neg_months/len(monthly_returns)*100:.0f}%), "
              f"avg {avg_clusters:.1f} clusters/month")

    return monthly_returns, months_traded


def main():
    log_returns_df = pd.read_pickle('data/log_returns_by_month.pkl')
    print(f"Log returns: {log_returns_df.shape[0]} stocks, {log_returns_df.shape[1]} months")

    methods = {}
    for name in ['kmeans_20', 'kmeans_30', 'dbscan_0.1', 'agglo_0.5']:
        d = f'./res/clusters/{name}'
        if os.path.exists(d) and len(glob(f'{d}/*.csv')) > 0:
            methods[name] = d

    orca_dir = './res/pinn/clustering'
    if os.path.exists(orca_dir) and len(glob(f'{orca_dir}/*.csv')) > 0:
        methods['ORCA'] = orca_dir

    if not methods:
        print("No cluster files found.")
        return

    print(f"\nBacktesting {len(methods)} methods: {list(methods.keys())}")
    print(f"Algorithm 1: γ=1.0, 10% stop-loss, per-cluster long-short, monthly rebalancing\n")

    all_results = {}
    all_monthly = {}

    for name, cluster_dir in methods.items():
        returns, months = backtest_method(cluster_dir, log_returns_df, start_month='2000-01')
        metrics = compute_metrics(returns)
        all_results[name] = metrics
        all_monthly[name] = pd.Series(returns, index=months)

    # Print results table
    print(f"\n{'='*80}")
    print(f"  RESULTS — Algorithm 1 Backtest (γ=1.0, K=30, monthly rebalancing)")
    print(f"  Per-cluster long-short, equal-weighted across clusters")
    print(f"{'='*80}\n")

    print(f"{'Method':<15} {'AR':>8} {'Vol':>8} {'MDD':>8} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8}")
    print("-" * 72)
    for name, m in all_results.items():
        print(f"{name:<15} {m['AR']:>8.4f} {m['Vol']:>8.4f} {m['MDD']:>8.4f} "
              f"{m['Sharpe']:>8.4f} {m['Sortino']:>8.4f} {m['Calmar']:>8.4f}")

    print("-" * 72)
    print(f"{'Paper K-means':<15} {'0.3054':>8} {'0.1063':>8} {'0.1904':>8} {'2.5909':>8} {'3.0118':>8} {'1.4459':>8}")
    print(f"{'Paper DBSCAN':<15} {'0.2010':>8} {'0.1012':>8} {'0.1870':>8} {'1.6893':>8} {'2.0438':>8} {'0.9144':>8}")
    print(f"{'Paper Agglo':<15} {'0.1462':>8} {'0.0773':>8} {'0.1948':>8} {'1.5027':>8} {'1.7852':>8} {'0.5965':>8}")
    print(f"{'Paper ORCA':<15} {'0.3680':>8} {'0.1176':>8} {'0.1718':>8} {'2.8747':>8} {'3.3411':>8} {'1.9669':>8}")

    # Save results
    cum_returns = pd.DataFrame()
    for name, series in all_monthly.items():
        cum_returns[name] = series.cumsum()
    cum_returns.to_csv('./res/cumulative_returns.csv')

    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_results).T
    metrics_df.to_csv('./res/metrics.csv')
    print(f"\nMetrics saved to ./res/metrics.csv")
    print(f"Cumulative returns saved to ./res/cumulative_returns.csv")

    # --- Constraint 6: Survivorship Bias Check ---
    print(f"\n{'='*80}")
    print(f"  SURVIVORSHIP BIAS CHECK")
    print(f"  Stocks present vs missing/delisted per decade")
    print(f"{'='*80}\n")

    all_months = sorted([str(c) for c in log_returns_df.columns])
    decades = {
        '2000-2004': [m for m in all_months if '2000' <= m < '2005'],
        '2005-2009': [m for m in all_months if '2005' <= m < '2010'],
        '2010-2014': [m for m in all_months if '2010' <= m < '2015'],
        '2015-2019': [m for m in all_months if '2015' <= m < '2020'],
        '2020-2023': [m for m in all_months if '2020' <= m <= '2024'],
    }

    all_permnos = set(log_returns_df.index.astype(str))
    print(f"{'Period':<12} {'Avg Stocks':>12} {'Ever Present':>14} {'Delisted/Missing':>18} {'Attrition':>10}")
    print("-" * 70)

    prev_stocks = None
    for period, months in decades.items():
        if not months:
            continue
        # Count stocks with non-NaN returns each month
        monthly_counts = []
        stocks_in_period = set()
        for m in months:
            if m in log_returns_df.columns:
                active = log_returns_df[m].dropna().index.astype(str)
                monthly_counts.append(len(active))
                stocks_in_period.update(active)

        avg_count = np.mean(monthly_counts) if monthly_counts else 0

        if prev_stocks is not None:
            disappeared = prev_stocks - stocks_in_period
            attrition = len(disappeared) / len(prev_stocks) * 100 if len(prev_stocks) > 0 else 0
            print(f"{period:<12} {avg_count:>12.0f} {len(stocks_in_period):>14} {len(disappeared):>18} {attrition:>9.1f}%")
        else:
            print(f"{period:<12} {avg_count:>12.0f} {len(stocks_in_period):>14} {'—':>18} {'—':>10}")

        prev_stocks = stocks_in_period

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 6))
        for col in cum_returns.columns:
            if cum_returns[col].abs().sum() < 0.01:
                continue
            wealth = np.exp(cum_returns[col].astype(float))
            ax.plot(range(len(wealth)), wealth, label=col, linewidth=1.5)
        ax.set_xlabel('Months (from 2000-01)')
        ax.set_ylabel('Cumulative Return (wealth)')
        ax.set_title('Cumulative Returns — Algorithm 1 Backtest (γ=1.0)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        n = len(cum_returns)
        ax.set_xticks(list(range(0, n, 48)))
        ax.set_xticklabels([f'{2000+i*4}' for i in range(len(range(0, n, 48)))])
        plt.tight_layout()
        plt.savefig('./res/cumulative_returns.png', dpi=150)
        print(f"Plot saved to ./res/cumulative_returns.png")
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == '__main__':
    main()
