"""
Stage 2: Threshold Baselines for Statistical Arbitrage Execution.

Stage 1 (clustering) decides WHICH stocks to group.
Stage 2 (this file) decides WHEN to enter/exit trades within those groups.

Threshold methods (simplest → most complex):
  1. STATIC: Fixed γ × global_σ — paper default (γ=1.0)
  2. STATIC SWEEP: Grid search over γ to find optimal static threshold
  3. OU-AWARE: Per-cluster threshold = κ × σ_OU / √(2θ) from OU stationary dist
  4. RULE-BASED ADAPTIVE: κ varies with θ (fast reversion → tighter threshold)

All methods use the same Algorithm 1 framework from backtest.py.
Transaction costs: 10 bps per side. Stop-loss: -10%.
"""
import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

TC_PER_SIDE = 0.0010


# ---------------------------------------------------------------------------
# Core: compute one month's return given a threshold method
# ---------------------------------------------------------------------------

def compute_monthly_return_with_threshold(
    cluster_file, next_month_returns, threshold_fn, trading_data=None,
    stoploss=-0.10
):
    """
    Algorithm 1 but with a pluggable threshold function.

    threshold_fn(cluster_id, spread_series, global_sigma, ou_params) -> float
        Returns the threshold for THIS cluster.
        ou_params is a dict with keys 'cluster_mu', 'cluster_sigma', 'cluster_theta'
        (None if not available, e.g. for non-ORCA methods).
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

    # Load OU params from trading data if available
    ou_by_cluster = {}
    if trading_data is not None:
        for cid in trading_data['clusters'].unique():
            mask = trading_data['clusters'] == cid
            ou_by_cluster[cid] = {
                'cluster_mu': trading_data.loc[mask, 'cluster_mu'].iloc[0]
                    if 'cluster_mu' in trading_data.columns else None,
                'cluster_sigma': trading_data.loc[mask, 'cluster_sigma'].iloc[0]
                    if 'cluster_sigma' in trading_data.columns else None,
                'cluster_theta': trading_data.loc[mask, 'cluster_theta'].iloc[0]
                    if 'cluster_theta' in trading_data.columns else None,
            }

    # Step 1: Compute spreads per cluster
    cluster_spreads = {}
    all_spread_values = []

    for cluster_id in df['clusters'].unique():
        cluster_df = df[df['clusters'] == cluster_id]
        if len(cluster_df) < 5:
            continue
        median_mom = cluster_df['MOM1'].median()
        spread = cluster_df['MOM1'] - median_mom
        cluster_spreads[cluster_id] = spread
        all_spread_values.extend(spread.values)

    if len(all_spread_values) < 10:
        return 0.0, 0, 0, 0

    global_sigma = np.std(all_spread_values)
    if global_sigma < 1e-8:
        return 0.0, 0, 0, 0

    # Step 2: Per-cluster long-short with pluggable threshold
    cluster_returns = []
    total_long = 0
    total_short = 0

    for cluster_id, spread in cluster_spreads.items():
        ou_params = ou_by_cluster.get(cluster_id, None)
        threshold = threshold_fn(cluster_id, spread, global_sigma, ou_params)

        if threshold < 1e-8:
            continue

        long_idx = spread[spread < -threshold].index.astype(str)
        short_idx = spread[spread > threshold].index.astype(str)

        if len(long_idx) == 0 or len(short_idx) == 0:
            continue

        long_rets = next_month_returns.reindex(long_idx).dropna()
        short_rets = next_month_returns.reindex(short_idx).dropna()

        if len(long_rets) == 0 or len(short_rets) == 0:
            continue

        if stoploss is not None:
            sl_log = np.log1p(stoploss)
            long_rets = long_rets.clip(lower=sl_log)

        cluster_ret = long_rets.mean() - short_rets.mean()
        cluster_ret -= 2 * TC_PER_SIDE

        cluster_returns.append(cluster_ret)
        total_long += len(long_rets)
        total_short += len(short_rets)

    if len(cluster_returns) == 0:
        return 0.0, 0, 0, 0

    portfolio_return = np.mean(cluster_returns)
    return portfolio_return, total_long, total_short, len(cluster_returns)


# ---------------------------------------------------------------------------
# Threshold functions
# ---------------------------------------------------------------------------

def make_static_threshold(gamma):
    """Method 1: Fixed γ × global σ (paper default)."""
    def fn(cluster_id, spread, global_sigma, ou_params):
        return gamma * global_sigma
    fn.__name__ = f'static_γ={gamma}'
    return fn


def make_ou_threshold(kappa):
    """
    Method 3: OU-aware per-cluster threshold.

    From OU theory: dS = θ(μ - S)dt + σdW
    Stationary distribution: N(μ, σ²/(2θ))
    Stationary std = σ_OU / √(2θ)

    Threshold = κ × σ_OU / √(2θ) for each cluster.
    Falls back to κ × global_sigma if OU params unavailable.
    """
    def fn(cluster_id, spread, global_sigma, ou_params):
        if ou_params is None:
            return kappa * global_sigma

        sigma_ou = ou_params.get('cluster_sigma')
        theta_ou = ou_params.get('cluster_theta')

        if sigma_ou is None or theta_ou is None:
            return kappa * global_sigma
        if theta_ou < 0.01 or sigma_ou < 1e-6:
            return kappa * global_sigma  # degenerate OU → fall back

        stationary_std = sigma_ou / np.sqrt(2 * theta_ou)
        return kappa * stationary_std
    fn.__name__ = f'ou_κ={kappa}'
    return fn


def make_adaptive_threshold(kappa_base=1.0):
    """
    Method 4: Rule-based adaptive threshold.

    Idea: mean-reversion speed θ tells us how aggressive to be.
    - High θ (fast reversion) → tighter threshold (trade more, reversion is reliable)
    - Low θ (slow reversion)  → wider threshold (be cautious, spread may not revert)

    κ_effective = kappa_base × (θ_median / θ_cluster)^0.5

    This scales the threshold inversely with sqrt of relative θ.
    Also falls back to static if OU params unavailable.
    """
    # We'll compute θ_median across all clusters in the first call each month,
    # then cache it. Use a mutable container for closure.
    state = {'theta_values': [], 'theta_median': None, 'current_month': None}

    def fn(cluster_id, spread, global_sigma, ou_params):
        if ou_params is None:
            return kappa_base * global_sigma

        sigma_ou = ou_params.get('cluster_sigma')
        theta_ou = ou_params.get('cluster_theta')

        if sigma_ou is None or theta_ou is None or theta_ou < 0.01:
            return kappa_base * global_sigma

        # Collect theta values to compute median (reset each month via backtest loop)
        state['theta_values'].append(theta_ou)
        theta_med = np.median(state['theta_values']) if len(state['theta_values']) > 1 else theta_ou

        # Scale κ: fast reversion → smaller κ → tighter threshold
        kappa_eff = kappa_base * np.sqrt(theta_med / max(theta_ou, 0.01))
        # Clamp to reasonable range
        kappa_eff = np.clip(kappa_eff, 0.3, 3.0)

        stationary_std = sigma_ou / np.sqrt(2 * theta_ou)
        return kappa_eff * stationary_std

    def reset():
        state['theta_values'] = []
        state['theta_median'] = None

    fn.__name__ = f'adaptive_κ={kappa_base}'
    fn.reset = reset
    return fn


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def compute_metrics(monthly_returns):
    """Same as backtest.py — AR, Vol, MDD, Sharpe, Sortino, Calmar."""
    returns = np.array(monthly_returns, dtype=float)
    returns = returns[np.isfinite(returns)]

    if len(returns) < 12:
        return {k: np.nan for k in ['AR', 'Vol', 'MDD', 'Sharpe', 'Sortino', 'Calmar']}

    total_log_return = np.sum(returns)
    n_years = len(returns) / 12.0
    annual_log_return = total_log_return / n_years
    ar = np.expm1(annual_log_return)

    vol = np.std(returns, ddof=1) * np.sqrt(12)

    cum_log = np.cumsum(returns)
    wealth = np.exp(cum_log)
    running_max = np.maximum.accumulate(wealth)
    drawdowns = (wealth - running_max) / running_max
    mdd = abs(np.min(drawdowns))

    mean_monthly = np.mean(returns)
    std_monthly = np.std(returns, ddof=1)
    sharpe = (mean_monthly / std_monthly) * np.sqrt(12) if std_monthly > 1e-8 else 0

    downside = returns[returns < 0]
    if len(downside) > 1:
        downside_dev = np.sqrt(np.mean(downside**2)) * np.sqrt(12)
    else:
        downside_dev = vol * 0.5
    sortino = (mean_monthly * 12) / downside_dev if downside_dev > 1e-8 else 0

    calmar = ar / mdd if mdd > 1e-4 else 0

    return {'AR': ar, 'Vol': vol, 'MDD': mdd, 'Sharpe': sharpe,
            'Sortino': sortino, 'Calmar': calmar}


def backtest_threshold_method(cluster_dir, log_returns_df, threshold_fn,
                               trading_data_dir=None, start_month='2000-01'):
    """Run Algorithm 1 with a given threshold function."""
    cluster_files = sorted(glob(f'{cluster_dir}/*.csv'))
    available_months = [os.path.splitext(os.path.basename(f))[0] for f in cluster_files]
    return_months = sorted([str(c) for c in log_returns_df.columns])
    trading_months = [m for m in available_months if m >= start_month and m in return_months]

    monthly_returns = []
    months_traded = []

    for month in trading_months:
        month_idx = return_months.index(month) if month in return_months else -1
        if month_idx < 0 or month_idx + 1 >= len(return_months):
            continue
        next_month = return_months[month_idx + 1]
        next_returns = log_returns_df[next_month]

        cluster_file = os.path.join(cluster_dir, f'{month}.csv')
        if not os.path.exists(cluster_file):
            continue

        # Load trading data (OU params) if available
        trading_data = None
        if trading_data_dir:
            td_file = os.path.join(trading_data_dir, f'{month}.csv')
            if os.path.exists(td_file):
                trading_data = pd.read_csv(td_file)

        # Reset adaptive state each month
        if hasattr(threshold_fn, 'reset'):
            threshold_fn.reset()

        ret, n_long, n_short, n_clusters = compute_monthly_return_with_threshold(
            cluster_file, next_returns, threshold_fn, trading_data
        )
        monthly_returns.append(ret)
        months_traded.append(next_month)

    return monthly_returns, months_traded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log_returns_df = pd.read_pickle('data/log_returns_by_month.pkl')
    print(f"Log returns: {log_returns_df.shape[0]} stocks, {log_returns_df.shape[1]} months")

    # ---- Define clustering sources (Stage 1) ----
    # Use best-performing baseline for threshold comparison
    # (Can swap in ORCA once training completes)
    cluster_sources = {}

    # Check for ORCA clusters
    orca_cluster_dir = './res/pinn/clustering'
    orca_trading_dir = './res/pinn/trading_data'
    if os.path.exists(orca_cluster_dir) and len(glob(f'{orca_cluster_dir}/*.csv')) > 50:
        cluster_sources['ORCA'] = {
            'cluster_dir': orca_cluster_dir,
            'trading_data_dir': orca_trading_dir,
        }
        print(f"ORCA: {len(glob(f'{orca_cluster_dir}/*.csv'))} months available")

    # All Stage 1 baselines
    for baseline in ['kmeans_20', 'kmeans_30', 'dbscan_0.1', 'agglo_0.5']:
        bdir = f'./res/clusters/{baseline}'
        if os.path.exists(bdir) and len(glob(f'{bdir}/*.csv')) > 50:
            cluster_sources[baseline] = {
                'cluster_dir': bdir,
                'trading_data_dir': None,  # No OU params for baselines
            }

    if not cluster_sources:
        print("No cluster files found. Run Stage 1 first.")
        return

    # ---- Define threshold methods (Stage 2) ----
    threshold_methods = {}

    # Method 1: Static sweep
    for gamma in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        threshold_methods[f'static_γ={gamma}'] = make_static_threshold(gamma)

    # Method 3: OU-aware (only meaningful with ORCA)
    for kappa in [0.5, 1.0, 1.5, 2.0]:
        threshold_methods[f'ou_κ={kappa}'] = make_ou_threshold(kappa)

    # Method 4: Rule-based adaptive
    for kappa in [0.75, 1.0, 1.5]:
        threshold_methods[f'adaptive_κ={kappa}'] = make_adaptive_threshold(kappa)

    # ---- Run all combinations ----
    all_results = {}

    for source_name, source_config in cluster_sources.items():
        print(f"\n{'='*60}")
        print(f"  Stage 1: {source_name}")
        print(f"{'='*60}")

        for method_name, threshold_fn in threshold_methods.items():
            # Skip OU/adaptive methods for non-ORCA sources (they just fall back to static)
            if source_config['trading_data_dir'] is None and ('ou_' in method_name or 'adaptive_' in method_name):
                continue

            label = f"{source_name} + {method_name}"
            returns, months = backtest_threshold_method(
                cluster_dir=source_config['cluster_dir'],
                log_returns_df=log_returns_df,
                threshold_fn=threshold_fn,
                trading_data_dir=source_config['trading_data_dir'],
            )
            metrics = compute_metrics(returns)
            all_results[label] = metrics
            n_months = len(returns)
            neg = sum(1 for r in returns if r < 0)
            print(f"  {method_name:<22} | Sharpe {metrics['Sharpe']:>6.2f} | "
                  f"AR {metrics['AR']:>6.3f} | MDD {metrics['MDD']:>6.3f} | "
                  f"{n_months} months ({neg} neg)")

    # ---- Summary table ----
    print(f"\n{'='*80}")
    print(f"  STAGE 2 RESULTS — Threshold Method Comparison")
    print(f"{'='*80}\n")

    print(f"{'Method':<40} {'AR':>7} {'Vol':>7} {'MDD':>7} {'Sharpe':>7} {'Sortino':>8} {'Calmar':>7}")
    print("-" * 85)

    # Sort by Sharpe
    sorted_results = sorted(all_results.items(), key=lambda x: x[1].get('Sharpe', 0), reverse=True)

    for label, m in sorted_results:
        print(f"{label:<40} {m['AR']:>7.4f} {m['Vol']:>7.4f} {m['MDD']:>7.4f} "
              f"{m['Sharpe']:>7.2f} {m['Sortino']:>8.2f} {m['Calmar']:>7.2f}")

    # ---- Save results ----
    os.makedirs('./res/stage2', exist_ok=True)
    metrics_df = pd.DataFrame(all_results).T
    metrics_df.to_csv('./res/stage2/threshold_comparison.csv')
    print(f"\nResults saved to ./res/stage2/threshold_comparison.csv")

    # ---- Key takeaway for DQN design ----
    if sorted_results:
        best_label, best_metrics = sorted_results[0]
        worst_label, worst_metrics = sorted_results[-1]
        print(f"\n--- Takeaway for DQN reward design ---")
        print(f"Best:  {best_label} (Sharpe {best_metrics['Sharpe']:.2f})")
        print(f"Worst: {worst_label} (Sharpe {worst_metrics['Sharpe']:.2f})")
        print(f"Sharpe range: {worst_metrics['Sharpe']:.2f} to {best_metrics['Sharpe']:.2f}")
        print(f"→ DQN reward should beat best static baseline Sharpe of {best_metrics['Sharpe']:.2f}")


if __name__ == '__main__':
    main()
