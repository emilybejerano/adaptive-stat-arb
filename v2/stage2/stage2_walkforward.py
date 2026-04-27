"""
Stage 2 Walk-Forward Validation: DQN vs Static Thresholds.

Gold standard for time-series evaluation — no look-ahead bias.
Each window: train on expanding history, test on next unseen period.

Windows:
  1. Train 2000-2007, Test 2008-2011  (GFC + recovery)
  2. Train 2000-2011, Test 2012-2015  (bull market)
  3. Train 2000-2015, Test 2016-2019  (late cycle)
  4. Train 2000-2019, Test 2020-2023  (COVID + post-COVID)

Reports per-window and aggregate Sharpe for DQN vs each static γ.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

from stage2_dqn_env import ThresholdEnv, GAMMA_CHOICES


def evaluate_agent(agent, env, deterministic=True):
    obs, _ = env.reset()
    months, gammas, returns = [], [], []
    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        months.append(info['month'])
        gammas.append(info['gamma'])
        returns.append(info['return'])
        done = terminated or truncated
    return pd.DataFrame({'month': months, 'gamma': gammas, 'return': returns})


def evaluate_static(env, gamma):
    action = GAMMA_CHOICES.index(gamma)
    obs, _ = env.reset()
    months, returns = [], []
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        months.append(info['month'])
        returns.append(info['return'])
        done = terminated or truncated
    return pd.DataFrame({'month': months, 'gamma': gamma, 'return': returns})


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


def train_dqn(cluster_dir, log_returns_df, train_start, train_end):
    """Train a fresh DQN on the given period."""
    env = ThresholdEnv(cluster_dir, log_returns_df,
                       start_month=train_start, end_month=train_end,
                       reward_lambda=0.5)

    n_months = len(env.trading_months)
    if n_months < 24:
        return None, env

    model = DQN(
        "MlpPolicy", env,
        learning_rate=1e-3,
        buffer_size=5000,
        learning_starts=50,
        batch_size=32,
        gamma=0.95,
        tau=0.1,
        target_update_interval=50,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[64, 32]),
        verbose=0,
        seed=42,
    )

    # Scale timesteps to training period length
    episodes = 500
    total_timesteps = n_months * episodes
    model.learn(total_timesteps=total_timesteps)
    return model, env


def main():
    cluster_dir = sys.argv[1] if len(sys.argv) > 1 else './res/clusters/agglo_0.5'
    method_name = os.path.basename(cluster_dir)

    print(f"Loading data...")
    log_returns_df = pd.read_pickle('data/log_returns_by_month.pkl')
    print(f"Stage 1: {cluster_dir} ({method_name})")

    # Walk-forward windows (expanding train, fixed test)
    windows = [
        ('2000-01', '2007-12', '2008-01', '2011-12', 'GFC+recovery'),
        ('2000-01', '2011-12', '2012-01', '2015-12', 'Bull market'),
        ('2000-01', '2015-12', '2016-01', '2019-12', 'Late cycle'),
        ('2000-01', '2019-12', '2020-01', '2023-12', 'COVID+post'),
    ]

    all_dqn_test_returns = []
    all_static_test_returns = {g: [] for g in GAMMA_CHOICES}
    window_results = []

    for train_start, train_end, test_start, test_end, label in windows:
        print(f"\n{'='*60}")
        print(f"  Window: Train {train_start}→{train_end}, Test {test_start}→{test_end} ({label})")
        print(f"{'='*60}")

        # Train fresh DQN
        print(f"  Training DQN...", end=' ', flush=True)
        model, train_env = train_dqn(cluster_dir, log_returns_df, train_start, train_end)

        if model is None:
            print("SKIP (too few months)")
            continue

        print(f"done ({len(train_env.trading_months)} train months)")

        # Create test env
        test_env = ThresholdEnv(cluster_dir, log_returns_df,
                                start_month=test_start, end_month=test_end,
                                reward_lambda=0.5)
        print(f"  Test: {len(test_env.trading_months)} months")

        # Evaluate DQN
        dqn_df = evaluate_agent(model, test_env, deterministic=True)
        dqn_metrics = compute_metrics(dqn_df['return'].values)
        all_dqn_test_returns.extend(dqn_df['return'].values)

        gamma_counts = dqn_df['gamma'].value_counts().sort_index()
        print(f"  DQN:         Sharpe {dqn_metrics['Sharpe']:>6.2f} | "
              f"AR {dqn_metrics['AR']:>.3f} | MDD {dqn_metrics['MDD']:>.3f} | "
              f"γ dist: {dict(gamma_counts)}")

        row = {'window': label, 'test': f'{test_start}→{test_end}',
               'DQN_Sharpe': dqn_metrics['Sharpe'], 'DQN_AR': dqn_metrics['AR'],
               'DQN_MDD': dqn_metrics['MDD']}

        # Evaluate static baselines
        for gamma in GAMMA_CHOICES:
            static_df = evaluate_static(test_env, gamma)
            static_metrics = compute_metrics(static_df['return'].values)
            all_static_test_returns[gamma].extend(static_df['return'].values)
            print(f"  γ={gamma:<4}:      Sharpe {static_metrics['Sharpe']:>6.2f} | "
                  f"AR {static_metrics['AR']:>.3f} | MDD {static_metrics['MDD']:>.3f}")
            row[f'Static_{gamma}_Sharpe'] = static_metrics['Sharpe']
            row[f'Static_{gamma}_AR'] = static_metrics['AR']

        window_results.append(row)

    # --- Aggregate results ---
    print(f"\n{'='*60}")
    print(f"  AGGREGATE — All test windows combined ({method_name})")
    print(f"{'='*60}\n")

    agg_dqn = compute_metrics(all_dqn_test_returns)
    print(f"  {'Method':<25} {'Sharpe':>7} {'AR':>7} {'MDD':>7}")
    print(f"  {'-'*50}")
    print(f"  {'DQN Adaptive':<25} {agg_dqn['Sharpe']:>7.2f} {agg_dqn['AR']:>7.3f} {agg_dqn['MDD']:>7.3f}")

    for gamma in GAMMA_CHOICES:
        agg = compute_metrics(all_static_test_returns[gamma])
        print(f"  {'Static γ='+str(gamma):<25} {agg['Sharpe']:>7.2f} {agg['AR']:>7.3f} {agg['MDD']:>7.3f}")

    # --- Per-window summary table ---
    print(f"\n  Per-window Sharpe:")
    print(f"  {'Window':<20} {'DQN':>7}", end='')
    for g in GAMMA_CHOICES:
        print(f" {'γ='+str(g):>8}", end='')
    print(f" {'DQN wins?':>10}")
    print(f"  {'-'*75}")

    dqn_wins = 0
    for row in window_results:
        best_static = max(row.get(f'Static_{g}_Sharpe', 0) for g in GAMMA_CHOICES)
        wins = row['DQN_Sharpe'] > best_static
        if wins:
            dqn_wins += 1
        print(f"  {row['window']:<20} {row['DQN_Sharpe']:>7.2f}", end='')
        for g in GAMMA_CHOICES:
            print(f" {row.get(f'Static_{g}_Sharpe', 0):>8.2f}", end='')
        print(f" {'YES' if wins else 'no':>10}")

    print(f"\n  DQN beats best static in {dqn_wins}/{len(window_results)} windows")

    # --- Save ---
    os.makedirs('./res/stage2', exist_ok=True)
    results_df = pd.DataFrame(window_results)
    results_df.to_csv(f'./res/stage2/walkforward_{method_name}.csv', index=False)

    # --- Plot ---
    _plot_walkforward(window_results, all_dqn_test_returns, all_static_test_returns, method_name)
    print(f"\nResults saved to ./res/stage2/walkforward_{method_name}.csv")


def _plot_walkforward(window_results, dqn_returns, static_returns, method_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Per-window Sharpe bar chart
    ax = axes[0]
    windows = [r['window'] for r in window_results]
    x = np.arange(len(windows))
    width = 0.15

    dqn_sharpes = [r['DQN_Sharpe'] for r in window_results]
    ax.bar(x - 2*width, dqn_sharpes, width, label='DQN', color='steelblue', zorder=3)

    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, gamma in enumerate(GAMMA_CHOICES):
        sharpes = [r.get(f'Static_{gamma}_Sharpe', 0) for r in window_results]
        ax.bar(x + (i-1)*width, sharpes, width, label=f'γ={gamma}',
               color=colors[i], alpha=0.7, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(windows, rotation=15, ha='right')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Per-Window Sharpe: DQN vs Static')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Cumulative wealth across all test windows
    ax = axes[1]
    dqn_wealth = np.exp(np.cumsum(dqn_returns))
    ax.plot(dqn_wealth, label='DQN', linewidth=2, color='steelblue')

    for i, gamma in enumerate(GAMMA_CHOICES):
        wealth = np.exp(np.cumsum(static_returns[gamma]))
        ax.plot(wealth, label=f'γ={gamma}', color=colors[i], alpha=0.7, linewidth=1)

    ax.set_xlabel('Month (all test windows concatenated)')
    ax.set_ylabel('Wealth')
    ax.set_title('Cumulative Returns — Walk-Forward Test')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Walk-Forward Validation — {method_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'./res/stage2/walkforward_{method_name}.png', dpi=150, bbox_inches='tight')
    print(f"Saved ./res/stage2/walkforward_{method_name}.png")


if __name__ == '__main__':
    main()
