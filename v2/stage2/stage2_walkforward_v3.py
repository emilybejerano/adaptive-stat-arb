"""
Stage 2 Walk-Forward v3: Fixed DQN with differential Sharpe reward.

Key fixes:
  1. Differential Sharpe reward (directly optimizes Sharpe)
  2. γ_RL=0 (bandit — no temporal credit assignment)
  3. Precomputed returns (10x faster training)
  4. State noise (prevents memorization)
  5. Sit-out action
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

from stage2_dqn_env_v3 import ThresholdEnvV3, GAMMA_CHOICES_V3
from stage2_dqn_env import ThresholdEnv, GAMMA_CHOICES  # v1 for static baselines


def evaluate_agent(agent, env, deterministic=True):
    env.training = False  # disable noise for evaluation
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
    env.training = True
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


def train_dqn_v3(cluster_dir, log_returns_df, train_start, train_end):
    env = ThresholdEnvV3(cluster_dir, log_returns_df,
                         start_month=train_start, end_month=train_end,
                         training=True)

    n_months = len(env.trading_months)
    if n_months < 24:
        return None, env

    model = DQN(
        "MlpPolicy", env,
        learning_rate=5e-4,        # slightly lower lr for stability
        buffer_size=10000,
        learning_starts=100,
        batch_size=64,
        gamma=0.0,                 # FIX: bandit — no temporal credit
        tau=0.05,                  # slower target updates
        target_update_interval=100,
        exploration_fraction=0.4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(
            net_arch=[64, 64],     # wider network
        ),
        verbose=0,
        seed=42,
    )

    # More episodes since precomputed returns make it faster
    episodes = 800
    total_timesteps = n_months * episodes
    model.learn(total_timesteps=total_timesteps)
    return model, env


def main():
    cluster_dir = sys.argv[1] if len(sys.argv) > 1 else './res/clusters/agglo_0.5'
    method_name = os.path.basename(cluster_dir)

    print(f"Loading data...")
    log_returns_df = pd.read_pickle('data/log_returns_by_month.pkl')
    print(f"Stage 1: {cluster_dir} ({method_name})")
    print(f"DQN v3: differential Sharpe reward, gamma_RL=0 (bandit), sit-out\n")

    windows = [
        ('2000-01', '2007-12', '2008-01', '2011-12', 'GFC+recovery'),
        ('2000-01', '2011-12', '2012-01', '2015-12', 'Bull market'),
        ('2000-01', '2015-12', '2016-01', '2019-12', 'Late cycle'),
        ('2000-01', '2019-12', '2020-01', '2023-12', 'COVID+post'),
    ]

    all_dqn_returns = []
    all_static_returns = {g: [] for g in GAMMA_CHOICES}
    window_results = []

    for train_start, train_end, test_start, test_end, label in windows:
        print(f"{'='*60}")
        print(f"  Window: Train {train_start}-{train_end}, Test {test_start}-{test_end} ({label})")
        print(f"{'='*60}")

        print(f"  Training DQN v3...", end=' ', flush=True)
        model, train_env = train_dqn_v3(cluster_dir, log_returns_df, train_start, train_end)

        if model is None:
            print("SKIP")
            continue

        print(f"done ({len(train_env.trading_months)} train months)")

        # Test envs
        test_env_v3 = ThresholdEnvV3(cluster_dir, log_returns_df,
                                      start_month=test_start, end_month=test_end,
                                      training=False)
        test_env_v1 = ThresholdEnv(cluster_dir, log_returns_df,
                                    start_month=test_start, end_month=test_end)

        print(f"  Test: {len(test_env_v3.trading_months)} months")

        # DQN v3
        dqn_df = evaluate_agent(model, test_env_v3, deterministic=True)
        dqn_metrics = compute_metrics(dqn_df['return'].values)
        all_dqn_returns.extend(dqn_df['return'].values)

        gamma_counts = dqn_df['gamma'].value_counts()
        sit_count = (dqn_df['gamma'] == 'sit_out').sum()
        print(f"  DQN v3:      Sharpe {dqn_metrics['Sharpe']:>6.2f} | "
              f"AR {dqn_metrics['AR']:>.3f} | MDD {dqn_metrics['MDD']:>.3f} | "
              f"sat out {sit_count}/{len(dqn_df)}")
        print(f"    actions: {dict(gamma_counts)}")

        row = {'window': label,
               'DQN_v3_Sharpe': dqn_metrics['Sharpe'],
               'DQN_v3_AR': dqn_metrics['AR'],
               'DQN_v3_MDD': dqn_metrics['MDD'],
               'sit_out': sit_count}

        # Static baselines
        for gamma in GAMMA_CHOICES:
            static_df = evaluate_static(test_env_v1, gamma)
            static_m = compute_metrics(static_df['return'].values)
            all_static_returns[gamma].extend(static_df['return'].values)
            row[f'Static_{gamma}_Sharpe'] = static_m['Sharpe']
            print(f"  static {gamma:<4}: Sharpe {static_m['Sharpe']:>6.2f} | "
                  f"AR {static_m['AR']:>.3f} | MDD {static_m['MDD']:>.3f}")

        window_results.append(row)
        print()

    # --- Aggregate ---
    print(f"{'='*60}")
    print(f"  AGGREGATE ({method_name})")
    print(f"{'='*60}\n")

    agg_dqn = compute_metrics(all_dqn_returns)
    print(f"  {'Method':<25} {'Sharpe':>7} {'AR':>7} {'MDD':>7}")
    print(f"  {'-'*50}")
    print(f"  {'DQN v3 (diff Sharpe)':<25} {agg_dqn['Sharpe']:>7.2f} {agg_dqn['AR']:>7.3f} {agg_dqn['MDD']:>7.3f}")

    for gamma in GAMMA_CHOICES:
        agg = compute_metrics(all_static_returns[gamma])
        print(f"  {'Static g='+str(gamma):<25} {agg['Sharpe']:>7.2f} {agg['AR']:>7.3f} {agg['MDD']:>7.3f}")

    # Per-window summary
    print(f"\n  Per-window Sharpe:")
    print(f"  {'Window':<15} {'DQN v3':>7} {'SitOut':>6}", end='')
    for g in GAMMA_CHOICES:
        print(f" {'g='+str(g):>7}", end='')
    print(f" {'Wins?':>6}")
    print(f"  {'-'*70}")

    dqn_wins = 0
    for row in window_results:
        best_static = max(row.get(f'Static_{g}_Sharpe', 0) for g in GAMMA_CHOICES)
        wins = row['DQN_v3_Sharpe'] > best_static
        if wins:
            dqn_wins += 1
        print(f"  {row['window']:<15} {row['DQN_v3_Sharpe']:>7.2f} {row['sit_out']:>6}", end='')
        for g in GAMMA_CHOICES:
            print(f" {row.get(f'Static_{g}_Sharpe', 0):>7.2f}", end='')
        print(f" {'YES' if wins else 'no':>6}")

    print(f"\n  DQN v3 beats best static in {dqn_wins}/{len(window_results)} windows")

    # Save
    os.makedirs('./res/stage2', exist_ok=True)
    pd.DataFrame(window_results).to_csv(f'./res/stage2/walkforward_v3_{method_name}.csv', index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    windows_labels = [r['window'] for r in window_results]
    x = np.arange(len(windows_labels))
    width = 0.15
    ax.bar(x - 2*width, [r['DQN_v3_Sharpe'] for r in window_results],
           width, label='DQN v3', color='steelblue', zorder=3)
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, gamma in enumerate(GAMMA_CHOICES):
        sharpes = [r.get(f'Static_{gamma}_Sharpe', 0) for r in window_results]
        ax.bar(x + (i-1)*width, sharpes, width, label=f'g={gamma}',
               color=colors[i], alpha=0.7, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(windows_labels, rotation=15, ha='right')
    ax.set_ylabel('Sharpe')
    ax.set_title('DQN v3 (Diff Sharpe + Bandit) vs Static')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    dqn_wealth = np.exp(np.cumsum(all_dqn_returns))
    ax.plot(dqn_wealth, label='DQN v3', linewidth=2, color='steelblue')
    for i, gamma in enumerate(GAMMA_CHOICES):
        wealth = np.exp(np.cumsum(all_static_returns[gamma]))
        ax.plot(wealth, label=f'g={gamma}', color=colors[i], alpha=0.7)
    ax.set_xlabel('Month')
    ax.set_ylabel('Wealth')
    ax.set_title('Cumulative Returns')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Walk-Forward v3 — {method_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'./res/stage2/walkforward_v3_{method_name}.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved ./res/stage2/walkforward_v3_{method_name}.png")


if __name__ == '__main__':
    main()
