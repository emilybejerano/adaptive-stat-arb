"""
Stage 2 DQN Training: Adaptive Threshold Selection via Stable Baselines3.

Trains a DQN agent to pick γ ∈ {0.5, 1.0, 1.25, 2.0} each month.
Compares against static baselines on held-out test period.

Splits:
  Train:    2000-01 to 2014-12  (180 months)
  Validate: 2015-01 to 2018-12  (48 months)
  Test:     2019-01 to 2023-12  (60 months)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from stage2_dqn_env import ThresholdEnv, GAMMA_CHOICES

# ---------------------------------------------------------------------------
# Callback to track training progress
# ---------------------------------------------------------------------------

class EpisodeLogger(BaseCallback):
    """Log episode returns during training."""
    def __init__(self):
        super().__init__()
        self.episode_returns = []
        self.episode_lengths = []

    def _on_step(self):
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.episode_returns.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
        return True


# ---------------------------------------------------------------------------
# Evaluation: run a trained agent through an env, collect actions + returns
# ---------------------------------------------------------------------------

def evaluate_agent(agent, env, deterministic=True):
    """Roll out the agent and collect per-month results."""
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
    """Run a fixed γ through the env for comparison."""
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
    """Compute AR, Vol, MDD, Sharpe from monthly log returns."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cluster_dir_override=None):
    print("Loading data...")
    log_returns_df = pd.read_pickle('data/log_returns_by_month.pkl')

    # Use override or default to best Stage 1 baseline
    cluster_dir = cluster_dir_override or './res/clusters/agglo_0.5'
    if not os.path.exists(cluster_dir):
        print(f"No cluster data found at {cluster_dir}. Run Stage 1 first.")
        return
    method_name = os.path.basename(cluster_dir)

    print(f"Stage 1 clusters: {cluster_dir}")
    print(f"Log returns: {log_returns_df.shape[0]} stocks × {log_returns_df.shape[1]} months")

    # --- Create environments for each split ---
    print("\nCreating environments...")
    train_env = ThresholdEnv(cluster_dir, log_returns_df,
                             start_month='2000-01', end_month='2014-12',
                             reward_lambda=0.5)
    val_env = ThresholdEnv(cluster_dir, log_returns_df,
                           start_month='2015-01', end_month='2018-12',
                           reward_lambda=0.5)
    test_env = ThresholdEnv(cluster_dir, log_returns_df,
                            start_month='2019-01', end_month='2023-12',
                            reward_lambda=0.5)

    print(f"  Train:    {len(train_env.trading_months)} months")
    print(f"  Validate: {len(val_env.trading_months)} months")
    print(f"  Test:     {len(test_env.trading_months)} months")

    # --- Train DQN ---
    print("\n" + "="*60)
    print("  Training DQN Agent")
    print("="*60)

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=1e-3,
        buffer_size=5000,
        learning_starts=50,
        batch_size=32,
        gamma=0.95,            # discount factor (NOT the trading threshold)
        tau=0.1,               # soft update
        target_update_interval=50,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[64, 32]),
        verbose=1,
        seed=42,
    )

    # Train for multiple episodes (each episode = one pass through train period)
    # 180 months/episode × 500 episodes = 90,000 steps
    total_timesteps = 180 * 500
    print(f"Training for {total_timesteps} timesteps (~500 episodes)...")
    logger = EpisodeLogger()
    model.learn(total_timesteps=total_timesteps, callback=logger)

    # Save model
    os.makedirs('./res/stage2', exist_ok=True)
    model.save(f'./res/stage2/dqn_{method_name}')
    print(f"Model saved to ./res/stage2/dqn_{method_name}.zip")

    # --- Evaluate on all splits ---
    print("\n" + "="*60)
    print("  Evaluation")
    print("="*60)

    results = {}

    for split_name, env in [('train', train_env), ('val', val_env), ('test', test_env)]:
        print(f"\n--- {split_name.upper()} period ---")

        # DQN agent
        dqn_df = evaluate_agent(model, env, deterministic=True)
        dqn_metrics = compute_metrics(dqn_df['return'].values)
        gamma_counts = dqn_df['gamma'].value_counts().sort_index()
        results[f'DQN ({split_name})'] = dqn_metrics

        print(f"  DQN:         Sharpe {dqn_metrics['Sharpe']:>6.2f} | "
              f"AR {dqn_metrics['AR']:>.3f} | MDD {dqn_metrics['MDD']:>.3f}")
        print(f"    γ choices: {dict(gamma_counts)}")

        # Static baselines
        for gamma in GAMMA_CHOICES:
            static_df = evaluate_static(env, gamma)
            static_metrics = compute_metrics(static_df['return'].values)
            label = f'Static γ={gamma} ({split_name})'
            results[label] = static_metrics
            print(f"  γ={gamma:<4}:      Sharpe {static_metrics['Sharpe']:>6.2f} | "
                  f"AR {static_metrics['AR']:>.3f} | MDD {static_metrics['MDD']:>.3f}")

    # --- Summary: TEST period comparison ---
    print("\n" + "="*60)
    print("  FINAL: TEST PERIOD (2019-2023)")
    print("="*60)
    print(f"\n{'Method':<25} {'AR':>7} {'Vol':>7} {'MDD':>7} {'Sharpe':>7}")
    print("-" * 55)

    test_keys = [k for k in results if 'test' in k]
    test_sorted = sorted(test_keys, key=lambda k: results[k]['Sharpe'], reverse=True)
    for k in test_sorted:
        m = results[k]
        label = k.replace(' (test)', '')
        print(f"{label:<25} {m['AR']:>7.4f} {m['Vol']:>7.4f} {m['MDD']:>7.4f} {m['Sharpe']:>7.2f}")

    # --- Plot results ---
    _plot_results(model, train_env, val_env, test_env, logger)
    print("\nPlots saved to ./res/stage2/")


def _plot_results(model, train_env, val_env, test_env, logger):
    """Generate 3-panel figure: training curve, γ distribution, cumulative returns."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Training reward curve
    ax = axes[0, 0]
    if logger.episode_returns:
        window = min(20, len(logger.episode_returns))
        smoothed = pd.Series(logger.episode_returns).rolling(window, min_periods=1).mean()
        ax.plot(smoothed, linewidth=1)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Return')
        ax.set_title('Training Reward Curve')
        ax.grid(True, alpha=0.3)

    # Panel 2: γ choices over time (test period)
    ax = axes[0, 1]
    test_df = evaluate_agent(model, test_env, deterministic=True)
    ax.scatter(range(len(test_df)), test_df['gamma'], c=test_df['gamma'],
               cmap='viridis', s=20, alpha=0.7)
    ax.set_xlabel('Month (test period)')
    ax.set_ylabel('γ chosen')
    ax.set_title('DQN Threshold Choices (Test Period)')
    ax.set_yticks(GAMMA_CHOICES)
    ax.grid(True, alpha=0.3)

    # Panel 3: Cumulative returns (test period)
    ax = axes[1, 0]
    dqn_wealth = np.exp(np.cumsum(test_df['return'].values))
    ax.plot(dqn_wealth, label='DQN Adaptive', linewidth=2, color='blue')

    for gamma in GAMMA_CHOICES:
        static_df = evaluate_static(test_env, gamma)
        static_wealth = np.exp(np.cumsum(static_df['return'].values))
        ax.plot(static_wealth, label=f'Static γ={gamma}', linewidth=1, alpha=0.6)

    ax.set_xlabel('Month (test period)')
    ax.set_ylabel('Wealth')
    ax.set_title('Cumulative Returns — Test Period (2019-2023)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: γ distribution bar chart
    ax = axes[1, 1]
    gamma_counts = test_df['gamma'].value_counts().sort_index()
    ax.bar([str(g) for g in gamma_counts.index], gamma_counts.values, color='steelblue')
    ax.set_xlabel('γ threshold')
    ax.set_ylabel('Months chosen')
    ax.set_title('DQN Action Distribution (Test)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Stage 2: DQN Adaptive Threshold Selection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    method_name = os.path.basename(test_env.cluster_dir)
    plt.savefig(f'./res/stage2/dqn_results_{method_name}.png', dpi=150, bbox_inches='tight')
    print(f"Saved ./res/stage2/dqn_results_{method_name}.png")

    # Save test results to CSV
    test_df.to_csv(f'./res/stage2/dqn_test_actions_{method_name}.csv', index=False)


if __name__ == '__main__':
    import sys
    cluster_dir = sys.argv[1] if len(sys.argv) > 1 else None
    main(cluster_dir_override=cluster_dir)
