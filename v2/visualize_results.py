"""
Generate 4 key figures for the report.

  Figure 1: Cumulative returns (Stage 1 — strategy works)
  Figure 2: Sharpe vs gamma (inverted U — 1.25 beats 1.0)
  Figure 3: Diagnosis (why RL fails — correlation + oracle)
  Figure 4: Blend vs static walk-forward (the result)

Run from v3/ directory:  python visualize_results.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = './results'


def figure1():
    """Cumulative returns — the strategy works."""
    cum = pd.read_csv(f'{RESULTS_DIR}/cumulative_returns.csv', index_col=0)
    metrics = pd.read_csv(f'{RESULTS_DIR}/metrics.csv', index_col=0)

    fig, ax = plt.subplots(figsize=(10, 5))

    labels = {'kmeans_20': 'K-means (20)', 'kmeans_30': 'K-means (30)',
              'dbscan_0.1': 'DBSCAN', 'agglo_0.5': 'Agglomerative', 'ORCA': 'ORCA'}
    colors = {'kmeans_20': '#1f77b4', 'kmeans_30': '#2ca02c',
              'dbscan_0.1': '#ff7f0e', 'agglo_0.5': '#d62728', 'ORCA': '#9467bd'}

    for col in cum.columns:
        if cum[col].abs().sum() < 0.01:
            continue
        wealth = np.exp(cum[col].astype(float))
        sharpe = metrics.loc[col, 'Sharpe'] if col in metrics.index else 0
        ax.plot(range(len(wealth)), wealth,
                label=f'{labels.get(col, col)} (Sharpe {sharpe:.2f})',
                color=colors.get(col, 'gray'), linewidth=1.5)

    ax.set_xlabel('Months (from 2000)')
    ax.set_ylabel('Cumulative Wealth (log scale)')
    ax.set_title('Stage 1: Cluster-Based Mean-Reversion Strategy\n(Algorithm 1, γ=1.0, 10 bps TC, 10% stop-loss)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    n = len(cum)
    ticks = list(range(0, n, 48))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{2000 + i*4}' for i in range(len(ticks))])

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/figure1_cumulative_returns.png', dpi=150, bbox_inches='tight')
    print('Saved figure1_cumulative_returns.png')


def figure2():
    """Sharpe vs gamma — inverted U, 1.25 beats 1.0."""
    sweep = pd.read_csv(f'{RESULTS_DIR}/threshold_comparison.csv', index_col=0)

    fig, ax = plt.subplots(figsize=(8, 5))

    gamma_values = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    colors = {'agglo_0.5': '#d62728', 'kmeans_20': '#1f77b4',
              'kmeans_30': '#2ca02c', 'dbscan_0.1': '#ff7f0e'}
    labels_map = {'agglo_0.5': 'Agglomerative', 'kmeans_20': 'K-means (20)',
                  'kmeans_30': 'K-means (30)', 'dbscan_0.1': 'DBSCAN'}

    # Parse sweep results
    methods = {}
    for idx in sweep.index:
        parts = idx.split(' + ')
        if len(parts) == 2:
            stage1, stage2 = parts
            if 'ORCA' in stage1:
                continue
            if stage1 not in methods:
                methods[stage1] = {}
            methods[stage1][stage2] = sweep.loc[idx, 'Sharpe']

    for method_name, gammas in methods.items():
        sharpes = []
        for g in gamma_values:
            key = f'static_\u03b3={g}'
            sharpes.append(gammas.get(key, 0))
        if any(s > 0 for s in sharpes):
            ax.plot(gamma_values, sharpes, 'o-',
                    color=colors.get(method_name, 'gray'),
                    label=labels_map.get(method_name, method_name), linewidth=2, markersize=8)

    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axvline(x=1.25, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(1.01, ax.get_ylim()[0] + 0.05, 'Paper\n(γ=1.0)', fontsize=9, color='gray')
    ax.text(1.26, ax.get_ylim()[0] + 0.05, 'Ours\n(γ=1.25)', fontsize=9, color='red')

    ax.set_xlabel('Threshold γ', fontsize=11)
    ax.set_ylabel('Sharpe Ratio', fontsize=11)
    ax.set_title('Static Threshold Sweep: γ=1.25 Beats Paper Default\n(Higher γ = fewer but more extreme trades)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/figure2_threshold_sweep.png', dpi=150, bbox_inches='tight')
    print('Saved figure2_threshold_sweep.png')


def figure3():
    """Diagnosis — why RL fails."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel 1: Oracle best gamma distribution
    ax = axes[0]
    oracle_dist = {'γ=0.5': 36, 'γ=1.0': 40, 'γ=1.25': 55, 'γ=2.0': 157}
    bars = ax.bar(oracle_dist.keys(), oracle_dist.values(),
                  color=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'])
    ax.set_ylabel('Months where γ is best (out of 288)')
    ax.set_title('Oracle: γ=2.0 wins 55% of months...')
    ax.grid(True, alpha=0.3, axis='y')

    # Add annotation
    ax.annotate('But γ=2.0 has worst Sharpe (3.02)\nbecause its losses are large',
                xy=(3, 157), xytext=(1.5, 170),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))

    # Panel 2: Correlation matrix
    ax = axes[1]
    corr = np.array([[1.0, 0.92, 0.879, 0.684],
                     [0.92, 1.0, 0.951, 0.737],
                     [0.879, 0.951, 1.0, 0.771],
                     [0.684, 0.737, 0.771, 1.0]])
    im = ax.imshow(corr, cmap='RdYlBu_r', vmin=0.5, vmax=1.0)
    gamma_labels = ['γ=0.5', 'γ=1.0', 'γ=1.25', 'γ=2.0']
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(gamma_labels, fontsize=10)
    ax.set_yticklabels(gamma_labels, fontsize=10)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                    fontsize=10, color='white' if corr[i, j] > 0.85 else 'black')
    ax.set_title('...but all γ move together (r=0.68-0.95)\n→ DQN cannot distinguish actions')
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Why RL Fails: Best γ Is Unpredictable (changes every 1.6 months, near-random)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/figure3_diagnosis.png', dpi=150, bbox_inches='tight')
    print('Saved figure3_diagnosis.png')


def figure4():
    """Blend vs static — the punchline."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: K-means 20 walk-forward bar chart
    ax = axes[0]
    blend_km20 = pd.read_csv(f'{RESULTS_DIR}/blend_kmeans_20.csv')

    windows = blend_km20['window'].tolist()
    x = np.arange(len(windows))
    width = 0.2

    # Best static per window
    best_static = []
    for _, row in blend_km20.iterrows():
        best_static.append(max(row['Static_1.0'], row['Static_1.25'], row['Static_2.0']))

    ax.bar(x - width/2, best_static, width, label='Best Static γ', color='#aaaaaa', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, blend_km20['Triple'], width, label='Triple Blend', color='#d62728', edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(windows, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('Sharpe Ratio', fontsize=11)
    ax.set_title('Walk-Forward: Triple Blend Wins 4/4\n(K-means 20 clusters)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add "WIN" labels
    for i in range(len(windows)):
        if blend_km20['Triple'].iloc[i] > best_static[i]:
            ax.text(i, blend_km20['Triple'].iloc[i] + 0.05, 'WIN',
                    ha='center', fontsize=8, color='#d62728', fontweight='bold')

    # Panel 2: Final method comparison
    ax = axes[1]
    methods = ['Paper γ=1.0', 'Static γ=1.25', 'Triple Blend', 'DQN+OU', 'DQN v3']
    sharpes = [2.87, 3.13, 3.18, 2.47, 1.97]
    colors = ['gray', '#aaaaaa', '#d62728', '#1f77b4', '#87CEEB']

    bars = ax.barh(range(len(methods)), sharpes, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=11)
    ax.set_xlabel('Aggregate Sharpe Ratio', fontsize=11)
    ax.set_title('All Methods Compared')
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(x=2.87, color='gray', linestyle='--', alpha=0.5)

    # Label values
    for i, (bar, val) in enumerate(zip(bars, sharpes)):
        ax.text(val + 0.03, i, f'{val:.2f}', va='center', fontsize=10)

    plt.suptitle('Result: Diversification Beats Prediction', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/figure4_blend_result.png', dpi=150, bbox_inches='tight')
    print('Saved figure4_blend_result.png')


if __name__ == '__main__':
    print('Generating 4 report figures...\n')
    figure1()
    figure2()
    figure3()
    figure4()
    print('\nDone. Figures in results/')
