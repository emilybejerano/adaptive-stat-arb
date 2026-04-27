"""Plot backtest results: metrics table + cumulative returns."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.size'] = 11

# Results from backtest.py
results = {
    'K-means (20)':  {'AR': 0.6446, 'Vol': 0.1409, 'MDD': 0.0940, 'Sharpe': 3.5243, 'Sortino': 8.0088, 'Calmar': 6.8567},
    'K-means (30)':  {'AR': 0.6571, 'Vol': 0.1379, 'MDD': 0.0582, 'Sharpe': 3.6555, 'Sortino': 10.8280, 'Calmar': 11.2954},
    'DBSCAN':        {'AR': 0.6771, 'Vol': 0.1903, 'MDD': 0.0990, 'Sharpe': 2.7128, 'Sortino': 6.0615, 'Calmar': 6.8411},
    'Agglomerative': {'AR': 0.6797, 'Vol': 0.1470, 'MDD': 0.0753, 'Sharpe': 3.5210, 'Sortino': 10.2463, 'Calmar': 9.0319},
}

# Paper reference
paper = {
    'K-means*':  {'AR': 0.3054, 'Vol': 0.1063, 'MDD': 0.1904, 'Sharpe': 2.5909, 'Sortino': 3.0118, 'Calmar': 1.4459},
    'DBSCAN*':   {'AR': 0.2010, 'Vol': 0.1012, 'MDD': 0.1870, 'Sharpe': 1.6893, 'Sortino': 2.0438, 'Calmar': 0.9144},
    'Agglo*':    {'AR': 0.1462, 'Vol': 0.0773, 'MDD': 0.1948, 'Sharpe': 1.5027, 'Sortino': 1.7852, 'Calmar': 0.5965},
    'ORCA*':     {'AR': 0.3680, 'Vol': 0.1176, 'MDD': 0.1718, 'Sharpe': 2.8747, 'Sortino': 3.3411, 'Calmar': 1.9669},
}

# --- Figure 1: Metrics Table ---
fig, (ax_table, ax_plot) = plt.subplots(2, 1, figsize=(12, 10),
                                         gridspec_kw={'height_ratios': [1.2, 1.5]})

# Build table data
metrics = ['AR', 'Vol', 'MDD', 'Sharpe', 'Sortino', 'Calmar']
row_labels = list(results.keys()) + [''] + list(paper.keys())
cell_text = []
cell_colors = []

for name in results:
    row = [f"{results[name][m]:.4f}" for m in metrics]
    cell_text.append(row)
    cell_colors.append(['#e6f3ff'] * len(metrics))

# Separator row
cell_text.append(['—'] * len(metrics))
cell_colors.append(['#f0f0f0'] * len(metrics))

for name in paper:
    row = [f"{paper[name][m]:.4f}" for m in metrics]
    cell_text.append(row)
    cell_colors.append(['#fff3e6'] * len(metrics))

ax_table.axis('off')
ax_table.set_title('Baseline Comparison — Algorithm 1 Backtest\n(* = paper reported values)',
                    fontsize=14, fontweight='bold', pad=20)

table = ax_table.table(
    cellText=cell_text,
    rowLabels=row_labels,
    colLabels=metrics,
    cellColours=cell_colors,
    loc='center',
    cellLoc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.6)

# Bold header
for j in range(len(metrics)):
    table[0, j].set_text_props(fontweight='bold')

# Bold row labels
for i in range(len(row_labels)):
    table[i + 1, -1].set_text_props(fontweight='bold')

# Highlight best Sharpe in our results
best_sharpe_idx = max(range(len(results)), key=lambda i: list(results.values())[i]['Sharpe'])
table[best_sharpe_idx + 1, 3].set_facecolor('#b3d9ff')

# --- Figure 2: Cumulative Returns ---
cum_returns = pd.read_csv('./res/cumulative_returns.csv', index_col=0)

colors = {'kmeans_20': '#1f77b4', 'kmeans_30': '#2ca02c',
          'dbscan_0.1': '#ff7f0e', 'agglo_0.5': '#d62728', 'ORCA': '#9467bd'}
labels = {'kmeans_20': 'K-means (20)', 'kmeans_30': 'K-means (30)',
          'dbscan_0.1': 'DBSCAN', 'agglo_0.5': 'Agglomerative', 'ORCA': 'ORCA'}

for col in cum_returns.columns:
    if col == 'ORCA' and cum_returns[col].abs().sum() < 0.01:
        continue  # Skip ORCA if not trained yet
    wealth = np.exp(cum_returns[col].astype(float))
    ax_plot.plot(range(len(wealth)), wealth,
                 label=labels.get(col, col),
                 color=colors.get(col, 'gray'),
                 linewidth=1.5)

ax_plot.set_xlabel('Months (from 2000-01)')
ax_plot.set_ylabel('Cumulative Return (wealth)')
ax_plot.set_title('Cumulative Returns — Cluster-Based Mean-Reversion Strategy', fontsize=13, fontweight='bold')
ax_plot.legend(loc='upper left')
ax_plot.grid(True, alpha=0.3)
ax_plot.set_yscale('log')

# Add year labels on x-axis
n_months = len(cum_returns)
year_ticks = list(range(0, n_months, 48))
year_labels = [f'{2000 + i*4}' for i in range(len(year_ticks))]
ax_plot.set_xticks(year_ticks)
ax_plot.set_xticklabels(year_labels)

plt.tight_layout()
plt.savefig('./res/baseline_results.png', dpi=150, bbox_inches='tight')
print("Saved to ./res/baseline_results.png")
plt.close()
