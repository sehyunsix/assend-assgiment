#!/usr/bin/env python3
"""
Micro-level Recovery Analysis
==============================

Analyzes orderbook recovery in very fine time granularity after liquidation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (20, 14)
plt.rcParams['font.size'] = 10


def load_data(output_dir: str):
    """Load all necessary data."""
    output_dir = Path(output_dir)

    metrics_df = pd.read_csv(output_dir / "orderbook_metrics.csv")
    metrics_df = metrics_df.sort_values('timestamp').reset_index(drop=True)

    with open(output_dir / "liquidation_summary.json", 'r') as f:
        liq_summary = json.load(f)

    return metrics_df, liq_summary


def get_micro_stages(metrics_df, liquidation_ts, micro_stages):
    """Get metrics for very fine time stages."""
    stage_data = {}

    # Before period
    before_window = 10 * 1_000_000  # 10 seconds before
    before_mask = (
        (metrics_df['timestamp'] >= liquidation_ts - before_window) &
        (metrics_df['timestamp'] < liquidation_ts)
    )
    stage_data['Before'] = metrics_df[before_mask].copy()
    stage_data['Before']['stage'] = 'Before'

    # Micro stages after liquidation
    for start_sec, end_sec, label in micro_stages:
        start_us = start_sec * 1_000_000
        end_us = end_sec * 1_000_000

        mask = (
            (metrics_df['timestamp'] >= liquidation_ts + start_us) &
            (metrics_df['timestamp'] < liquidation_ts + end_us)
        )
        stage_df = metrics_df[mask].copy()
        stage_df['stage'] = label
        stage_data[label] = stage_df

    return stage_data


def plot_micro_recovery(metrics_df, clusters, output_dir):
    """Plot micro-level recovery analysis."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"

    if not clusters:
        print("No clusters found")
        return

    # Define micro recovery stages (very fine granularity)
    micro_stages = [
        (0, 1, '0-1s'),
        (1, 2, '1-2s'),
        (2, 3, '2-3s'),
        (3, 5, '3-5s'),
        (5, 10, '5-10s'),
        (10, 20, '10-20s'),
        (20, 30, '30-30s'),
        (30, 60, '30-60s')
    ]

    # Collect data
    all_stage_data = {stage[2]: [] for stage in micro_stages}
    all_stage_data['Before'] = []

    for i, cluster in enumerate(clusters[:10]):
        start_ts = int(cluster['start_ts'])
        stage_data = get_micro_stages(metrics_df, start_ts, micro_stages)

        for stage_label, df in stage_data.items():
            if len(df) > 0:
                df['cluster_id'] = i
                all_stage_data[stage_label].append(df)

    # Combine
    combined_stages = []
    for stage_label, dfs in all_stage_data.items():
        if dfs:
            combined_stages.append(pd.concat(dfs, ignore_index=True))

    if not combined_stages:
        print("No data found")
        return

    combined = pd.concat(combined_stages, ignore_index=True)

    # Stage order
    stage_order = ['Before'] + [s[2] for s in micro_stages]

    # Color gradient from blue to red to green
    n_stages = len(stage_order) - 1
    colors = ['#3498db']  # Before in blue
    colors.extend(plt.cm.RdYlGn(np.linspace(0.1, 0.9, n_stages)))

    # ========================================
    # Figure 1: Micro-level line plots
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    metrics_to_plot = [
        ('spread_bps', 'Spread (bps)', axes[0, 0], 'lower is better'),
        ('depth_50bps_total', 'Market Depth (BTC)', axes[0, 1], 'higher is better'),
        ('order_imbalance', 'Order Imbalance (abs)', axes[1, 0], 'lower is better'),
        ('stability_score', 'Stability Score', axes[1, 1], 'higher is better')
    ]

    for metric, ylabel, ax, direction in metrics_to_plot:
        stage_means = []
        stage_stds = []
        stage_counts = []
        stage_labels_clean = []

        for stage in stage_order:
            stage_df = combined[combined['stage'] == stage]
            if len(stage_df) > 5:  # Minimum data points
                if metric == 'order_imbalance':
                    mean_val = stage_df[metric].abs().mean()
                    std_val = stage_df[metric].abs().std()
                else:
                    mean_val = stage_df[metric].mean()
                    std_val = stage_df[metric].std()

                stage_means.append(mean_val)
                stage_stds.append(std_val)
                stage_counts.append(len(stage_df))
                stage_labels_clean.append(stage)

        if not stage_means:
            continue

        x_pos = np.arange(len(stage_means))

        # Plot with error bars
        ax.errorbar(x_pos, stage_means, yerr=stage_stds,
                   marker='o', markersize=12, linewidth=3, capsize=10,
                   color='#2c3e50', ecolor='#95a5a6', capthick=2.5,
                   markerfacecolor='#e74c3c', markeredgewidth=2)

        # Fill confidence interval
        upper = np.array(stage_means) + np.array(stage_stds)
        lower = np.array(stage_means) - np.array(stage_stds)
        ax.fill_between(x_pos, lower, upper, alpha=0.2, color='#3498db')

        # Baseline
        baseline = stage_means[0]
        ax.axhline(baseline, color='red', linestyle='--', linewidth=2.5,
                  alpha=0.7, label=f'Baseline: {baseline:.3f}')

        # Vertical line after "Before"
        ax.axvline(0.5, color='red', linestyle=':', linewidth=2, alpha=0.5)
        ax.text(0.5, ax.get_ylim()[1] * 0.95, 'Liquidation ↓',
               ha='center', va='top', fontsize=11, color='red', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Annotate changes
        for i in range(1, len(stage_means)):
            change_pct = ((stage_means[i] - baseline) / baseline * 100) if baseline != 0 else 0

            # Determine if change is good or bad
            if direction == 'lower is better':
                is_good = change_pct < 0
            else:
                is_good = change_pct > 0

            color = 'green' if is_good else 'red'
            y_offset = 10 if i % 2 == 0 else -15

            ax.annotate(f'{change_pct:+.1f}%',
                       xy=(i, stage_means[i]),
                       xytext=(0, y_offset), textcoords='offset points',
                       fontsize=9, color=color, fontweight='bold',
                       ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor=color, alpha=0.7))

        ax.set_xticks(x_pos)
        ax.set_xticklabels(stage_labels_clean, rotation=45, ha='right')
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.set_title(f'{ylabel} - Micro Recovery Timeline', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add data point counts
        ax2 = ax.twinx()
        ax2.bar(x_pos, stage_counts, alpha=0.15, color='gray', width=0.6)
        ax2.set_ylabel('Data Points', fontsize=10, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

    plt.suptitle('Micro-Level Recovery Analysis: Second-by-Second Changes After Liquidation',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(figures_dir / "micro_recovery_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: micro_recovery_analysis.png")

    # ========================================
    # Figure 2: Heat map of changes
    # ========================================
    fig, ax = plt.subplots(figsize=(16, 10))

    # Prepare data matrix
    metrics_list = ['spread_bps', 'depth_50bps_total', 'order_imbalance', 'stability_score']
    metric_labels = ['Spread (bps)', 'Depth (BTC)', 'Imbalance', 'Stability']

    change_matrix = []

    for metric in metrics_list:
        row = []
        baseline = None

        for stage in stage_order:
            stage_df = combined[combined['stage'] == stage]
            if len(stage_df) > 0:
                if metric == 'order_imbalance':
                    val = stage_df[metric].abs().mean()
                else:
                    val = stage_df[metric].mean()

                if baseline is None:
                    baseline = val
                    row.append(0)  # Baseline = 0% change
                else:
                    change_pct = ((val - baseline) / baseline * 100) if baseline != 0 else 0
                    row.append(change_pct)
            else:
                row.append(np.nan)

        change_matrix.append(row)

    # Create DataFrame for heatmap
    change_df = pd.DataFrame(change_matrix,
                             index=metric_labels,
                             columns=stage_order)

    # Plot heatmap
    sns.heatmap(change_df, annot=True, fmt='.1f', cmap='RdYlGn_r', center=0,
                cbar_kws={'label': 'Change from Baseline (%)'}, ax=ax,
                linewidths=1, linecolor='white', square=False,
                vmin=-70, vmax=70)

    ax.set_title('Percentage Change Heat Map: All Metrics Across Micro Time Stages',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Time Stage After Liquidation', fontsize=13, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=13, fontweight='bold')

    # Add vertical line after "Before"
    ax.axvline(1, color='red', linewidth=3, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(figures_dir / "micro_recovery_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: micro_recovery_heatmap.png")

    # ========================================
    # Print detailed statistics table
    # ========================================
    print("\n" + "=" * 100)
    print("MICRO-LEVEL RECOVERY ANALYSIS (Second-by-Second)")
    print("=" * 100)

    for metric_name, metric_col in [('Spread (bps)', 'spread_bps'),
                                     ('Market Depth (BTC)', 'depth_50bps_total'),
                                     ('Order Imbalance (abs)', 'order_imbalance'),
                                     ('Stability Score', 'stability_score')]:
        print(f"\n{metric_name}:")
        print("┌────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
        print("│   Stage    │   Mean   │   Std    │   Min    │   Max    │ Change%  │")
        print("├────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")

        baseline = None
        for stage in stage_order:
            stage_df = combined[combined['stage'] == stage]
            if len(stage_df) > 0:
                if metric_col == 'order_imbalance':
                    vals = stage_df[metric_col].abs()
                else:
                    vals = stage_df[metric_col]

                mean_val = vals.mean()
                std_val = vals.std()
                min_val = vals.min()
                max_val = vals.max()

                if baseline is None:
                    baseline = mean_val
                    change_pct = 0
                else:
                    change_pct = ((mean_val - baseline) / baseline * 100) if baseline != 0 else 0

                print(f"│ {stage:>10} │ {mean_val:>8.4f} │ {std_val:>8.4f} │ {min_val:>8.4f} │ {max_val:>8.4f} │ {change_pct:>+7.1f}% │")

        print("└────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")

    # ========================================
    # Recovery velocity analysis
    # ========================================
    print("\n" + "=" * 100)
    print("RECOVERY VELOCITY ANALYSIS")
    print("=" * 100)
    print("\nHow fast each metric recovers to baseline:")
    print("┌────────────────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ Metric                 │  Time to 90% │  Time to 95% │  Time to 99% │")
    print("├────────────────────────┼──────────────┼──────────────┼──────────────┤")

    for metric_name, metric_col in [('Spread (bps)', 'spread_bps'),
                                     ('Market Depth (BTC)', 'depth_50bps_total'),
                                     ('Stability Score', 'stability_score')]:

        # Get baseline
        baseline_df = combined[combined['stage'] == 'Before']
        if len(baseline_df) > 0:
            baseline = baseline_df[metric_col].mean()

            # Find when each threshold is reached
            thresholds = {'90%': 0.1, '95%': 0.05, '99%': 0.01}
            times = {}

            for thresh_name, thresh_val in thresholds.items():
                found = False
                for stage in [s[2] for s in micro_stages]:
                    stage_df = combined[combined['stage'] == stage]
                    if len(stage_df) > 0:
                        mean_val = stage_df[metric_col].mean()
                        deviation = abs((mean_val - baseline) / baseline) if baseline != 0 else 0

                        if deviation <= thresh_val:
                            times[thresh_name] = stage
                            found = True
                            break

                if not found:
                    times[thresh_name] = ">60s"

            print(f"│ {metric_name:<22} │ {times['90%']:>12} │ {times['95%']:>12} │ {times['99%']:>12} │")

    print("└────────────────────────┴──────────────┴──────────────┴──────────────┘")


def main():
    output_dir = "../../output/phase1"

    print("Loading data...")
    metrics_df, liq_summary = load_data(output_dir)

    # Find clusters
    min_ts = metrics_df['timestamp'].min()
    max_ts = metrics_df['timestamp'].max()

    clusters = []
    for cluster in liq_summary.get('large_cluster_details', []):
        start_ts = int(cluster['start_ts'])
        if min_ts <= start_ts <= max_ts:
            clusters.append(cluster)

    print(f"Found {len(clusters)} clusters in time range")

    if clusters:
        print("\nGenerating micro-level recovery visualizations...")
        plot_micro_recovery(metrics_df, clusters, output_dir)
    else:
        print("No clusters found")

    print("\nDone!")


if __name__ == '__main__':
    main()

