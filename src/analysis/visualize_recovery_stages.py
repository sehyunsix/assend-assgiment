#!/usr/bin/env python3
"""
Visualize Recovery Stages After Liquidation
===========================================

Analyzes orderbook recovery in multiple time stages after liquidation.
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
plt.rcParams['figure.figsize'] = (18, 12)
plt.rcParams['font.size'] = 11


def load_data(output_dir: str):
    """Load all necessary data."""
    output_dir = Path(output_dir)

    metrics_df = pd.read_csv(output_dir / "orderbook_metrics.csv")
    metrics_df = metrics_df.sort_values('timestamp').reset_index(drop=True)

    with open(output_dir / "liquidation_summary.json", 'r') as f:
        liq_summary = json.load(f)

    return metrics_df, liq_summary


def get_metrics_by_stage(metrics_df, liquidation_ts, stages):
    """
    Get metrics for different time stages after liquidation.

    Args:
        metrics_df: DataFrame with metrics
        liquidation_ts: Liquidation timestamp
        stages: List of tuples [(start_sec, end_sec, label), ...]

    Returns:
        Dictionary of {label: dataframe}
    """
    stage_data = {}

    # Get baseline (before)
    before_window = 60 * 1_000_000  # 60 seconds
    before_mask = (
        (metrics_df['timestamp'] >= liquidation_ts - before_window) &
        (metrics_df['timestamp'] < liquidation_ts)
    )
    stage_data['Before'] = metrics_df[before_mask].copy()
    stage_data['Before']['stage'] = 'Before'

    # Get each stage after liquidation
    for start_sec, end_sec, label in stages:
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


def plot_recovery_stages(metrics_df, clusters, output_dir):
    """Plot recovery in multiple stages."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"

    if not clusters:
        print("No clusters found")
        return

    # Define recovery stages
    stages = [
        (0, 5, 'Immediate (0-5s)'),
        (5, 15, 'Short (5-15s)'),
        (15, 30, 'Medium (15-30s)'),
        (30, 60, 'Long (30-60s)')
    ]

    # Collect data from all clusters
    all_stage_data = {stage[2]: [] for stage in stages}
    all_stage_data['Before'] = []

    for i, cluster in enumerate(clusters[:10]):
        start_ts = int(cluster['start_ts'])
        stage_data = get_metrics_by_stage(metrics_df, start_ts, stages)

        for stage_label, df in stage_data.items():
            if len(df) > 0:
                df['cluster_id'] = i
                all_stage_data[stage_label].append(df)

    # Combine all data
    combined_stages = []
    for stage_label, dfs in all_stage_data.items():
        if dfs:
            combined_stages.append(pd.concat(dfs, ignore_index=True))

    if not combined_stages:
        print("No data found")
        return

    combined = pd.concat(combined_stages, ignore_index=True)

    # Define stage order
    stage_order = ['Before', 'Immediate (0-5s)', 'Short (5-15s)',
                   'Medium (15-30s)', 'Long (30-60s)']
    stage_colors = ['#3498db', '#e74c3c', '#f39c12', '#27ae60', '#9b59b6']

    # ========================================
    # Figure 1: Box plots by recovery stage
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Spread
    ax = axes[0, 0]
    sns.boxplot(data=combined, x='stage', y='spread_bps', ax=ax,
                order=stage_order, palette=stage_colors)
    ax.set_xlabel('')
    ax.set_ylabel('Spread (bps)')
    ax.set_title('Bid-Ask Spread Recovery by Stage', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

    # Add mean line
    stage_means = combined.groupby('stage')['spread_bps'].mean()
    for i, stage in enumerate(stage_order):
        if stage in stage_means.index:
            ax.hlines(stage_means[stage], i-0.3, i+0.3, colors='red',
                     linestyles='--', linewidth=2, alpha=0.7)

    # Market Depth
    ax = axes[0, 1]
    sns.boxplot(data=combined, x='stage', y='depth_50bps_total', ax=ax,
                order=stage_order, palette=stage_colors)
    ax.set_xlabel('')
    ax.set_ylabel('Market Depth (BTC within 50 bps)')
    ax.set_title('Market Depth Recovery by Stage', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

    # Order Imbalance
    ax = axes[1, 0]
    sns.boxplot(data=combined, x='stage', y='order_imbalance', ax=ax,
                order=stage_order, palette=stage_colors)
    ax.set_xlabel('')
    ax.set_ylabel('Order Imbalance')
    ax.set_title('Order Imbalance Recovery by Stage', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Stability Score
    ax = axes[1, 1]
    sns.boxplot(data=combined, x='stage', y='stability_score', ax=ax,
                order=stage_order, palette=stage_colors)
    ax.set_xlabel('')
    ax.set_ylabel('Stability Score')
    ax.set_title('Stability Score Recovery by Stage', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Orderbook Recovery Stages After Large Liquidations',
                 fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(figures_dir / "recovery_stages_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: recovery_stages_boxplot.png")

    # ========================================
    # Figure 2: Mean values progression
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    metrics_info = [
        ('spread_bps', 'Spread (bps)', axes[0, 0], 'lower'),
        ('depth_50bps_total', 'Market Depth (BTC)', axes[0, 1], 'higher'),
        ('order_imbalance', 'Order Imbalance (abs)', axes[1, 0], 'lower'),
        ('stability_score', 'Stability Score', axes[1, 1], 'higher')
    ]

    for metric, ylabel, ax, better in metrics_info:
        stage_stats = []

        for stage in stage_order:
            stage_df = combined[combined['stage'] == stage]
            if len(stage_df) > 0:
                if metric == 'order_imbalance':
                    mean_val = stage_df[metric].abs().mean()
                    std_val = stage_df[metric].abs().std()
                else:
                    mean_val = stage_df[metric].mean()
                    std_val = stage_df[metric].std()

                stage_stats.append({
                    'stage': stage,
                    'mean': mean_val,
                    'std': std_val,
                    'count': len(stage_df)
                })

        if not stage_stats:
            continue

        stats_df = pd.DataFrame(stage_stats)
        x_pos = np.arange(len(stats_df))

        # Plot line with error bars
        ax.errorbar(x_pos, stats_df['mean'], yerr=stats_df['std'],
                   marker='o', markersize=10, linewidth=2.5, capsize=8,
                   color='#2c3e50', ecolor='#95a5a6', capthick=2)

        # Fill area
        ax.fill_between(x_pos,
                        stats_df['mean'] - stats_df['std'],
                        stats_df['mean'] + stats_df['std'],
                        alpha=0.2, color='#3498db')

        # Baseline reference (Before stage)
        baseline = stats_df[stats_df['stage'] == 'Before']['mean'].values[0]
        ax.axhline(baseline, color='red', linestyle='--', linewidth=2,
                  alpha=0.7, label=f'Baseline: {baseline:.3f}')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(stats_df['stage'], rotation=45, ha='right')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{ylabel} Mean Progression', fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Add percentage change annotations
        for i in range(1, len(stats_df)):
            change_pct = ((stats_df.iloc[i]['mean'] - baseline) / baseline * 100) if baseline != 0 else 0
            color = 'green' if (better == 'higher' and change_pct > 0) or (better == 'lower' and change_pct < 0) else 'red'

            ax.annotate(f'{change_pct:+.1f}%',
                       xy=(i, stats_df.iloc[i]['mean']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, color=color, fontweight='bold')

    plt.suptitle('Recovery Stage Mean Progression (with Std Dev)',
                 fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(figures_dir / "recovery_stages_progression.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: recovery_stages_progression.png")

    # ========================================
    # Figure 3: Detailed timeline with stages
    # ========================================
    fig, axes = plt.subplots(4, 1, figsize=(18, 16), sharex=True)

    # Process first few clusters for detailed view
    for cluster_id in combined['cluster_id'].unique()[:3]:
        cluster_df = combined[combined['cluster_id'] == cluster_id].copy()

        # Calculate relative time
        liq_ts = int(clusters[cluster_id]['start_ts'])
        cluster_df['rel_time'] = (cluster_df['timestamp'] - liq_ts) / 1_000_000
        cluster_df = cluster_df.sort_values('rel_time')

        color = plt.cm.tab10(cluster_id)

        # Plot each metric
        axes[0].plot(cluster_df['rel_time'], cluster_df['spread_bps'],
                    color=color, alpha=0.6, linewidth=1.5, label=f'Cluster {cluster_id}')
        axes[1].plot(cluster_df['rel_time'], cluster_df['depth_50bps_total'],
                    color=color, alpha=0.6, linewidth=1.5)
        axes[2].plot(cluster_df['rel_time'], cluster_df['order_imbalance'],
                    color=color, alpha=0.6, linewidth=1.5)
        axes[3].plot(cluster_df['rel_time'], cluster_df['stability_score'],
                    color=color, alpha=0.6, linewidth=1.5)

    # Add stage boundaries and labels
    stage_boundaries = [0, 5, 15, 30, 60]
    stage_labels = ['Immediate\n(0-5s)', 'Short\n(5-15s)', 'Medium\n(15-30s)', 'Long\n(30-60s)']
    stage_colors_fill = ['#e74c3c', '#f39c12', '#27ae60', '#9b59b6']

    for ax in axes:
        # Before period
        ax.axvspan(-60, 0, alpha=0.15, color='#3498db', label='Before')

        # After stages
        for i in range(len(stage_boundaries)-1):
            ax.axvspan(stage_boundaries[i], stage_boundaries[i+1],
                      alpha=0.1, color=stage_colors_fill[i])

        ax.axvline(0, color='red', linestyle='--', linewidth=2.5,
                  alpha=0.8, label='Liquidation', zorder=10)

    # Add stage labels at top
    for i, (start, end, label) in enumerate(zip(stage_boundaries[:-1],
                                                 stage_boundaries[1:],
                                                 stage_labels)):
        mid = (start + end) / 2
        axes[0].text(mid, axes[0].get_ylim()[1] * 0.95, label,
                    ha='center', va='top', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor=stage_colors_fill[i],
                             alpha=0.3))

    axes[0].set_ylabel('Spread (bps)', fontsize=12)
    axes[0].set_title('Spread Timeline with Recovery Stages', fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper right')

    axes[1].set_ylabel('Depth (BTC)', fontsize=12)
    axes[1].set_title('Market Depth Timeline', fontsize=13, fontweight='bold')

    axes[2].set_ylabel('Order Imbalance', fontsize=12)
    axes[2].set_title('Order Imbalance Timeline', fontsize=13, fontweight='bold')
    axes[2].axhline(0, color='gray', linestyle=':', alpha=0.5)

    axes[3].set_ylabel('Stability Score', fontsize=12)
    axes[3].set_title('Stability Score Timeline', fontsize=13, fontweight='bold')
    axes[3].set_xlabel('Time Relative to Liquidation (seconds)', fontsize=12)

    plt.suptitle('Detailed Recovery Timeline by Stage',
                 fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(figures_dir / "recovery_stages_timeline.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: recovery_stages_timeline.png")

    # ========================================
    # Print summary statistics
    # ========================================
    print("\n" + "=" * 80)
    print("RECOVERY STAGE ANALYSIS")
    print("=" * 80)

    for metric_name, metric_col in [('Spread (bps)', 'spread_bps'),
                                     ('Depth (BTC)', 'depth_50bps_total'),
                                     ('Order Imbalance', 'order_imbalance'),
                                     ('Stability Score', 'stability_score')]:
        print(f"\n{metric_name}:")
        print("┌─────────────────────┬──────────┬──────────┬──────────┬──────────┐")
        print("│ Stage               │   Mean   │   Std    │  Count   │ Change%  │")
        print("├─────────────────────┼──────────┼──────────┼──────────┼──────────┤")

        baseline = None
        for stage in stage_order:
            stage_df = combined[combined['stage'] == stage]
            if len(stage_df) > 0:
                if metric_col == 'order_imbalance':
                    mean_val = stage_df[metric_col].abs().mean()
                    std_val = stage_df[metric_col].abs().std()
                else:
                    mean_val = stage_df[metric_col].mean()
                    std_val = stage_df[metric_col].std()

                if baseline is None:
                    baseline = mean_val
                    change_pct = 0
                else:
                    change_pct = ((mean_val - baseline) / baseline * 100) if baseline != 0 else 0

                print(f"│ {stage:<19} │ {mean_val:>8.4f} │ {std_val:>8.4f} │ {len(stage_df):>8} │ {change_pct:>+7.1f}% │")

        print("└─────────────────────┴──────────┴──────────┴──────────┴──────────┘")


def main():
    output_dir = "../../output/phase1"

    print("Loading data...")
    metrics_df, liq_summary = load_data(output_dir)

    # Find clusters in time range
    min_ts = metrics_df['timestamp'].min()
    max_ts = metrics_df['timestamp'].max()

    clusters = []
    for cluster in liq_summary.get('large_cluster_details', []):
        start_ts = int(cluster['start_ts'])
        if min_ts <= start_ts <= max_ts:
            clusters.append(cluster)

    print(f"Found {len(clusters)} clusters in time range")

    if clusters:
        print("\nGenerating recovery stage visualizations...")
        plot_recovery_stages(metrics_df, clusters, output_dir)
    else:
        print("No clusters found in metrics time range")

    print("\nDone!")


if __name__ == '__main__':
    main()

