#!/usr/bin/env python3
"""
Visualize Liquidation Impact on Orderbook Stability
====================================================

Creates visualizations comparing orderbook stability before and after
large liquidation events.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def load_data(output_dir: str):
    """Load all necessary data files."""
    output_dir = Path(output_dir)

    # Load orderbook metrics
    metrics_df = pd.read_csv(output_dir / "orderbook_metrics.csv")
    metrics_df = metrics_df.sort_values('timestamp').reset_index(drop=True)

    # Load liquidation summary
    with open(output_dir / "liquidation_summary.json", 'r') as f:
        liq_summary = json.load(f)

    # Load liquidation report for individual events
    liq_report = pd.read_csv(output_dir / "liquidation_report.csv")

    return metrics_df, liq_summary, liq_report


def find_large_liquidations_in_range(liq_summary, metrics_df):
    """Find large liquidation clusters within the metrics time range."""
    min_ts = metrics_df['timestamp'].min()
    max_ts = metrics_df['timestamp'].max()

    clusters_in_range = []
    for cluster in liq_summary.get('large_cluster_details', []):
        start_ts = int(cluster['start_ts'])
        if min_ts <= start_ts <= max_ts:
            clusters_in_range.append(cluster)

    return clusters_in_range


def get_metrics_around_liquidation(
    metrics_df,
    liquidation_ts,
    before_sec=60,
    after_sec=60
):
    """Get metrics before and after a liquidation event."""
    before_window = before_sec * 1_000_000
    after_window = after_sec * 1_000_000

    before_mask = (
        (metrics_df['timestamp'] >= liquidation_ts - before_window) &
        (metrics_df['timestamp'] < liquidation_ts)
    )
    after_mask = (
        (metrics_df['timestamp'] > liquidation_ts) &
        (metrics_df['timestamp'] <= liquidation_ts + after_window)
    )

    before_df = metrics_df[before_mask].copy()
    after_df = metrics_df[after_mask].copy()

    # Add relative time
    before_df['relative_time_sec'] = (before_df['timestamp'] - liquidation_ts) / 1_000_000
    after_df['relative_time_sec'] = (after_df['timestamp'] - liquidation_ts) / 1_000_000

    return before_df, after_df


def plot_liquidation_impact_comparison(metrics_df, clusters, output_dir):
    """Create comparison plots for liquidation impact."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    if not clusters:
        print("No clusters found in metrics time range")
        return

    # Collect all before/after data
    all_before = []
    all_after = []

    for i, cluster in enumerate(clusters[:10]):  # Limit to first 10
        start_ts = int(cluster['start_ts'])
        before_df, after_df = get_metrics_around_liquidation(metrics_df, start_ts)

        if len(before_df) > 0 and len(after_df) > 0:
            before_df['cluster_id'] = i
            after_df['cluster_id'] = i
            before_df['period'] = 'Before'
            after_df['period'] = 'After'
            all_before.append(before_df)
            all_after.append(after_df)

    if not all_before:
        print("No data found around liquidation events")
        return

    before_combined = pd.concat(all_before, ignore_index=True)
    after_combined = pd.concat(all_after, ignore_index=True)
    combined = pd.concat([before_combined, after_combined], ignore_index=True)

    # ========================================
    # Figure 1: Box plot comparison of key metrics
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Spread comparison
    ax = axes[0, 0]
    sns.boxplot(data=combined, x='period', y='spread_bps', ax=ax,
                palette=['#3498db', '#e74c3c'], order=['Before', 'After'])
    ax.set_ylabel('Spread (bps)')
    ax.set_xlabel('')
    ax.set_title('Bid-Ask Spread: Before vs After Liquidation')

    # Add statistics annotation
    before_mean = before_combined['spread_bps'].mean()
    after_mean = after_combined['spread_bps'].mean()
    change_pct = ((after_mean - before_mean) / before_mean * 100) if before_mean > 0 else 0
    ax.annotate(f'Before: {before_mean:.3f} bps\nAfter: {after_mean:.3f} bps\nChange: {change_pct:+.1f}%',
                xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Market Depth comparison
    ax = axes[0, 1]
    sns.boxplot(data=combined, x='period', y='depth_50bps_total', ax=ax,
                palette=['#3498db', '#e74c3c'], order=['Before', 'After'])
    ax.set_ylabel('Market Depth (BTC within 50 bps)')
    ax.set_xlabel('')
    ax.set_title('Market Depth: Before vs After Liquidation')

    before_mean = before_combined['depth_50bps_total'].mean()
    after_mean = after_combined['depth_50bps_total'].mean()
    change_pct = ((after_mean - before_mean) / before_mean * 100) if before_mean > 0 else 0
    ax.annotate(f'Before: {before_mean:.2f} BTC\nAfter: {after_mean:.2f} BTC\nChange: {change_pct:+.1f}%',
                xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Order Imbalance comparison
    ax = axes[1, 0]
    sns.boxplot(data=combined, x='period', y='order_imbalance', ax=ax,
                palette=['#3498db', '#e74c3c'], order=['Before', 'After'])
    ax.set_ylabel('Order Imbalance')
    ax.set_xlabel('')
    ax.set_title('Order Imbalance: Before vs After Liquidation')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    before_mean = before_combined['order_imbalance'].mean()
    after_mean = after_combined['order_imbalance'].mean()
    ax.annotate(f'Before: {before_mean:+.3f}\nAfter: {after_mean:+.3f}',
                xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Stability Score comparison
    ax = axes[1, 1]
    sns.boxplot(data=combined, x='period', y='stability_score', ax=ax,
                palette=['#3498db', '#e74c3c'], order=['Before', 'After'])
    ax.set_ylabel('Stability Score')
    ax.set_xlabel('')
    ax.set_title('Stability Score: Before vs After Liquidation')

    before_mean = before_combined['stability_score'].mean()
    after_mean = after_combined['stability_score'].mean()
    change_pct = ((after_mean - before_mean) / before_mean * 100) if before_mean > 0 else 0
    ax.annotate(f'Before: {before_mean:.3f}\nAfter: {after_mean:.3f}\nChange: {change_pct:+.1f}%',
                xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Orderbook Stability Comparison: Before vs After Large Liquidations',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "liquidation_impact_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir / 'liquidation_impact_boxplot.png'}")

    # ========================================
    # Figure 2: Time series around liquidation events
    # ========================================
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

    # Plot each cluster's data
    colors = plt.cm.tab10(np.linspace(0, 1, len(set(combined['cluster_id']))))

    for cluster_id in combined['cluster_id'].unique():
        cluster_data = combined[combined['cluster_id'] == cluster_id].sort_values('relative_time_sec')
        color = colors[cluster_id % len(colors)]
        alpha = 0.6

        # Spread
        axes[0].plot(cluster_data['relative_time_sec'], cluster_data['spread_bps'],
                    color=color, alpha=alpha, linewidth=1, label=f'Cluster {cluster_id}')

        # Depth
        axes[1].plot(cluster_data['relative_time_sec'], cluster_data['depth_50bps_total'],
                    color=color, alpha=alpha, linewidth=1)

        # Imbalance
        axes[2].plot(cluster_data['relative_time_sec'], cluster_data['order_imbalance'],
                    color=color, alpha=alpha, linewidth=1)

        # Stability
        axes[3].plot(cluster_data['relative_time_sec'], cluster_data['stability_score'],
                    color=color, alpha=alpha, linewidth=1)

    # Add vertical line at liquidation time
    for ax in axes:
        ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Liquidation')
        ax.axvspan(-60, 0, alpha=0.1, color='blue', label='Before')
        ax.axvspan(0, 60, alpha=0.1, color='red', label='After')

    axes[0].set_ylabel('Spread (bps)')
    axes[0].set_title('Bid-Ask Spread Around Liquidation Events')
    axes[0].legend(loc='upper right', fontsize=8, ncol=3)

    axes[1].set_ylabel('Depth (BTC)')
    axes[1].set_title('Market Depth Around Liquidation Events')

    axes[2].set_ylabel('Order Imbalance')
    axes[2].set_title('Order Imbalance Around Liquidation Events')
    axes[2].axhline(0, color='gray', linestyle=':', alpha=0.5)

    axes[3].set_ylabel('Stability Score')
    axes[3].set_title('Stability Score Around Liquidation Events')
    axes[3].set_xlabel('Time Relative to Liquidation (seconds)')

    plt.suptitle('Orderbook Metrics Time Series Around Large Liquidations',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(figures_dir / "liquidation_impact_timeseries.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir / 'liquidation_impact_timeseries.png'}")

    # ========================================
    # Figure 3: Distribution comparison (KDE plots)
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Spread distribution
    ax = axes[0, 0]
    sns.kdeplot(data=before_combined, x='spread_bps', ax=ax, color='#3498db',
                fill=True, alpha=0.3, label='Before Liquidation')
    sns.kdeplot(data=after_combined, x='spread_bps', ax=ax, color='#e74c3c',
                fill=True, alpha=0.3, label='After Liquidation')
    ax.set_xlabel('Spread (bps)')
    ax.set_title('Spread Distribution: Before vs After')
    ax.legend()

    # Depth distribution
    ax = axes[0, 1]
    sns.kdeplot(data=before_combined, x='depth_50bps_total', ax=ax, color='#3498db',
                fill=True, alpha=0.3, label='Before Liquidation')
    sns.kdeplot(data=after_combined, x='depth_50bps_total', ax=ax, color='#e74c3c',
                fill=True, alpha=0.3, label='After Liquidation')
    ax.set_xlabel('Market Depth (BTC)')
    ax.set_title('Depth Distribution: Before vs After')
    ax.legend()

    # Imbalance distribution
    ax = axes[1, 0]
    sns.kdeplot(data=before_combined, x='order_imbalance', ax=ax, color='#3498db',
                fill=True, alpha=0.3, label='Before Liquidation')
    sns.kdeplot(data=after_combined, x='order_imbalance', ax=ax, color='#e74c3c',
                fill=True, alpha=0.3, label='After Liquidation')
    ax.set_xlabel('Order Imbalance')
    ax.set_title('Imbalance Distribution: Before vs After')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.legend()

    # Stability Score distribution
    ax = axes[1, 1]
    sns.kdeplot(data=before_combined, x='stability_score', ax=ax, color='#3498db',
                fill=True, alpha=0.3, label='Before Liquidation')
    sns.kdeplot(data=after_combined, x='stability_score', ax=ax, color='#e74c3c',
                fill=True, alpha=0.3, label='After Liquidation')
    ax.set_xlabel('Stability Score')
    ax.set_title('Stability Score Distribution: Before vs After')
    ax.legend()

    plt.suptitle('Metric Distribution Comparison: Before vs After Liquidations',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "liquidation_impact_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir / 'liquidation_impact_distributions.png'}")

    # ========================================
    # Figure 4: Summary statistics bar chart
    # ========================================
    fig, ax = plt.subplots(figsize=(14, 8))

    metrics = ['Spread (bps)', 'Depth (BTC)', 'Order Imbalance', 'Stability Score']
    before_means = [
        before_combined['spread_bps'].mean(),
        before_combined['depth_50bps_total'].mean(),
        abs(before_combined['order_imbalance'].mean()),
        before_combined['stability_score'].mean()
    ]
    after_means = [
        after_combined['spread_bps'].mean(),
        after_combined['depth_50bps_total'].mean(),
        abs(after_combined['order_imbalance'].mean()),
        after_combined['stability_score'].mean()
    ]

    # Normalize for comparison
    before_norm = [b / max(b, a) if max(b, a) > 0 else 0 for b, a in zip(before_means, after_means)]
    after_norm = [a / max(b, a) if max(b, a) > 0 else 0 for b, a in zip(before_means, after_means)]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, before_means, width, label='Before Liquidation', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, after_means, width, label='After Liquidation', color='#e74c3c', alpha=0.8)

    ax.set_ylabel('Value')
    ax.set_title('Average Orderbook Metrics: Before vs After Large Liquidations')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Add value labels on bars
    def add_labels(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

    add_labels(bars1, before_means)
    add_labels(bars2, after_means)

    plt.tight_layout()
    plt.savefig(figures_dir / "liquidation_impact_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir / 'liquidation_impact_summary.png'}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"\nData points - Before: {len(before_combined)}, After: {len(after_combined)}")
    print(f"Number of liquidation clusters analyzed: {len(set(combined['cluster_id']))}")

    print("\n┌────────────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ Metric             │    Before    │    After     │   Change %   │")
    print("├────────────────────┼──────────────┼──────────────┼──────────────┤")

    for metric, before, after in zip(metrics, before_means, after_means):
        change = ((after - before) / before * 100) if before != 0 else 0
        print(f"│ {metric:<18} │ {before:>12.4f} │ {after:>12.4f} │ {change:>+11.1f}% │")

    print("└────────────────────┴──────────────┴──────────────┴──────────────┘")


def main():
    output_dir = "../../output/phase1"

    print("Loading data...")
    metrics_df, liq_summary, liq_report = load_data(output_dir)

    print(f"Metrics time range: {metrics_df['timestamp'].min()} - {metrics_df['timestamp'].max()}")
    print(f"Total large clusters: {len(liq_summary.get('large_cluster_details', []))}")

    # Find clusters within time range
    clusters = find_large_liquidations_in_range(liq_summary, metrics_df)
    print(f"Clusters in metrics time range: {len(clusters)}")

    print("\nGenerating visualizations...")
    plot_liquidation_impact_comparison(metrics_df, clusters, output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()

