#!/usr/bin/env python3
"""
Ultra Micro-level Recovery Analysis
====================================

Analyzes orderbook recovery in microsecond/millisecond granularity.
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
plt.rcParams['figure.figsize'] = (22, 16)
plt.rcParams['font.size'] = 10


def load_data(output_dir: str):
    """Load data."""
    output_dir = Path(output_dir)

    metrics_df = pd.read_csv(output_dir / "orderbook_metrics.csv")
    metrics_df = metrics_df.sort_values('timestamp').reset_index(drop=True)

    with open(output_dir / "liquidation_summary.json", 'r') as f:
        liq_summary = json.load(f)

    return metrics_df, liq_summary


def get_ultra_micro_stages(metrics_df, liquidation_ts, ultra_stages):
    """Get metrics for ultra-fine time stages (millisecond level)."""
    stage_data = {}

    # Before period (1 second before)
    before_window = 1_000_000  # 1 second in microseconds
    before_mask = (
        (metrics_df['timestamp'] >= liquidation_ts - before_window) &
        (metrics_df['timestamp'] < liquidation_ts)
    )
    stage_data['Before'] = metrics_df[before_mask].copy()
    stage_data['Before']['stage'] = 'Before'

    # Ultra micro stages
    for start_ms, end_ms, label in ultra_stages:
        start_us = start_ms * 1000  # Convert ms to μs
        end_us = end_ms * 1000

        mask = (
            (metrics_df['timestamp'] >= liquidation_ts + start_us) &
            (metrics_df['timestamp'] < liquidation_ts + end_us)
        )
        stage_df = metrics_df[mask].copy()
        stage_df['stage'] = label
        stage_data[label] = stage_df

    return stage_data


def plot_ultra_micro_recovery(metrics_df, clusters, output_dir):
    """Plot ultra-micro level recovery."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"

    if not clusters:
        print("No clusters found")
        return

    # Define ultra-micro stages in milliseconds
    ultra_stages = [
        (0, 10, '0-10ms'),          # 0-10 milliseconds
        (10, 50, '10-50ms'),        # 10-50 ms
        (50, 100, '50-100ms'),      # 50-100 ms
        (100, 200, '100-200ms'),    # 100-200 ms (0.1-0.2s)
        (200, 500, '200-500ms'),    # 200-500 ms (0.2-0.5s)
        (500, 1000, '500ms-1s'),    # 0.5-1 second
        (1000, 2000, '1-2s'),       # 1-2 seconds
        (2000, 5000, '2-5s'),       # 2-5 seconds
        (5000, 10000, '5-10s'),     # 5-10 seconds
    ]

    # Collect data
    all_stage_data = {stage[2]: [] for stage in ultra_stages}
    all_stage_data['Before'] = []

    for i, cluster in enumerate(clusters[:10]):
        start_ts = int(cluster['start_ts'])
        stage_data = get_ultra_micro_stages(metrics_df, start_ts, ultra_stages)

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
    stage_order = ['Before'] + [s[2] for s in ultra_stages]

    # ========================================
    # Figure 1: Ultra-micro timeline
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))

    metrics_to_plot = [
        ('spread_bps', 'Spread (bps)', axes[0, 0]),
        ('depth_50bps_total', 'Market Depth (BTC)', axes[0, 1]),
        ('order_imbalance', 'Order Imbalance (abs)', axes[1, 0]),
        ('stability_score', 'Stability Score', axes[1, 1])
    ]

    for metric, ylabel, ax in metrics_to_plot:
        stage_means = []
        stage_stds = []
        stage_counts = []
        stage_labels_clean = []

        for stage in stage_order:
            stage_df = combined[combined['stage'] == stage]
            if len(stage_df) > 2:  # Minimum data points
                if metric == 'order_imbalance':
                    mean_val = stage_df[metric].abs().mean()
                    std_val = stage_df[metric].abs().std()
                else:
                    mean_val = stage_df[metric].mean()
                    std_val = stage_df[metric].std()

                stage_means.append(mean_val)
                stage_stds.append(std_val if not pd.isna(std_val) else 0)
                stage_counts.append(len(stage_df))
                stage_labels_clean.append(stage)

        if not stage_means:
            continue

        x_pos = np.arange(len(stage_means))

        # Main line plot
        ax.plot(x_pos, stage_means, 'o-', linewidth=3, markersize=10,
               color='#2c3e50', markerfacecolor='#e74c3c',
               markeredgewidth=2, markeredgecolor='#2c3e50')

        # Error bars
        ax.errorbar(x_pos, stage_means, yerr=stage_stds,
                   fmt='none', ecolor='#95a5a6', capsize=8, capthick=2,
                   alpha=0.6, zorder=1)

        # Fill area
        upper = np.array(stage_means) + np.array(stage_stds)
        lower = np.array(stage_means) - np.array(stage_stds)
        ax.fill_between(x_pos, lower, upper, alpha=0.15, color='#3498db')

        # Baseline
        baseline = stage_means[0]
        ax.axhline(baseline, color='red', linestyle='--', linewidth=2.5,
                  alpha=0.7, label=f'Baseline: {baseline:.3f}', zorder=0)

        # Liquidation marker
        ax.axvline(0.5, color='red', linestyle=':', linewidth=3, alpha=0.7, zorder=0)
        ax.text(0.5, ax.get_ylim()[1] * 0.97, 'LIQUIDATION ↓',
               ha='center', va='top', fontsize=12, color='red',
               fontweight='bold', rotation=0,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow',
                        edgecolor='red', alpha=0.8, linewidth=2))

        # Millisecond zones
        ms_boundary = None
        for i, label in enumerate(stage_labels_clean):
            if label == '500ms-1s' and ms_boundary is None:
                ms_boundary = i

        if ms_boundary:
            ax.axvspan(-0.5, ms_boundary - 0.5, alpha=0.08, color='orange',
                      label='Sub-second zone')

        # Annotate key changes
        for i in range(1, len(stage_means)):
            if i <= 3:  # Focus on first few stages
                change_pct = ((stage_means[i] - baseline) / baseline * 100) if baseline != 0 else 0

                ax.annotate(f'{change_pct:+.1f}%',
                           xy=(i, stage_means[i]),
                           xytext=(0, 15), textcoords='offset points',
                           fontsize=10, color='darkgreen' if change_pct < 0 else 'darkred',
                           fontweight='bold', ha='center',
                           bbox=dict(boxstyle='round,pad=0.4',
                                   facecolor='white', edgecolor='gray', alpha=0.9))

        ax.set_xticks(x_pos)
        ax.set_xticklabels(stage_labels_clean, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_title(f'{ylabel} - Ultra-Micro Recovery (ms to seconds)',
                    fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Secondary axis for data counts
        ax2 = ax.twinx()
        ax2.bar(x_pos, stage_counts, alpha=0.12, color='gray', width=0.7)
        ax2.set_ylabel('Sample Count', fontsize=11, color='gray', style='italic')
        ax2.tick_params(axis='y', labelcolor='gray')

        # Add text for key millisecond stages
        for i, (label, count) in enumerate(zip(stage_labels_clean, stage_counts)):
            if 'ms' in label and count > 0:
                ax2.text(i, count, f'n={count}', ha='center', va='bottom',
                        fontsize=8, color='gray', alpha=0.7)

    plt.suptitle('Ultra-Micro Recovery Analysis: Millisecond to Second Granularity After Liquidation',
                 fontsize=20, fontweight='bold', y=0.998)
    plt.tight_layout()
    plt.savefig(figures_dir / "ultra_micro_recovery.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: ultra_micro_recovery.png")

    # ========================================
    # Figure 2: Millisecond-focused zoom
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    # Filter to only millisecond stages
    ms_stages = [s for s in stage_order if 'ms' in s or s == 'Before' or s == '500ms-1s']

    for metric, ylabel, ax in metrics_to_plot:
        stage_means = []
        stage_labels_clean = []

        for stage in ms_stages:
            stage_df = combined[combined['stage'] == stage]
            if len(stage_df) > 0:
                if metric == 'order_imbalance':
                    mean_val = stage_df[metric].abs().mean()
                else:
                    mean_val = stage_df[metric].mean()

                stage_means.append(mean_val)
                stage_labels_clean.append(stage)

        if len(stage_means) <= 1:
            continue

        x_pos = np.arange(len(stage_means))

        # Create gradient colors
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(stage_means)))
        colors[0] = [0.2, 0.4, 0.8, 1.0]  # Blue for "Before"

        # Bar plot
        bars = ax.bar(x_pos, stage_means, color=colors, edgecolor='black',
                     linewidth=1.5, alpha=0.8)

        # Baseline line
        baseline = stage_means[0]
        ax.axhline(baseline, color='red', linestyle='--', linewidth=3,
                  alpha=0.8, label=f'Baseline: {baseline:.3f}')

        # Value labels on bars
        for i, (bar, val) in enumerate(zip(bars, stage_means)):
            height = bar.get_height()
            change_pct = ((val - baseline) / baseline * 100) if baseline != 0 and i > 0 else 0

            if i > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}\n({change_pct:+.1f}%)',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}\n(baseline)',
                       ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(stage_labels_clean, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.set_title(f'{ylabel} in Millisecond Range (< 1 second)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Millisecond-Level Focus: First Second After Liquidation',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(figures_dir / "millisecond_focus.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: millisecond_focus.png")

    # ========================================
    # Print statistics
    # ========================================
    print("\n" + "=" * 120)
    print("ULTRA-MICRO LEVEL RECOVERY ANALYSIS (Millisecond → Second)")
    print("=" * 120)

    for metric_name, metric_col in [('Spread (bps)', 'spread_bps'),
                                     ('Market Depth (BTC)', 'depth_50bps_total'),
                                     ('Order Imbalance (abs)', 'order_imbalance'),
                                     ('Stability Score', 'stability_score')]:
        print(f"\n{metric_name}:")
        print("┌──────────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
        print("│    Stage     │   Mean   │   Std    │   Min    │   Max    │  Count   │ Change%  │")
        print("├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")

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
                count = len(stage_df)

                if baseline is None:
                    baseline = mean_val
                    change_pct = 0
                else:
                    change_pct = ((mean_val - baseline) / baseline * 100) if baseline != 0 else 0

                # Highlight millisecond stages
                stage_display = f"🔬{stage}" if 'ms' in stage else stage

                print(f"│ {stage_display:>12} │ {mean_val:>8.4f} │ {std_val:>8.4f} │ {min_val:>8.4f} │ {max_val:>8.4f} │ {count:>8} │ {change_pct:>+7.1f}% │")

        print("└──────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")

    # ========================================
    # Millisecond breakdown summary
    # ========================================
    print("\n" + "=" * 120)
    print("MILLISECOND BREAKDOWN SUMMARY")
    print("=" * 120)

    print("\n⚡ IMMEDIATE RESPONSE (0-100ms):")
    print("┌─────────────────────┬────────────────┬────────────────┬────────────────┬────────────────┐")
    print("│ Time Window         │ Spread Change  │  Depth Change  │ Imbal. Change  │ Stabil. Change │")
    print("├─────────────────────┼────────────────┼────────────────┼────────────────┼────────────────┤")

    ms_windows = ['0-10ms', '10-50ms', '50-100ms']
    baseline_vals = {}

    # Get baselines
    before_df = combined[combined['stage'] == 'Before']
    if len(before_df) > 0:
        baseline_vals['spread_bps'] = before_df['spread_bps'].mean()
        baseline_vals['depth_50bps_total'] = before_df['depth_50bps_total'].mean()
        baseline_vals['order_imbalance'] = before_df['order_imbalance'].abs().mean()
        baseline_vals['stability_score'] = before_df['stability_score'].mean()

    for window in ms_windows:
        window_df = combined[combined['stage'] == window]
        if len(window_df) > 0:
            spread_change = ((window_df['spread_bps'].mean() - baseline_vals['spread_bps']) /
                           baseline_vals['spread_bps'] * 100) if baseline_vals['spread_bps'] != 0 else 0
            depth_change = ((window_df['depth_50bps_total'].mean() - baseline_vals['depth_50bps_total']) /
                          baseline_vals['depth_50bps_total'] * 100) if baseline_vals['depth_50bps_total'] != 0 else 0
            imbal_change = ((window_df['order_imbalance'].abs().mean() - baseline_vals['order_imbalance']) /
                          baseline_vals['order_imbalance'] * 100) if baseline_vals['order_imbalance'] != 0 else 0
            stab_change = ((window_df['stability_score'].mean() - baseline_vals['stability_score']) /
                         baseline_vals['stability_score'] * 100) if baseline_vals['stability_score'] != 0 else 0

            print(f"│ {window:>19} │ {spread_change:>+12.1f}% │ {depth_change:>+12.1f}% │ {imbal_change:>+12.1f}% │ {stab_change:>+12.1f}% │")

    print("└─────────────────────┴────────────────┴────────────────┴────────────────┴────────────────┘")

    print("\n🔍 Key Insight: Response time measured in MILLISECONDS after liquidation!")


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
    print(f"Data timestamp precision: microseconds (μs)")
    print(f"Analysis granularity: milliseconds (ms) to seconds (s)")

    if clusters:
        print("\nGenerating ultra-micro level visualizations...")
        plot_ultra_micro_recovery(metrics_df, clusters, output_dir)
    else:
        print("No clusters found")

    print("\nDone!")


if __name__ == '__main__':
    main()

