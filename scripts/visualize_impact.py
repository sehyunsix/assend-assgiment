import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.analysis.liquidation_analyzer import LiquidationAnalyzer
from src.analysis.impact_analyzer import LiquidationImpactAnalyzer

def visualize_liquidation_impact():
    output_dir = Path("output/phase1")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    metrics_df = pd.read_csv("output/phase1/orderbook_metrics.csv")
    if 'mid_price' not in metrics_df.columns:
        metrics_df['mid_price'] = 100000.0 # Dummy price
    
    # liquidation_report.csv might be simplified, let's load it and add dummy 'side' if missing
    liq_df = pd.read_csv("output/phase1/liquidation_report.csv")
    if 'side' not in liq_df.columns:
        liq_df['side'] = 'sell' # Default to sell for visualization
    
    # Initialize Analyzers
    liq_analyzer = LiquidationAnalyzer(liq_df)
    clusters = liq_analyzer.cluster_liquidations(time_window_us=5_000_000)
    
    # Convert clusters to dict format expected by ImpactAnalyzer
    cluster_dicts = [
        {
            'start_ts': c.start_timestamp,
            'end_ts': c.end_timestamp,
            'total_value': c.total_value,
            'event_count': c.event_count,
            'dominant_side': c.dominant_side
        } for c in clusters if c.total_value > 100000 # Only significant ones
    ]
    
    if not cluster_dicts:
        print("No significant clusters found for visualization.")
        return

    impact_analyzer = LiquidationImpactAnalyzer(
        orderbook_metrics=metrics_df,
        liquidation_clusters=cluster_dicts,
        before_window_us=30_000_000, # 30s
        after_window_us=120_000_000, # 120s
        recovery_threshold_pct=1.2    # 20% above baseline
    )
    
    analyses = impact_analyzer.analyze_all_clusters()
    if not analyses:
        print("No successful impact analyses.")
        return
    
    # Pick the most impactful cluster (highest spread change)
    target = max(analyses, key=lambda a: a.spread_change_pct)
    
    # Setup Plot Styling
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 12})
    sns.set_theme(style="whitegrid", palette="muted")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    
    # Window for plotting
    plot_start = target.start_ts - 30_000_000
    plot_end = target.end_ts + 120_000_000
    
    mask = (metrics_df['timestamp'] >= plot_start) & (metrics_df['timestamp'] <= plot_end)
    plot_df = metrics_df[mask].copy()
    plot_df['time_sec'] = (plot_df['timestamp'] - target.start_ts) / 1_000_000
    
    # 1. Spread Plot
    axes[0].plot(plot_df['time_sec'], plot_df['spread_bps'], label='Spread (bps)', color='#D32F2F', linewidth=2)
    axes[0].fill_between(plot_df['time_sec'], plot_df['spread_bps'], target.before_spread_bps_mean, 
                         where=(plot_df['spread_bps'] > target.before_spread_bps_mean), color='#D32F2F', alpha=0.1)
    
    axes[0].axvspan(0, (target.end_ts - target.start_ts)/1_000_000, color='#9E9E9E', alpha=0.2, label='Liquidation Event')
    axes[0].axhline(target.before_spread_bps_mean, color='#388E3C', linestyle='--', linewidth=2, label='Baseline')
    
    if target.recovery_time_us:
        recovery_sec = target.recovery_time_us/1_000_000
        axes[0].axvline(recovery_sec, color='#1976D2', linestyle=':', linewidth=2, label=f'Recovery ({recovery_sec:.1f}s)')
        # Annotation for recovery
        axes[0].annotate('Stabilized', xy=(recovery_sec, target.before_spread_bps_mean), xytext=(recovery_sec+5, target.before_spread_bps_mean+5),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    axes[0].set_ylabel('Spread (bps)')
    axes[0].set_title(f"Market Resilience Profile (Liq: ${target.total_value:,.0f})", fontweight='bold', pad=20)
    axes[0].legend(loc='upper right', frameon=True, shadow=True)

    # 2. Depth Plot
    axes[1].plot(plot_df['time_sec'], plot_df['depth_50bps_total'], label='Depth (BTC)', color='#388E3C', linewidth=2)
    axes[1].axvspan(0, (target.end_ts - target.start_ts)/1_000_000, color='#9E9E9E', alpha=0.2)
    axes[1].set_ylabel('Depth (BTC)')
    axes[1].legend(loc='upper right', frameon=True)

    # 3. Imbalance Plot
    axes[2].plot(plot_df['time_sec'], plot_df['order_imbalance'], label='Imbalance', color='#0288D1', linewidth=2)
    axes[2].axvspan(0, (target.end_ts - target.start_ts)/1_000_000, color='#9E9E9E', alpha=0.2)
    axes[2].axhline(0, color='black', linewidth=1, alpha=0.5)
    axes[2].set_ylabel('Imbalance α')
    axes[2].set_xlabel('Time from Liquidation Start (seconds)')
    axes[2].legend(loc='upper right', frameon=True)

    plt.tight_layout()
    save_path = output_dir / "liquidation_impact.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Impact visualization saved to {save_path}")

    # --- High Resolution Microsecond Plot ---
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    high_res_window_us = 5_000_000 
    hr_plot_start = target.end_ts - 500_000 
    hr_plot_end = target.end_ts + high_res_window_us
    
    hr_mask = (metrics_df['timestamp'] >= hr_plot_start) & (metrics_df['timestamp'] <= hr_plot_end)
    hr_df = metrics_df[hr_mask].copy()
    hr_df['time_ms'] = (hr_df['timestamp'] - target.end_ts) / 1000
    
    # 1. Spread (Micro)
    axes2[0].step(hr_df['time_ms'], hr_df['spread_bps'], where='post', color='#D32F2F', linewidth=2, label='Ticks (Step)')
    axes2[0].axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.6, label='Liquidation End')
    axes2[0].set_ylabel('Spread (bps)')
    axes2[0].set_title(f"Micro-Stability Analysis (Zoom: 5000ms Post-Event)", fontweight='bold', pad=20)
    axes2[0].legend(loc='upper right', frameon=True)

    # 2. Depth (Micro)
    axes2[1].step(hr_df['time_ms'], hr_df['depth_50bps_total'], where='post', color='#388E3C', linewidth=2)
    axes2[1].axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.6)
    axes2[1].set_ylabel('Depth (BTC)')

    # 3. Imbalance (Micro)
    axes2[2].step(hr_df['time_ms'], hr_df['order_imbalance'], where='post', color='#0288D1', linewidth=2)
    axes2[2].axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.6)
    axes2[2].set_ylabel('α Imbalance')
    axes2[2].set_xlabel('Time from Impact (milliseconds)')

    plt.tight_layout()
    hr_save_path = output_dir / "liquidation_impact_micro.png"
    plt.savefig(hr_save_path, dpi=150, bbox_inches='tight')
    print(f"Microsecond impact visualization saved to {hr_save_path}")

if __name__ == "__main__":
    visualize_liquidation_impact()
