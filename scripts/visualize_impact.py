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
    
    # Setup Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Window for plotting
    plot_start = target.start_ts - 30_000_000
    plot_end = target.end_ts + 120_000_000
    
    mask = (metrics_df['timestamp'] >= plot_start) & (metrics_df['timestamp'] <= plot_end)
    plot_df = metrics_df[mask].copy()
    plot_df['time_sec'] = (plot_df['timestamp'] - target.start_ts) / 1_000_000
    
    # 1. Spread Plot
    axes[0].plot(plot_df['time_sec'], plot_df['spread_bps'], label='Spread (bps)', color='#e74c3c', linewidth=1.5)
    axes[0].axvspan(0, (target.end_ts - target.start_ts)/1_000_000, color='gray', alpha=0.3, label='Liquidation Cluster')
    axes[0].axhline(target.before_spread_bps_mean, color='green', linestyle='--', alpha=0.6, label='Baseline')
    if target.recovery_time_us:
        axes[0].axvline(target.recovery_time_us/1_000_000, color='blue', linestyle=':', label='Recovered')
    axes[0].set_ylabel('Spread (bps)')
    axes[0].legend(loc='upper right')
    axes[0].set_title(f"Liquidation Impact Analysis (Value: ${target.total_value:,.0f})")

    # 2. Depth Plot
    axes[1].plot(plot_df['time_sec'], plot_df['depth_50bps_total'], label='Depth (BTC)', color='#2ecc71', linewidth=1.5)
    axes[1].axvspan(0, (target.end_ts - target.start_ts)/1_000_000, color='gray', alpha=0.3)
    axes[1].set_ylabel('Depth (BTC)')
    axes[1].legend(loc='upper right')

    # 3. Imbalance Plot
    axes[2].plot(plot_df['time_sec'], plot_df['order_imbalance'], label='Imbalance', color='#3498db', linewidth=1.5)
    axes[2].axvspan(0, (target.end_ts - target.start_ts)/1_000_000, color='gray', alpha=0.3)
    axes[2].axhline(0, color='black', alpha=0.3)
    axes[2].set_ylabel('Order Imbalance')
    axes[2].set_xlabel('Time from Liquidation Start (sec)')
    axes[2].legend(loc='upper right')

    plt.tight_layout()
    save_path = output_dir / "liquidation_impact.png"
    plt.savefig(save_path, dpi=120)
    print(f"Impact visualization saved to {save_path}")

if __name__ == "__main__":
    visualize_liquidation_impact()
