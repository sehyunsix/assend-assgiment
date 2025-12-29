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
    
    # --- Aggregated Impact Visualization ---
    # We will overlay all analyzed clusters to see the "average" recovery path
    
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
    
    # Define window for aggregation (seconds)
    pre_event_sec = 20
    post_event_sec = 100
    
    all_spreads = []
    all_depths = []
    all_imbalances = []
    common_time = np.linspace(-pre_event_sec, post_event_sec, 240) # 0.5s resolution approx
    
    for adj_idx, a in enumerate(analyses):
        # Extract window for this cluster
        c_start = a.start_ts - (pre_event_sec * 1_000_000)
        c_end = a.start_ts + (post_event_sec * 1_000_000)
        
        c_mask = (metrics_df['timestamp'] >= c_start) & (metrics_df['timestamp'] <= c_end)
        c_df = metrics_df[c_mask].copy()
        if c_df.empty: continue
        
        c_df['rel_time'] = (c_df['timestamp'] - a.start_ts) / 1_000_000
        
        # Interpolate to common time grid for averaging
        s_interp = np.interp(common_time, c_df['rel_time'], c_df['spread_bps'])
        d_interp = np.interp(common_time, c_df['rel_time'], c_df['depth_50bps_total'])
        i_interp = np.interp(common_time, c_df['rel_time'], c_df['order_imbalance'])
        
        all_spreads.append(s_interp)
        all_depths.append(d_interp)
        all_imbalances.append(i_interp)
        
        # Plot individual lines (faded)
        label = "Individual Events" if adj_idx == 0 else None
        axes[0].plot(common_time, s_interp, color='red', alpha=0.15, linewidth=1, label=label)
        axes[1].plot(common_time, d_interp, color='green', alpha=0.15, linewidth=1)
        axes[2].plot(common_time, i_interp, color='blue', alpha=0.15, linewidth=1)

    # Calculate and plot averages
    avg_spread = np.mean(all_spreads, axis=0)
    avg_depth = np.mean(all_depths, axis=0)
    avg_imbalance = np.mean(all_imbalances, axis=0)
    
    axes[0].plot(common_time, avg_spread, color='#D32F2F', linewidth=3, label='Average Profile')
    axes[1].plot(common_time, avg_depth, color='#388E3C', linewidth=3, label='Average Profile')
    axes[2].plot(common_time, avg_imbalance, color='#1976D2', linewidth=3, label='Average Profile')
    
    # Annotations
    for ax in axes:
        ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvspan(0, 5, color='gray', alpha=0.1) # Highlight the impact zone
        ax.legend(loc='upper right')

    axes[0].set_title("Aggregated Liquidation Impact (Overlaid Events)", fontweight='bold')
    axes[0].set_ylabel("Spread (bps)")
    axes[1].set_ylabel("Depth (BTC)")
    axes[2].set_ylabel("Imbalance α")
    axes[2].set_xlabel("Seconds from Liquidation Start")
    
    plt.tight_layout()
    save_path = output_dir / "liquidation_impact_aggregated.png"
    plt.savefig(save_path, dpi=150)
    print(f"Aggregated impact visualization saved to {save_path}")

if __name__ == "__main__":
    visualize_liquidation_impact()
