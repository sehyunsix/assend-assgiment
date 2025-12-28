#!/usr/bin/env python3
"""
Full-scale Orderbook Analysis
=============================
Processes the entire 9.5GB orderbook.csv using Dask and performs
comprehensive liquidation impact analysis.
"""

import os
import pandas as pd
import numpy as np
import dask.dataframe as dd
from pathlib import Path
import json
import time
from tqdm import tqdm
from data_loader import DataLoader
from orderbook_metrics import OrderbookMetrics
from liquidation_analyzer import LiquidationAnalyzer

def run_full_analysis():
    base_dir = Path("/Users/yuksehyun/project/asend-assigment")
    data_dir = base_dir / "data" / "research"
    output_dir = base_dir / "output" / "phase1"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting Full-scale Analysis...")
    start_time = time.time()

    # Initialize Components
    loader = DataLoader(str(data_dir))
    metrics_calc = OrderbookMetrics()

    # 1. Load and Analyze Liquidation Clusters
    print("Loading and analyzing liquidation clusters...")
    liq_df = loader.load_liquidations()
    liq_analyzer = LiquidationAnalyzer(liq_df)

    # Identify and cluster liquidations
    clusters = liq_analyzer.cluster_liquidations(time_window_us=5_000_000) # 5s
    print(f"Found {len(clusters)} liquidation clusters in total.")

    # Categorize clusters by size (as suggested in report.md)
    categorized_clusters = {
        'small': [c for c in clusters if c.total_value < 50000],
        'medium': [c for c in clusters if 50000 <= c.total_value < 200000],
        'large': [c for c in clusters if c.total_value >= 200000]
    }
    print(f"Cluster Distribution: Small={len(categorized_clusters['small'])}, "
          f"Medium={len(categorized_clusters['medium'])}, Large={len(categorized_clusters['large'])}")

    # 2. Process Orderbook Data in Chunks using Dask
    print("Loading 9.5GB orderbook data with Dask...")
    ob_dd = loader.load_orderbook()

    # Baseline: Sample a larger portion (e.g., first 10 million rows)
    print("Calculating global baseline stats from 10M rows...")
    baseline_sample = ob_dd.head(10000000)
    baseline_metrics = metrics_calc.calculate_metrics_for_timestamps(baseline_sample)

    # Calculate stability score for baseline
    baseline_metrics['stability_score'] = OrderbookMetrics.calculate_stability_score(baseline_metrics)

    global_stats = {
        'spread_bps_mean': float(baseline_metrics['spread_bps'].mean()),
        'depth_mean': float(baseline_metrics['depth_50bps_total'].mean()),
        'imbalance_mean': float(baseline_metrics['order_imbalance'].mean()),
        'stability_mean': float(baseline_metrics['stability_score'].mean()),
        'total_rows_file': 115060085
    }
    print(f"Global Baseline: Spread={global_stats['spread_bps_mean']:.4f} bps, "
          f"Depth={global_stats['depth_mean']:.2f} BTC, Stability={global_stats['stability_mean']:.4f}")

    # 3. Analyze Impact for ALL clusters
    print("Analyzing impact for all clusters (Parallel extraction)...")
    all_impacts = []

    # We focus on the clusters found
    significant_clusters = [c for c in clusters if c.total_value > 1000] # At least $1k

    for i, cluster in enumerate(tqdm(significant_clusters)):
        ts = cluster.start_timestamp
        window_before = 60_000_000 # 60s
        window_after = 60_000_000

        # Get data around cluster
        window_df = loader.get_orderbook_at_timestamp(ts, window_us=window_after)

        if len(window_df) > 10:
            metrics_df = metrics_calc.calculate_metrics_for_timestamps(window_df)
            metrics_df['stability_score'] = OrderbookMetrics.calculate_stability_score(metrics_df)

            # Use LiquidationAnalyzer's built-in analysis
            impact_data = liq_analyzer.analyze_orderbook_around_liquidation(
                metrics_df, ts, before_window_us=window_before, after_window_us=window_after
            )

            if impact_data:
                # Add category and stability
                cat = 'small' if cluster.total_value < 50000 else ('medium' if cluster.total_value < 200000 else 'large')

                before_stab = metrics_df[metrics_df['timestamp'] < ts]['stability_score'].mean()
                after_stab = metrics_df[metrics_df['timestamp'] >= ts]['stability_score'].mean()

                impact = {
                    'cluster_id': i,
                    'category': cat,
                    'total_usd': cluster.total_value,
                    'spread_before': impact_data['before']['spread_bps_mean'],
                    'spread_after': impact_data['after']['spread_bps_mean'],
                    'depth_before': impact_data['before']['depth_50bps_mean'],
                    'depth_after': impact_data['after']['depth_50bps_mean'],
                    'stability_before': before_stab,
                    'stability_after': after_stab,
                    'spread_change_pct': impact_data['changes']['spread_bps_change_pct'],
                    'depth_change_pct': impact_data['changes']['depth_change_pct']
                }
                all_impacts.append(impact)

    impact_df = pd.DataFrame(all_impacts)
    impact_df.to_csv(output_dir / "full_liquidation_impact.csv", index=False)

    # 4. Summary Statistics by Category
    summary = {
        'global_baseline': global_stats,
        'impact_by_category': impact_df.groupby('category').mean().to_dict() if not impact_df.empty else {},
        'cluster_counts': {k: len(v) for k, v in categorized_clusters.items()},
        'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration_seconds': time.time() - start_time
    }

    with open(output_dir / "full_analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"✅ Full analysis complete! Results saved to {output_dir}")
    print(f"⏱ Total time: {summary['duration_seconds']:.2f} seconds")

if __name__ == "__main__":
    run_full_analysis()
