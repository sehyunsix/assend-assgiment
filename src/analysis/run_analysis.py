#!/usr/bin/env python3
"""
Phase 1: Orderbook Stability Analysis
=====================================

This script performs the complete analysis of orderbook stability metrics
in relation to liquidation events using clean (research) data.

Usage:
    python run_analysis.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]
"""

import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import DataLoader
from orderbook_metrics import OrderbookMetrics
from liquidation_analyzer import LiquidationAnalyzer

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12


def analyze_liquidations(data_loader: DataLoader, output_dir: Path) -> pd.DataFrame:
    """Analyze liquidation events and identify large ones."""
    logger.info("=" * 60)
    logger.info("STEP 1: Analyzing Liquidation Events")
    logger.info("=" * 60)

    # Load liquidations
    liquidations_df = data_loader.load_liquidations()
    analyzer = LiquidationAnalyzer(liquidations_df)

    # Get summary statistics
    stats = analyzer.get_summary_stats()
    logger.info("\n=== Liquidation Summary Statistics ===")
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:,.2f}")
        else:
            logger.info(f"  {key}: {value}")

    # Identify large liquidations
    large_liquidations = analyzer.identify_large_liquidations(percentile=90)
    logger.info(f"\nLarge liquidations (top 10%):")
    logger.info(f"  Count: {len(large_liquidations)}")
    logger.info(f"  Total value: ${large_liquidations['value'].sum():,.2f}")

    # Cluster liquidations
    clusters = analyzer.cluster_liquidations(time_window_us=5_000_000)
    large_clusters = analyzer.get_large_clusters(clusters, min_events=2, percentile=75)

    logger.info(f"\nLiquidation Clusters:")
    logger.info(f"  Total clusters: {len(clusters)}")
    logger.info(f"  Large clusters (>= 2 events, top 25% value): {len(large_clusters)}")

    # Generate report
    report = analyzer.generate_liquidation_report()
    report.to_csv(output_dir / "liquidation_report.csv", index=False)
    logger.info(f"\nSaved liquidation report to {output_dir / 'liquidation_report.csv'}")

    # Save summary
    summary = {
        'statistics': stats,
        'large_liquidation_count': len(large_liquidations),
        'total_clusters': len(clusters),
        'large_clusters': len(large_clusters),
        'large_cluster_details': [
            {
                'start_ts': c.start_timestamp,
                'end_ts': c.end_timestamp,
                'total_value': c.total_value,
                'event_count': c.event_count,
                'dominant_side': c.dominant_side
            }
            for c in large_clusters[:20]  # Top 20
        ]
    }

    with open(output_dir / "liquidation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    return liquidations_df





def calculate_orderbook_metrics_optimized(
    data_loader: DataLoader,
    output_dir: Path
) -> pd.DataFrame:
    """Optimized metric calculation using Dask parallel processing."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Calculating Optimized Orderbook Metrics")
    logger.info("=" * 60)

    # Convert to parquet first for 10x faster I/O if not exists
    data_loader.convert_csv_to_parquet()

    orderbook_ddf = data_loader.load_orderbook()
    metrics_calculator = OrderbookMetrics()

    # Use Dask's map_partitions to run vectorized batch processing on all cores
    logger.info(f"Processing {orderbook_ddf.npartitions} partitions in parallel...")

    with ProgressBar():
        # map_partitions applies our vectorized batch function to each chunk of the large file
        metrics_ddf = orderbook_ddf.map_partitions(
            metrics_calculator.calculate_metrics_batch,
            meta={
                'timestamp': 'int64',
                'best_bid': 'float64',
                'bid_levels': 'int64',
                'best_ask': 'float64',
                'ask_levels': 'int64',
                'bid_volume_total': 'float64',
                'ask_volume_total': 'float64',
                'local_timestamp': 'int64',
                'mid_price': 'float64',
                'spread': 'float64',
                'spread_bps': 'float64',
                'order_imbalance': 'float64',
                'depth_10bps_ask': 'float64',
                'depth_10bps_bid': 'float64',
                'depth_50bps_ask': 'float64',
                'depth_50bps_bid': 'float64',
                'depth_10bps_total': 'float64',
                'depth_50bps_total': 'float64'
            }
        )

        # Compute everything in parallel
        metrics_df = metrics_ddf.compute()

    logger.info(f"Calculated metrics for {len(metrics_df):,} timestamps")

    # Add optimized stability score
    # Ensure vpin and price_impact are present for the optimized score
    if 'vpin' not in metrics_df.columns:
        metrics_df['vpin'] = 0.5
    if 'price_impact' not in metrics_df.columns:
        metrics_df['price_impact'] = 0.0

    metrics_df['stability_score'] = OrderbookMetrics.calculate_stability_score(metrics_df)

    # Detect anomalies
    metrics_df = OrderbookMetrics.detect_anomalies(metrics_df)

    # Save metrics
    metrics_df.to_csv(output_dir / "orderbook_metrics.csv", index=False)
    logger.info(f"Saved optimized metrics to {output_dir / 'orderbook_metrics.csv'}")

    return metrics_df

    # Print summary statistics
    logger.info("\n=== Orderbook Metrics Summary ===")
    logger.info(f"  Spread (bps) - Mean: {metrics_df['spread_bps'].mean():.4f}, "
               f"Std: {metrics_df['spread_bps'].std():.4f}, "
               f"Max: {metrics_df['spread_bps'].max():.4f}")
    logger.info(f"  Order Imbalance - Mean: {metrics_df['order_imbalance'].mean():.4f}, "
               f"Std: {metrics_df['order_imbalance'].std():.4f}")
    logger.info(f"  Depth 50bps - Mean: {metrics_df['depth_50bps_total'].mean():.4f}, "
               f"Min: {metrics_df['depth_50bps_total'].min():.4f}")
    logger.info(f"  Stability Score - Mean: {metrics_df['stability_score'].mean():.4f}")
    logger.info(f"  Anomalies detected: {metrics_df['is_anomaly'].sum():,} "
               f"({100*metrics_df['is_anomaly'].mean():.2f}%)")

    return metrics_df


def visualize_results(
    metrics_df: pd.DataFrame,
    liquidations_df: pd.DataFrame,
    output_dir: Path
):
    """Create visualizations of the analysis results."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Creating Visualizations")
    logger.info("=" * 60)

    # Create figures directory
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # 1. Spread distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Spread histogram
    ax = axes[0, 0]
    ax.hist(metrics_df['spread_bps'], bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(metrics_df['spread_bps'].mean(), color='red', linestyle='--',
               label=f'Mean: {metrics_df["spread_bps"].mean():.4f}')
    ax.axvline(metrics_df['spread_bps'].median(), color='orange', linestyle='--',
               label=f'Median: {metrics_df["spread_bps"].median():.4f}')
    ax.set_xlabel('Spread (bps)')
    ax.set_ylabel('Frequency')
    ax.set_title('Bid-Ask Spread Distribution')
    ax.legend()

    # Order imbalance histogram
    ax = axes[0, 1]
    ax.hist(metrics_df['order_imbalance'], bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', label='Balanced')
    ax.set_xlabel('Order Imbalance')
    ax.set_ylabel('Frequency')
    ax.set_title('Order Imbalance Distribution')
    ax.legend()

    # Market depth distribution
    ax = axes[1, 0]
    ax.hist(metrics_df['depth_50bps_total'], bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(metrics_df['depth_50bps_total'].mean(), color='red', linestyle='--',
               label=f'Mean: {metrics_df["depth_50bps_total"].mean():.2f}')
    ax.set_xlabel('Depth (50 bps)')
    ax.set_ylabel('Frequency')
    ax.set_title('Market Depth Distribution (50 bps)')
    ax.legend()

    # Stability score distribution
    ax = axes[1, 1]
    ax.hist(metrics_df['stability_score'], bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(metrics_df['stability_score'].mean(), color='red', linestyle='--',
               label=f'Mean: {metrics_df["stability_score"].mean():.4f}')
    ax.set_xlabel('Stability Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Orderbook Stability Score Distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig(figures_dir / "metrics_distributions.png", dpi=150)
    plt.close()
    logger.info(f"Saved metrics_distributions.png")

    # 2. Time series of metrics
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

    # Convert timestamp to datetime-like for better plotting
    metrics_df['time_sec'] = (metrics_df['timestamp'] - metrics_df['timestamp'].min()) / 1_000_000

    # Spread over time
    ax = axes[0]
    ax.plot(metrics_df['time_sec'], metrics_df['spread_bps'], alpha=0.7, linewidth=0.5)
    ax.set_ylabel('Spread (bps)')
    ax.set_title('Bid-Ask Spread Over Time')

    # Mark anomalies
    anomalies = metrics_df[metrics_df['anomaly_wide_spread']]
    ax.scatter(anomalies['time_sec'], anomalies['spread_bps'],
               color='red', s=20, alpha=0.5, label='Wide spread anomaly')
    ax.legend()

    # Order imbalance over time
    ax = axes[1]
    ax.plot(metrics_df['time_sec'], metrics_df['order_imbalance'], alpha=0.7, linewidth=0.5)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Order Imbalance')
    ax.set_title('Order Imbalance Over Time')

    # Depth over time
    ax = axes[2]
    ax.plot(metrics_df['time_sec'], metrics_df['depth_50bps_total'], alpha=0.7, linewidth=0.5)
    ax.set_ylabel('Depth (50 bps)')
    ax.set_title('Market Depth Over Time')

    # Mark liquidations on depth chart
    if len(liquidations_df) > 0:
        liq_times = (liquidations_df['timestamp'] - metrics_df['timestamp'].min()) / 1_000_000
        valid_liq = liq_times[(liq_times >= 0) & (liq_times <= metrics_df['time_sec'].max())]
        for lt in valid_liq:
            ax.axvline(lt, color='red', alpha=0.3, linewidth=0.5)

    # Stability score over time
    ax = axes[3]
    ax.plot(metrics_df['time_sec'], metrics_df['stability_score'], alpha=0.7, linewidth=0.5)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Stability Score')
    ax.set_title('Orderbook Stability Score Over Time')

    plt.tight_layout()
    plt.savefig(figures_dir / "metrics_timeseries.png", dpi=150)
    plt.close()
    logger.info(f"Saved metrics_timeseries.png")

    # 3. Correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    corr_cols = ['spread_bps', 'order_imbalance', 'depth_10bps_total',
                 'depth_50bps_total', 'bid_levels', 'ask_levels',
                 'bid_volume_total', 'ask_volume_total', 'stability_score']
    corr_matrix = metrics_df[corr_cols].corr()

    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                fmt='.2f', ax=ax, square=True)
    ax.set_title('Correlation Matrix of Orderbook Metrics')

    plt.tight_layout()
    plt.savefig(figures_dir / "correlation_matrix.png", dpi=150)
    plt.close()
    logger.info(f"Saved correlation_matrix.png")

    # 4. Liquidation value distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(liquidations_df['value'], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Liquidation Value ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Liquidation Value Distribution')
    ax.axvline(liquidations_df['value'].mean(), color='red', linestyle='--',
               label=f'Mean: ${liquidations_df["value"].mean():,.0f}')
    ax.legend()

    ax = axes[1]
    side_counts = liquidations_df['side'].value_counts()
    ax.bar(side_counts.index, side_counts.values, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Side')
    ax.set_ylabel('Count')
    ax.set_title('Liquidations by Side')

    plt.tight_layout()
    plt.savefig(figures_dir / "liquidation_distributions.png", dpi=150)
    plt.close()
    logger.info(f"Saved liquidation_distributions.png")

    logger.info(f"All visualizations saved to {figures_dir}")


def generate_summary_report(
    metrics_df: pd.DataFrame,
    liquidations_df: pd.DataFrame,
    output_dir: Path
):
    """Generate a summary JSON report."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Generating Summary Report")
    logger.info("=" * 60)

    # Calculate percentiles for metric thresholds
    spread_percentiles = {
        'p50': metrics_df['spread_bps'].quantile(0.50),
        'p75': metrics_df['spread_bps'].quantile(0.75),
        'p90': metrics_df['spread_bps'].quantile(0.90),
        'p95': metrics_df['spread_bps'].quantile(0.95),
        'p99': metrics_df['spread_bps'].quantile(0.99),
    }

    depth_percentiles = {
        'p01': metrics_df['depth_50bps_total'].quantile(0.01),
        'p05': metrics_df['depth_50bps_total'].quantile(0.05),
        'p10': metrics_df['depth_50bps_total'].quantile(0.10),
        'p25': metrics_df['depth_50bps_total'].quantile(0.25),
        'p50': metrics_df['depth_50bps_total'].quantile(0.50),
    }

    stability_percentiles = {
        'p01': metrics_df['stability_score'].quantile(0.01),
        'p05': metrics_df['stability_score'].quantile(0.05),
        'p10': metrics_df['stability_score'].quantile(0.10),
        'p25': metrics_df['stability_score'].quantile(0.25),
        'p50': metrics_df['stability_score'].quantile(0.50),
    }

    summary = {
        'analysis_timestamp': datetime.now().isoformat(),
        'data_summary': {
            'orderbook_snapshots': len(metrics_df),
            'liquidation_events': len(liquidations_df),
            'time_range_seconds': (
                metrics_df['timestamp'].max() - metrics_df['timestamp'].min()
            ) / 1_000_000,
        },
        'spread_analysis': {
            'mean_bps': float(metrics_df['spread_bps'].mean()),
            'std_bps': float(metrics_df['spread_bps'].std()),
            'min_bps': float(metrics_df['spread_bps'].min()),
            'max_bps': float(metrics_df['spread_bps'].max()),
            'percentiles': {k: float(v) for k, v in spread_percentiles.items()},
        },
        'depth_analysis': {
            'mean_50bps': float(metrics_df['depth_50bps_total'].mean()),
            'std_50bps': float(metrics_df['depth_50bps_total'].std()),
            'min_50bps': float(metrics_df['depth_50bps_total'].min()),
            'max_50bps': float(metrics_df['depth_50bps_total'].max()),
            'percentiles': {k: float(v) for k, v in depth_percentiles.items()},
        },
        'imbalance_analysis': {
            'mean': float(metrics_df['order_imbalance'].mean()),
            'std': float(metrics_df['order_imbalance'].std()),
            'min': float(metrics_df['order_imbalance'].min()),
            'max': float(metrics_df['order_imbalance'].max()),
        },
        'stability_analysis': {
            'mean_score': float(metrics_df['stability_score'].mean()),
            'std_score': float(metrics_df['stability_score'].std()),
            'percentiles': {k: float(v) for k, v in stability_percentiles.items()},
        },
        'anomaly_analysis': {
            'total_anomalies': int(metrics_df['is_anomaly'].sum()),
            'anomaly_rate': float(metrics_df['is_anomaly'].mean()),
            'wide_spread_anomalies': int(metrics_df['anomaly_wide_spread'].sum()),
            'high_imbalance_anomalies': int(metrics_df['anomaly_high_imbalance'].sum()),
            'depth_drop_anomalies': int(metrics_df['anomaly_depth_drop'].sum()),
        },
        'suggested_thresholds': {
            'description': 'Suggested thresholds for decision engine',
            'spread_warning_bps': float(spread_percentiles['p90']),
            'spread_critical_bps': float(spread_percentiles['p99']),
            'depth_warning': float(depth_percentiles['p10']),
            'depth_critical': float(depth_percentiles['p01']),
            'stability_warning': float(stability_percentiles['p10']),
            'stability_critical': float(stability_percentiles['p01']),
            'imbalance_threshold': 0.5,
        },
    }

    with open(output_dir / "analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved summary report to {output_dir / 'analysis_summary.json'}")

    # Print key findings
    logger.info("\n=== KEY FINDINGS ===")
    logger.info(f"\n1. SPREAD ANALYSIS:")
    logger.info(f"   - Normal spread: {spread_percentiles['p50']:.4f} bps (median)")
    logger.info(f"   - Warning level (p90): {spread_percentiles['p90']:.4f} bps")
    logger.info(f"   - Critical level (p99): {spread_percentiles['p99']:.4f} bps")

    logger.info(f"\n2. DEPTH ANALYSIS:")
    logger.info(f"   - Normal depth: {depth_percentiles['p50']:.2f} BTC (median)")
    logger.info(f"   - Warning level (p10): {depth_percentiles['p10']:.2f} BTC")
    logger.info(f"   - Critical level (p01): {depth_percentiles['p01']:.2f} BTC")

    logger.info(f"\n3. STABILITY SCORE:")
    logger.info(f"   - Mean score: {metrics_df['stability_score'].mean():.4f}")
    logger.info(f"   - Warning level (p10): {stability_percentiles['p10']:.4f}")
    logger.info(f"   - Critical level (p01): {stability_percentiles['p01']:.4f}")

    logger.info(f"\n4. ANOMALY DETECTION:")
    logger.info(f"   - Total anomalies: {metrics_df['is_anomaly'].sum():,} "
               f"({100*metrics_df['is_anomaly'].mean():.2f}%)")

    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Phase 1: Orderbook Stability Analysis')
    parser.add_argument('--data-dir', type=str, default='data/research',
                       help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='output/phase1',
                       help='Path to output directory')
    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 1: Orderbook Stability Analysis")
    logger.info("=" * 60)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Initialize data loader
    data_loader = DataLoader(str(data_dir))

    # Step 1: Analyze liquidations
    liquidations_df = analyze_liquidations(data_loader, output_dir)

    # Step 2: Sampling (Optional, let's process all using optimized method)


    # Step 3: Calculate metrics (OPTIMIZED)
    metrics_df = calculate_orderbook_metrics_optimized(data_loader, output_dir)

    # Step 4: Create visualizations
    visualize_results(metrics_df, liquidations_df, output_dir)

    # Step 5: Generate summary report
    summary = generate_summary_report(metrics_df, liquidations_df, output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Analysis Complete!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")

    return summary


if __name__ == '__main__':
    main()

