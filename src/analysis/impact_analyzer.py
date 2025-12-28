"""
Liquidation Impact Analyzer
============================

Analyzes the impact of liquidation events on orderbook stability,
including pre/post comparison and recovery time estimation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImpactAnalysis:
    """Results of analyzing liquidation impact on orderbook."""
    cluster_id: int
    start_ts: int
    end_ts: int
    total_value: float
    event_count: int
    dominant_side: str

    # Before metrics (baseline)
    before_spread_bps_mean: float
    before_spread_bps_std: float
    before_depth_mean: float
    before_imbalance_mean: float
    before_mid_price: float
    before_sample_count: int

    # After metrics
    after_spread_bps_mean: float
    after_spread_bps_std: float
    after_depth_mean: float
    after_imbalance_mean: float
    after_mid_price: float
    after_sample_count: int

    # Changes
    spread_change_pct: float
    depth_change_pct: float
    price_change_pct: float

    # Recovery
    recovery_time_us: Optional[int]
    recovered: bool


class LiquidationImpactAnalyzer:
    """
    Analyzes the impact of liquidation clusters on orderbook state.
    """

    def __init__(
        self,
        orderbook_metrics: pd.DataFrame,
        liquidation_clusters: List[Dict],
        before_window_us: int = 60_000_000,  # 60 seconds
        after_window_us: int = 60_000_000,   # 60 seconds
        recovery_threshold_pct: float = 1.5  # Within 150% of baseline spread
    ):
        """
        Initialize the impact analyzer.

        Args:
            orderbook_metrics: DataFrame with orderbook metrics (from Phase 1)
            liquidation_clusters: List of cluster dictionaries from liquidation_summary.json
            before_window_us: Time window before liquidation to measure baseline (microseconds)
            after_window_us: Time window after liquidation to measure impact (microseconds)
            recovery_threshold_pct: Spread must be within this factor of baseline to be "recovered"
        """
        self.metrics = orderbook_metrics.copy()
        self.clusters = liquidation_clusters
        self.before_window_us = before_window_us
        self.after_window_us = after_window_us
        self.recovery_threshold_pct = recovery_threshold_pct

        # Ensure metrics are sorted by timestamp
        self.metrics = self.metrics.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Initialized ImpactAnalyzer with {len(self.clusters)} clusters")
        logger.info(f"Metrics time range: {self.metrics['timestamp'].min()} - {self.metrics['timestamp'].max()}")

    def analyze_single_cluster(self, cluster: Dict, cluster_id: int) -> Optional[ImpactAnalysis]:
        """
        Analyze impact of a single liquidation cluster.

        Args:
            cluster: Dictionary with cluster information
            cluster_id: Index of the cluster

        Returns:
            ImpactAnalysis object or None if insufficient data
        """
        start_ts = int(cluster['start_ts'])
        end_ts = int(cluster['end_ts'])

        # Get metrics before liquidation
        before_mask = (
            (self.metrics['timestamp'] >= start_ts - self.before_window_us) &
            (self.metrics['timestamp'] < start_ts)
        )
        before_metrics = self.metrics[before_mask]

        # Get metrics after liquidation
        after_mask = (
            (self.metrics['timestamp'] > end_ts) &
            (self.metrics['timestamp'] <= end_ts + self.after_window_us)
        )
        after_metrics = self.metrics[after_mask]

        # Need sufficient data for comparison
        if len(before_metrics) < 10 or len(after_metrics) < 10:
            logger.debug(f"Cluster {cluster_id}: Insufficient data (before={len(before_metrics)}, after={len(after_metrics)})")
            return None

        # Calculate baseline metrics
        before_spread_mean = before_metrics['spread_bps'].mean()
        before_spread_std = before_metrics['spread_bps'].std()
        before_depth_mean = before_metrics['depth_50bps_total'].mean()
        before_imbalance_mean = before_metrics['order_imbalance'].mean()
        before_mid_price = before_metrics['mid_price'].mean()

        # Calculate post-liquidation metrics
        after_spread_mean = after_metrics['spread_bps'].mean()
        after_spread_std = after_metrics['spread_bps'].std()
        after_depth_mean = after_metrics['depth_50bps_total'].mean()
        after_imbalance_mean = after_metrics['order_imbalance'].mean()
        after_mid_price = after_metrics['mid_price'].mean()

        # Calculate changes
        spread_change_pct = ((after_spread_mean - before_spread_mean) / before_spread_mean * 100
                           if before_spread_mean > 0 else 0)
        depth_change_pct = ((after_depth_mean - before_depth_mean) / before_depth_mean * 100
                          if before_depth_mean > 0 else 0)
        price_change_pct = ((after_mid_price - before_mid_price) / before_mid_price * 100
                          if before_mid_price > 0 else 0)

        # Estimate recovery time
        recovery_time, recovered = self._estimate_recovery_time(
            end_ts,
            before_spread_mean
        )

        return ImpactAnalysis(
            cluster_id=cluster_id,
            start_ts=start_ts,
            end_ts=end_ts,
            total_value=cluster['total_value'],
            event_count=cluster['event_count'],
            dominant_side=cluster['dominant_side'],
            before_spread_bps_mean=before_spread_mean,
            before_spread_bps_std=before_spread_std,
            before_depth_mean=before_depth_mean,
            before_imbalance_mean=before_imbalance_mean,
            before_mid_price=before_mid_price,
            before_sample_count=len(before_metrics),
            after_spread_bps_mean=after_spread_mean,
            after_spread_bps_std=after_spread_std,
            after_depth_mean=after_depth_mean,
            after_imbalance_mean=after_imbalance_mean,
            after_mid_price=after_mid_price,
            after_sample_count=len(after_metrics),
            spread_change_pct=spread_change_pct,
            depth_change_pct=depth_change_pct,
            price_change_pct=price_change_pct,
            recovery_time_us=recovery_time,
            recovered=recovered
        )

    def _estimate_recovery_time(
        self,
        liquidation_end_ts: int,
        baseline_spread: float,
        max_search_window_us: int = 300_000_000  # 5 minutes
    ) -> Tuple[Optional[int], bool]:
        """
        Estimate how long it takes for spread to return to baseline.

        Args:
            liquidation_end_ts: End timestamp of liquidation cluster
            baseline_spread: Baseline spread before liquidation
            max_search_window_us: Maximum time to search for recovery

        Returns:
            Tuple of (recovery_time_us, recovered_bool)
        """
        recovery_threshold = baseline_spread * self.recovery_threshold_pct

        # Get metrics after liquidation
        after_mask = (
            (self.metrics['timestamp'] > liquidation_end_ts) &
            (self.metrics['timestamp'] <= liquidation_end_ts + max_search_window_us)
        )
        after_metrics = self.metrics[after_mask].sort_values('timestamp')

        # Find first timestamp where spread returns to acceptable level
        for _, row in after_metrics.iterrows():
            if row['spread_bps'] <= recovery_threshold:
                recovery_time = int(row['timestamp'] - liquidation_end_ts)
                return recovery_time, True

        return None, False

    def analyze_all_clusters(self) -> List[ImpactAnalysis]:
        """
        Analyze all liquidation clusters.

        Returns:
            List of ImpactAnalysis objects
        """
        results = []

        for i, cluster in enumerate(self.clusters):
            analysis = self.analyze_single_cluster(cluster, i)
            if analysis:
                results.append(analysis)

            if (i + 1) % 10 == 0:
                logger.info(f"Analyzed {i + 1}/{len(self.clusters)} clusters")

        logger.info(f"Successfully analyzed {len(results)}/{len(self.clusters)} clusters")
        return results

    def generate_impact_report(self, analyses: List[ImpactAnalysis]) -> pd.DataFrame:
        """
        Generate a DataFrame report from impact analyses.

        Args:
            analyses: List of ImpactAnalysis objects

        Returns:
            DataFrame with impact analysis results
        """
        records = []
        for a in analyses:
            records.append({
                'cluster_id': a.cluster_id,
                'start_ts': a.start_ts,
                'end_ts': a.end_ts,
                'total_value': a.total_value,
                'event_count': a.event_count,
                'dominant_side': a.dominant_side,
                'before_spread_bps_mean': a.before_spread_bps_mean,
                'before_spread_bps_std': a.before_spread_bps_std,
                'before_depth_mean': a.before_depth_mean,
                'before_imbalance_mean': a.before_imbalance_mean,
                'before_mid_price': a.before_mid_price,
                'before_sample_count': a.before_sample_count,
                'after_spread_bps_mean': a.after_spread_bps_mean,
                'after_spread_bps_std': a.after_spread_bps_std,
                'after_depth_mean': a.after_depth_mean,
                'after_imbalance_mean': a.after_imbalance_mean,
                'after_mid_price': a.after_mid_price,
                'after_sample_count': a.after_sample_count,
                'spread_change_pct': a.spread_change_pct,
                'depth_change_pct': a.depth_change_pct,
                'price_change_pct': a.price_change_pct,
                'recovery_time_us': a.recovery_time_us,
                'recovery_time_sec': a.recovery_time_us / 1_000_000 if a.recovery_time_us else None,
                'recovered': a.recovered
            })

        return pd.DataFrame(records)

    def generate_recovery_summary(self, analyses: List[ImpactAnalysis]) -> Dict:
        """
        Generate summary statistics for recovery analysis.

        Args:
            analyses: List of ImpactAnalysis objects

        Returns:
            Dictionary with recovery statistics
        """
        recovery_times = [a.recovery_time_us for a in analyses if a.recovered and a.recovery_time_us]
        recovered_count = sum(1 for a in analyses if a.recovered)

        # Group by liquidation size
        small_liquidations = [a for a in analyses if a.total_value < 50000]
        medium_liquidations = [a for a in analyses if 50000 <= a.total_value < 200000]
        large_liquidations = [a for a in analyses if a.total_value >= 200000]

        def get_recovery_stats(group: List[ImpactAnalysis]) -> Dict:
            if not group:
                return {'count': 0, 'recovered_count': 0, 'recovery_rate': 0}

            recovered = [a for a in group if a.recovered]
            times = [a.recovery_time_us for a in recovered if a.recovery_time_us]

            return {
                'count': len(group),
                'recovered_count': len(recovered),
                'recovery_rate': len(recovered) / len(group) if group else 0,
                'avg_recovery_time_sec': np.mean(times) / 1_000_000 if times else None,
                'median_recovery_time_sec': np.median(times) / 1_000_000 if times else None,
                'max_recovery_time_sec': max(times) / 1_000_000 if times else None,
                'avg_spread_change_pct': np.mean([a.spread_change_pct for a in group]),
                'avg_depth_change_pct': np.mean([a.depth_change_pct for a in group]),
                'avg_price_change_pct': np.mean([a.price_change_pct for a in group]),
            }

        return {
            'total_analyzed': len(analyses),
            'total_recovered': recovered_count,
            'overall_recovery_rate': recovered_count / len(analyses) if analyses else 0,
            'overall_stats': {
                'avg_recovery_time_sec': np.mean(recovery_times) / 1_000_000 if recovery_times else None,
                'median_recovery_time_sec': np.median(recovery_times) / 1_000_000 if recovery_times else None,
                'min_recovery_time_sec': min(recovery_times) / 1_000_000 if recovery_times else None,
                'max_recovery_time_sec': max(recovery_times) / 1_000_000 if recovery_times else None,
                'percentile_90_sec': np.percentile(recovery_times, 90) / 1_000_000 if recovery_times else None,
            },
            'by_size': {
                'small_under_50k': get_recovery_stats(small_liquidations),
                'medium_50k_200k': get_recovery_stats(medium_liquidations),
                'large_over_200k': get_recovery_stats(large_liquidations),
            },
            'by_side': {
                'sell_dominant': get_recovery_stats([a for a in analyses if a.dominant_side == 'sell']),
                'buy_dominant': get_recovery_stats([a for a in analyses if a.dominant_side == 'buy']),
            },
            'impact_correlation': {
                'description': 'Correlation between liquidation value and orderbook impact',
                'spread_impact_by_value': self._calculate_value_impact_correlation(analyses, 'spread_change_pct'),
                'depth_impact_by_value': self._calculate_value_impact_correlation(analyses, 'depth_change_pct'),
            }
        }

    def _calculate_value_impact_correlation(
        self,
        analyses: List[ImpactAnalysis],
        impact_field: str
    ) -> float:
        """Calculate correlation between liquidation value and impact metric."""
        if len(analyses) < 3:
            return 0.0

        values = [a.total_value for a in analyses]
        impacts = [getattr(a, impact_field) for a in analyses]

        # Handle NaN values
        valid_pairs = [(v, i) for v, i in zip(values, impacts) if not np.isnan(i)]
        if len(valid_pairs) < 3:
            return 0.0

        values, impacts = zip(*valid_pairs)
        return float(np.corrcoef(values, impacts)[0, 1])


def run_impact_analysis(
    metrics_path: str,
    liquidation_summary_path: str,
    output_dir: str
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run the full impact analysis pipeline.

    Args:
        metrics_path: Path to orderbook_metrics.csv
        liquidation_summary_path: Path to liquidation_summary.json
        output_dir: Output directory for results

    Returns:
        Tuple of (impact_df, recovery_summary)
    """
    output_dir = Path(output_dir)

    # Load data
    logger.info("Loading orderbook metrics...")
    metrics_df = pd.read_csv(metrics_path)

    logger.info("Loading liquidation summary...")
    with open(liquidation_summary_path, 'r') as f:
        liq_summary = json.load(f)

    clusters = liq_summary.get('large_cluster_details', [])
    logger.info(f"Found {len(clusters)} large clusters to analyze")

    # Initialize analyzer
    analyzer = LiquidationImpactAnalyzer(
        orderbook_metrics=metrics_df,
        liquidation_clusters=clusters,
        before_window_us=60_000_000,  # 60 seconds
        after_window_us=60_000_000,   # 60 seconds
        recovery_threshold_pct=1.5    # Within 150% of baseline
    )

    # Run analysis
    logger.info("Analyzing liquidation impacts...")
    analyses = analyzer.analyze_all_clusters()

    # Generate reports
    impact_df = analyzer.generate_impact_report(analyses)
    recovery_summary = analyzer.generate_recovery_summary(analyses)

    # Save results
    impact_df.to_csv(output_dir / "liquidation_impact_analysis.csv", index=False)
    logger.info(f"Saved impact analysis to {output_dir / 'liquidation_impact_analysis.csv'}")

    with open(output_dir / "recovery_time_analysis.json", 'w') as f:
        json.dump(recovery_summary, f, indent=2, default=str)
    logger.info(f"Saved recovery analysis to {output_dir / 'recovery_time_analysis.json'}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("IMPACT ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Clusters analyzed: {len(analyses)}")
    logger.info(f"Recovery rate: {recovery_summary['overall_recovery_rate']:.1%}")

    if recovery_summary['overall_stats']['avg_recovery_time_sec']:
        logger.info(f"Average recovery time: {recovery_summary['overall_stats']['avg_recovery_time_sec']:.2f} seconds")
        logger.info(f"Median recovery time: {recovery_summary['overall_stats']['median_recovery_time_sec']:.2f} seconds")

    logger.info("\nBy liquidation size:")
    for size, stats in recovery_summary['by_size'].items():
        if stats['count'] > 0:
            logger.info(f"  {size}: {stats['count']} clusters, "
                       f"{stats['recovery_rate']:.1%} recovered, "
                       f"avg spread change: {stats['avg_spread_change_pct']:.1f}%")

    return impact_df, recovery_summary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze liquidation impact on orderbook')
    parser.add_argument('--metrics', type=str, default='output/phase1/orderbook_metrics.csv')
    parser.add_argument('--liquidations', type=str, default='output/phase1/liquidation_summary.json')
    parser.add_argument('--output', type=str, default='output/phase1')
    args = parser.parse_args()

    run_impact_analysis(args.metrics, args.liquidations, args.output)

