"""
Liquidation Analyzer Module
===========================

Analyzes the relationship between liquidation events and orderbook stability.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class LiquidationCluster:
    """Represents a cluster of liquidation events."""
    start_timestamp: int
    end_timestamp: int
    total_value: float
    event_count: int
    dominant_side: str
    events: pd.DataFrame


class LiquidationAnalyzer:
    """
    Analyzes liquidation events and their impact on orderbook stability.

    Features:
    - Identify large liquidation events
    - Cluster consecutive liquidations
    - Analyze orderbook state before/after liquidations
    """

    def __init__(self, liquidations_df: pd.DataFrame):
        """
        Initialize with liquidations data.

        Args:
            liquidations_df: DataFrame with liquidation events
        """
        self.liquidations = liquidations_df.copy()
        self._preprocess()

    def _preprocess(self):
        """Preprocess liquidation data."""
        # Calculate liquidation value if not present
        if 'value' not in self.liquidations.columns:
            self.liquidations['value'] = (
                self.liquidations['price'] * self.liquidations['amount']
            )

        # Sort by timestamp
        self.liquidations = self.liquidations.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Preprocessed {len(self.liquidations)} liquidation events")
        logger.info(f"Total liquidation value: ${self.liquidations['value'].sum():,.2f}")

    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics for liquidations.

        Returns:
            Dictionary with summary statistics
        """
        return {
            'total_events': len(self.liquidations),
            'total_value': self.liquidations['value'].sum(),
            'mean_value': self.liquidations['value'].mean(),
            'median_value': self.liquidations['value'].median(),
            'max_value': self.liquidations['value'].max(),
            'std_value': self.liquidations['value'].std(),
            'buy_count': len(self.liquidations[self.liquidations['side'] == 'buy']),
            'sell_count': len(self.liquidations[self.liquidations['side'] == 'sell']),
            'buy_value': self.liquidations[self.liquidations['side'] == 'buy']['value'].sum(),
            'sell_value': self.liquidations[self.liquidations['side'] == 'sell']['value'].sum(),
            'time_span_hours': (
                self.liquidations['timestamp'].max() -
                self.liquidations['timestamp'].min()
            ) / (1_000_000 * 3600),  # Convert microseconds to hours
        }

    def identify_large_liquidations(
        self,
        percentile: float = 90.0,
        min_value: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Identify large liquidation events.

        Args:
            percentile: Percentile threshold for "large" liquidations
            min_value: Minimum value threshold (overrides percentile if provided)

        Returns:
            DataFrame with large liquidation events
        """
        if min_value is not None:
            threshold = min_value
        else:
            threshold = np.percentile(self.liquidations['value'], percentile)

        large_liquidations = self.liquidations[
            self.liquidations['value'] >= threshold
        ].copy()

        logger.info(f"Identified {len(large_liquidations)} large liquidations "
                   f"(threshold: ${threshold:,.2f})")

        return large_liquidations

    def cluster_liquidations(
        self,
        time_window_us: int = 5_000_000
    ) -> List[LiquidationCluster]:
        """
        Cluster consecutive liquidation events within a time window.

        Args:
            time_window_us: Maximum gap between events to consider them clustered
                           (default 5 seconds in microseconds)

        Returns:
            List of LiquidationCluster objects
        """
        if len(self.liquidations) == 0:
            return []

        clusters = []
        current_cluster_events = []

        for idx, row in self.liquidations.iterrows():
            if not current_cluster_events:
                current_cluster_events.append(row)
            else:
                # Check if this event is within the time window of the previous
                last_ts = current_cluster_events[-1]['timestamp']
                if row['timestamp'] - last_ts <= time_window_us:
                    current_cluster_events.append(row)
                else:
                    # Save current cluster and start new one
                    cluster_df = pd.DataFrame(current_cluster_events)
                    buy_value = cluster_df[cluster_df['side'] == 'buy']['value'].sum()
                    sell_value = cluster_df[cluster_df['side'] == 'sell']['value'].sum()

                    clusters.append(LiquidationCluster(
                        start_timestamp=cluster_df['timestamp'].min(),
                        end_timestamp=cluster_df['timestamp'].max(),
                        total_value=cluster_df['value'].sum(),
                        event_count=len(cluster_df),
                        dominant_side='buy' if buy_value > sell_value else 'sell',
                        events=cluster_df
                    ))
                    current_cluster_events = [row]

        # Don't forget the last cluster
        if current_cluster_events:
            cluster_df = pd.DataFrame(current_cluster_events)
            buy_value = cluster_df[cluster_df['side'] == 'buy']['value'].sum()
            sell_value = cluster_df[cluster_df['side'] == 'sell']['value'].sum()

            clusters.append(LiquidationCluster(
                start_timestamp=cluster_df['timestamp'].min(),
                end_timestamp=cluster_df['timestamp'].max(),
                total_value=cluster_df['value'].sum(),
                event_count=len(cluster_df),
                dominant_side='buy' if buy_value > sell_value else 'sell',
                events=cluster_df
            ))

        logger.info(f"Created {len(clusters)} liquidation clusters")

        return clusters

    def get_large_clusters(
        self,
        clusters: List[LiquidationCluster],
        min_events: int = 3,
        min_value: Optional[float] = None,
        percentile: float = 90.0
    ) -> List[LiquidationCluster]:
        """
        Filter clusters to get only large/significant ones.

        Args:
            clusters: List of LiquidationCluster objects
            min_events: Minimum number of events in cluster
            min_value: Minimum total value threshold
            percentile: Percentile threshold for total value

        Returns:
            List of large clusters
        """
        if not clusters:
            return []

        # Filter by minimum events
        filtered = [c for c in clusters if c.event_count >= min_events]

        # Filter by value
        if min_value is None:
            values = [c.total_value for c in filtered]
            min_value = np.percentile(values, percentile) if values else 0

        large_clusters = [c for c in filtered if c.total_value >= min_value]

        logger.info(f"Found {len(large_clusters)} large clusters "
                   f"(min_events={min_events}, min_value=${min_value:,.2f})")

        return large_clusters

    def analyze_orderbook_around_liquidation(
        self,
        orderbook_metrics: pd.DataFrame,
        liquidation_ts: int,
        before_window_us: int = 60_000_000,  # 60 seconds before
        after_window_us: int = 60_000_000    # 60 seconds after
    ) -> Dict:
        """
        Analyze orderbook metrics around a liquidation event.

        Args:
            orderbook_metrics: DataFrame with orderbook metrics
            liquidation_ts: Timestamp of liquidation event
            before_window_us: Time window before liquidation (microseconds)
            after_window_us: Time window after liquidation (microseconds)

        Returns:
            Dictionary with before/after analysis
        """
        # Get metrics before liquidation
        before_mask = (
            (orderbook_metrics['timestamp'] >= liquidation_ts - before_window_us) &
            (orderbook_metrics['timestamp'] < liquidation_ts)
        )
        before_metrics = orderbook_metrics[before_mask]

        # Get metrics after liquidation
        after_mask = (
            (orderbook_metrics['timestamp'] > liquidation_ts) &
            (orderbook_metrics['timestamp'] <= liquidation_ts + after_window_us)
        )
        after_metrics = orderbook_metrics[after_mask]

        if before_metrics.empty or after_metrics.empty:
            return None

        return {
            'liquidation_ts': liquidation_ts,
            'before': {
                'spread_bps_mean': before_metrics['spread_bps'].mean(),
                'spread_bps_std': before_metrics['spread_bps'].std(),
                'order_imbalance_mean': before_metrics['order_imbalance'].mean(),
                'depth_50bps_mean': before_metrics['depth_50bps_total'].mean(),
                'mid_price_mean': before_metrics['mid_price'].mean(),
                'count': len(before_metrics)
            },
            'after': {
                'spread_bps_mean': after_metrics['spread_bps'].mean(),
                'spread_bps_std': after_metrics['spread_bps'].std(),
                'order_imbalance_mean': after_metrics['order_imbalance'].mean(),
                'depth_50bps_mean': after_metrics['depth_50bps_total'].mean(),
                'mid_price_mean': after_metrics['mid_price'].mean(),
                'count': len(after_metrics)
            },
            'changes': {
                'spread_bps_change': (
                    after_metrics['spread_bps'].mean() - before_metrics['spread_bps'].mean()
                ),
                'spread_bps_change_pct': (
                    (after_metrics['spread_bps'].mean() - before_metrics['spread_bps'].mean()) /
                    before_metrics['spread_bps'].mean() * 100
                    if before_metrics['spread_bps'].mean() != 0 else 0
                ),
                'depth_change_pct': (
                    (after_metrics['depth_50bps_total'].mean() - before_metrics['depth_50bps_total'].mean()) /
                    before_metrics['depth_50bps_total'].mean() * 100
                    if before_metrics['depth_50bps_total'].mean() != 0 else 0
                ),
                'price_change_pct': (
                    (after_metrics['mid_price'].mean() - before_metrics['mid_price'].mean()) /
                    before_metrics['mid_price'].mean() * 100
                    if before_metrics['mid_price'].mean() != 0 else 0
                ),
            }
        }



    def generate_liquidation_report(self) -> pd.DataFrame:
        """
        Generate a comprehensive report of all liquidations.

        Returns:
            DataFrame with enhanced liquidation data
        """
        report = self.liquidations.copy()

        # Add percentile rank
        report['value_percentile'] = report['value'].rank(pct=True) * 100

        # Add time delta to previous liquidation
        report['time_since_prev_us'] = report['timestamp'].diff()
        report['time_since_prev_sec'] = report['time_since_prev_us'] / 1_000_000

        # Add cumulative value
        report['cumulative_value'] = report['value'].cumsum()

        # Add rolling metrics
        report['rolling_5_value'] = report['value'].rolling(window=5, min_periods=1).sum()
        report['rolling_10_value'] = report['value'].rolling(window=10, min_periods=1).sum()

        return report

