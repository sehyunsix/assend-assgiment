"""
Orderbook Metrics Module
========================

Defines and calculates orderbook stability metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OrderbookSnapshot:
    """Represents a processed orderbook snapshot at a specific timestamp."""
    timestamp: int
    local_timestamp: int
    best_bid: float
    best_ask: float
    bid_volume_total: float
    ask_volume_total: float
    bid_levels: int
    ask_levels: int
    mid_price: float
    spread: float
    spread_bps: float
    order_imbalance: float
    depth_10bps_bid: float
    depth_10bps_ask: float
    depth_50bps_bid: float
    depth_50bps_ask: float
    obwa_spread: float = 0.0
    obwa_spread_bps: float = 0.0


class OrderbookMetrics:
    """
    Calculate orderbook stability metrics.

    Metrics:
    - Bid-Ask Spread: best_ask - best_bid
    - Spread Ratio (bps): spread / mid_price * 10000
    - Market Depth: Total volume within N bps of mid price
    - Order Imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
    - Depth Imbalance: bid/ask volume ratio at various levels
    """

    def __init__(self):
        self.metrics_cache: Dict[int, OrderbookSnapshot] = {}

    def calculate_obwa(self, df: pd.DataFrame, side: str, levels: int = 10) -> float:
        """Calculate Order Book Weighted Average price for a side."""
        side_df = df[df['side'] == side].copy()
        if side_df.empty:
            return 0.0

        # Sort by price: descending for bids, ascending for asks
        ascending = (side == 'ask')
        side_df = side_df.sort_values('price', ascending=ascending).head(levels)

        if side_df['amount'].sum() == 0:
            return 0.0

        return (side_df['price'] * side_df['amount']).sum() / side_df['amount'].sum()

    def process_orderbook_snapshot(
        self,
        df: pd.DataFrame,
        timestamp: int
    ) -> Optional[OrderbookSnapshot]:
        """
        Process a single orderbook snapshot and calculate metrics.

        Args:
            df: DataFrame containing orderbook data for a single timestamp
            timestamp: The timestamp being processed

        Returns:
            OrderbookSnapshot with calculated metrics, or None if invalid
        """
        if df.empty:
            return None

        # Separate bids and asks
        bids = df[df['side'] == 'bid'].copy()
        asks = df[df['side'] == 'ask'].copy()

        if bids.empty or asks.empty:
            return None

        # Best bid/ask
        best_bid = bids['price'].max()
        best_ask = asks['price'].min()

        # Validate: best_bid should be less than best_ask (no crossed market)
        if best_bid >= best_ask:
            logger.warning(f"Crossed market at {timestamp}: bid={best_bid}, ask={best_ask}")
            return None

        # Calculate mid price and spread
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_bps = (spread / mid_price) * 10000

        # OBWA Spread (top 10 levels)
        obwa_bid = self.calculate_obwa(df, 'bid', levels=10)
        obwa_ask = self.calculate_obwa(df, 'ask', levels=10)
        obwa_spread = obwa_ask - obwa_bid if (obwa_ask > 0 and obwa_bid > 0) else spread
        obwa_spread_bps = (obwa_spread / mid_price) * 10000 if mid_price > 0 else 0

        # Total volumes
        bid_volume_total = bids['amount'].sum()
        ask_volume_total = asks['amount'].sum()

        # Order imbalance
        total_volume = bid_volume_total + ask_volume_total
        order_imbalance = (bid_volume_total - ask_volume_total) / total_volume if total_volume > 0 else 0

        # Market depth at different levels (10 bps and 50 bps from mid)
        depth_10bps_bid = bids[bids['price'] >= mid_price * (1 - 0.001)]['amount'].sum()
        depth_10bps_ask = asks[asks['price'] <= mid_price * (1 + 0.001)]['amount'].sum()
        depth_50bps_bid = bids[bids['price'] >= mid_price * (1 - 0.005)]['amount'].sum()
        depth_50bps_ask = asks[asks['price'] <= mid_price * (1 + 0.005)]['amount'].sum()

        return OrderbookSnapshot(
            timestamp=timestamp,
            local_timestamp=df['local_timestamp'].iloc[0],
            best_bid=best_bid,
            best_ask=best_ask,
            bid_volume_total=bid_volume_total,
            ask_volume_total=ask_volume_total,
            bid_levels=len(bids),
            ask_levels=len(asks),
            mid_price=mid_price,
            spread=spread,
            spread_bps=spread_bps,
            order_imbalance=order_imbalance,
            depth_10bps_bid=depth_10bps_bid,
            depth_10bps_ask=depth_10bps_ask,
            depth_50bps_bid=depth_50bps_bid,
            depth_50bps_ask=depth_50bps_ask,
            obwa_spread=obwa_spread,
            obwa_spread_bps=obwa_spread_bps
        )

    def calculate_window_spread(
        self,
        metrics_df: pd.DataFrame,
        window_size: int = 30,
        freq: str = 'S'
    ) -> pd.Series:
        """
        Calculate Window Spread: max(mid_price) - min(mid_price) over a window.

        Args:
            metrics_df: DataFrame with mid_price and timestamp
            window_size: Window size in seconds
            freq: Frequency string for rolling window

        Returns:
            Series with window spread values
        """
        # Ensure timestamp is datetime for rolling window if needed,
        # but here we assume metrics_df is sorted by timestamp (us)
        # 1s = 1,000,000 us
        window_us = window_size * 1_000_000

        # Using a rolling window on index if it's regularly spaced,
        # or use rolling with time offset if converted to datetime
        temp_df = metrics_df.copy()
        temp_df['dt'] = pd.to_datetime(temp_df['timestamp'], unit='us')
        temp_df = temp_df.set_index('dt')

        rolling = temp_df['mid_price'].rolling(window=f'{window_size}s')
        window_spread = rolling.max() - rolling.min()

        return window_spread.values

    def calculate_price_impact(
        self,
        trades_df: pd.DataFrame,
        metrics_df: pd.DataFrame,
        lookahead_seconds: int = 30
    ) -> pd.DataFrame:
        """
        Calculate Price Impact: Q_t * (M_{t+x} - M_t) / M_t

        Args:
            trades_df: DataFrame with trades (timestamp, side, amount, price)
            metrics_df: DataFrame with orderbook metrics (timestamp, mid_price)
            lookahead_seconds: x in seconds

        Returns:
            DataFrame with price impact per trade
        """
        if trades_df.empty or metrics_df.empty:
            return pd.DataFrame()

        # Merge trades with mid-price at trade time
        # Use merge_asof to find the closest mid-price at or before trade time
        trades = trades_df.sort_values('timestamp')
        metrics = metrics_df[['timestamp', 'mid_price']].sort_values('timestamp')

        impact_df = pd.merge_asof(
            trades,
            metrics,
            on='timestamp',
            direction='backward'
        )

        # Find mid-price at t + lookahead
        lookahead_us = lookahead_seconds * 1_000_000
        metrics_future = metrics.copy()
        metrics_future['timestamp_shifted'] = metrics_future['timestamp'] - lookahead_us

        impact_df = pd.merge_asof(
            impact_df,
            metrics_future[['timestamp_shifted', 'mid_price']].rename(columns={'mid_price': 'mid_price_future'}),
            left_on='timestamp',
            right_on='timestamp_shifted',
            direction='forward'
        )

        # Calculate impact
        # Q_t: Buy = 1, Sell = -1. In our data 'side' might be 'buy'/'sell' or 'bid'/'ask'
        # liquidations.csv uses 'buy'/'sell'. trades.csv might be different.
        # Assuming 'side' column exists and needs mapping
        side_map = {'buy': 1, 'sell': -1, 'bid': -1, 'ask': 1} # For liquidations: Sell = Long Liq (Price down), Buy = Short Liq (Price up)
        q_t = impact_df['side'].map(side_map).fillna(0)

        impact_df['price_impact'] = q_t * (impact_df['mid_price_future'] - impact_df['mid_price']) / impact_df['mid_price']

        return impact_df

    def calculate_vpin(
        self,
        trades_df: pd.DataFrame,
        bucket_size_vol: float,
        num_buckets: int = 50
    ) -> float:
        """
        Calculate VPIN (Volume-Synchronized Probability of Informed Trading).

        Args:
            trades_df: DataFrame with trades (timestamp, side, amount)
            bucket_size_vol: V, fixed volume per bucket
            num_buckets: n, number of buckets to use

        Returns:
            VPIN value
        """
        if trades_df.empty:
            return 0.0

        # 1. Assign trades to volume buckets
        trades = trades_df.copy()
        trades['cum_vol'] = trades['amount'].cumsum()
        trades['bucket'] = (trades['cum_vol'] // bucket_size_vol).astype(int)

        # 2. For each bucket, calculate |V_buy - V_sell|
        # Map side to buy/sell volume
        trades['v_buy'] = np.where(trades['side'] == 'buy', trades['amount'], 0)
        trades['v_sell'] = np.where(trades['side'] == 'sell', trades['amount'], 0)

        bucket_agg = trades.groupby('bucket').agg({
            'v_buy': 'sum',
            'v_sell': 'sum'
        }).head(num_buckets)

        if len(bucket_agg) < num_buckets:
            logger.warning(f"Not enough trades for {num_buckets} buckets. Using {len(bucket_agg)}.")
            n = len(bucket_agg)
        else:
            n = num_buckets

        if n == 0:
            return 0.0

        oi_sum = (bucket_agg['v_buy'] - bucket_agg['v_sell']).abs().sum()
        vpin = oi_sum / (n * bucket_size_vol)

        return vpin

    def calculate_hybrid_indicator(
        self,
        window_spread: float,
        manual_spread: float,
        obwa_spread: float
    ) -> float:
        """
        Calculate Hybrid Indicator: 0.62 * Window + 0.19 * MS + 0.19 * OBWA
        """
        return 0.62 * window_spread + 0.19 * manual_spread + 0.19 * obwa_spread

    def calculate_metrics_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate metrics for a batch of orderbook data (multiple timestamps)
        using vectorized groupby operations. Much faster than loop-based processing.
        """
        if df.empty:
            return pd.DataFrame()

        # 1. Basic price info per timestamp
        # Filter bids and asks
        bids = df[df['side'] == 'bid']
        asks = df[df['side'] == 'ask']

        # Group by timestamp to get best prices
        bid_stats = bids.groupby('timestamp')['price'].agg(['max', 'count']).rename(columns={'max': 'best_bid', 'count': 'bid_levels'})
        ask_stats = asks.groupby('timestamp')['price'].agg(['min', 'count']).rename(columns={'min': 'best_ask', 'count': 'ask_levels'})

        # Total volumes
        bid_vols = bids.groupby('timestamp')['amount'].sum().rename('bid_volume_total')
        ask_vols = asks.groupby('timestamp')['amount'].sum().rename('ask_volume_total')

        # Merge basic stats
        metrics = pd.concat([bid_stats, ask_stats, bid_vols, ask_vols], axis=1).dropna()

        # Add local_timestamp (take the first one for each timestamp)
        local_ts = df.groupby('timestamp')['local_timestamp'].first().rename('local_timestamp')
        metrics = metrics.join(local_ts)

        # Calculate derived metrics
        metrics['mid_price'] = (metrics['best_bid'] + metrics['best_ask']) / 2
        metrics['spread'] = metrics['best_ask'] - metrics['best_bid']
        metrics['spread_bps'] = (metrics['spread'] / metrics['mid_price']) * 10000

        # Filter out crossed markets
        metrics = metrics[metrics['best_bid'] < metrics['best_ask']]

        # Order imbalance
        total_vol = metrics['bid_volume_total'] + metrics['ask_volume_total']
        metrics['order_imbalance'] = (metrics['bid_volume_total'] - metrics['ask_volume_total']) / total_vol

        # 2. Market Depth (10bps, 50bps)
        # For efficiency, we join mid_price back to original df and filter
        df_with_mid = df.merge(metrics[['mid_price']], left_on='timestamp', right_index=True)

        # 10bps depth
        depth_10b = df_with_mid[
            ((df_with_mid['side'] == 'bid') & (df_with_mid['price'] >= df_with_mid['mid_price'] * 0.999)) |
            ((df_with_mid['side'] == 'ask') & (df_with_mid['price'] <= df_with_mid['mid_price'] * 1.001))
        ]
        depth_10_agg = depth_10b.groupby(['timestamp', 'side'])['amount'].sum().unstack().rename(columns={'bid': 'depth_10bps_bid', 'ask': 'depth_10bps_ask'})

        # 50bps depth
        depth_50b = df_with_mid[
            ((df_with_mid['side'] == 'bid') & (df_with_mid['price'] >= df_with_mid['mid_price'] * 0.995)) |
            ((df_with_mid['side'] == 'ask') & (df_with_mid['price'] <= df_with_mid['mid_price'] * 1.005))
        ]
        depth_50_agg = depth_50b.groupby(['timestamp', 'side'])['amount'].sum().unstack().rename(columns={'bid': 'depth_50bps_bid', 'ask': 'depth_50bps_ask'})

        # Merge depth back
        metrics = metrics.join(depth_10_agg).join(depth_50_agg).fillna(0)

        # Combined depth
        metrics['depth_10bps_total'] = metrics['depth_10bps_bid'] + metrics['depth_10bps_ask']
        metrics['depth_50bps_total'] = metrics['depth_50bps_bid'] + metrics['depth_50bps_ask']

        # 3. Advanced Metrics: OBWA (Top 10)
        # This is harder to vectorize perfectly without a loop over timestamps or complex aggregation
        # but we can optimize it by sorting first
        # For now, if performance is key, we might skip full OBWA in batch or use a simplified version
        # Let's use a groupby apply for OBWA if needed, but it might be slower

        return metrics.reset_index()

    def calculate_metrics_for_timestamps(
        self,
        orderbook_df: pd.DataFrame,
        timestamps: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Calculate orderbook metrics for multiple timestamps.

        Args:
            orderbook_df: DataFrame with orderbook data
            timestamps: Optional list of specific timestamps to process

        Returns:
            DataFrame with metrics for each timestamp
        """
        if timestamps is None:
            timestamps = orderbook_df['timestamp'].unique()

        results = []

        for ts in timestamps:
            snapshot_df = orderbook_df[orderbook_df['timestamp'] == ts]
            snapshot = self.process_orderbook_snapshot(snapshot_df, ts)

            if snapshot:
                results.append({
                    'timestamp': snapshot.timestamp,
                    'local_timestamp': snapshot.local_timestamp,
                    'best_bid': snapshot.best_bid,
                    'best_ask': snapshot.best_ask,
                    'mid_price': snapshot.mid_price,
                    'spread': snapshot.spread,
                    'spread_bps': snapshot.spread_bps,
                    'obwa_spread': snapshot.obwa_spread,
                    'obwa_spread_bps': snapshot.obwa_spread_bps,
                    'bid_volume_total': snapshot.bid_volume_total,
                    'ask_volume_total': snapshot.ask_volume_total,
                    'bid_levels': snapshot.bid_levels,
                    'ask_levels': snapshot.ask_levels,
                    'order_imbalance': snapshot.order_imbalance,
                    'depth_10bps_bid': snapshot.depth_10bps_bid,
                    'depth_10bps_ask': snapshot.depth_10bps_ask,
                    'depth_50bps_bid': snapshot.depth_50bps_bid,
                    'depth_50bps_ask': snapshot.depth_50bps_ask,
                    'depth_10bps_total': snapshot.depth_10bps_bid + snapshot.depth_10bps_ask,
                    'depth_50bps_total': snapshot.depth_50bps_bid + snapshot.depth_50bps_ask,
                })

        return pd.DataFrame(results)

    def aggregate_metrics_by_time_window(
        self,
        metrics_df: pd.DataFrame,
        window_us: int = 1_000_000
    ) -> pd.DataFrame:
        """
        Aggregate metrics into time windows.

        Args:
            metrics_df: DataFrame with per-timestamp metrics
            window_us: Window size in microseconds (default 1 second)

        Returns:
            DataFrame with aggregated metrics per window
        """
        # Create time window column
        metrics_df = metrics_df.copy()
        metrics_df['time_window'] = (metrics_df['timestamp'] // window_us) * window_us

        # Aggregate by time window
        agg_funcs = {
            'mid_price': ['mean', 'std', 'min', 'max'],
            'spread_bps': ['mean', 'std', 'max'],
            'order_imbalance': ['mean', 'std'],
            'depth_10bps_total': ['mean', 'min'],
            'depth_50bps_total': ['mean', 'min'],
            'bid_levels': 'mean',
            'ask_levels': 'mean',
        }

        aggregated = metrics_df.groupby('time_window').agg(agg_funcs)
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]

        return aggregated.reset_index()

    @staticmethod
    def calculate_stability_score(metrics_df: pd.DataFrame) -> pd.Series:
        """
        Calculate an optimized composite stability score.

        Weights:
        - Toxicity (40%): VPIN & Price Impact (Lower is better)
        - Inertia (30%): Market Depth (Higher is better)
        - Cost (20%): Spread (Lower is better)
        - Direction (10%): Order Imbalance (Closer to 0 is better)
        """
        df = metrics_df.copy()

        # 1. Normalize components (0-1)
        # For lower-is-better metrics, we use (1 - rank_pct)
        spread_score = 1 - df['spread_bps'].rank(pct=True)

        # Inertia (Depth) - higher is better
        depth_score = df['depth_50bps_total'].rank(pct=True)

        # Direction (Imbalance) - closer to 0 is better
        direction_score = 1 - df['order_imbalance'].abs().rank(pct=True)

        # Toxicity (VPIN + Price Impact) - lower is better
        # Handle cases where these might be missing
        vpin_score = 1 - df['vpin'].rank(pct=True) if 'vpin' in df.columns else pd.Series(0.5, index=df.index)
        impact_score = 1 - df['price_impact'].abs().rank(pct=True) if 'price_impact' in df.columns else pd.Series(0.5, index=df.index)
        toxicity_score = (vpin_score + impact_score) / 2

        # 2. Weighted average
        stability_score = (
            0.40 * toxicity_score +
            0.30 * depth_score +
            0.20 * spread_score +
            0.10 * direction_score
        )

        return stability_score

    def find_absorption_range(
        self,
        orderbook_df: pd.DataFrame,
        liquidation_volume: float,
        timestamp: int
    ) -> float:
        """
        Calculate how many bps the orderbook needs to absorb a specific volume.
        """
        snapshot = orderbook_df[orderbook_df['timestamp'] == timestamp]
        if snapshot.empty:
            return 0.0

        # Try both sides (sell volume hitting bids, buy volume hitting asks)
        results = []
        for side in ['bid', 'ask']:
            side_df = snapshot[snapshot['side'] == side].copy()
            ascending = (side == 'ask')
            side_df = side_df.sort_values('price', ascending=ascending)

            side_df['cum_amount'] = side_df['amount'].cumsum()
            match = side_df[side_df['cum_amount'] >= liquidation_volume].head(1)

            if not match.empty:
                best_price = side_df['price'].iloc[0]
                worst_price = match['price'].iloc[0]
                bps_impact = abs(worst_price - best_price) / best_price * 10000
                results.append(bps_impact)
            else:
                # If volume is larger than total depth, return max possible impact or large number
                results.append(1000.0) # Placeholder for extreme impact

        return max(results) if results else 0.0

    @staticmethod
    def calculate_dynamic_range(
        avg_spread_bps: float,
        price_volatility_bps: float,
        k: float = 2.0,
        beta: float = 1.5
    ) -> float:
        """
        Calculate dynamic observation range.
        Range_opt = k * Average_Spread + beta * Volatility
        """
        return k * avg_spread_bps + beta * price_volatility_bps

    @staticmethod
    def detect_anomalies(
        metrics_df: pd.DataFrame,
        spread_threshold_bps: float = 10.0,
        imbalance_threshold: float = 0.5,
        depth_drop_pct: float = 0.5
    ) -> pd.DataFrame:
        """
        Detect anomalies in orderbook metrics.

        Args:
            metrics_df: DataFrame with orderbook metrics
            spread_threshold_bps: Spread threshold in basis points
            imbalance_threshold: Order imbalance threshold (absolute)
            depth_drop_pct: Depth drop percentage threshold

        Returns:
            DataFrame with anomaly flags
        """
        anomalies = metrics_df.copy()

        # Wide spread anomaly
        anomalies['anomaly_wide_spread'] = anomalies['spread_bps'] > spread_threshold_bps

        # High imbalance anomaly
        anomalies['anomaly_high_imbalance'] = anomalies['order_imbalance'].abs() > imbalance_threshold

        # Depth drop anomaly (compared to rolling median)
        rolling_median = anomalies['depth_50bps_total'].rolling(window=100, min_periods=10).median()
        anomalies['anomaly_depth_drop'] = anomalies['depth_50bps_total'] < rolling_median * (1 - depth_drop_pct)

        # Combined anomaly flag
        anomalies['is_anomaly'] = (
            anomalies['anomaly_wide_spread'] |
            anomalies['anomaly_high_imbalance'] |
            anomalies['anomaly_depth_drop']
        )

        return anomalies

