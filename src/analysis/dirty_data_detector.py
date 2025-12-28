"""
Dirty Data Detector Module
==========================

Detects and classifies dirty data patterns in validation datasets.
Implements the Sanitization Policy: ACCEPT, REPAIR, QUARANTINE.
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from datetime import datetime
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SanitizationClass(Enum):
    """Classification for data sanitization."""
    ACCEPT = "ACCEPT"           # Normal data
    REPAIR = "REPAIR"           # Can be fixed (e.g., reorder)
    QUARANTINE = "QUARANTINE"   # Unreliable, exclude from decisions


class DirtyDataType(Enum):
    """Types of dirty data patterns."""
    OUT_OF_ORDER = "out_of_order_timestamp"
    DUPLICATE = "duplicate_event"
    FAT_FINGER = "fat_finger_price"
    CROSSED_MARKET = "crossed_market"
    DELAYED_EVENT = "delayed_event"
    MISSING_GAP = "missing_gap"
    INVALID_AMOUNT = "invalid_amount"
    STALE_DATA = "stale_data"


@dataclass
class DirtyDataEvent:
    """A detected dirty data event."""
    timestamp: int
    local_timestamp: int
    data_type: DirtyDataType
    sanitization: SanitizationClass
    details: str
    original_values: Dict
    suggested_repair: Optional[Dict] = None


class DirtyDataDetector:
    """
    Detects dirty data patterns in trading data.

    Detection patterns:
    1. Out-of-order timestamps
    2. Duplicate events
    3. Fat-finger prices (extreme deviations)
    4. Crossed market (bid >= ask)
    5. Event delays (local_timestamp >> timestamp)
    6. Missing gaps (large timestamp jumps)
    """

    def __init__(
        self,
        fat_finger_threshold_pct: float = 5.0,      # 5% price deviation
        delay_threshold_us: int = 1_000_000,         # 1 second delay
        gap_threshold_us: int = 5_000_000,           # 5 second gap
        duplicate_window_us: int = 1000,             # 1ms for duplicates
        baseline_price: Optional[float] = None       # Reference price for fat finger
    ):
        """
        Initialize the detector.

        Args:
            fat_finger_threshold_pct: Percentage deviation to consider fat finger
            delay_threshold_us: Microseconds delay to flag as delayed
            gap_threshold_us: Microseconds gap to flag as missing
            duplicate_window_us: Window for duplicate detection
            baseline_price: Optional baseline price for fat finger detection
        """
        self.fat_finger_threshold_pct = fat_finger_threshold_pct
        self.delay_threshold_us = delay_threshold_us
        self.gap_threshold_us = gap_threshold_us
        self.duplicate_window_us = duplicate_window_us
        self.baseline_price = baseline_price

        self.detected_events: List[DirtyDataEvent] = []
        self.statistics: Dict = {
            'total_rows': 0,
            'clean_rows': 0,
            'dirty_rows': 0,
            'by_type': {},
            'by_sanitization': {'ACCEPT': 0, 'REPAIR': 0, 'QUARANTINE': 0}
        }

    def detect_out_of_order(self, df: pd.DataFrame) -> List[DirtyDataEvent]:
        """Detect out-of-order timestamps."""
        events = []

        if 'timestamp' not in df.columns:
            return events

        # Find rows where timestamp is less than previous
        df_sorted = df.sort_index()
        timestamp_diff = df_sorted['timestamp'].diff()
        out_of_order_mask = timestamp_diff < 0

        for idx in df_sorted[out_of_order_mask].index:
            row = df_sorted.loc[idx]
            prev_ts = df_sorted.loc[idx - 1, 'timestamp'] if idx > 0 else None

            events.append(DirtyDataEvent(
                timestamp=int(row['timestamp']),
                local_timestamp=int(row.get('local_timestamp', row['timestamp'])),
                data_type=DirtyDataType.OUT_OF_ORDER,
                sanitization=SanitizationClass.REPAIR,
                details=f"timestamp {row['timestamp']} < previous {prev_ts}",
                original_values={'timestamp': row['timestamp'], 'prev_timestamp': prev_ts},
                suggested_repair={'action': 'reorder_by_local_timestamp'}
            ))

        return events

    def detect_duplicates(self, df: pd.DataFrame) -> List[DirtyDataEvent]:
        """Detect duplicate events."""
        events = []

        # Check for exact duplicates
        duplicate_mask = df.duplicated(keep='first')

        for idx in df[duplicate_mask].index:
            row = df.loc[idx]
            events.append(DirtyDataEvent(
                timestamp=int(row['timestamp']),
                local_timestamp=int(row.get('local_timestamp', row['timestamp'])),
                data_type=DirtyDataType.DUPLICATE,
                sanitization=SanitizationClass.REPAIR,
                details="exact duplicate row",
                original_values=row.to_dict(),
                suggested_repair={'action': 'remove_duplicate'}
            ))

        return events

    def detect_fat_finger(self, df: pd.DataFrame) -> List[DirtyDataEvent]:
        """Detect fat finger prices (extreme deviations from baseline)."""
        events = []

        if 'price' not in df.columns:
            return events

        # Use rolling median as baseline if not provided
        if self.baseline_price is None:
            rolling_median = df['price'].rolling(window=100, min_periods=1).median()
        else:
            rolling_median = pd.Series([self.baseline_price] * len(df), index=df.index)

        # Calculate percentage deviation
        deviation_pct = abs(df['price'] - rolling_median) / rolling_median * 100
        fat_finger_mask = deviation_pct > self.fat_finger_threshold_pct

        for idx in df[fat_finger_mask].index:
            row = df.loc[idx]
            median = rolling_median.loc[idx]
            dev = deviation_pct.loc[idx]

            events.append(DirtyDataEvent(
                timestamp=int(row['timestamp']),
                local_timestamp=int(row.get('local_timestamp', row['timestamp'])),
                data_type=DirtyDataType.FAT_FINGER,
                sanitization=SanitizationClass.QUARANTINE,
                details=f"price {row['price']:.2f} deviates {dev:.1f}% from median {median:.2f}",
                original_values={'price': row['price'], 'median': median, 'deviation_pct': dev}
            ))

        return events

    def detect_crossed_market(self, orderbook_df: pd.DataFrame) -> List[DirtyDataEvent]:
        """Detect crossed market conditions (bid >= ask)."""
        events = []

        if 'side' not in orderbook_df.columns or 'price' not in orderbook_df.columns:
            return events

        # Group by timestamp
        for ts, group in orderbook_df.groupby('timestamp'):
            bids = group[group['side'] == 'bid']
            asks = group[group['side'] == 'ask']

            if bids.empty or asks.empty:
                continue

            best_bid = bids['price'].max()
            best_ask = asks['price'].min()

            if best_bid >= best_ask:
                events.append(DirtyDataEvent(
                    timestamp=int(ts),
                    local_timestamp=int(group['local_timestamp'].iloc[0]),
                    data_type=DirtyDataType.CROSSED_MARKET,
                    sanitization=SanitizationClass.QUARANTINE,
                    details=f"crossed market: bid={best_bid:.2f} >= ask={best_ask:.2f}",
                    original_values={'best_bid': best_bid, 'best_ask': best_ask}
                ))

        return events

    def detect_delayed_events(self, df: pd.DataFrame) -> List[DirtyDataEvent]:
        """Detect events with significant delay (local_timestamp >> timestamp)."""
        events = []

        if 'timestamp' not in df.columns or 'local_timestamp' not in df.columns:
            return events

        delay = df['local_timestamp'] - df['timestamp']
        delayed_mask = delay > self.delay_threshold_us

        for idx in df[delayed_mask].index:
            row = df.loc[idx]
            delay_us = row['local_timestamp'] - row['timestamp']
            delay_sec = delay_us / 1_000_000

            events.append(DirtyDataEvent(
                timestamp=int(row['timestamp']),
                local_timestamp=int(row['local_timestamp']),
                data_type=DirtyDataType.DELAYED_EVENT,
                sanitization=SanitizationClass.REPAIR,
                details=f"event delayed by {delay_sec:.2f} seconds",
                original_values={'delay_us': delay_us, 'delay_sec': delay_sec},
                suggested_repair={'action': 'use_local_timestamp_for_ordering'}
            ))

        return events

    def detect_gaps(self, df: pd.DataFrame) -> List[DirtyDataEvent]:
        """Detect large gaps in timestamps (possible missing data)."""
        events = []

        if 'timestamp' not in df.columns:
            return events

        df_sorted = df.sort_values('timestamp')
        gaps = df_sorted['timestamp'].diff()
        gap_mask = gaps > self.gap_threshold_us

        for idx in df_sorted[gap_mask].index:
            row = df_sorted.loc[idx]
            gap = gaps.loc[idx]
            gap_sec = gap / 1_000_000

            events.append(DirtyDataEvent(
                timestamp=int(row['timestamp']),
                local_timestamp=int(row.get('local_timestamp', row['timestamp'])),
                data_type=DirtyDataType.MISSING_GAP,
                sanitization=SanitizationClass.QUARANTINE,
                details=f"gap of {gap_sec:.2f} seconds before this event",
                original_values={'gap_us': gap, 'gap_sec': gap_sec}
            ))

        return events

    def detect_invalid_amounts(self, df: pd.DataFrame) -> List[DirtyDataEvent]:
        """Detect invalid amounts (negative, zero, or extremely large)."""
        events = []

        if 'amount' not in df.columns:
            return events

        # Invalid: negative, zero, or extremely large (> 1000 BTC)
        invalid_mask = (df['amount'] <= 0) | (df['amount'] > 1000)

        for idx in df[invalid_mask].index:
            row = df.loc[idx]

            events.append(DirtyDataEvent(
                timestamp=int(row['timestamp']),
                local_timestamp=int(row.get('local_timestamp', row['timestamp'])),
                data_type=DirtyDataType.INVALID_AMOUNT,
                sanitization=SanitizationClass.QUARANTINE,
                details=f"invalid amount: {row['amount']}",
                original_values={'amount': row['amount']}
            ))

        return events

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        data_type: str = "unknown"
    ) -> Dict:
        """
        Analyze a DataFrame for all dirty data patterns.

        Args:
            df: DataFrame to analyze
            data_type: Type of data (orderbook, trades, liquidations, ticker)

        Returns:
            Dictionary with detection results
        """
        logger.info(f"Analyzing {len(df)} rows of {data_type} data...")

        all_events = []

        # Run all detectors
        all_events.extend(self.detect_out_of_order(df))
        all_events.extend(self.detect_duplicates(df))
        all_events.extend(self.detect_delayed_events(df))
        all_events.extend(self.detect_gaps(df))

        if 'price' in df.columns:
            all_events.extend(self.detect_fat_finger(df))

        if 'amount' in df.columns:
            all_events.extend(self.detect_invalid_amounts(df))

        if data_type == 'orderbook' and 'side' in df.columns:
            all_events.extend(self.detect_crossed_market(df))

        # Update statistics
        self.detected_events.extend(all_events)
        self.statistics['total_rows'] += len(df)

        # Count by type and sanitization
        dirty_timestamps = set()
        for event in all_events:
            dirty_timestamps.add(event.timestamp)

            type_key = event.data_type.value
            if type_key not in self.statistics['by_type']:
                self.statistics['by_type'][type_key] = 0
            self.statistics['by_type'][type_key] += 1

            self.statistics['by_sanitization'][event.sanitization.value] += 1

        self.statistics['dirty_rows'] += len(dirty_timestamps)
        self.statistics['clean_rows'] = self.statistics['total_rows'] - self.statistics['dirty_rows']

        logger.info(f"Detected {len(all_events)} dirty data events in {data_type}")

        return {
            'data_type': data_type,
            'total_rows': len(df),
            'events_detected': len(all_events),
            'unique_dirty_timestamps': len(dirty_timestamps),
            'by_type': {e.data_type.value: sum(1 for x in all_events if x.data_type == e.data_type)
                       for e in all_events}
        }

    def generate_sanitization_log(self) -> List[Dict]:
        """Generate sanitization log entries."""
        return [
            {
                'timestamp': e.timestamp,
                'local_timestamp': e.local_timestamp,
                'type': e.data_type.value,
                'sanitization': e.sanitization.value,
                'details': e.details,
                'original_values': e.original_values,
                'suggested_repair': e.suggested_repair
            }
            for e in self.detected_events
        ]

    def generate_report(self) -> Dict:
        """Generate comprehensive dirty data report."""
        return {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_rows_analyzed': self.statistics['total_rows'],
                'clean_rows': self.statistics['clean_rows'],
                'dirty_rows': self.statistics['dirty_rows'],
                'dirty_rate': self.statistics['dirty_rows'] / self.statistics['total_rows']
                             if self.statistics['total_rows'] > 0 else 0,
            },
            'by_pattern_type': self.statistics['by_type'],
            'by_sanitization_class': self.statistics['by_sanitization'],
            'sanitization_policy': {
                'ACCEPT': 'Normal data, use as-is',
                'REPAIR': 'Can be fixed (reorder, remove duplicate)',
                'QUARANTINE': 'Unreliable, exclude from decision making'
            },
            'detection_thresholds': {
                'fat_finger_threshold_pct': self.fat_finger_threshold_pct,
                'delay_threshold_seconds': self.delay_threshold_us / 1_000_000,
                'gap_threshold_seconds': self.gap_threshold_us / 1_000_000,
            }
        }


def run_dirty_data_analysis(
    validation_dir: str,
    output_dir: str,
    sample_size: int = 1_000_000
) -> Tuple[Dict, List[Dict]]:
    """
    Run dirty data analysis on validation data.

    Args:
        validation_dir: Path to validation data directory
        output_dir: Output directory for results
        sample_size: Number of rows to sample from large files

    Returns:
        Tuple of (report, sanitization_log)
    """
    validation_dir = Path(validation_dir)
    output_dir = Path(output_dir)

    detector = DirtyDataDetector(
        fat_finger_threshold_pct=5.0,
        delay_threshold_us=1_000_000,
        gap_threshold_us=5_000_000,
    )

    results = []

    # Analyze each file
    files_to_analyze = [
        ('liquidations.csv', 'liquidations', None),  # Small file, read all
        ('ticker.csv', 'ticker', None),              # Medium file
        ('trades.csv', 'trades', sample_size),       # Large file, sample
        ('orderbook.csv', 'orderbook', sample_size), # Very large file, sample
    ]

    for filename, data_type, limit in files_to_analyze:
        filepath = validation_dir / filename

        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing {filename}...")
        logger.info(f"{'='*60}")

        try:
            if limit:
                # Use head for large files
                logger.info(f"Reading first {limit:,} rows...")
                df = pd.read_csv(filepath, nrows=limit)
            else:
                df = pd.read_csv(filepath)

            result = detector.analyze_dataframe(df, data_type)
            results.append(result)

        except Exception as e:
            logger.error(f"Error analyzing {filename}: {e}")
            results.append({'data_type': data_type, 'error': str(e)})

    # Generate reports
    report = detector.generate_report()
    report['file_results'] = results

    sanitization_log = detector.generate_sanitization_log()

    # Save results
    with open(output_dir / "dirty_data_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Saved dirty data report to {output_dir / 'dirty_data_report.json'}")

    with open(output_dir / "sanitization_log.jsonl", 'w') as f:
        for entry in sanitization_log:
            f.write(json.dumps(entry, default=str) + '\n')
    logger.info(f"Saved {len(sanitization_log)} sanitization entries to sanitization_log.jsonl")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("DIRTY DATA ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total rows analyzed: {report['summary']['total_rows_analyzed']:,}")
    logger.info(f"Clean rows: {report['summary']['clean_rows']:,}")
    logger.info(f"Dirty rows: {report['summary']['dirty_rows']:,}")
    logger.info(f"Dirty rate: {report['summary']['dirty_rate']:.2%}")

    logger.info("\nBy pattern type:")
    for pattern, count in report['by_pattern_type'].items():
        logger.info(f"  {pattern}: {count:,}")

    logger.info("\nBy sanitization class:")
    for cls, count in report['by_sanitization_class'].items():
        logger.info(f"  {cls}: {count:,}")

    return report, sanitization_log


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Detect dirty data patterns')
    parser.add_argument('--validation-dir', type=str, default='data/validation')
    parser.add_argument('--output', type=str, default='output/phase1')
    parser.add_argument('--sample-size', type=int, default=1000000)
    args = parser.parse_args()

    run_dirty_data_analysis(args.validation_dir, args.output, args.sample_size)

