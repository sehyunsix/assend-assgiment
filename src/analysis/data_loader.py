"""
Data Loader Module
==================

Handles loading large CSV files using dask for efficient parallel processing.
"""

import dask.dataframe as dd
import pandas as pd
from pathlib import Path
from typing import Optional
import logging


logger = logging.getLogger(__name__)


class DataLoader:
    """
    Efficient data loader for large Binance Futures CSV files.
    Uses dask for parallel processing of large files.
    """

    # Column data types for each file type
    ORDERBOOK_DTYPES = {
        'exchange': 'category',
        'symbol': 'category',
        'timestamp': 'int64',
        'local_timestamp': 'int64',
        'is_snapshot': 'bool',
        'side': 'category',
        'price': 'float64',
        'amount': 'float64'
    }



    LIQUIDATIONS_DTYPES = {
        'exchange': 'category',
        'symbol': 'category',
        'timestamp': 'int64',
        'local_timestamp': 'int64',
        'id': 'object',  # Can be empty
        'side': 'category',
        'price': 'float64',
        'amount': 'float64'
    }



    def __init__(self, data_dir: str = "data/research"):
        """
        Initialize data loader.

        Args:
            data_dir: Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self._orderbook_ddf: Optional[dd.DataFrame] = None
        self._liquidations_df: Optional[pd.DataFrame] = None

    def load_orderbook(self, blocksize: str = "256MB") -> dd.DataFrame:
        """
        Load orderbook data. Prefers parquet if available, otherwise CSV.
        """
        parquet_path = self.data_dir / "orderbook.parquet"
        if parquet_path.exists():
            if self._orderbook_ddf is None:
                logger.info(f"Loading orderbook from {parquet_path}...")
                self._orderbook_ddf = dd.read_parquet(parquet_path)
            return self._orderbook_ddf

        if self._orderbook_ddf is None:
            filepath = self.data_dir / "orderbook.csv"
            logger.info(f"Loading orderbook from {filepath}...")
            self._orderbook_ddf = dd.read_csv(
                filepath,
                dtype=self.ORDERBOOK_DTYPES,
                blocksize=blocksize,
                assume_missing=True
            )
            logger.info(f"Orderbook loaded with {self._orderbook_ddf.npartitions} partitions")
        return self._orderbook_ddf

    def convert_csv_to_parquet(self, blocksize: str = "256MB"):
        """Convert orderbook.csv to parquet for much faster I/O."""
        csv_path = self.data_dir / "orderbook.csv"
        parquet_path = self.data_dir / "orderbook.parquet"

        if parquet_path.exists():
            logger.info(f"Parquet file already exists at {parquet_path}")
            return

        logger.info(f"Converting {csv_path} to Parquet...")
        ddf = dd.read_csv(
            csv_path,
            dtype=self.ORDERBOOK_DTYPES,
            blocksize=blocksize,
            assume_missing=True
        )

        # Repartition to a reasonable number of files if needed
        # ddf = ddf.repartition(npartitions=20)

        ddf.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
        logger.info(f"Successfully converted to {parquet_path}")


    def load_liquidations(self) -> pd.DataFrame:
        """
        Load liquidations.csv using pandas (small file).

        Returns:
            Pandas DataFrame
        """
        if self._liquidations_df is None:
            filepath = self.data_dir / "liquidations.csv"
            logger.info(f"Loading liquidations from {filepath}...")
            self._liquidations_df = pd.read_csv(filepath)
            # Calculate liquidation value
            self._liquidations_df['value'] = (
                self._liquidations_df['price'] * self._liquidations_df['amount']
            )
            logger.info(f"Liquidations loaded: {len(self._liquidations_df)} events")
        return self._liquidations_df

    def get_orderbook_at_timestamp(
        self,
        target_ts: int,
        window_us: int = 1_000_000
    ) -> pd.DataFrame:
        """
        Get orderbook snapshot near a specific timestamp.

        Args:
            target_ts: Target timestamp in microseconds
            window_us: Time window in microseconds (default 1 second)

        Returns:
            Pandas DataFrame with orderbook data
        """
        orderbook = self.load_orderbook()

        # Filter to time window
        mask = (
            (orderbook['timestamp'] >= target_ts - window_us) &
            (orderbook['timestamp'] <= target_ts + window_us)
        )

        return orderbook[mask].compute()

