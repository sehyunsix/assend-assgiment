import pandas as pd
import numpy as np
from pathlib import Path
import json

def prepare_test_data_with_jitter():
    output_dir = Path("output/phase1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create mock orderbook metrics with jitter
    n_samples = 1000
    base_ts = 1760000000000000
    interval = 100000 # 100ms
    
    timestamps = np.arange(base_ts, base_ts + n_samples * interval, interval)
    
    # Add local_timestamp (processing time) with some delay
    # Normal delay 10-50ms, but some spikes
    delays = np.random.uniform(10000, 50000, size=n_samples)
    spike_indices = np.random.choice(n_samples, size=n_samples // 20, replace=False)
    delays[spike_indices] += np.random.uniform(500000, 1500000, size=len(spike_indices))
    
    local_timestamps = timestamps + delays.astype(int)
    
    # Introduce some out-of-order event time (late events)
    order = np.arange(n_samples)
    late_indices = np.random.choice(n_samples, size=n_samples // 10, replace=False)
    # Move some events to be processed much later
    # (Actually we keep them in local_timestamp order but their event_time is 'old')
    # Actually the ExperimentRunner iterates through the DataFrame.
    # If we want out-of-order, we should sort by local_timestamp and see some old event_time.
    
    metrics_df = pd.DataFrame({
        'timestamp': timestamps,
        'local_timestamp': local_timestamps,
        'spread_bps': np.random.uniform(2.0, 15.0, size=n_samples),
        'depth_50bps_total': np.random.uniform(5.0, 20.0, size=n_samples),
        'order_imbalance': np.random.uniform(-0.9, 0.9, size=n_samples)
    })
    
    # Sort by local_timestamp to simulate arrival order
    metrics_df = metrics_df.sort_values('local_timestamp').reset_index(drop=True)
    
    # 2. Create mock liquidations
    liq_indices = np.random.choice(n_samples, size=50, replace=False)
    liquidations_df = pd.DataFrame({
        'timestamp': timestamps[liq_indices],
        'value': np.random.uniform(10000, 500000, size=50)
    })
    
    metrics_df.to_csv(output_dir / "orderbook_metrics.csv", index=False)
    liquidations_df.to_csv(output_dir / "liquidation_report.csv", index=False)
    
    print(f"Jittered test data prepared in {output_dir}")

if __name__ == "__main__":
    prepare_test_data_with_jitter()
