"""
Experiment Runner Module
========================

Runs multiple sets of DecisionEngine parameters on historical data and records results.
Supports grid search and structured output management.
"""

import os
import json
import logging
import itertools
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import pandas as pd

from src.analysis.decision_engine import DecisionEngine, DecisionConditions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        self.base_config = self.config.get("base_config", {})
        self.parameter_grid = self.config.get("parameter_grid", {})
        self.data_config = self.config.get("data", {})
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"experiment/output/{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment config to output dir for reproducibility
        with open(self.output_dir / "experiment_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
            
        self.metrics_df: Optional[pd.DataFrame] = None
        self.liquidations_df: Optional[pd.DataFrame] = None

    def load_data(self):
        """Load data once for all experiments."""
        metrics_path = self.data_config.get("metrics_path", "output/phase1/orderbook_metrics.csv")
        liquidations_path = self.data_config.get("liquidations_path", "output/phase1/liquidation_report.csv")
        
        logger.info(f"Loading metrics from {metrics_path}...")
        self.metrics_df = pd.read_csv(metrics_path).sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Loading liquidations from {liquidations_path}...")
        self.liquidations_df = pd.read_csv(liquidations_path).sort_values('timestamp').reset_index(drop=True)

    def generate_configs(self) -> List[Dict[str, Any]]:
        """Generate a list of configurations based on the parameter grid."""
        keys = self.parameter_grid.keys()
        values = self.parameter_grid.values()
        
        combinations = list(itertools.product(*values))
        configs = []
        
        for combo in combinations:
            conf = self.base_config.copy()
            for k, v in zip(keys, combo):
                conf[k] = v
            configs.append(conf)
            
        return configs if configs else [self.base_config]

    def run_single_experiment(self, run_idx: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment configuration."""
        run_dir = self.output_dir / f"run_{run_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        conditions = DecisionConditions.from_dict(config)
        engine = DecisionEngine(conditions)
        
        liquidation_window_us = conditions.liquidation_window_us
        watermark_interval_us = conditions.watermark_interval_us
        allowed_lateness_us = conditions.allowed_lateness_us
        
        current_watermark = 0
        state_records = []
        
        # Sort metrics by timestamp to simulate stream if not already sorted
        # In a real stream, we'd handle out-of-order here.
        # For simulation, we'll use 'event_time' or 'local_timestamp' as proxy for processing time.
        
        for _, row in self.metrics_df.iterrows():
            ts = row['timestamp']
            processing_ts = row.get('local_timestamp', ts) if conditions.time_alignment_mode == "processing_time" else ts
            
            # 1. Allowed Lateness Check
            # If event arrival is too late compared to current watermark, we might drop or quarantine it.
            # Here we simulate dropping late events.
            if current_watermark > 0 and ts < current_watermark - allowed_lateness_us:
                continue

            # 2. Watermark Check: Only evaluate at intervals
            if current_watermark == 0 or processing_ts >= current_watermark + watermark_interval_us:
                current_watermark = processing_ts
                
                # Filter liquidations in window
                recent_liq_mask = (
                    (self.liquidations_df['timestamp'] >= ts - liquidation_window_us) &
                    (self.liquidations_df['timestamp'] <= ts)
                )
                recent_liqs = self.liquidations_df[recent_liq_mask]
                recent_count = int(len(recent_liqs))
                recent_value = float(recent_liqs['value'].sum()) if 'value' in recent_liqs.columns else 0.0
                
                # Evaluate state
                state = engine.evaluate(
                    timestamp=ts,
                    spread_bps=row['spread_bps'],
                    depth=row['depth_50bps_total'],
                    imbalance=row['order_imbalance'],
                    recent_liquidation_count=recent_count,
                    recent_liquidation_value=recent_value
                )
                
                state_records.append({
                    'timestamp': ts,
                    'decision': state.decision_permission.value,
                    'trigger': state.trigger
                })

        # Save results for this run
        summary = engine.get_decision_summary()
        summary['config'] = config
        
        # Action logging logic (decisions.jsonl)
        actions = []
        last_decision = "ALLOWED"
        decision_start_ts = 0
        
        for i, rec in enumerate(state_records):
            current_ts = rec['timestamp']
            current_dec = rec['decision']
            
            state_changed = current_dec != last_decision
            duration_ms = (current_ts - decision_start_ts) // 1000 if decision_start_ts > 0 else 0
            
            if state_changed:
                if last_decision != "ALLOWED" and current_dec == "ALLOWED":
                    actions.append({
                        "ts": current_ts,
                        "action": "RESUME",
                        "reason": f"recovered_from_{last_decision}",
                        "duration_ms": duration_ms
                    })
                elif current_dec != "ALLOWED":
                    action_type = "HALT" if current_dec == "HALTED" else "RESTRICT"
                    actions.append({
                        "ts": current_ts,
                        "action": action_type,
                        "reason": rec['trigger'],
                        "duration_ms": 0 # Start of new state
                    })
                
                last_decision = current_dec
                decision_start_ts = current_ts
            elif current_dec != "ALLOWED":
                # Heartbeat/Periodic log for active restrictions
                action_type = "HALT" if current_dec == "HALTED" else "RESTRICT"
                actions.append({
                    "ts": current_ts,
                    "action": action_type,
                    "reason": rec['trigger'],
                    "duration_ms": duration_ms
                })

        # Create nested folders
        for folder in ["research", "validation"]:
            path = run_dir / folder
            path.mkdir(parents=True, exist_ok=True)
            
            # Save summary.json
            with open(path / "summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
                
            # Save state_transitions.jsonl
            transitions = engine.get_state_transitions()
            with open(path / "state_transitions.jsonl", 'w') as f:
                for t in transitions:
                    f.write(json.dumps(t) + '\n')
            
            # Save decisions.jsonl
            with open(path / "decisions.jsonl", 'w') as f:
                for a in actions:
                    f.write(json.dumps(a) + '\n')
                
        return {
            'run_id': run_idx,
            'config': config,
            'summary': summary
        }

    def run_all(self):
        """Run all generated experiment configurations."""
        self.load_data()
        configs = self.generate_configs()
        
        logger.info(f"Starting {len(configs)} experiments...")
        results = []
        
        # tqdm for total experiment progress
        for i, config in enumerate(tqdm(configs, desc="Experiments Progress")):
            result = self.run_single_experiment(i, config)
            results.append(result)
            
        # Generate aggregate summary
        aggregate_summary = {
            'experiment_timestamp': self.output_dir.name,
            'total_runs': len(results),
            'runs': results
        }
        
        with open(self.output_dir / "aggregate_summary.json", 'w') as f:
            json.dump(aggregate_summary, f, indent=2)
            
        logger.info(f"All experiments completed. Results saved to {self.output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run a batch of experiments')
    parser.add_argument('--config', type=str, default='configs/experiment_config.json', help='Path to experiment config')
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.config)
    runner.run_all()
