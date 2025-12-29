import asyncio
import json
import logging

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from decision_engine import DecisionEngine, DataTrustState, HypothesisValidityState, DecisionPermission, DecisionConditions, SystemState
from dirty_data_detector import DirtyDataDetector, SanitizationClass
from binance_client import BinanceWebSocketClient
from orderbook_metrics import OrderbookMetrics

logger = logging.getLogger(__name__)

class EngineRunner:
    """
    Unified runner for the Decision Engine.
    Handles data ingestion, synchronization, sanitization, and state management.
    """
    
    def __init__(
        self, 
        mode: str, 
        output_dir: str = "output",
        watermark_interval_us: int = 1_000_000, # 1s
        config_path: Optional[str] = None
    ):
        self.mode = mode.lower() # 'historical' or 'realtime'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load conditions from config if provided
        conditions = self._load_config(config_path) if config_path else None
        self.engine = DecisionEngine(conditions=conditions)
        self.detector = DirtyDataDetector()
        self.metrics_calc = OrderbookMetrics()
        
        self.watermark_interval_us = watermark_interval_us
        self.current_watermark = 0
        
        # Internal state for metrics pooling
        self.latest_metrics = {
            'spread_bps': 0.0,
            'depth': 0.0,
            'imbalance': 0.0,
            'liq_count': 0,
            'liq_value': 0.0
        }
        self.liq_buffer = [] # List of (ts, value)
        
        # Files for output
        (self.output_dir / "transitions").parent.mkdir(parents=True, exist_ok=True)
        self.files = {
            'transitions': open(self.output_dir / "state_transitions.jsonl", "a"),
            'decisions': open(self.output_dir / "decisions.jsonl", "a")
        }
        
        self.start_time = datetime.now()
        self.event_count = 0
        self.quarantine_count = 0
        
        # Tracking for duration_ms
        self.last_decision = DecisionPermission.ALLOWED
        self.decision_start_ts = 0
        
        # Guard for duplicate logs at same timestamp
        self.last_logged_ts = 0
        self.last_logged_decision = DecisionPermission.ALLOWED

    def _load_config(self, path: str) -> Optional[Dict]:
        """Load decision conditions from a JSON file."""
        try:
            config_p = Path(path)
            if config_p.exists():
                with open(config_p, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Config file not found: {path}. Using defaults.")
        except Exception as e:
            logger.error(f"Error loading config: {e}. Using defaults.")
        return None

    async def process_event(self, event: Dict):
        """
        Sanitize and process a single event.
        Logic is identical for both Historical and Realtime.
        """
        self.event_count += 1
        
        # Unwrap combined stream payload if nested
        if 'data' in event and 'stream' in event:
            payload = event['data']
        else:
            payload = event

        ts = payload.get('timestamp', payload.get('E', payload.get('_processing_ts')))
        if not ts: return

        # 1. Sanitization Policy Check
        sanitization_result, reason = self._check_sanitization(payload, ts)
        
        if sanitization_result == SanitizationClass.QUARANTINE:
            self.quarantine_count += 1
            
            # Create a forced UNTRUSTED/HALTED state in the engine
            quarantine_state = SystemState(
                timestamp=ts,
                data_trust=DataTrustState.UNTRUSTED,
                hypothesis_validity=HypothesisValidityState.VALID,
                decision_permission=DecisionPermission.HALTED,
                current_spread_bps=self.latest_metrics['spread_bps'],
                current_depth=self.latest_metrics['depth'],
                current_imbalance=self.latest_metrics['imbalance'],
                recent_liquidation_count=self.latest_metrics['liq_count'],
                recent_liquidation_value=self.latest_metrics['liq_value'],
                trigger=f"quarantine:{reason}"
            )
            
            # Only append and log if this is actually a new state at this timestamp
            should_log = (not self.engine.current_state or 
                          self.engine.current_state.decision_permission != DecisionPermission.HALTED or
                          self.engine.current_state.timestamp != ts)
            
            if should_log:
                self.engine.state_history.append(quarantine_state)
                self.engine.current_state = quarantine_state
                self._flush_engine_logs(ts)
            
            if self.quarantine_count % 100 == 1: # Log every 100th quarantine to avoid burst
                logger.warning(f"⚠️ QUARANTINE ({reason}) detected at {ts}. Total: {self.quarantine_count}")
            return

        # 2. Update state from event
        self._update_metrics(payload, ts)

        # 3. Decision Engine Evaluation (Time Alignment)
        if ts > self.current_watermark + self.watermark_interval_us:
            self.current_watermark = ts
            
            # Simple liquidation window clearing (just as a placeholder for better windowing)
            # In a production system, we'd use a sliding window buffer.
            if self.event_count % 100 == 0:
                logger.debug(f"Metrics: Spread={self.latest_metrics['spread_bps']:.2f}bps, Depth={self.latest_metrics['depth']:.2f}")

            # Toxicity: Simple proxy (abs imbalance * spread relative to normal)
            toxicity = abs(self.latest_metrics['imbalance']) * (self.latest_metrics['spread_bps'] / 5.0)
            toxicity = min(1.0, toxicity)

            state = self.engine.evaluate(
                timestamp=ts,
                spread_bps=self.latest_metrics['spread_bps'],
                depth=self.latest_metrics['depth'],
                imbalance=self.latest_metrics['imbalance'],
                recent_liquidation_count=self.latest_metrics['liq_count'],
                recent_liquidation_value=self.latest_metrics['liq_value'],
                toxicity=toxicity
            )
            
            # Use current_state history to write logs if changed
            # (DecisionEngine.evaluate already tracks transitions)
            if self.engine._state_changed(state):
                logger.info(f"🔄 State Change: {state.data_trust.value} / {state.hypothesis_validity.value} -> {state.decision_permission.value} | Trigger: {state.trigger}")
            
            self._flush_engine_logs(ts)

        self.event_count += 1
        if self.event_count % 1000 == 0:
            logger.info(f"📊 Heartbeat: Processed {self.event_count} events. Target TS: {ts} | Spread: {self.latest_metrics['spread_bps']:.2f} bps | Depth: {self.latest_metrics['depth']:.2f} BTC")

    def _check_sanitization(self, event: Dict, ts: int) -> Tuple[SanitizationClass, str]:
        """Improved sanitization check that respects event types."""
        e_type = event.get('e', '')
        
        # aggTrade: p=price, q=quantity
        if e_type == 'aggTrade':
            price = float(event.get('p', 0))
            quantity = float(event.get('q', 0))
            if price <= 0: return SanitizationClass.QUARANTINE, "invalid_trade_price"
            if quantity > 1000: return SanitizationClass.QUARANTINE, "fat_finger_trade_quantity"
            
        # forceOrder (liquidation): o['p'], o['q']
        elif e_type == 'forceOrder':
            o = event.get('o', {})
            price = float(o.get('p', 0))
            quantity = float(o.get('q', 0))
            if price <= 0: return SanitizationClass.QUARANTINE, "invalid_liq_price"
            if quantity > 1000: return SanitizationClass.QUARANTINE, "fat_finger_liq_quantity"
            
        # depth: b and a are lists of [price, qty]
        elif 'depth' in event.get('stream', '') or 'lastUpdateId' in event:
            bids = event.get('b', [])
            asks = event.get('a', [])
            if bids and asks:
                try:
                    bp, ap = float(bids[0][0]), float(asks[0][0])
                    if bp >= ap: return SanitizationClass.QUARANTINE, "crossed_market"
                    if bp <= 0 or ap <= 0: return SanitizationClass.QUARANTINE, "zero_or_negative_orderbook_price"
                except (ValueError, IndexError): 
                    return SanitizationClass.QUARANTINE, "malformed_orderbook_data"
                
        return SanitizationClass.ACCEPT, "ok"

    def _update_metrics(self, event: Dict, ts: int):
        event_type = event.get('e', event.get('stream', ''))
        
        if 'aggTrade' in event_type:
            pass
        elif 'depth' in event_type:
            bids = event.get('b', [])
            asks = event.get('a', [])
            if bids and asks:
                # Convert to DataFrame for OrderbookMetrics
                data = []
                for b in bids: data.append({'side': 'bid', 'price': float(b[0]), 'amount': float(b[1]), 'local_timestamp': ts})
                for a in asks: data.append({'side': 'ask', 'price': float(a[0]), 'amount': float(a[1]), 'local_timestamp': ts})
                df = pd.DataFrame(data)
                
                snapshot = self.metrics_calc.process_orderbook_snapshot(df, ts)
                if snapshot:
                    # Use OBWA spread if possible for stability
                    self.latest_metrics['spread_bps'] = snapshot.obwa_spread_bps
                    # Use 50bps depth as the primary metric
                    self.latest_metrics['depth'] = snapshot.depth_50bps_bid + snapshot.depth_50bps_ask
                    self.latest_metrics['imbalance'] = snapshot.order_imbalance
        elif 'forceOrder' in event_type or 'liquidationOrder' in event_type:
            o = event.get('o', {})
            val = float(o.get('p', 0)) * float(o.get('q', 0))
            self.liq_buffer.append((ts, val))
            
        # Maintain liquidation window
        window_us = self.engine.conditions.liquidation_window_us
        self.liq_buffer = [item for item in self.liq_buffer if ts - item[0] <= window_us]
        self.latest_metrics['liq_count'] = len(self.liq_buffer)
        self.latest_metrics['liq_value'] = sum(item[1] for item in self.liq_buffer)

    def _flush_engine_logs(self, event_ts: int):
        if not self.engine.state_history: return
        last_state = self.engine.state_history[-1]
        
        # Strict deduplication: NEVER log twice at the same timestamp
        if event_ts <= self.last_logged_ts:
            return
        
        state_changed = last_state.decision_permission != self.last_decision
        
        # Calculate current cumulative duration in the current state
        duration_ms = (event_ts - self.decision_start_ts) // 1000 if self.decision_start_ts > 0 else 0

        # 1. Transitions Log (Optional: keep only on change to avoid bloating)
        if state_changed:
            transition_entry = {
                "ts": event_ts,
                "data_trust": last_state.data_trust.value,
                "hypothesis": last_state.hypothesis_validity.value,
                "decision": last_state.decision_permission.value,
                "trigger": last_state.trigger
            }
            self.files['transitions'].write(json.dumps(transition_entry) + "\n")
            self.files['transitions'].flush()

        # 2. Decisions Log (Continuous / Periodic as requested)
        # Check for Recovery (Transition to ALLOWED)
        if (self.last_decision != DecisionPermission.ALLOWED and 
            last_state.decision_permission == DecisionPermission.ALLOWED):
            
            resume_entry = {
                "ts": event_ts,
                "action": "RESUME",
                "reason": f"recovered_from_{self.last_decision.value}",
                "duration_ms": duration_ms
            }
            self.files['decisions'].write(json.dumps(resume_entry) + "\n")
            self.files['decisions'].flush()
        
        # Check for Active Restriction (Log every interval)
        elif last_state.decision_permission != DecisionPermission.ALLOWED:
            action = "HALT" if last_state.decision_permission == DecisionPermission.HALTED else "RESTRICT"
            
            decision_entry = {
                "ts": event_ts,
                "action": action,
                "reason": last_state.trigger,
                "duration_ms": duration_ms
            }
            self.files['decisions'].write(json.dumps(decision_entry) + "\n")
            self.files['decisions'].flush()
        
        # Update tracking for NEXT interval
        self.last_logged_ts = event_ts
        self.last_logged_decision = last_state.decision_permission
        if state_changed:
            self.last_decision = last_state.decision_permission
            self.decision_start_ts = event_ts


    async def run_historical(self, data_path: str):
        """Simulation of historical data ingestion."""
        logger.info(f"Running Historical Validation on {data_path}")
        from data_loader import DataLoader
        from collections import defaultdict
        
        loader = DataLoader(data_path)
        
        logger.info("Loading historical data samples...")
        ob_df = loader.load_orderbook().head(10000)
        liq_df = loader.load_liquidations()
        
        # Aggregate orderbook by timestamp
        ob_by_ts = defaultdict(lambda: {'bids': [], 'asks': []})
        for _, row in ob_df.iterrows():
            ts = int(row['timestamp'])
            entry = [str(row['price']), str(row['amount'])]
            if row['side'] == 'bid':
                ob_by_ts[ts]['bids'].append(entry)
            else:
                ob_by_ts[ts]['asks'].append(entry)
        
        # Create aggregated orderbook events
        events = []
        for ts, data in ob_by_ts.items():
            if data['bids'] and data['asks']:
                # Sort bids descending, asks ascending
                data['bids'].sort(key=lambda x: float(x[0]), reverse=True)
                data['asks'].sort(key=lambda x: float(x[0]))
                events.append({
                    'stream': 'btcusdt@depth5',
                    'e': 'depthUpdate',
                    'timestamp': ts,
                    'b': data['bids'][:5],
                    'a': data['asks'][:5],
                })
            
        # Add liquidation events
        for _, row in liq_df.iterrows():
            events.append({
                'stream': 'btcusdt@forceOrder',
                'e': 'forceOrder',
                'timestamp': int(row['timestamp']),
                'o': {'p': str(row['price']), 'q': str(row['amount'])}
            })
            
        events.sort(key=lambda x: x['timestamp'])
        
        logger.info(f"Processing {len(events)} historical events...")
        for event in events:
            await self.process_event(event)

    async def run_realtime(self):
        """Realtime WebSocket connection."""
        logger.info("Running Realtime Validation...")
        client = BinanceWebSocketClient()
        client.add_callback(self.process_event)
        await client.start()

    def save_summary(self):
        summary = {
            "mode": self.mode,
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "total_events": self.event_count,
            "quarantined_events": self.quarantine_count,
            "final_state": self.engine.get_decision_summary() if self.engine.state_history else "N/A"
        }
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=4)
        
    def close(self):
        for f in self.files.values():
            f.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["historical", "realtime"])
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="output")
    parser.add_argument("--config", help="Path to decision conditions config file")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    runner = EngineRunner(args.mode, output_dir=args.output, config_path=args.config)
    
    async def main():
        try:
            if args.mode == "historical":
                await runner.run_historical(args.data)
            else:
                await runner.run_realtime()
        except KeyboardInterrupt:
            pass
        finally:
            runner.save_summary()
            runner.close()
    
    asyncio.run(main())
