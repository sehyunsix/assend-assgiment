"""
Decision Engine Module
======================

Defines the state transition logic for the trading decision system.
Implements Data Trust State, Hypothesis Validity State, and Decision Permission.
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class DataTrustState(Enum):
    """Data trust level based on orderbook metrics."""
    TRUSTED = "TRUSTED"
    DEGRADED = "DEGRADED"
    UNTRUSTED = "UNTRUSTED"


class HypothesisValidityState(Enum):
    """Validity of trading hypothesis based on market conditions."""
    VALID = "VALID"
    WEAKENING = "WEAKENING"
    INVALID = "INVALID"


class DecisionPermission(Enum):
    """Permission level for making trading decisions."""
    ALLOWED = "ALLOWED"
    RESTRICTED = "RESTRICTED"
    HALTED = "HALTED"


@dataclass
class DecisionConditions:
    """Threshold conditions for decision engine."""
    # Spread thresholds (in bps)
    spread_trusted_max: float = 3.2      # p90
    spread_degraded_max: float = 11.6    # p99

    # Depth thresholds (in BTC within 50 bps)
    depth_trusted_min: float = 12.7      # p10
    depth_degraded_min: float = 1.7      # p01

    # Imbalance thresholds (absolute value)
    imbalance_trusted_max: float = 0.7
    imbalance_degraded_max: float = 0.85

    # Toxicity thresholds (0-1 score)
    toxicity_trusted_max: float = 0.3
    toxicity_degraded_max: float = 0.6

    # Liquidation-based thresholds
    liquidation_cluster_threshold: int = 2      # Events to trigger WEAKENING
    liquidation_cascade_threshold: int = 5      # Events to trigger INVALID
    liquidation_value_threshold: float = 100000  # USD value threshold

    # Time windows (microseconds)
    liquidation_window_us: int = 10_000_000     # 10 seconds for clustering
    recovery_window_us: int = 60_000_000        # 60 seconds for recovery check
    min_state_duration_us: int = 100_000       # 100ms minimum stay in HALTED/RESTRICTED

    # Time Alignment Policy
    watermark_interval_us: int = 1_000_000     # 1 second default watermark
    allowed_lateness_us: int = 500_000         # 0.5 second allowed lateness
    time_alignment_mode: str = "event_time"    # "event_time" or "processing_time"

    @staticmethod
    def from_dict(data: Dict) -> 'DecisionConditions':
        """Create conditions from a dictionary, with defaults for missing keys."""
        return DecisionConditions(
            spread_trusted_max=data.get('spread_trusted_max', 3.2),
            spread_degraded_max=data.get('spread_degraded_max', 11.6),
            depth_trusted_min=data.get('depth_trusted_min', 12.7),
            depth_degraded_min=data.get('depth_degraded_min', 1.7),
            imbalance_trusted_max=data.get('imbalance_trusted_max', 0.7),
            imbalance_degraded_max=data.get('imbalance_degraded_max', 0.85),
            liquidation_cluster_threshold=data.get('liquidation_cluster_threshold', 2),
            liquidation_cascade_threshold=data.get('liquidation_cascade_threshold', 5),
            liquidation_value_threshold=data.get('liquidation_value_threshold', 100000),
            liquidation_window_us=data.get('liquidation_window_us', 10_000_000),
            recovery_window_us=data.get('recovery_window_us', 60_000_000),
            min_state_duration_us=data.get('min_state_duration_us', 100_000),
            watermark_interval_us=data.get('watermark_interval_us', 1_000_000),
            allowed_lateness_us=data.get('allowed_lateness_us', 500_000),
            time_alignment_mode=data.get('time_alignment_mode', 'event_time')
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'spread_thresholds': {
                'trusted_max_bps': self.spread_trusted_max,
                'degraded_max_bps': self.spread_degraded_max,
            },
            'depth_thresholds': {
                'trusted_min_btc': self.depth_trusted_min,
                'degraded_min_btc': self.depth_degraded_min,
            },
            'imbalance_thresholds': {
                'trusted_max': self.imbalance_trusted_max,
                'degraded_max': self.imbalance_degraded_max,
            },
            'liquidation_thresholds': {
                'cluster_events': self.liquidation_cluster_threshold,
                'cascade_events': self.liquidation_cascade_threshold,
                'value_threshold_usd': self.liquidation_value_threshold,
            },
            'time_windows': {
                'liquidation_window_seconds': self.liquidation_window_us / 1_000_000,
                'recovery_window_seconds': self.recovery_window_us / 1_000_000,
                'min_state_duration_ms': self.min_state_duration_us / 1_000,
                'watermark_interval_ms': self.watermark_interval_us / 1_000,
                'allowed_lateness_ms': self.allowed_lateness_us / 1_000,
                'mode': self.time_alignment_mode
            }
        }


@dataclass
class SystemState:
    """Current state of the decision engine."""
    timestamp: int
    data_trust: DataTrustState
    hypothesis_validity: HypothesisValidityState
    decision_permission: DecisionPermission

    # Metrics that led to this state
    current_spread_bps: float
    current_depth: float
    current_imbalance: float
    current_toxicity: float

    # Liquidation context
    recent_liquidation_count: int
    recent_liquidation_value: float

    # Trigger reason
    trigger: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'ts': self.timestamp,
            'data_trust': self.data_trust.value,
            'hypothesis': self.hypothesis_validity.value,
            'decision': self.decision_permission.value,
            'metrics': {
                'spread_bps': self.current_spread_bps,
                'depth': self.current_depth,
                'imbalance': self.current_imbalance,
                'toxicity': self.current_toxicity,
            },
            'liquidation_context': {
                'recent_count': self.recent_liquidation_count,
                'recent_value': self.recent_liquidation_value,
            },
            'trigger': self.trigger
        }


class DecisionEngine:
    """
    Decision Engine for determining trading permission based on market state.

    The engine evaluates:
    1. Data Trust State: Based on orderbook metrics (spread, depth, imbalance)
    2. Hypothesis Validity: Based on liquidation events and market conditions
    3. Decision Permission: Combination of the above two states
    """

    def __init__(self, conditions: Optional[Union[DecisionConditions, Dict]] = None):
        """
        Initialize the decision engine.

        Args:
            conditions: Threshold conditions (object or dict)
        """
        if isinstance(conditions, dict):
            self.conditions = DecisionConditions.from_dict(conditions)
        else:
            self.conditions = conditions or DecisionConditions()
        
        self.state_history: List[SystemState] = []
        self.current_state: Optional[SystemState] = None

        logger.info("DecisionEngine initialized with conditions:")
        logger.info(f"  Spread thresholds: TRUSTED < {self.conditions.spread_trusted_max} bps, "
                   f"DEGRADED < {self.conditions.spread_degraded_max} bps")
        logger.info(f"  Depth thresholds: TRUSTED > {self.conditions.depth_trusted_min} BTC, "
                   f"DEGRADED > {self.conditions.depth_degraded_min} BTC")

    def evaluate_data_trust(
        self,
        spread_bps: float,
        depth: float,
        imbalance: float,
        toxicity: float = 0.0
    ) -> Tuple[DataTrustState, str]:
        """
        Evaluate data trust state based on orderbook metrics and toxicity.

        Args:
            spread_bps: Current bid-ask spread in basis points
            depth: Current market depth (BTC within 50 bps)
            imbalance: Current order imbalance (-1 to 1)
            toxicity: Current toxicity score (0-1)

        Returns:
            Tuple of (DataTrustState, reason_string)
        """
        reasons = []

        # Check each metric
        spread_state = DataTrustState.TRUSTED
        if spread_bps > self.conditions.spread_degraded_max:
            spread_state = DataTrustState.UNTRUSTED
            reasons.append(f"spread={spread_bps:.2f}bps>UNTRUSTED")
        elif spread_bps > self.conditions.spread_trusted_max:
            spread_state = DataTrustState.DEGRADED
            reasons.append(f"spread={spread_bps:.2f}bps>DEGRADED")

        depth_state = DataTrustState.TRUSTED
        if depth < self.conditions.depth_degraded_min:
            depth_state = DataTrustState.UNTRUSTED
            reasons.append(f"depth={depth:.2f}BTC<UNTRUSTED")
        elif depth < self.conditions.depth_trusted_min:
            depth_state = DataTrustState.DEGRADED
            reasons.append(f"depth={depth:.2f}BTC<DEGRADED")

        imbalance_state = DataTrustState.TRUSTED
        abs_imbalance = abs(imbalance)
        if abs_imbalance > self.conditions.imbalance_degraded_max:
            imbalance_state = DataTrustState.UNTRUSTED
            reasons.append(f"imbalance={imbalance:.2f}>UNTRUSTED")
        elif abs_imbalance > self.conditions.imbalance_trusted_max:
            imbalance_state = DataTrustState.DEGRADED
            reasons.append(f"imbalance={imbalance:.2f}>DEGRADED")

        toxicity_state = DataTrustState.TRUSTED
        if toxicity > self.conditions.toxicity_degraded_max:
            toxicity_state = DataTrustState.UNTRUSTED
            reasons.append(f"toxicity={toxicity:.2f}>UNTRUSTED")
        elif toxicity > self.conditions.toxicity_trusted_max:
            toxicity_state = DataTrustState.DEGRADED
            reasons.append(f"toxicity={toxicity:.2f}>DEGRADED")

        # Overall state is the worst of the four
        states = [spread_state, depth_state, imbalance_state, toxicity_state]

        if DataTrustState.UNTRUSTED in states:
            return DataTrustState.UNTRUSTED, "; ".join(reasons) if reasons else "metrics_untrusted"
        elif DataTrustState.DEGRADED in states:
            return DataTrustState.DEGRADED, "; ".join(reasons) if reasons else "metrics_degraded"
        else:
            return DataTrustState.TRUSTED, "metrics_normal"

    def evaluate_hypothesis_validity(
        self,
        recent_liquidation_count: int,
        recent_liquidation_value: float
    ) -> Tuple[HypothesisValidityState, str]:
        """
        Evaluate hypothesis validity based on liquidation activity.

        Args:
            recent_liquidation_count: Number of liquidations in recent window
            recent_liquidation_value: Total value of recent liquidations

        Returns:
            Tuple of (HypothesisValidityState, reason_string)
        """
        # Check for liquidation cascade
        if (recent_liquidation_count >= self.conditions.liquidation_cascade_threshold or
            recent_liquidation_value >= self.conditions.liquidation_value_threshold * 2):
            return HypothesisValidityState.INVALID, \
                f"liquidation_cascade(count={recent_liquidation_count}, value=${recent_liquidation_value:,.0f})"

        # Check for liquidation cluster
        if (recent_liquidation_count >= self.conditions.liquidation_cluster_threshold or
            recent_liquidation_value >= self.conditions.liquidation_value_threshold):
            return HypothesisValidityState.WEAKENING, \
                f"liquidation_cluster(count={recent_liquidation_count}, value=${recent_liquidation_value:,.0f})"

        return HypothesisValidityState.VALID, "no_significant_liquidations"

    def determine_decision_permission(
        self,
        data_trust: DataTrustState,
        hypothesis: HypothesisValidityState
    ) -> DecisionPermission:
        """
        Determine decision permission based on data trust and hypothesis validity.

        Decision Matrix:

        |                | VALID     | WEAKENING   | INVALID  |
        |----------------|-----------|-------------|----------|
        | TRUSTED        | ALLOWED   | RESTRICTED  | HALTED   |
        | DEGRADED       | RESTRICTED| RESTRICTED  | HALTED   |
        | UNTRUSTED      | HALTED    | HALTED      | HALTED   |

        Args:
            data_trust: Current data trust state
            hypothesis: Current hypothesis validity state

        Returns:
            DecisionPermission
        """
        # UNTRUSTED data -> always HALTED
        if data_trust == DataTrustState.UNTRUSTED:
            return DecisionPermission.HALTED

        # INVALID hypothesis -> always HALTED
        if hypothesis == HypothesisValidityState.INVALID:
            return DecisionPermission.HALTED

        # TRUSTED + VALID -> ALLOWED
        if data_trust == DataTrustState.TRUSTED and hypothesis == HypothesisValidityState.VALID:
            return DecisionPermission.ALLOWED

        # Everything else -> RESTRICTED
        return DecisionPermission.RESTRICTED

    def evaluate(
        self,
        timestamp: int,
        spread_bps: float,
        depth: float,
        imbalance: float,
        recent_liquidation_count: int = 0,
        recent_liquidation_value: float = 0.0,
        toxicity: float = 0.0
    ) -> SystemState:
        """
        Evaluate current market state and determine decision permission.

        Args:
            timestamp: Current timestamp (microseconds)
            spread_bps: Current spread in basis points
            depth: Current market depth
            imbalance: Current order imbalance
            recent_liquidation_count: Number of recent liquidations
            recent_liquidation_value: Value of recent liquidations

        Returns:
            SystemState with current evaluation
        """
        # 1. Evaluate Data Trust
        data_trust, trust_reason = self.evaluate_data_trust(spread_bps, depth, imbalance, toxicity)
        
        # 2. Evaluate Hypothesis Validity
        hypothesis, hypo_reason = self.evaluate_hypothesis_validity(
            recent_liquidation_count, recent_liquidation_value
        )
        decision = self.determine_decision_permission(data_trust, hypothesis)
        
        # Enforce Minimum State Duration (Cooldown)
        # If current state is HALTED/RESTRICTED and we try to move to ALLOWED, check duration
        if (self.current_state and 
            self.current_state.decision_permission != DecisionPermission.ALLOWED and
            decision == DecisionPermission.ALLOWED):
            
            # Find when we first entered this restricted state
            # (In state_history, states are only added when they CHANGE)
            last_transition = self.state_history[-1]
            elapsed_us = timestamp - last_transition.timestamp
            
            if elapsed_us < self.conditions.min_state_duration_us:
                # Override: Stay in the previous restricted state
                decision = self.current_state.decision_permission
                trust_reason = f"{trust_reason} (cooldown_active:{elapsed_us//1000}ms)"

        # Build trigger reason
        trigger = f"trust:{trust_reason}|hypothesis:{hypo_reason}"

        # Create state
        state = SystemState(
            timestamp=timestamp,
            data_trust=data_trust,
            hypothesis_validity=hypothesis,
            decision_permission=decision,
            current_spread_bps=spread_bps,
            current_depth=depth,
            current_imbalance=imbalance,
            current_toxicity=toxicity,
            recent_liquidation_count=recent_liquidation_count,
            recent_liquidation_value=recent_liquidation_value,
            trigger=trigger
        )

        # Track state changes
        if self.current_state is None or self._state_changed(state):
            self.state_history.append(state)
            logger.debug(f"State transition at {timestamp}: "
                        f"{data_trust.value}/{hypothesis.value} -> {decision.value}")

        self.current_state = state
        return state

    def _state_changed(self, new_state: SystemState) -> bool:
        """Check if state has changed from current."""
        if self.current_state is None:
            return True
        return (
            new_state.data_trust != self.current_state.data_trust or
            new_state.hypothesis_validity != self.current_state.hypothesis_validity or
            new_state.decision_permission != self.current_state.decision_permission
        )

    def get_state_transitions(self) -> List[Dict]:
        """Get all state transitions as list of dictionaries."""
        return [s.to_dict() for s in self.state_history]

    def get_decision_summary(self) -> Dict:
        """Get summary of decisions made."""
        if not self.state_history:
            return {'error': 'no_states_evaluated'}

        decisions = [s.decision_permission for s in self.state_history]

        return {
            'total_transitions': len(self.state_history),
            'decision_counts': {
                'ALLOWED': sum(1 for d in decisions if d == DecisionPermission.ALLOWED),
                'RESTRICTED': sum(1 for d in decisions if d == DecisionPermission.RESTRICTED),
                'HALTED': sum(1 for d in decisions if d == DecisionPermission.HALTED),
            },
            'data_trust_counts': {
                'TRUSTED': sum(1 for s in self.state_history if s.data_trust == DataTrustState.TRUSTED),
                'DEGRADED': sum(1 for s in self.state_history if s.data_trust == DataTrustState.DEGRADED),
                'UNTRUSTED': sum(1 for s in self.state_history if s.data_trust == DataTrustState.UNTRUSTED),
            },
            'hypothesis_counts': {
                'VALID': sum(1 for s in self.state_history if s.hypothesis_validity == HypothesisValidityState.VALID),
                'WEAKENING': sum(1 for s in self.state_history if s.hypothesis_validity == HypothesisValidityState.WEAKENING),
                'INVALID': sum(1 for s in self.state_history if s.hypothesis_validity == HypothesisValidityState.INVALID),
            }
        }


def run_decision_engine_analysis(
    metrics_path: str,
    liquidations_path: str,
    output_dir: str,
    conditions: Optional[DecisionConditions] = None
) -> Tuple[List[Dict], Dict]:
    """
    Run decision engine analysis on historical data.

    Args:
        metrics_path: Path to orderbook_metrics.csv
        liquidations_path: Path to liquidation_report.csv
        output_dir: Output directory
        conditions: Optional custom conditions

    Returns:
        Tuple of (state_transitions, summary)
    """
    output_dir = Path(output_dir)

    # Load data
    logger.info("Loading orderbook metrics...")
    metrics_df = pd.read_csv(metrics_path)
    metrics_df = metrics_df.sort_values('timestamp').reset_index(drop=True)

    logger.info("Loading liquidations...")
    liquidations_df = pd.read_csv(liquidations_path)
    liquidations_df = liquidations_df.sort_values('timestamp').reset_index(drop=True)

    # Initialize engine
    conditions = conditions or DecisionConditions()
    engine = DecisionEngine(conditions)

    # Save conditions
    conditions_dict = {
        'generated_at': datetime.now().isoformat(),
        'description': 'Decision Engine thresholds derived from Phase 1 analysis',
        'conditions': conditions.to_dict(),
        'state_definitions': {
            'DataTrustState': {
                'TRUSTED': 'All metrics within normal range',
                'DEGRADED': 'One or more metrics in warning range',
                'UNTRUSTED': 'One or more metrics in critical range',
            },
            'HypothesisValidityState': {
                'VALID': 'No significant liquidation activity',
                'WEAKENING': 'Liquidation cluster detected',
                'INVALID': 'Liquidation cascade detected',
            },
            'DecisionPermission': {
                'ALLOWED': 'Trading decisions permitted',
                'RESTRICTED': 'Reduced position sizes, increased caution',
                'HALTED': 'No new positions, data collection only',
            }
        },
        'decision_matrix': {
            'description': 'Permission = f(DataTrust, HypothesisValidity)',
            'rules': [
                'UNTRUSTED -> HALTED (regardless of hypothesis)',
                'INVALID -> HALTED (regardless of trust)',
                'TRUSTED + VALID -> ALLOWED',
                'DEGRADED or WEAKENING -> RESTRICTED',
            ]
        }
    }

    with open(output_dir / "decision_conditions.json", 'w') as f:
        json.dump(conditions_dict, f, indent=2)
    logger.info(f"Saved decision conditions to {output_dir / 'decision_conditions.json'}")

    # Process each timestamp
    logger.info("Evaluating decision states...")
    liquidation_window_us = conditions.liquidation_window_us

    state_records = []


    for idx, row in metrics_df.iterrows():
        if idx % 10000 == 0:
            logger.info(f"Processing {idx}/{len(metrics_df)} timestamps...")

        ts = row['timestamp']

        # Count recent liquidations
        recent_liq_mask = (
            (liquidations_df['timestamp'] >= ts - liquidation_window_us) &
            (liquidations_df['timestamp'] <= ts)
        )
        recent_liqs = liquidations_df[recent_liq_mask]
        recent_count = len(recent_liqs)
        recent_value = recent_liqs['value'].sum() if 'value' in recent_liqs.columns else 0

        # Evaluate state
        state = engine.evaluate(
            timestamp=ts,
            spread_bps=row['spread_bps'],
            depth=row['depth_50bps_total'],
            imbalance=row['order_imbalance'],
            recent_liquidation_count=recent_count,
            recent_liquidation_value=recent_value
        )

        # Record for output
        state_records.append({
            'timestamp': ts,
            'data_trust': state.data_trust.value,
            'hypothesis': state.hypothesis_validity.value,
            'decision': state.decision_permission.value,
            'spread_bps': state.current_spread_bps,
            'depth': state.current_depth,
            'imbalance': state.current_imbalance,
            'liq_count': state.recent_liquidation_count,
            'liq_value': state.recent_liquidation_value,
        })

    # Save state records
    states_df = pd.DataFrame(state_records)
    states_df.to_csv(output_dir / "decision_states.csv", index=False)

    # Save state transitions (only changes)
    transitions = engine.get_state_transitions()
    with open(output_dir / "state_transitions.jsonl", 'w') as f:
        for t in transitions:
            f.write(json.dumps(t) + '\n')
    logger.info(f"Saved {len(transitions)} state transitions to state_transitions.jsonl")

    # Generate summary
    summary = engine.get_decision_summary()
    summary['conditions'] = conditions.to_dict()

    # Add time-based analysis
    states_df['decision_numeric'] = states_df['decision'].map({
        'ALLOWED': 1, 'RESTRICTED': 0.5, 'HALTED': 0
    })

    summary['time_analysis'] = {
        'total_timestamps': len(states_df),
        'allowed_pct': (states_df['decision'] == 'ALLOWED').mean() * 100,
        'restricted_pct': (states_df['decision'] == 'RESTRICTED').mean() * 100,
        'halted_pct': (states_df['decision'] == 'HALTED').mean() * 100,
    }

    with open(output_dir / "decision_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("DECISION ENGINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total state transitions: {summary['total_transitions']}")
    logger.info(f"\nDecision distribution:")
    logger.info(f"  ALLOWED: {summary['time_analysis']['allowed_pct']:.1f}%")
    logger.info(f"  RESTRICTED: {summary['time_analysis']['restricted_pct']:.1f}%")
    logger.info(f"  HALTED: {summary['time_analysis']['halted_pct']:.1f}%")

    return transitions, summary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Decision Engine analysis')
    parser.add_argument('--metrics', type=str, default='output/phase1/orderbook_metrics.csv')
    parser.add_argument('--liquidations', type=str, default='output/phase1/liquidation_report.csv')
    parser.add_argument('--output', type=str, default='output/phase1')
    args = parser.parse_args()

    run_decision_engine_analysis(args.metrics, args.liquidations, args.output)

