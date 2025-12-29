#!/usr/bin/env python3
"""
Stability Score Analyzer
Analyzes state transitions and calculates stability scores by time period.
"""

import json
import sys
from datetime import datetime
from collections import defaultdict
from pathlib import Path

def load_transitions(filepath: str) -> list:
    """Load state transitions from JSONL file."""
    transitions = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                transitions.append(json.loads(line))
    return transitions

def calculate_stability_score(transitions: list) -> dict:
    """
    Calculate stability score.
    Score = (ALLOWED events) / (Total events) * 100
    """
    total = len(transitions)
    if total == 0:
        return {"score": 0, "total": 0, "allowed": 0, "restricted": 0, "halted": 0}
    
    allowed = sum(1 for t in transitions if t.get('decision') == 'ALLOWED')
    restricted = sum(1 for t in transitions if t.get('decision') == 'RESTRICTED')
    halted = sum(1 for t in transitions if t.get('decision') == 'HALTED')
    
    score = (allowed / total) * 100
    
    return {
        "score": round(score, 2),
        "total": total,
        "allowed": allowed,
        "restricted": restricted,
        "halted": halted
    }

def group_by_time_window(transitions: list, window_seconds: int = 60) -> dict:
    """Group transitions by time window."""
    window_us = window_seconds * 1_000_000
    groups = defaultdict(list)
    
    for t in transitions:
        event_ts = t.get('event_ts', 0)
        window_key = (event_ts // window_us) * window_us
        groups[window_key].append(t)
    
    return dict(sorted(groups.items()))

def format_timestamp(ts_us: int) -> str:
    """Convert microsecond timestamp to readable format."""
    try:
        ts_sec = ts_us / 1_000_000
        return datetime.fromtimestamp(ts_sec).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return f"TS:{ts_us}"

def print_stability_report(filepath: str, window_seconds: int = 60):
    """Print stability report by time window."""
    print(f"\n{'='*70}")
    print(f"📊 STABILITY SCORE ANALYSIS")
    print(f"{'='*70}")
    print(f"File: {filepath}")
    print(f"Window Size: {window_seconds} seconds")
    print(f"{'='*70}\n")
    
    transitions = load_transitions(filepath)
    
    if not transitions:
        print("No transitions found!")
        return
    
    # Overall statistics
    overall = calculate_stability_score(transitions)
    print(f"📈 OVERALL STABILITY")
    print(f"   Score: {overall['score']:.1f}%")
    print(f"   Total Events: {overall['total']}")
    print(f"   ✅ ALLOWED: {overall['allowed']} ({overall['allowed']/overall['total']*100:.1f}%)")
    print(f"   ⚠️  RESTRICTED: {overall['restricted']} ({overall['restricted']/overall['total']*100:.1f}%)")
    print(f"   🛑 HALTED: {overall['halted']} ({overall['halted']/overall['total']*100:.1f}%)")
    print()
    
    # By time window
    groups = group_by_time_window(transitions, window_seconds)
    
    print(f"📅 STABILITY BY TIME WINDOW ({window_seconds}s intervals)")
    print(f"{'-'*70}")
    print(f"{'Time Window':<25} {'Score':>8} {'Total':>7} {'✅':>6} {'⚠️':>6} {'🛑':>6}")
    print(f"{'-'*70}")
    
    for ts, group in groups.items():
        stats = calculate_stability_score(group)
        time_str = format_timestamp(ts)
        
        # Color indicator based on score
        if stats['score'] >= 80:
            indicator = "🟢"
        elif stats['score'] >= 50:
            indicator = "🟡"
        else:
            indicator = "🔴"
        
        print(f"{indicator} {time_str:<22} {stats['score']:>7.1f}% {stats['total']:>6} {stats['allowed']:>6} {stats['restricted']:>6} {stats['halted']:>6}")
    
    print(f"{'-'*70}")
    
    # Trigger analysis
    print(f"\n📋 TOP TRIGGERS (causing non-ALLOWED states)")
    print(f"{'-'*70}")
    
    trigger_counts = defaultdict(int)
    for t in transitions:
        if t.get('decision') != 'ALLOWED':
            trigger = t.get('trigger', 'unknown')
            # Simplify trigger for grouping
            if 'quarantine:' in trigger:
                trigger = trigger.split('|')[0] if '|' in trigger else trigger
            trigger_counts[trigger] += 1
    
    for trigger, count in sorted(trigger_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"   {count:>5}x  {trigger[:60]}")
    
    print(f"\n{'='*70}\n")

def main():
    # Default paths
    default_paths = [
        "output/historical/state_transitions.jsonl",
        "output/realtime/state_transitions.jsonl",
    ]
    
    filepath = sys.argv[1] if len(sys.argv) > 1 else None
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    
    if filepath:
        if Path(filepath).exists():
            print_stability_report(filepath, window)
        else:
            print(f"File not found: {filepath}")
    else:
        # Try default paths
        for path in default_paths:
            if Path(path).exists():
                print_stability_report(path, window)

if __name__ == "__main__":
    main()
