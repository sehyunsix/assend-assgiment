import json
import pandas as pd
import sys
from pathlib import Path

def calculate_scorecard(output_dir):
    decisions_path = Path(output_dir) / "decisions.jsonl"
    transitions_path = Path(output_dir) / "state_transitions.jsonl"
    
    if not decisions_path.exists():
        print(f"Error: {decisions_path} not found.")
        return

    # Load decisions
    decisions = []
    with open(decisions_path, "r") as f:
        for line in f:
            if line.strip():
                decisions.append(json.loads(line))
    
    df_d = pd.DataFrame(decisions)
    if df_d.empty:
        print("No decisions recorded.")
        return

    # Load transitions for more detail
    transitions = []
    if transitions_path.exists():
        with open(transitions_path, "r") as f:
            for line in f:
                if line.strip():
                    transitions.append(json.loads(line))
    df_t = pd.DataFrame(transitions)

    print(f"\n===== Stability Scorecard: {output_dir} =====")
    
    # 1. Basic Stats
    total_halts = len(df_d[df_d['action'] == 'HALT'])
    total_restricts = len(df_d[df_d['action'] == 'RESTRICT'])
    total_resumes = len(df_d[df_d['action'] == 'RESUME'])
    
    print(f"Actions: HALT={total_halts}, RESTRICT={total_restricts}, RESUME={total_resumes}")
    
    # 2. Flicker Rate (Transitions per internal time)
    if not df_t.empty:
        duration_us = df_t['ts'].max() - df_t['ts'].min()
        duration_hours = duration_us / (1_000_000 * 3600)
        flicker_rate = len(df_t) / duration_hours if duration_hours > 0 else 0
        print(f"Flicker Rate: {flicker_rate:.2f} transitions/hour")

    # 3. Halt Duration Analysis
    resumes = df_d[df_d['action'] == 'RESUME']
    if not resumes.empty:
        avg_halt_ms = resumes['duration_ms'].mean()
        max_halt_ms = resumes['duration_ms'].max()
        print(f"Average Protection Period: {avg_halt_ms:.2f} ms")
        print(f"Max Protection Period: {max_halt_ms:.2f} ms")

    # 4. Reason Analysis
    print("\nTop Halt/Restrict Triggers:")
    triggers = df_d[df_d['action'] != 'RESUME']['reason'].value_counts().head(3)
    for reason, count in triggers.items():
        print(f" - {reason}: {count}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scorecard.py <output_directory>")
    else:
        calculate_scorecard(sys.argv[1])
