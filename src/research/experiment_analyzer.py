"""
Experiment Analyzer Module
==========================

Analyzes the aggregate summary of an experiment batch and provides insights.
"""

import json
import pandas as pd
from pathlib import Path
import sys

def analyze_batch(batch_dir):
    summary_path = Path(batch_dir) / "aggregate_summary.json"
    if not summary_path.exists():
        print(f"Error: {summary_path} not found.")
        return

    with open(summary_path, 'r') as f:
        data = json.load(f)

    runs = data.get("runs", [])
    if not runs:
        print("No runs found in summary.")
        return

    analysis_data = []
    for run in runs:
        config = run['config']
        summary = run['summary']
        
        # Flatten config and summary
        row = {**config}
        
        # Add decision percentages
        total_decisions = sum(summary['decision_counts'].values())
        if total_decisions > 0:
            for state, count in summary['decision_counts'].items():
                row[f'{state}_pct'] = round((count / total_decisions) * 100, 2)
        
        row['total_transitions'] = summary['total_transitions']
        analysis_data.append(row)

    df = pd.DataFrame(analysis_data)
    
    # Identify variable parameters
    variable_params = [col for col in df.columns if df[col].nunique() > 1 and not col.endswith('_pct')]
    
    print(f"\n===== Batch Analysis: {batch_dir} =====")
    print(f"Total Runs: {len(df)}")
    print(f"Variable Parameters: {variable_params}")
    
    # Display comparison table
    display_cols = variable_params + ['ALLOWED_pct', 'RESTRICTED_pct', 'HALTED_pct', 'total_transitions']
    print("\nComparison Table:")
    print(df[display_cols].to_string(index=False))

    # Calculate sensitivity (rough estimate)
    if variable_params:
        print("\nSensitivity Observation:")
        for param in variable_params:
            if pd.api.types.is_numeric_dtype(df[param]):
                corr = df[param].corr(df['ALLOWED_pct'])
                print(f" - {param} correlation with ALLOWED_pct: {corr:.4f}")
            else:
                print(f" - {param} (non-numeric): Impact see comparison table")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/research/experiment_analyzer.py <batch_directory>")
    else:
        analyze_batch(sys.argv[1])
