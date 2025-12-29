import json
import pandas as pd

def analyze(file_path):
    try:
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        df = pd.DataFrame(data)
        if df.empty:
            print(f"File {file_path} is empty.")
            return

        print(f"\n=== Analysis for {file_path} ===")
        print(f"Total entries: {len(df)}")
        
        if 'action' in df.columns:
            print("\nAction counts:")
            print(df['action'].value_counts())
            
            print("\nTop 5 Reasons for RESTRICT/HALT:")
            print(df[df['action'] != 'RESUME']['reason'].value_counts().head(5))
            
            if 'duration_ms' in df.columns:
                print("\nDuration stats for RESUME (ms):")
                resume_df = df[df['action'] == 'RESUME']
                if not resume_df.empty:
                    print(resume_df['duration_ms'].describe())
                    
                    very_short = df[(df['action'] == 'RESUME') & (df['duration_ms'] < 100)]
                    if not very_short.empty:
                        print(f"\nWARNING: Found {len(very_short)} RESUME events with duration < 100ms")
                        print(very_short[['ts', 'duration_ms', 'reason']].head())
                else:
                    print("No RESUME actions found.")
        
        if 'data_trust' in df.columns:
            print("\nData Trust distribution:")
            print(df['data_trust'].value_counts())

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")

analyze('output/historical/validation/decisions.jsonl')
