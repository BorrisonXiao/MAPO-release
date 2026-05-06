#!/usr/bin/env python3
"""
Robust JSON/JSONL to Parquet Converter (v2)

This script attempts multiple strategies to load a JSON file, ignoring the file extension
to ensure maximum compatibility.

Usage:
    python json2parquet_v2.py -i input_file.json -o output.parquet
"""

import pandas as pd
import argparse
import os
import json
import sys
import ast

def try_load_standard_json(path):
    """Strategy 1: specific for standard JSON list of objects."""
    try:
        # Try pandas first (fastest)
        return pd.read_json(path)
    except ValueError:
        # Try python json standard lib (more robust error messages)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.DataFrame(data)

def try_load_jsonl(path):
    """Strategy 2: specific for JSON Lines (one object per line)."""
    try:
        # Try pandas with lines=True
        return pd.read_json(path, lines=True)
    except ValueError:
        # Try manual line-by-line parsing (handles occasional empty lines better)
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"  Warning: Skipped malformed JSON on line {i+1}: {e}")
        if not data:
            raise ValueError("No valid JSON lines found.")
        return pd.DataFrame(data)

def inspect_file_head(path, n_chars=200):
    """Helper to show the user what the file looks like on failure."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read(n_chars)
            print("\n--- FILE PREVIEW (First 200 chars) ---")
            print(content)
            print("--------------------------------------\n")
    except Exception:
        print("(Could not read file preview)")

def main():
    parser = argparse.ArgumentParser(description='Robust JSON/JSONL to Parquet Converter')
    parser.add_argument('-i', '--input', required=True, help='Input file path')
    parser.add_argument('-o', '--output', required=True, help='Output Parquet file path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    print(f"Processing: {args.input}")
    df = None
    
    # --- Attempt 1: Standard JSON ([{...}, {...}]) ---
    try:
        df = try_load_standard_json(args.input)
        print("-> Success: Loaded as Standard JSON")
    except Exception as e_json:
        # --- Attempt 2: JSON Lines ({...}\n{...}) ---
        try:
            df = try_load_jsonl(args.input)
            print("-> Success: Loaded as JSON Lines (JSONL)")
        except Exception as e_jsonl:
            print("\n❌ FAILED to load data.")
            print(f"1. Standard JSON error: {e_json}")
            print(f"2. JSONL error: {e_jsonl}")
            
            inspect_file_head(args.input)
            sys.exit(1)

    # Clean up and normalize columns
    if len(df) == 0:
        print("Error: DataFrame is empty.")
        sys.exit(1)

    # Fix 'choices' column if it loaded as string representation of lists
    if 'choices' in df.columns:
        sample = df['choices'].iloc[0] if len(df) > 0 else None
        if isinstance(sample, str) and sample.strip().startswith('['):
            print("Parsing stringified 'choices' column...")
            try:
                df['choices'] = df['choices'].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
            except Exception as e:
                print(f"Warning: Failed to parse 'choices': {e}")

    # Save to Parquet
    try:
        df.to_parquet(args.output, engine='pyarrow', index=False)
        print(f"✅ Successfully converted {len(df)} records to '{args.output}'")
        
        # Check column names for the user
        cols = df.columns.tolist()
        if 'model_prediction' in cols and 'model_output' not in cols:
            print(f"ℹ️  NOTE: Use flag '--model_output_column model_prediction' in evaluation.")
            
    except Exception as e:
        print(f"Error saving parquet: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()