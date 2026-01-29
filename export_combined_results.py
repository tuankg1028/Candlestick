#!/usr/bin/env python3
"""
Combine results from database/exp_results/ and results_export/{Dataset}/

This script merges:
- database/exp_results/{COIN}_exp{N}_results.csv (custom CNN results)
- results_export/{Dataset}/{COIN}_exp{N}_results.csv (pre-trained model results)

And exports to:
- results_export/Combined/{Dataset}/{COIN}_exp{N}_results.csv

Merge logic:
- exp1 files: database/exp_results/{COIN}_exp1_results.csv + results_export/{Dataset}/{COIN}_exp1_results.csv
- exp2 files: database/exp_results/{COIN}_exp2_results.csv + results_export/{Dataset}/{COIN}_exp2_results.csv
"""

import os
import csv
from pathlib import Path

# Project configuration
PROJECT_DIR = "/home/nckh/son/Candlestick"
DATABASE_DIR = os.path.join(PROJECT_DIR, "database")
EXPORT_DIR = os.path.join(PROJECT_DIR, "results_export")
COMBINED_DIR = os.path.join(EXPORT_DIR, "Combined")

# Dataset configuration with their exp_results paths
DATASETS = {
    "Regular": {
        "exp_results_path": os.path.join(DATABASE_DIR, "exp_results")
    },
    "Fullimage": {
        "exp_results_path": os.path.join(DATABASE_DIR, "crypto_research_minute_fullimage", "exp_results")
    },
    "Irregular": {
        "exp_results_path": os.path.join(DATABASE_DIR, "crypto_research_minute_irregular", "exp_results")
    }
}
COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"]

# CSV columns (all files should have these columns)
COLUMNS = [
    "Coin", "Experiment", "Model", "Window_Size", "Period", "Month",
    "Dataset", "Accuracy", "F1", "Recall", "AUROC", "AUPRC"
]


def read_csv_file(filepath, use_cnn_label=False):
    """Read results from a CSV file.

    Args:
        filepath: Path to CSV file
        use_cnn_label: If True, use "CNN" instead of "custom" for missing Model column
    """
    results = []
    if not os.path.exists(filepath):
        return results

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Ensure all columns exist
            for col in COLUMNS:
                if col not in row:
                    if col == "Model":
                        row[col] = "CNN" if use_cnn_label else "custom"
                    else:
                        row[col] = ""
            results.append(row)

    return results


def merge_results_by_key(results_list, key_columns):
    """Merge results, keeping only one entry per unique key combination."""
    seen = set()
    merged = []

    for result in results_list:
        # Create key from specified columns
        key_parts = []
        for col in key_columns:
            key_parts.append(str(result.get(col, "")))
        key = "|".join(key_parts)

        if key not in seen:
            seen.add(key)
            merged.append(result)

    return merged


def main():
    """Main export function."""
    # Create combined directory structure
    for dataset in DATASETS:
        os.makedirs(os.path.join(COMBINED_DIR, dataset), exist_ok=True)

    print("="*60)
    print("Combined Results Export")
    print("="*60)
    print("Merging:")
    print("  Regular:   database/exp_results/ + results_export/Regular/")
    print("  Fullimage: database/crypto_research_minute_fullimage/exp_results/ + results_export/Fullimage/")
    print("  Irregular: database/crypto_research_minute_irregular/exp_results/ + results_export/Irregular/")
    print(f"Output: {COMBINED_DIR}/{{Dataset}}/")
    print("="*60)

    all_stats = {}

    for dataset in DATASETS.keys():
        exp_results_path = DATASETS[dataset]["exp_results_path"]

        print(f"\n{'='*60}")
        print(f"Processing {dataset} dataset")
        print(f"exp_results: {exp_results_path}")
        print(f"{'='*60}")

        dataset_stats = {}

        for coin in COINS:
            for exp_num in ["exp1", "exp2"]:
                combined = []
                source_counts = []

                # Read from results_export/{Dataset}/{COIN}_exp{N}_results.csv
                source_file = os.path.join(EXPORT_DIR, dataset, f"{coin}_{exp_num}_results.csv")
                if os.path.exists(source_file):
                    results = read_csv_file(source_file)
                    combined.extend(results)
                    source_counts.append(f"{len(results)} from results_export/{dataset}/")

                # Read from dataset-specific exp_results/{COIN}_exp{N}_results.csv
                # Use "CNN" label for Fullimage exp_results (they don't have Model column)
                exp_results_file = os.path.join(exp_results_path, f"{coin}_{exp_num}_results.csv")
                if os.path.exists(exp_results_file):
                    use_cnn_label = (dataset == "Fullimage")
                    results = read_csv_file(exp_results_file, use_cnn_label=use_cnn_label)
                    combined.extend(results)
                    source_counts.append(f"{len(results)} from {dataset}/exp_results/")

                # Merge results by key
                key_columns = ["Coin", "Experiment", "Model", "Window_Size", "Period", "Month", "Dataset"]
                merged = merge_results_by_key(combined, key_columns)

                # Write combined CSV
                if merged:
                    # Sort by Model, Window_Size, Period
                    merged_sorted = sorted(merged, key=lambda x: (x.get("Model", ""), x.get("Window_Size", ""), x.get("Period", "")))
                    output_path = os.path.join(COMBINED_DIR, dataset, f"{coin}_{exp_num}_results.csv")
                    with open(output_path, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=COLUMNS)
                        writer.writeheader()
                        writer.writerows(merged_sorted)
                    sources = " + ".join(source_counts)
                    print(f"  {coin}_{exp_num}: {sources} => {len(merged)} unique (merged from {len(combined)})")
                else:
                    print(f"  {coin}_{exp_num}: No results")

                dataset_stats[f"{coin}_{exp_num}"] = len(merged) if merged else 0

        all_stats[dataset] = dataset_stats

    # Print summary
    print("\n" + "="*60)
    print("EXPORT SUMMARY")
    print("="*60)

    for dataset in DATASETS:
        total = sum(all_stats[dataset].values())
        print(f"\n{dataset}: {total} total results")

    print("\nStructure created:")
    print("results_export/Combined/")
    for dataset in DATASETS:
        print(f"  {dataset}/")
        for coin in COINS:
            print(f"    {coin}_exp1_results.csv")
            print(f"    {coin}_exp2_results.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
