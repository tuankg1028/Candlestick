#!/usr/bin/env python3
"""
Export all dataset training results to per-coin CSV format.
Generates separate folders for each dataset, each containing exp1 and exp2 files.

Structure:
results_export/
├── Regular/
│   ├── BTCUSDT_exp1_results.csv  (Experiment: I - matching lengths)
│   ├── BTCUSDT_exp2_results.csv  (Experiment: II - 1-week training)
│   └── ...
├── Fullimage/
│   ├── BTCUSDT_exp1_results.csv  (Experiment: I)
│   ├── BTCUSDT_exp2_results.csv  (Experiment: II)
│   └── ...
└── Irregular/
    ├── BTCUSDT_exp1_results.csv  (Experiment: I)
    ├── BTCUSDT_exp2_results.csv  (Experiment: II)
    └── ...
"""

import os
import re
import csv
from pathlib import Path

# Dataset configuration
DATASETS = {
    "Regular": {
        "dir": "/home/nckh/son/Candlestick/database",
        "subpath": ""
    },
    "Fullimage": {
        "dir": "/home/nckh/son/Candlestick/database",
        "subpath": "crypto_research_minute_fullimage"
    },
    "Irregular": {
        "dir": "/home/nckh/son/Candlestick/database",
        "subpath": "crypto_research_minute_irregular"
    }
}

COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"]

# Final columns (in order)
FINAL_COLUMNS = [
    "Coin", "Experiment", "Model", "Window_Size", "Period", "Month", "Dataset",
    "Accuracy", "F1", "Recall", "AUROC", "AUPRC"
]


def parse_filename(filename):
    """
    Parse result filename to extract metadata.
    Returns: dict with metadata, or None if not matched
    """
    # Pattern for irregular dataset with exp2 (1-week training) - suffix format _exp2_95pct
    # e.g., results_edgenext_test_2024-03_1m_21days_w15_exp2_95pct.txt
    pattern_irregular_exp2 = r"results_(\w+)_(train|test)_(\d{4})-(\d{2})_(\d+m)_(\d+days|1week)_w(\d+)_exp2_(\d+)pct\.txt"
    match = re.match(pattern_irregular_exp2, filename)

    if match:
        model, result_type, year, month, interval, period, window, missing_pct = match.groups()
        period_days = 7 if period == "1week" else int(period.replace("days", ""))
        return {
            "model": model,
            "type": result_type,
            "period_days": period_days,
            "window": int(window),
            "year": int(year),
            "month": int(month),
            "interval": interval,
            "missing": float(missing_pct) / 100.0,
            "design": "exp2"  # 1-week training design
        }

    # Pattern for irregular dataset with exp2 - alt format _60pct_exp2
    # e.g., results_test_2024-04_1m_14days_w15_60pct_exp2.txt
    pattern_irregular_exp2_alt = r"results_(?:\w+_)??(train|test)_(\d{4})-(\d{2})_(\d+m)_(\d+days|1week)_w(\d+)_(\d+)pct_exp2\.txt"
    match = re.match(pattern_irregular_exp2_alt, filename)

    if match:
        result_type, year, month, interval, period, window, missing_pct = match.groups()
        period_days = 7 if period == "1week" else int(period.replace("days", ""))
        # Extract model name if present (optional prefix before result_type)
        model_match = re.match(r"results_(\w+)_(?:train|test)", filename)
        model = model_match.group(1) if model_match else "custom"
        return {
            "model": model,
            "type": result_type,
            "period_days": period_days,
            "window": int(window),
            "year": int(year),
            "month": int(month),
            "interval": interval,
            "missing": float(missing_pct) / 100.0,
            "design": "exp2"  # 1-week training design
        }

    # Pattern for irregular dataset exp1 (no exp2 suffix) - with model name
    # e.g., results_edgenext_test_2024-03_1m_7days_w5_60pct.txt
    pattern_irregular_exp1 = r"results_(\w+)_(train|test)_(\d{4})-(\d{2})_(\d+m)_(\d+days|1week)_w(\d+)_(\d+)pct\.txt"
    match = re.match(pattern_irregular_exp1, filename)

    if match:
        model, result_type, year, month, interval, period, window, missing_pct = match.groups()
        period_days = 7 if period == "1week" else int(period.replace("days", ""))
        return {
            "model": model,
            "type": result_type,
            "period_days": period_days,
            "window": int(window),
            "year": int(year),
            "month": int(month),
            "interval": interval,
            "missing": float(missing_pct) / 100.0,
            "design": "exp1"  # matching lengths design
        }

    # Pattern for irregular dataset exp1 - without model name (old format)
    # e.g., results_test_2024-12_1m_14days_w30_95pct.txt
    pattern_irregular_exp1_old = r"results_(train|test)_(\d{4})-(\d{2})_(\d+m)_(\d+days|1week)_w(\d+)_(\d+)pct\.txt"
    match = re.match(pattern_irregular_exp1_old, filename)

    if match:
        result_type, year, month, interval, period, window, missing_pct = match.groups()
        period_days = 7 if period == "1week" else int(period.replace("days", ""))
        return {
            "model": "custom",  # Default model for old format files
            "type": result_type,
            "period_days": period_days,
            "window": int(window),
            "year": int(year),
            "month": int(month),
            "interval": interval,
            "missing": float(missing_pct) / 100.0,
            "design": "exp1"  # matching lengths design
        }

    # Pattern for regular/fullimage datasets with exp2 suffix
    # e.g., results_levit_train_2024-06_1m_1week_w5_exp2.txt
    pattern_exp2 = r"results_(\w+)_(train|test)_(\d{4})-(\d{2})_(\d+m)_(\d+days|1week)_w(\d+)_exp2\.txt"
    match = re.match(pattern_exp2, filename)

    if match:
        model, result_type, year, month, interval, period, window = match.groups()
        period_days = 7 if period == "1week" else int(period.replace("days", ""))
        return {
            "model": model,
            "type": result_type,
            "period_days": period_days,
            "window": int(window),
            "year": int(year),
            "month": int(month),
            "interval": interval,
            "missing": None,
            "design": "exp2"  # 1-week training design
        }

    # Pattern for regular/fullimage datasets without exp2 suffix (exp1)
    # e.g., results_mobilenetv3_test_2024-03_1m_14days_w5.txt
    pattern_exp1 = r"results_(\w+)_(train|test)_(\d{4})-(\d{2})_(\d+m)_(\d+days|1week)_w(\d+)\.txt"
    match = re.match(pattern_exp1, filename)

    if match:
        model, result_type, year, month, interval, period, window = match.groups()
        period_days = 7 if period == "1week" else int(period.replace("days", ""))
        return {
            "model": model,
            "type": result_type,
            "period_days": period_days,
            "window": int(window),
            "year": int(year),
            "month": int(month),
            "interval": interval,
            "missing": None,
            "design": "exp1"  # matching lengths design
        }

    return None


def parse_result_file(filepath):
    """
    Parse result file to extract metrics.
    Returns: dict with Accuracy, F1, Recall, AUROC, AUPRC
    """
    metrics = {
        "Accuracy": None,
        "F1": None,
        "Recall": None,
        "AUROC": None,
        "AUPRC": None
    }
    # Case-insensitive mapping
    metric_map = {
        "Accuracy": "Accuracy", "accuracy": "Accuracy",
        "F1": "F1", "f1": "F1",
        "Recall": "Recall", "recall": "Recall",
        "Auroc": "AUROC", "AUROC": "AUROC", "auroc": "AUROC",
        "Auprc": "AUPRC", "AUPRC": "AUPRC", "auprc": "AUPRC"
    }

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                for metric_name, standard_name in metric_map.items():
                    if line.startswith(f"{metric_name}:"):
                        value = line.split(":", 1)[1].strip()
                        try:
                            metrics[standard_name] = float(value)
                        except ValueError:
                            metrics[standard_name] = value
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

    return metrics


def build_result_row(coin, experiment, metadata, metrics):
    """Build a result row with the desired column format."""
    # Format month as YYYY-MM
    month_str = f"{metadata['year']}-{metadata['month']:02d}"

    # Format period with "days" suffix
    period_str = f"{metadata['period_days']}days"

    # Capitalize dataset (Train/Test)
    dataset_str = metadata["type"].capitalize()

    result_row = {
        "Coin": coin,
        "Experiment": experiment,
        "Model": metadata["model"],
        "Window_Size": metadata["window"],
        "Period": period_str,
        "Month": month_str,
        "Dataset": dataset_str,
        "Accuracy": metrics["Accuracy"],
        "F1": metrics["F1"],
        "Recall": metrics["Recall"],
        "AUROC": metrics["AUROC"],
        "AUPRC": metrics["AUPRC"]
    }
    return result_row


def export_dataset(dataset_name, dataset_info, output_dir):
    """
    Export results for a specific dataset.
    Creates exp1 and exp2 CSV files for each coin.
    """
    base_dir = dataset_info["dir"]
    subpath = dataset_info["subpath"]

    # Build the dataset directory path
    if subpath:
        dataset_dir = os.path.join(base_dir, subpath)
    else:
        dataset_dir = base_dir

    # Create output folder for this dataset
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing {dataset_name} dataset")
    print(f"Directory: {dataset_dir}")
    print(f"Output: {dataset_output_dir}")
    print(f"{'='*60}")

    coin_counts = {"exp1": {}, "exp2": {}}

    for coin in COINS:
        coin_result_dir = os.path.join(dataset_dir, coin, "results")

        if not os.path.exists(coin_result_dir):
            print(f"  {coin}: No results directory")
            coin_counts["exp1"][coin] = 0
            coin_counts["exp2"][coin] = 0
            continue

        # Separate results by design (exp1 or exp2)
        results_exp1 = []
        results_exp2 = []

        # Find all result files
        for filename in os.listdir(coin_result_dir):
            if not filename.startswith("results_") or not filename.endswith(".txt"):
                continue

            # Parse filename
            metadata = parse_filename(filename)
            if not metadata:
                print(f"  {coin}: Warning - could not parse {filename}")
                continue

            # Parse result file
            filepath = os.path.join(coin_result_dir, filename)
            metrics = parse_result_file(filepath)

            # Build result row with correct experiment label
            experiment = "I" if metadata["design"] == "exp1" else "II"
            result_row = build_result_row(coin, experiment, metadata, metrics)

            # Add to appropriate list
            if metadata["design"] == "exp1":
                results_exp1.append(result_row)
            else:
                results_exp2.append(result_row)

        coin_counts["exp1"][coin] = len(results_exp1)
        coin_counts["exp2"][coin] = len(results_exp2)

        # Write exp1 CSV
        if results_exp1:
            # Sort by Model, Window_Size, Period
            results_exp1_sorted = sorted(results_exp1, key=lambda x: (x.get("Model", ""), x.get("Window_Size", ""), x.get("Period", "")))
            output_csv = os.path.join(dataset_output_dir, f"{coin}_exp1_results.csv")
            with open(output_csv, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=FINAL_COLUMNS)
                writer.writeheader()
                writer.writerows(results_exp1_sorted)
            print(f"  {coin}_exp1: {len(results_exp1)} results")
        else:
            print(f"  {coin}_exp1: No results")

        # Write exp2 CSV
        if results_exp2:
            # Sort by Model, Window_Size, Period
            results_exp2_sorted = sorted(results_exp2, key=lambda x: (x.get("Model", ""), x.get("Window_Size", ""), x.get("Period", "")))
            output_csv = os.path.join(dataset_output_dir, f"{coin}_exp2_results.csv")
            with open(output_csv, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=FINAL_COLUMNS)
                writer.writeheader()
                writer.writerows(results_exp2_sorted)
            print(f"  {coin}_exp2: {len(results_exp2)} results")
        else:
            print(f"  {coin}_exp2: No results")

    return coin_counts


def export_all_results(output_dir="/home/nckh/son/Candlestick/results_export"):
    """
    Export all datasets to organized folder structure.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("Exporting All Dataset Results")
    print(f"Output directory: {output_dir}")
    print("="*60)

    # Summary for all datasets
    all_summary = {}

    for dataset_name, dataset_info in DATASETS.items():
        coin_counts = export_dataset(dataset_name, dataset_info, output_dir)
        all_summary[dataset_name] = coin_counts

    # Print overall summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)

    grand_total = 0
    for dataset_name, dataset_info in DATASETS.items():
        coin_counts = all_summary[dataset_name]
        total_exp1 = sum(coin_counts["exp1"].values())
        total_exp2 = sum(coin_counts["exp2"].values())
        total = total_exp1 + total_exp2
        grand_total += total
        print(f"\n{dataset_name}:")
        print(f"  exp1: {total_exp1} results")
        print(f"  exp2: {total_exp2} results")
        print(f"  Total: {total} results")

    print(f"\n{'='*60}")
    print(f"GRAND TOTAL: {grand_total} results exported")
    print(f"{'='*60}")

    # List generated structure
    print("\nGenerated structure:")
    print("results_export/")
    for dataset_name in DATASETS.keys():
        print(f"  {dataset_name}/")
        for coin in COINS:
            print(f"    {coin}_exp1_results.csv")
            print(f"    {coin}_exp2_results.csv")

    return all_summary


if __name__ == "__main__":
    export_all_results()
    print("\nDone!")
