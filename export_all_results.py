#!/usr/bin/env python3
"""
Export all dataset training results to per-coin CSV format.
Generates separate CSV files for each coin: {COIN}_exp{N}_results.csv
- exp1: Regular dataset (Experiment I)
- exp2: Fullimage dataset (Experiment II)
- exp3: Irregular dataset (Experiment III)

Output format matches: database/crypto_research_minute_fullimage/exp_results/
"""

import os
import re
import csv
from pathlib import Path

# Dataset configuration
DATASETS = {
    "1": {
        "name": "Regular",
        "experiment": "I",
        "dir": "/home/nckh/son/Candlestick/database",
        "subpath": ""
    },
    "2": {
        "name": "Fullimage",
        "experiment": "II",
        "dir": "/home/nckh/son/Candlestick/database",
        "subpath": "crypto_research_minute_fullimage"
    },
    "3": {
        "name": "Irregular",
        "experiment": "III",
        "dir": "/home/nckh/son/Candlestick/database",
        "subpath": "crypto_research_minute_irregular"
    }
}

COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"]
MODELS = ["edgenext", "mobilenetv3", "ghostnet", "levit", "custom"]

# Final columns (in order)
FINAL_COLUMNS = [
    "Coin", "Experiment", "Model", "Window_Size", "Period", "Month", "Dataset",
    "Accuracy", "F1", "Recall", "AUROC", "AUPRC"
]


def parse_filename(filename):
    """
    Parse result filename to extract metadata.
    Examples:
      - results_mobilenetv3_test_2024-03_1m_14days_w5.txt
      - results_levit_train_2024-06_1m_1week_w5_exp2.txt
      - results_edgenext_test_2024-03_1m_7days_w5_m0.6.txt (irregular)
    Returns: dict with metadata
    """
    # Pattern for regular/fullimage datasets
    pattern = r"results_(\w+)_(train|test)_(\d{4})-(\d{2})_(\d+m)_(\d+days|1week)_w(\d+)(?:_exp\d+)?\.txt"
    match = re.match(pattern, filename)

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
            "missing": None
        }

    # Pattern for irregular dataset (with missing ratio)
    pattern_irregular = r"results_(\w+)_(train|test)_(\d{4})-(\d{2})_(\d+m)_(\d+days|1week)_w(\d+)_m([\d.]+)\.txt"
    match_irregular = re.match(pattern_irregular, filename)

    if match_irregular:
        model, result_type, year, month, interval, period, window, missing = match_irregular.groups()
        period_days = 7 if period == "1week" else int(period.replace("days", ""))
        return {
            "model": model,
            "type": result_type,
            "period_days": period_days,
            "window": int(window),
            "year": int(year),
            "month": int(month),
            "interval": interval,
            "missing": float(missing)
        }

    return None


def parse_result_file(filepath):
    """
    Parse result file to extract metrics.
    Returns: dict with Accuracy, F1, Recall, Auroc, Auprc
    """
    metrics = {
        "Accuracy": None,
        "F1": None,
        "Recall": None,
        "Auroc": None,
        "Auprc": None
    }

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                for metric in metrics.keys():
                    if line.startswith(f"{metric}:"):
                        value = line.split(":")[1].strip()
                        try:
                            metrics[metric] = float(value)
                        except ValueError:
                            metrics[metric] = value
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
        "AUROC": metrics["Auroc"],
        "AUPRC": metrics["Auprc"]
    }
    return result_row


def export_dataset(exp_id, dataset_info, output_dir):
    """
    Export results for a specific dataset.
    Returns: dict of {coin: result_count}
    """
    dataset_name = dataset_info["name"]
    experiment = dataset_info["experiment"]
    base_dir = dataset_info["dir"]
    subpath = dataset_info["subpath"]

    # Build the dataset directory path
    if subpath:
        dataset_dir = os.path.join(base_dir, subpath)
    else:
        dataset_dir = base_dir

    print(f"\n{'='*60}")
    print(f"Processing {dataset_name} dataset (exp{exp_id}, Experiment {experiment})")
    print(f"Directory: {dataset_dir}")
    print(f"{'='*60}")

    coin_counts = {}

    for coin in COINS:
        coin_result_dir = os.path.join(dataset_dir, coin, "results")

        if not os.path.exists(coin_result_dir):
            print(f"Warning: {coin_result_dir} does not exist")
            coin_counts[coin] = 0
            continue

        results = []

        # Find all result files
        for filename in os.listdir(coin_result_dir):
            if not filename.startswith("results_") or not filename.endswith(".txt"):
                continue

            # Parse filename
            metadata = parse_filename(filename)
            if not metadata:
                continue

            # Parse result file
            filepath = os.path.join(coin_result_dir, filename)
            metrics = parse_result_file(filepath)

            # Build result row
            result_row = build_result_row(coin, experiment, metadata, metrics)
            results.append(result_row)

        coin_counts[coin] = len(results)

        # Write per-coin CSV
        if results:
            output_csv = os.path.join(output_dir, f"{coin}_exp{exp_id}_results.csv")
            with open(output_csv, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=FINAL_COLUMNS)
                writer.writeheader()
                writer.writerows(results)

            print(f"  {coin}: {len(results)} results -> {output_csv}")
        else:
            print(f"  {coin}: No results found")

    return coin_counts


def export_all_results(output_dir="/home/nckh/son/Candlestick/results_export"):
    """
    Export all datasets to per-coin CSV files.
    Output format: {COIN}_exp{N}_results.csv
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("Exporting All Dataset Results to Per-Coin CSV Format")
    print(f"Output directory: {output_dir}")
    print("="*60)

    # Summary for all datasets
    all_summary = {}

    for exp_id, dataset_info in DATASETS.items():
        coin_counts = export_dataset(exp_id, dataset_info, output_dir)
        all_summary[exp_id] = coin_counts

    # Print overall summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)

    for exp_id, dataset_info in DATASETS.items():
        dataset_name = dataset_info["name"]
        coin_counts = all_summary[exp_id]
        total = sum(coin_counts.values())
        print(f"\n{dataset_name} (exp{exp_id}): {total} total results")

    # Grand total
    grand_total = sum(sum(counts.values()) for counts in all_summary.values())
    print(f"\n{'='*60}")
    print(f"GRAND TOTAL: {grand_total} results exported")
    print(f"{'='*60}")

    # List generated files
    print("\nGenerated files:")
    for coin in COINS:
        print(f"  {coin}_exp1_results.csv  (Regular)")
        print(f"  {coin}_exp2_results.csv  (Fullimage)")
        print(f"  {coin}_exp3_results.csv  (Irregular)")

    return all_summary


if __name__ == "__main__":
    export_all_results()
    print("\nDone!")
