#!/usr/bin/env python3
"""
Export Regular dataset training results to CSV format.
Parses result files from src/crypto_research_minute directory and outputs CSV.
"""

import os
import re
import csv
from pathlib import Path

def parse_filename(filename):
    """
    Parse result filename to extract metadata.
    Examples:
      - results_mobilenetv3_test_2024-03_1m_14days_w5.txt
      - results_levit_train_2024-06_1m_1week_w5_exp2.txt
    Returns: dict with metadata
    """
    # Pattern for regular period (e.g., 14days) or 1week
    pattern = r"results_(\w+)_(train|test)_(\d{4})-(\d{2})_(\d+m)_(\d+days|1week)_w(\d+)(?:_exp\d+)?\.txt"
    match = re.match(pattern, filename)

    if match:
        model, result_type, year, month, interval, period, window = match.groups()
        # Convert "1week" to 7 days
        period_days = 7 if period == "1week" else int(period.replace("days", ""))
        return {
            "model": model,
            "type": result_type,  # 'train' or 'test'
            "period_days": period_days,
            "window": int(window),
            "year": int(year),
            "month": int(month),
            "interval": interval
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

def export_regular_to_csv(src_dir, output_csv):
    """
    Scan src/crypto_research_minute directory for results and export to CSV.
    """
    results = []
    coins = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"]
    models = ["mobilenetv3", "ghostnet", "levit"]

    # Find all result files for each coin
    for coin in coins:
        coin_result_dir = os.path.join(src_dir, coin, "results")

        if not os.path.exists(coin_result_dir):
            print(f"Warning: {coin_result_dir} does not exist")
            continue

        # Find all result files for our models
        for filename in os.listdir(coin_result_dir):
            # Check if file matches any of our models
            if not any(filename.startswith(f"results_{model}_") for model in models):
                continue

            if not filename.endswith(".txt"):
                continue

            # Parse filename
            metadata = parse_filename(filename)
            if not metadata:
                print(f"Warning: Could not parse filename: {filename}")
                continue

            # Parse result file
            filepath = os.path.join(coin_result_dir, filename)
            metrics = parse_result_file(filepath)

            # Build result row
            result_row = {
                "Coin": coin,
                "Dataset": "Regular",
                "Model": metadata["model"],
                "Window": metadata["window"],
                "Period_Days": metadata["period_days"],
                "Result_Type": metadata["type"],  # train or test
                "Test_Year": metadata["year"],
                "Test_Month": metadata["month"],
                "Accuracy": metrics["Accuracy"],
                "F1": metrics["F1"],
                "Recall": metrics["Recall"],
                "AUROC": metrics["Auroc"],
                "AUPRC": metrics["Auprc"],
                "Source_File": filename
            }

            results.append(result_row)

    # Write to CSV
    if results:
        fieldnames = ["Coin", "Dataset", "Model", "Window", "Period_Days", "Result_Type",
                     "Test_Year", "Test_Month", "Accuracy", "F1", "Recall",
                     "AUROC", "AUPRC", "Source_File"]

        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"Exported {len(results)} results to {output_csv}")

        # Print summary
        print("\nSummary by Coin:")
        for coin in coins:
            count = sum(1 for r in results if r["Coin"] == coin)
            print(f"  {coin}: {count} results")

        print("\nSummary by Model:")
        for model in models:
            count = sum(1 for r in results if r["Model"] == model)
            print(f"  {model}: {count} results")

        print(f"\nTotal: {len(results)} Regular dataset results")

        return results
    else:
        print("No results found!")
        return []

if __name__ == "__main__":
    src_dir = "/home/nckh/son/Candlestick/src/crypto_research_minute"
    output_csv = "/home/nckh/son/Candlestick/regular_results.csv"

    print("Exporting Regular dataset results to CSV...")
    print(f"Source directory: {src_dir}")
    print(f"Output file: {output_csv}")
    print("=" * 60)

    export_regular_to_csv(src_dir, output_csv)

    print("\nDone!")
