#!/usr/bin/env python3
"""
Regenerate comprehensive metrics CSV with Model column from existing benchmark results
"""

import json
import csv
import os
import glob
from datetime import datetime

def regenerate_csv_from_results():
    """Generate comprehensive metrics CSV with Model column from existing benchmark JSON"""
    
    # Find the most recent comprehensive benchmark JSON
    json_pattern = "benchmarks/results/comprehensive_benchmark_*.json"
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print("No comprehensive benchmark JSON files found")
        return
    
    # Use the most recent file
    latest_json = max(json_files, key=os.path.getctime)
    print(f"Using benchmark results from: {latest_json}")
    
    # Load the JSON data
    with open(latest_json, 'r') as f:
        data = json.load(f)
    
    # Extract metrics data
    metrics_data = []
    experiment_type_mapping = {
        "regular": "R",
        "fullimage": "F", 
        "irregular": "I"
    }
    
    combination_results = data.get("combination_results", {})
    
    for combo_key, combo_data in combination_results.items():
        if "results" not in combo_data or combo_data.get("status") in ["error", "skipped"]:
            continue
        
        # Parse combination key: coin_period_wX_experiment_type
        parts = combo_key.split("_")
        if len(parts) < 4:
            continue
            
        coin = parts[0]
        period = parts[1]
        window_spec = parts[2]
        experiment_type = parts[3]
        
        # Extract window size from "wX" format
        try:
            window_size = int(window_spec.replace("w", ""))
        except ValueError:
            continue
        
        # Map experiment type to single letter
        experiment_code = experiment_type_mapping.get(experiment_type, experiment_type[0].upper())
        
        # Extract month from timestamp (fallback to current date)
        timestamp = combo_data.get("timestamp")
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                month = dt.strftime('%Y-%m')
            except:
                month = datetime.now().strftime('%Y-%m')
        else:
            month = datetime.now().strftime('%Y-%m')
        
        # Process each model's results (Test dataset only)
        for model_name, model_result in combo_data["results"].items():
            if "error" in model_result:
                continue
            
            # Get model display name from model config or use model_name as fallback
            model_config = model_result.get("model_config", {})
            model_display_name = model_config.get("name", model_name)
            
            # Extract evaluation metrics (Test dataset)
            eval_metrics = model_result.get("evaluation_metrics", {})
            if eval_metrics:
                record = {
                    'Coin': coin,
                    'Experiment': experiment_code,
                    'Window_Size': window_size,
                    'Period': period,
                    'Month': month,
                    'Model': model_display_name,
                    'Accuracy': round(eval_metrics.get('accuracy', 0), 4),
                    'F1': round(eval_metrics.get('f1', 0), 4),
                    'Recall': round(eval_metrics.get('recall', 0), 4),
                    'AUROC': round(eval_metrics.get('auroc', 0), 4),
                    'AUPRC': round(eval_metrics.get('auprc', 0), 4)
                }
                metrics_data.append(record)
    
    if not metrics_data:
        print("No metrics data to export")
        return
    
    # Sort by Coin, Experiment, Window_Size, Period, Month, Model
    sort_key = lambda x: (x['Coin'], x['Experiment'], x['Window_Size'], 
                         x['Period'], x['Month'], x['Model'])
    metrics_data.sort(key=sort_key)
    
    # Write CSV file
    output_file = f"comprehensive_metrics_full_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    columns = ['Coin', 'Experiment', 'Window_Size', 'Period', 'Month', 'Model',
              'Accuracy', 'F1', 'Recall', 'AUROC', 'AUPRC']
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        
        for record in metrics_data:
            writer.writerow(record)
    
    # Summary
    models_count = len(set(record['Model'] for record in metrics_data))
    combinations_count = len(set((record['Coin'], record['Experiment'], record['Window_Size'], 
                                record['Period'], record['Month']) 
                               for record in metrics_data))
    
    print(f"✓ Exported {len(metrics_data)} records ({models_count} models × {combinations_count} combinations)")
    print(f"✓ CSV saved to: {output_file}")
    print(f"✓ Format: Coin,Experiment,Window_Size,Period,Month,Model,Accuracy,F1,Recall,AUROC,AUPRC")
    print(f"✓ Test dataset only, Dataset column removed, Model column added")
    
    return output_file

if __name__ == "__main__":
    regenerate_csv_from_results()