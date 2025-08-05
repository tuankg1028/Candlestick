#!/usr/bin/env python3
"""
Test script for the enhanced CSV export functionality in huggingface_benchmark.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from huggingface_benchmark import MetricsCollector
import tempfile
import csv

def test_each_model_separate_row():
    """Test that each model gets its own row in CSV"""
    print("Testing Each Model Gets Separate Row...")
    
    collector = MetricsCollector()
    
    # Add multiple models for same combination to test that each gets its own row
    models = ["resnet50", "vit_base", "efficientnet_b0"]
    
    for model in models:
        # Test metrics with different performance levels to distinguish models
        accuracy = 0.9995 - (models.index(model) * 0.01)  # Different accuracies
        test_metrics = {
            'accuracy': accuracy,
            'f1': accuracy - 0.001,
            'recall': accuracy - 0.002,
            'auroc': accuracy + 0.0005,
            'auprc': accuracy + 0.0003
        }
        
        collector.add_result(
            coin="ADAUSDT",
            experiment_type="irregular", 
            window_size=5,
            period="14days",
            month="2024-01",
            dataset_type="Test",
            metrics=test_metrics,
            model_name=model
        )
        
        # Train metrics (usually perfect but slightly different)
        train_accuracy = 1.0 - (models.index(model) * 0.001)
        train_metrics = {
            'accuracy': train_accuracy,
            'f1': train_accuracy,
            'recall': train_accuracy,
            'auroc': train_accuracy,
            'auprc': train_accuracy
        }
        
        collector.add_result(
            coin="ADAUSDT",
            experiment_type="irregular",
            window_size=5, 
            period="14days",
            month="2024-09",
            dataset_type="Train",
            metrics=train_metrics,
            model_name=model
        )
    
    print(f"Added {len(collector.metrics_data)} total records ({len(models)} models × 2 datasets)")
    
    # Export all models (default behavior)
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as tmp_file:
        collector.export_to_csv(tmp_file.name, include_all_models=True)
        
        # Read and analyze results
        with open(tmp_file.name, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            print(f"\n📊 CSV EXPORT CONTAINS {len(rows)} ROWS:")
            print("Header:", ','.join(reader.fieldnames))
            print("-" * 80)
            
            # Display each row to show that each model is included
            for i, row in enumerate(rows):
                print(f"Row {i+1}: {row['Coin']},{row['Experiment']},{row['Window_Size']},{row['Period']},{row['Month']},{row['Dataset']},{row['Accuracy']},{row['F1']},{row['Recall']},{row['AUROC']},{row['AUPRC']}")
            
            # Verify we have all models
            expected_rows = len(models) * 2  # Each model × 2 datasets (Train/Test)
            if len(rows) == expected_rows:
                print(f"\n✅ SUCCESS: Found {len(rows)} rows for {len(models)} models × 2 datasets = {expected_rows} expected rows")
                
                # Verify each model has different metrics (proving they're separate)
                test_rows = [r for r in rows if r['Dataset'] == 'Test']
                accuracies = [float(r['Accuracy']) for r in test_rows]
                if len(set(accuracies)) == len(models):
                    print("✅ Each model has different accuracy values - confirmed separate rows per model")
                else:
                    print("❌ Models have same accuracy - may not be recording separately")
                    
            else:
                print(f"❌ MISMATCH: Expected {expected_rows} rows, got {len(rows)}")
            
            # Verify format
            expected_columns = ['Coin', 'Experiment', 'Window_Size', 'Period', 'Month', 'Dataset', 'Accuracy', 'F1', 'Recall', 'AUROC', 'AUPRC']
            if reader.fieldnames == expected_columns:
                print("✅ CSV format matches specification exactly!")
            else:
                print("❌ CSV format mismatch!")
                print(f"Expected: {expected_columns}")
                print(f"Got: {reader.fieldnames}")
        
        os.unlink(tmp_file.name)

def test_multiple_combinations():
    """Test multiple coin/experiment combinations"""
    print("\n🔄 Testing Multiple Combinations...")
    
    collector = MetricsCollector()
    
    combinations = [
        ("ADAUSDT", "irregular", 5, "14days", "2024-01"),
        ("ADAUSDT", "irregular", 5, "21days", "2024-03"),
        ("BTCUSDT", "regular", 15, "7days", "2024-02"),
    ]
    
    for coin, exp_type, window, period, month in combinations:
        test_metrics = {
            'accuracy': 0.995,
            'f1': 0.994,
            'recall': 0.988,
            'auroc': 0.999,
            'auprc': 0.999
        }
        
        collector.add_result(
            coin=coin, experiment_type=exp_type, window_size=window,
            period=period, month=month, dataset_type="Test",
            metrics=test_metrics, model_name="best_model"
        )
    
    # Export and verify
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as tmp_file:
        collector.export_to_csv(tmp_file.name)
        
        with open(tmp_file.name, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            print(f"✅ Generated {len(rows)} combination records")
            for row in rows:
                print(f"   {row['Coin']},{row['Experiment']},{row['Window_Size']},{row['Period']},{row['Month']},{row['Dataset']}")
        
        os.unlink(tmp_file.name)

if __name__ == "__main__":
    print("🧪 Testing Enhanced CSV Export - Each Model Gets Its Own Row")
    print("=" * 70)
    
    test_each_model_separate_row()
    test_multiple_combinations()
    
    print("\n✨ All tests completed!")
    print("📋 CSV Format: Coin,Experiment,Window_Size,Period,Month,Dataset,Accuracy,F1,Recall,AUROC,AUPRC")
    print("🎯 Each model tested gets its own row for each combination and dataset type")