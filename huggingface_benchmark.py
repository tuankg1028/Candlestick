"""
HuggingFace Model Benchmarking Framework for Candlestick Image Classification
Integrates with existing candlestick analysis pipeline to benchmark 9 different models
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_recall_curve, auc
import csv
from collections import defaultdict

# Import from existing modules
from model_configs import BENCHMARK_MODELS, get_model_config, get_model_set, MODEL_SETS
from benchmark_utils import ModelAdapter, PerformanceMonitor, BenchmarkDataLoader, cleanup_gpu_memory, get_device_info

# Import data loading functions from standalone data loader
try:
    from data_loader import load_candlestick_data, COINS, TIME_LENGTHS, WINDOW_SIZES, list_available_data, find_candlestick_data
    DATA_LOADER_AVAILABLE = True
except ImportError:
    print("Error: Could not import data_loader.py. Please ensure the file exists.")
    COINS = {}
    TIME_LENGTHS = [7, 14, 21, 28]
    WINDOW_SIZES = [5, 15, 30]
    DATA_LOADER_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and manages benchmark metrics for CSV export"""
    
    def __init__(self):
        self.metrics_data = []
        self.experiment_type_mapping = {
            "regular": "R",
            "fullimage": "F", 
            "irregular": "I"
        }
    
    def add_result(self, coin: str, experiment_type: str, window_size: int, period: str,
                  month: str, dataset_type: str, metrics: Dict, model_name: str = None):
        """Add a single result to the metrics collection"""
        
        # Only add Test dataset results (skip Train dataset)
        if dataset_type != "Test":
            return
        
        # Map experiment type to single letter
        experiment_code = self.experiment_type_mapping.get(experiment_type, experiment_type[0].upper())
        
        record = {
            'Coin': coin,
            'Experiment': experiment_code,
            'Window_Size': window_size,
            'Period': period,
            'Month': month,
            'Model': model_name or 'Unknown',
            'Accuracy': round(metrics.get('accuracy', 0), 4),
            'F1': round(metrics.get('f1', 0), 4),
            'Recall': round(metrics.get('recall', 0), 4),
            'AUROC': round(metrics.get('auroc', 0), 4),
            'AUPRC': round(metrics.get('auprc', 0), 4)
        }
            
        self.metrics_data.append(record)
    
    def get_best_model_per_combination(self) -> List[Dict]:
        """Get the best performing model for each combination"""
        
        # Group by combination (coin, experiment, window_size, period, month)
        combinations = defaultdict(list)
        
        for record in self.metrics_data:
            key = (record['Coin'], record['Experiment'], record['Window_Size'], 
                  record['Period'], record['Month'])
            combinations[key].append(record)
        
        # Select best model per combination (highest accuracy)
        best_results = []
        for combo_key, records in combinations.items():
            if records:
                best_record = max(records, key=lambda x: x['Accuracy'])
                # Keep model name in final output
                best_results.append(best_record)
        
        return best_results
    
    def export_to_csv(self, filepath: str, include_all_models: bool = True):
        """Export metrics to CSV file with Model column - each model gets its own row"""
        
        # Always export all models by default (each model = separate row)
        if include_all_models:
            data_to_export = self.metrics_data.copy()
        else:
            data_to_export = self.get_best_model_per_combination()
        
        if not data_to_export:
            logger.warning("No metrics data to export")
            return
        
        # Sort by Coin, Experiment, Window_Size, Period, Month, Model
        sort_key = lambda x: (x['Coin'], x['Experiment'], x['Window_Size'], 
                             x['Period'], x['Month'], x.get('Model', ''))
        data_to_export.sort(key=sort_key)
        
        # New format: Coin,Experiment,Window_Size,Period,Month,Model,Accuracy,F1,Recall,AUROC,AUPRC
        # Dataset column removed, Model column added, Test records only
        columns = ['Coin', 'Experiment', 'Window_Size', 'Period', 'Month', 'Model',
                  'Accuracy', 'F1', 'Recall', 'AUROC', 'AUPRC']
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            
            for record in data_to_export:
                # Write one row per model including Model column
                filtered_record = {k: v for k, v in record.items() if k in columns}
                writer.writerow(filtered_record)
        
        # Log info about what was exported
        models_count = len(set(record.get('Model', 'unknown') for record in data_to_export))
        combinations_count = len(set((record['Coin'], record['Experiment'], record['Window_Size'], 
                                    record['Period'], record['Month']) 
                                   for record in data_to_export))
        
        logger.info(f"Exported {len(data_to_export)} records ({models_count} models × {combinations_count} combinations) to {filepath}")
        logger.info(f"CSV format: Test dataset only with Model column (Dataset column removed)")
    
    def clear(self):
        """Clear all collected metrics"""
        self.metrics_data.clear()

class HuggingFaceBenchmark:
    """Main benchmarking class for HuggingFace models"""
    
    def __init__(self, output_dir: str = "benchmarks"):
        self.output_dir = output_dir
        self.results_dir = os.path.join(output_dir, "results")
        
        # Create directory for CSV output only
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.device_info = get_device_info()
        self.benchmark_results = {}
        self.metrics_collector = MetricsCollector()
        
    def extract_month_from_period(self, period: str, timestamp: str = None) -> str:
        """Extract month information from period or timestamp"""
        # For now, use current timestamp or provided timestamp
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return dt.strftime('%Y-%m')
            except:
                pass
        
        # Fallback to current date
        return datetime.now().strftime('%Y-%m')
        
    def load_candlestick_data(self, coin: str, period: str, window_size: int, 
                            experiment_type: str = "regular") -> Tuple[np.ndarray, np.ndarray]:
        """Load candlestick image data using standalone data loader"""
        
        if not DATA_LOADER_AVAILABLE:
            raise ImportError("Data loader not available. Please ensure data_loader.py exists.")
        
        try:
            X, y = load_candlestick_data(coin, period, window_size, experiment_type)
            logger.info(f"Loaded {len(X)} samples with shape {X[0].shape if len(X) > 0 else 'N/A'}")
            return X, y
        except Exception as e:
            logger.error(f"Error loading candlestick data: {str(e)}")
            raise
    
    def train_model(self, model_adapter: ModelAdapter, train_loader: BenchmarkDataLoader,
                   val_loader: Optional[BenchmarkDataLoader] = None, epochs: int = 10) -> Dict:
        """Train a model and return training metrics"""
        
        model = model_adapter.model
        if model is None:
            raise ValueError("Model not loaded in adapter")
        
        # Set up training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        
        # Training metrics
        training_metrics = {
            "train_losses": [],
            "train_accuracies": [],
            "val_accuracies": [],
            "epoch_times": [],
            "memory_usage": []
        }
        
        model.train()
        performance_monitor = PerformanceMonitor()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            with performance_monitor.monitor():
                for batch_images, batch_labels in train_loader:
                    # Preprocess batch
                    batch_tensor = model_adapter.preprocess_batch(batch_images)
                    batch_labels = batch_labels.to(model_adapter.device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(batch_tensor)
                    
                    # Debug shapes to understand the error
                    if epoch == 0 and total_predictions == 0:  # Only print once
                        logger.info(f"Debug - outputs shape: {outputs.shape}, batch_labels shape: {batch_labels.shape}")
                        logger.info(f"Debug - outputs type: {type(outputs)}, batch_labels type: {type(batch_labels)}")
                    
                    loss = criterion(outputs, batch_labels)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Statistics
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_predictions += batch_labels.size(0)
                    correct_predictions += (predicted == batch_labels).sum().item()
            
            # Calculate metrics
            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = correct_predictions / total_predictions
            epoch_time = time.time() - epoch_start
            
            training_metrics["train_losses"].append(epoch_loss)
            training_metrics["train_accuracies"].append(epoch_accuracy)
            training_metrics["epoch_times"].append(epoch_time)
            training_metrics["memory_usage"].append(performance_monitor.get_metrics().get("peak_memory_gb", 0))
            
            # Validation
            if val_loader is not None:
                val_accuracy = self.evaluate_model(model_adapter, val_loader)["accuracy"]
                training_metrics["val_accuracies"].append(val_accuracy)
                scheduler.step(1 - val_accuracy)  # Step based on validation accuracy
            else:
                scheduler.step(epoch_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, "
                       f"Acc={epoch_accuracy:.4f}, Time={epoch_time:.1f}s")
        
        return training_metrics
    
    def evaluate_model(self, model_adapter: ModelAdapter, test_loader: BenchmarkDataLoader) -> Dict:
        """Evaluate model and return comprehensive metrics"""
        
        model = model_adapter.model
        if model is None:
            raise ValueError("Model not loaded in adapter")
        
        model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        performance_monitor = PerformanceMonitor()
        
        with performance_monitor.monitor():
            with torch.no_grad():
                for batch_images, batch_labels in test_loader:
                    # Preprocess batch
                    batch_tensor = model_adapter.preprocess_batch(batch_images)
                    
                    # Forward pass
                    outputs = model(batch_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    
                    # Store results
                    all_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of class 1
                    all_labels.extend(batch_labels.numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        
        metrics = {
            "accuracy": accuracy_score(all_labels, all_predictions),
            "f1": f1_score(all_labels, all_predictions),
            "recall": recall_score(all_labels, all_predictions),
            "auroc": roc_auc_score(all_labels, all_probabilities),
            "auprc": auc(*precision_recall_curve(all_labels, all_probabilities)[1::-1])
        }
        
        # Add performance metrics
        perf_metrics = performance_monitor.get_metrics()
        metrics.update({
            "inference_time": perf_metrics.get("execution_time", 0),
            "inference_memory_gb": perf_metrics.get("peak_memory_gb", 0)
        })
        
        return metrics
    
    def benchmark_single_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray, 
                             train_epochs: int = 10, 
                             coin: str = None, period: str = None, window_size: int = None, 
                             experiment_type: str = None) -> Dict:
        """Benchmark a single model"""
        
        logger.info(f"Benchmarking model: {model_name}")
        
        try:
            # Initialize model adapter
            model_adapter = ModelAdapter(model_name, num_classes=2)
            model_config = get_model_config(model_name)
            
            # Load model
            model_loading_start = time.time()
            model = model_adapter.load_model()
            model_loading_time = time.time() - model_loading_start
            
            # Create data loaders
            batch_size = model_config.batch_size_recommendation
            train_loader = BenchmarkDataLoader(X_train, y_train, batch_size)
            test_loader = BenchmarkDataLoader(X_test, y_test, batch_size)
            
            # Train model
            logger.info(f"Training {model_name} for {train_epochs} epochs...")
            training_metrics = self.train_model(model_adapter, train_loader, epochs=train_epochs)
            
            # Evaluate model
            logger.info(f"Evaluating {model_name}...")
            evaluation_metrics = self.evaluate_model(model_adapter, test_loader)
            
            # Compile results
            model_info = model_adapter.get_model_info()
            timestamp = datetime.now().isoformat()
            
            results = {
                "model_name": model_name,
                "model_config": model_config.__dict__,
                "model_info": model_info,
                "training_metrics": training_metrics,
                "evaluation_metrics": evaluation_metrics,
                "model_loading_time": model_loading_time,
                "timestamp": timestamp
            }
            
            # Add to metrics collector if combination parameters are provided
            if all(param is not None for param in [coin, period, window_size, experiment_type]):
                month = self.extract_month_from_period(period, timestamp)
                
                # Add training metrics - we should evaluate on training set properly
                # For now, using final epoch accuracy as approximation for all metrics
                if training_metrics.get("train_accuracies"):
                    final_train_accuracy = training_metrics["train_accuracies"][-1]
                    # For training set, we often see near-perfect performance
                    train_metrics = {
                        "accuracy": final_train_accuracy,
                        "f1": final_train_accuracy,
                        "recall": final_train_accuracy,  
                        "auroc": final_train_accuracy,
                        "auprc": final_train_accuracy
                    }
                    self.metrics_collector.add_result(
                        coin=coin, experiment_type=experiment_type, 
                        window_size=window_size, period=period, month=month,
                        dataset_type="Train", metrics=train_metrics, model_name=model_name
                    )
                
                # Add test metrics
                self.metrics_collector.add_result(
                    coin=coin, experiment_type=experiment_type,
                    window_size=window_size, period=period, month=month,
                    dataset_type="Test", metrics=evaluation_metrics, model_name=model_name
                )
            
            # Skip individual model results JSON - only keep metrics collection
            
            logger.info(f"✓ Completed benchmarking {model_name}")
            logger.info(f"  Accuracy: {evaluation_metrics['accuracy']:.4f}")
            logger.info(f"  F1 Score: {evaluation_metrics['f1']:.4f}")
            logger.info(f"  Training time: {sum(training_metrics['epoch_times']):.1f}s")
            
            # Cleanup
            del model_adapter, model
            cleanup_gpu_memory()
            
            return results
            
        except Exception as e:
            logger.error(f"✗ Error benchmarking {model_name}: {str(e)}")
            cleanup_gpu_memory()
            return {
                "model_name": model_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_benchmark_suite(self, model_set: str = "quick_test", 
                          coin: str = "BTCUSDT", period: str = "7days", 
                          window_size: int = 5, experiment_type: str = "regular",
                          train_epochs: int = 5, max_samples: int = 1000) -> Dict:
        """Run complete benchmark suite"""
        
        logger.info("=" * 80)
        logger.info("Starting HuggingFace Model Benchmark Suite")
        logger.info("=" * 80)
        
        # Get model list
        models_to_test = get_model_set(model_set)
        logger.info(f"Testing {len(models_to_test)} models: {models_to_test}")
        
        # Load data
        logger.info(f"Loading candlestick data: {coin}, {period}, window_size={window_size}")
        try:
            X, y = self.load_candlestick_data(coin, period, window_size, experiment_type)
            
            # Limit samples if specified
            if max_samples and len(X) > max_samples:
                indices = np.random.choice(len(X), max_samples, replace=False)
                X, y = X[indices], y[indices]
                logger.info(f"Limited to {max_samples} samples")
            
            # Train/test split
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            return {"error": f"Data loading failed: {str(e)}"}
        
        # Run benchmarks
        benchmark_results = {}
        total_start_time = time.time()
        
        for i, model_name in enumerate(models_to_test, 1):
            logger.info(f"\n[{i}/{len(models_to_test)}] Benchmarking {model_name}...")
            
            result = self.benchmark_single_model(
                model_name, X_train, y_train, X_test, y_test, train_epochs,
                coin=coin, period=period, window_size=window_size, experiment_type=experiment_type
            )
            benchmark_results[model_name] = result
        
        total_time = time.time() - total_start_time
        
        # Compile final results
        final_results = {
            "benchmark_config": {
                "model_set": model_set,
                "models_tested": models_to_test,
                "coin": coin,
                "period": period,
                "window_size": window_size,
                "experiment_type": experiment_type,
                "train_epochs": train_epochs,
                "max_samples": max_samples,
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            },
            "device_info": self.device_info,
            "results": benchmark_results,
            "total_benchmark_time": total_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # Skip JSON results file - only keep metrics collection
        logger.info("=" * 80)
        logger.info("Benchmark Suite Completed!")
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info("=" * 80)
        
        # Print summary
        self.print_benchmark_summary(final_results)
        
        return final_results
    
    def export_metrics_to_csv(self, filename: str = None, include_all_models: bool = False):
        """Export collected metrics to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"comprehensive_metrics_full_benchmark_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        self.metrics_collector.export_to_csv(filepath, include_all_models)
        return filepath
    
    def run_comprehensive_benchmark(self, model_set: str = "quick_test",
                                   coins: List[str] = None, periods: List[str] = None,
                                   window_sizes: List[int] = None, experiment_types: List[str] = None,
                                   train_epochs: int = 5, max_samples: int = 1000) -> Dict:
        """Run comprehensive benchmark across all specified combinations"""
        
        coins = list(COINS.keys())
        periods = [f"{days}days" for days in TIME_LENGTHS]
        window_sizes = WINDOW_SIZES
        experiment_types = ["regular", "fullimage", "irregular"]
        
        logger.info("=" * 100)
        logger.info("STARTING COMPREHENSIVE HUGGINGFACE BENCHMARK SUITE")
        logger.info("=" * 100)
        logger.info(f"Coins: {coins}")
        logger.info(f"Periods: {periods}")
        logger.info(f"Window sizes: {window_sizes}")
        logger.info(f"Experiment types: {experiment_types}")
        logger.info(f"Model set: {model_set}")
        
        # Calculate total combinations
        total_combinations = len(coins) * len(periods) * len(window_sizes) * len(experiment_types)
        logger.info(f"Total combinations to test: {total_combinations}")
        
        comprehensive_results = {}
        successful_runs = 0
        failed_runs = 0
        start_time = time.time()
        
        combination_count = 0
        
        for coin in coins:
            for period in periods:
                for window_size in window_sizes:
                    for experiment_type in experiment_types:
                        combination_count += 1
                        combination_key = f"{coin}_{period}_w{window_size}_{experiment_type}"
                        
                        logger.info(f"\n[{combination_count}/{total_combinations}] Testing combination: {combination_key}")
                        
                        try:
                            # Check if data exists before running benchmark
                            data_files = find_candlestick_data(coin, period, window_size, experiment_type)
                            if not data_files:
                                logger.warning(f"No data found for {combination_key}, skipping...")
                                comprehensive_results[combination_key] = {
                                    "status": "skipped",
                                    "reason": "no_data_available",
                                    "timestamp": datetime.now().isoformat()
                                }
                                continue
                            
                            # Run benchmark for this combination
                            result = self.run_benchmark_suite(
                                model_set=model_set,
                                coin=coin,
                                period=period,
                                window_size=window_size,
                                experiment_type=experiment_type,
                                train_epochs=train_epochs,
                                max_samples=max_samples
                            )
                            
                            if "error" not in result:
                                comprehensive_results[combination_key] = result
                                successful_runs += 1
                                logger.info(f"✓ Successfully completed {combination_key}")
                            else:
                                comprehensive_results[combination_key] = result
                                failed_runs += 1
                                logger.error(f"✗ Failed {combination_key}: {result.get('error', 'Unknown error')}")
                                
                        except Exception as e:
                            logger.error(f"✗ Exception in {combination_key}: {str(e)}")
                            comprehensive_results[combination_key] = {
                                "status": "error",
                                "error": str(e),
                                "timestamp": datetime.now().isoformat()
                            }
                            failed_runs += 1
        
        total_time = time.time() - start_time
        
        # Create comprehensive summary
        final_results = {
            "comprehensive_benchmark_config": {
                "model_set": model_set,
                "coins": coins,
                "periods": periods,
                "window_sizes": window_sizes,
                "experiment_types": experiment_types,
                "train_epochs": train_epochs,
                "max_samples": max_samples,
                "total_combinations": total_combinations,
                "successful_runs": successful_runs,
                "failed_runs": failed_runs
            },
            "device_info": self.device_info,
            "combination_results": comprehensive_results,
            "total_benchmark_time": total_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # Skip JSON results file - only export metrics to CSV
        logger.info("=" * 100)
        logger.info("COMPREHENSIVE BENCHMARK COMPLETED!")
        logger.info(f"Total time: {total_time/3600:.1f} hours ({total_time:.1f}s)")
        logger.info(f"Successful runs: {successful_runs}/{total_combinations}")
        logger.info(f"Failed runs: {failed_runs}/{total_combinations}")
        logger.info("=" * 100)
        
        # Print comprehensive summary
        self.print_comprehensive_summary(final_results)
        
        # Export only metrics to CSV (each model gets its own row for each combination)
        csv_file = self.export_metrics_to_csv(
            filename=f"comprehensive_metrics_full_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            include_all_models=True  # Always export all models - each model = separate row
        )
        logger.info(f"📊 SINGLE COMPREHENSIVE REPORT: {csv_file}")
        logger.info("📋 Format: Coin,Experiment,Window_Size,Period,Month,Model,Accuracy,F1,Recall,AUROC,AUPRC")
        logger.info("✨ Test dataset only, Model column included, Dataset column removed")
        
        return final_results

    def print_comprehensive_summary(self, results: Dict):
        """Print a comprehensive summary across all combinations"""
        print("\n" + "=" * 100)
        print("COMPREHENSIVE BENCHMARK SUMMARY")
        print("=" * 100)
        
        combination_results = results["combination_results"]
        successful_combinations = {k: v for k, v in combination_results.items() 
                                 if v.get("status") != "error" and v.get("status") != "skipped" and "error" not in v}
        
        if not successful_combinations:
            print("No successful benchmark results to display.")
            return
        
        # Aggregate results by different dimensions
        print(f"\nSUCCESSFUL COMBINATIONS: {len(successful_combinations)}")
        print("-" * 50)
        
        # Best performing models across all combinations
        all_model_results = []
        for combo_key, combo_result in successful_combinations.items():
            if "results" in combo_result:
                coin, period, window_spec, experiment = combo_key.split("_", 3)
                window_size = window_spec.replace("w", "")
                
                for model_name, model_result in combo_result["results"].items():
                    if "evaluation_metrics" in model_result:
                        metrics = model_result["evaluation_metrics"]
                        all_model_results.append({
                            "Combination": f"{coin}_{period}_w{window_size}_{experiment}",
                            "Model": model_result["model_config"]["name"],
                            "Accuracy": metrics["accuracy"],
                            "F1": metrics["f1"],
                            "AUROC": metrics["auroc"]
                        })
        
        if all_model_results:
            # Top 10 results by accuracy
            all_model_results.sort(key=lambda x: x["Accuracy"], reverse=True)
            print("\nTOP 10 RESULTS BY ACCURACY:")
            top_results_df = pd.DataFrame(all_model_results[:10])
            print(top_results_df.to_string(index=False))
            
            # Best model per combination
            combo_best = {}
            for result in all_model_results:
                combo = result["Combination"]
                if combo not in combo_best or result["Accuracy"] > combo_best[combo]["Accuracy"]:
                    combo_best[combo] = result
            
            print(f"\nBEST MODEL PER COMBINATION ({len(combo_best)} combinations):")
            combo_df = pd.DataFrame(list(combo_best.values()))
            combo_df = combo_df.sort_values("Accuracy", ascending=False)
            print(combo_df.to_string(index=False))
        
        print("=" * 100)

    def print_benchmark_summary(self, results: Dict):
        """Print a summary of benchmark results"""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        successful_results = {k: v for k, v in results["results"].items() if "error" not in v}
        
        if not successful_results:
            print("No successful benchmark results to display.")
            return
        
        # Create summary table
        summary_data = []
        for model_name, result in successful_results.items():
            if "evaluation_metrics" in result:
                metrics = result["evaluation_metrics"]
                training_time = sum(result["training_metrics"]["epoch_times"])
                params = result["model_info"]["total_parameters"]
                
                summary_data.append({
                    "Model": result["model_config"]["name"],
                    "Parameters": f"{params/1_000_000:.1f}M",
                    "Accuracy": f"{metrics['accuracy']:.4f}",
                    "F1 Score": f"{metrics['f1']:.4f}",
                    "AUROC": f"{metrics['auroc']:.4f}",
                    "Train Time (s)": f"{training_time:.1f}",
                    "Inference Time (s)": f"{metrics['inference_time']:.3f}"
                })
        
        # Sort by accuracy (descending)
        summary_data.sort(key=lambda x: float(x["Accuracy"]), reverse=True)
        
        # Print table
        if summary_data:
            df = pd.DataFrame(summary_data)
            print(df.to_string(index=False))
        
        print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="HuggingFace Model Benchmarking for Candlestick Classification")
    parser.add_argument("--model-set", choices=list(MODEL_SETS.keys()), default="full_benchmark",
                       help="Set of models to benchmark")
    
    # Single benchmark options
    parser.add_argument("--coin", default="BTCUSDT", help="Cryptocurrency to analyze")
    parser.add_argument("--period", default="7days", help="Time period for analysis")
    parser.add_argument("--window-size", type=int, default=5, help="Window size for candlestick images")
    parser.add_argument("--experiment-type", choices=["regular", "fullimage", "irregular"], 
                       default="regular", help="Experiment type")
    
    # Comprehensive benchmark options
    parser.add_argument("--comprehensive", action="store_true", 
                       help="Run comprehensive benchmark across all combinations")
    parser.add_argument("--coins", nargs="+", choices=list(COINS.keys()) if DATA_LOADER_AVAILABLE else [],
                       help="Coins to test (for comprehensive benchmark)")
    parser.add_argument("--periods", nargs="+", 
                       choices=[f"{days}days" for days in TIME_LENGTHS] if DATA_LOADER_AVAILABLE else [],
                       help="Periods to test (for comprehensive benchmark)")
    parser.add_argument("--window-sizes", nargs="+", type=int, choices=WINDOW_SIZES if DATA_LOADER_AVAILABLE else [],
                       help="Window sizes to test (for comprehensive benchmark)")
    parser.add_argument("--experiment-types", nargs="+", choices=["regular", "fullimage", "irregular"],
                       default=["regular", "fullimage", "irregular"],
                       help="Experiment types to test (for comprehensive benchmark)")
    
    # Training options
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max-samples", type=int, default=0, help="Maximum samples to use (0 for all)")
    parser.add_argument("--output-dir", default="benchmarks", help="Output directory")
    parser.add_argument("--list-data", action="store_true", help="List available data and exit")
    
    args = parser.parse_args()
    
    # Check data loader availability
    if not DATA_LOADER_AVAILABLE:
        print("❌ Error: Data loader not available.")
        print("Please ensure data_loader.py exists in the current directory.")
        return
    
    # List available data if requested
    if args.list_data:
        print("Listing available candlestick data...")
        list_available_data()
        return
    
    # Initialize benchmark
    benchmark = HuggingFaceBenchmark(args.output_dir)
    
    if args.comprehensive:
        # Run comprehensive benchmark
        print("🚀 Starting comprehensive benchmark...")
        results = benchmark.run_comprehensive_benchmark(
            model_set=args.model_set,
            coins=args.coins,
            periods=args.periods,
            window_sizes=args.window_sizes,
            experiment_types=args.experiment_types,
            train_epochs=args.epochs,
            max_samples=args.max_samples if args.max_samples > 0 else None
        )
        
        # Note: Comprehensive benchmark automatically generates the comprehensive_metrics CSV
    else:
        # Run single benchmark suite
        print("🚀 Starting single benchmark...")
        results = benchmark.run_benchmark_suite(
            model_set=args.model_set,
            coin=args.coin,
            period=args.period,
            window_size=args.window_size,
            experiment_type=args.experiment_type,
            train_epochs=args.epochs,
            max_samples=args.max_samples if args.max_samples > 0 else None
        )
        
        # Note: Single benchmarks don't generate CSV files
        # Use --comprehensive flag to generate comprehensive_metrics CSV

if __name__ == "__main__":
    main()