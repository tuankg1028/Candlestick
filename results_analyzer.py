"""
Results analyzer and visualization for HuggingFace model benchmarks
Creates comprehensive reports and visualizations of benchmark results
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import glob
from pathlib import Path
import torch
from packaging import version

# Set matplotlib style
plt.style.use('default')
sns.set_palette("husl")

def check_pytorch_version() -> Dict[str, str]:
    """Check PyTorch version and provide upgrade recommendations"""
    current_version = torch.__version__
    
    # Check if version is compatible (>= 2.6 to avoid CVE-2025-32434)
    is_compatible = version.parse(current_version) >= version.parse("2.6.0")
    
    result = {
        "current_version": current_version,
        "is_compatible": is_compatible,
        "recommendation": ""
    }
    
    if not is_compatible:
        result["recommendation"] = (
            f"⚠️  PyTorch {current_version} has security vulnerabilities (CVE-2025-32434). "
            f"Upgrade to PyTorch >= 2.6.0 with: pip install torch>=2.6.0"
        )
    else:
        result["recommendation"] = f"✓ PyTorch {current_version} is compatible and secure."
    
    return result

class BenchmarkAnalyzer:
    """Analyze and visualize benchmark results"""
    
    def __init__(self, results_dir: str = "benchmarks/results", reports_dir: str = "benchmarks/reports"):
        self.results_dir = results_dir
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)
        
        self.benchmark_data = {}
        self.comparison_df = None
        self.failed_models = {}  # Store failed model information
        self.is_comprehensive = True  # Track if this is comprehensive benchmark data
    
    def load_results(self, pattern: str = "benchmark_suite_*.json") -> Dict:
        """Load benchmark results from JSON files"""
        result_files = glob.glob(os.path.join(self.results_dir, pattern))
        
        if not result_files:
            raise FileNotFoundError(f"No benchmark results found matching pattern: {pattern}")
        
        # Load the most recent benchmark suite result
        latest_file = max(result_files, key=os.path.getctime)
        
        with open(latest_file, 'r') as f:
            self.benchmark_data = json.load(f)
        
        # Check if this is comprehensive benchmark data
        self.is_comprehensive = "combination_results" in self.benchmark_data
        
        print(f"Loaded benchmark results from: {latest_file}")
        if self.is_comprehensive:
            print("Detected comprehensive benchmark data with multiple experiment types")
        
        return self.benchmark_data
    
    def create_comparison_dataframe(self, experiment_type: str = None) -> pd.DataFrame:
        """Create a comparison DataFrame from benchmark results
        
        Args:
            experiment_type: Filter results by experiment type (for comprehensive benchmarks)
        """
        if not self.benchmark_data:
            raise ValueError("No benchmark data loaded. Call load_results() first.")
        
        # Handle comprehensive vs single benchmark data
        if self.is_comprehensive:
            results = self._extract_results_by_experiment_type(experiment_type)
        else:
            results = self.benchmark_data.get("results", {})
        
        comparison_data = []
        failed_models = {}
        
        for model_name, result in results.items():
            if "error" in result:
                # Store failed models for reporting
                failed_models[model_name] = result
                continue  # Skip failed models for comparison DataFrame
            
            # Extract key metrics
            eval_metrics = result.get("evaluation_metrics", {})
            train_metrics = result.get("training_metrics", {})
            model_info = result.get("model_info", {})
            model_config = result.get("model_config", {})
            
            # Calculate training time
            epoch_times = train_metrics.get("epoch_times", [])
            total_train_time = sum(epoch_times) if epoch_times else 0
            avg_epoch_time = np.mean(epoch_times) if epoch_times else 0
            
            # Calculate memory usage
            memory_usage = train_metrics.get("memory_usage", [])
            peak_memory = max(memory_usage) if memory_usage else 0
            
            # Get final training metrics
            train_losses = train_metrics.get("train_losses", [])
            train_accuracies = train_metrics.get("train_accuracies", [])
            final_train_loss = train_losses[-1] if train_losses else 0
            final_train_acc = train_accuracies[-1] if train_accuracies else 0
            
            row = {
                # Model Information
                "Model": model_config.get("name", model_name),
                "Model_ID": model_name,
                "Architecture": model_config.get("architecture_type", "unknown"),
                "Parameters": model_info.get("total_parameters", 0),
                "Parameters_M": model_info.get("total_parameters", 0) / 1_000_000,
                
                # Performance Metrics
                "Accuracy": eval_metrics.get("accuracy", 0),
                "F1_Score": eval_metrics.get("f1", 0),
                "Recall": eval_metrics.get("recall", 0),
                "AUROC": eval_metrics.get("auroc", 0),
                "AUPRC": eval_metrics.get("auprc", 0),
                
                # Training Metrics
                "Final_Train_Loss": final_train_loss,
                "Final_Train_Accuracy": final_train_acc,
                "Total_Train_Time": total_train_time,
                "Avg_Epoch_Time": avg_epoch_time,
                "Peak_Memory_GB": peak_memory,
                
                # Efficiency Metrics
                "Inference_Time": eval_metrics.get("inference_time", 0),
                "Inference_Memory_GB": eval_metrics.get("inference_memory_gb", 0),
                "Params_per_Accuracy": model_info.get("total_parameters", 0) / max(eval_metrics.get("accuracy", 0.001), 0.001),
                "Time_per_Accuracy": total_train_time / max(eval_metrics.get("accuracy", 0.001), 0.001),
                
                # Model Loading
                "Model_Loading_Time": result.get("model_loading_time", 0)
            }
            
            comparison_data.append(row)
        
        self.comparison_df = pd.DataFrame(comparison_data)
        self.failed_models = failed_models  # Store for report generation
        
        # Sort by accuracy (descending) if we have data
        if not self.comparison_df.empty:
            self.comparison_df = self.comparison_df.sort_values("Accuracy", ascending=False)
        
        return self.comparison_df
    
    def _extract_results_by_experiment_type(self, experiment_type: str = None) -> Dict:
        """Extract results from comprehensive benchmark data by experiment type"""
        if not self.is_comprehensive:
            return self.benchmark_data.get("results", {})
        
        combination_results = self.benchmark_data.get("combination_results", {})
        
        if experiment_type is None:
            # Return all results combined
            all_results = {}
            for combo_key, combo_data in combination_results.items():
                if "results" in combo_data:
                    # Add combination info to each model result
                    for model_name, model_result in combo_data["results"].items():
                        combo_model_key = f"{model_name}_{combo_key}"
                        all_results[combo_model_key] = model_result.copy()
                        all_results[combo_model_key]["combination_key"] = combo_key
            return all_results
        
        # Filter by specific experiment type
        filtered_results = {}
        for combo_key, combo_data in combination_results.items():
            if combo_key.endswith(f"_{experiment_type}") and "results" in combo_data:
                # Add experiment type results
                for model_name, model_result in combo_data["results"].items():
                    combo_model_key = f"{model_name}_{combo_key}"
                    filtered_results[combo_model_key] = model_result.copy()
                    filtered_results[combo_model_key]["combination_key"] = combo_key
        
        return filtered_results
    
    def get_available_experiment_types(self) -> List[str]:
        """Get list of available experiment types from comprehensive benchmark data"""
        if not self.is_comprehensive:
            # Check single benchmark config for experiment type
            config = self.benchmark_data.get("benchmark_config", {})
            exp_type = config.get("experiment_type", "regular")
            return [exp_type]
        
        experiment_types = set()
        combination_results = self.benchmark_data.get("combination_results", {})
        
        for combo_key in combination_results.keys():
            # Extract experiment type from combination key (format: coin_period_wX_experiment_type)
            parts = combo_key.split("_")
            if len(parts) >= 4:
                experiment_types.add(parts[-1])
        
        return sorted(list(experiment_types))
    
    def create_performance_comparison_plot(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive performance comparison plots"""
        if self.comparison_df is None:
            self.create_comparison_dataframe()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("HuggingFace Model Performance Comparison", fontsize=16, fontweight='bold')
        
        # 1. Accuracy vs Parameters
        ax1 = axes[0, 0]
        scatter = ax1.scatter(self.comparison_df["Parameters_M"], self.comparison_df["Accuracy"],
                            c=self.comparison_df["Total_Train_Time"], cmap="viridis",
                            s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel("Parameters (Millions)")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Accuracy vs Model Size")
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar for training time
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label("Training Time (s)")
        
        # Add model labels
        for idx, row in self.comparison_df.iterrows():
            ax1.annotate(row["Model_ID"], (row["Parameters_M"], row["Accuracy"]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        # 2. F1 Score comparison
        ax2 = axes[0, 1]
        bars = ax2.bar(range(len(self.comparison_df)), self.comparison_df["F1_Score"],
                      color=plt.cm.Set3(range(len(self.comparison_df))))
        ax2.set_xlabel("Models")
        ax2.set_ylabel("F1 Score")
        ax2.set_title("F1 Score by Model")
        ax2.set_xticks(range(len(self.comparison_df)))
        ax2.set_xticklabels(self.comparison_df["Model_ID"], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, self.comparison_df["F1_Score"]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Training Time vs Accuracy
        ax3 = axes[0, 2]
        ax3.scatter(self.comparison_df["Total_Train_Time"], self.comparison_df["Accuracy"],
                   s=self.comparison_df["Parameters_M"]*10, alpha=0.6,
                   c=self.comparison_df["Peak_Memory_GB"], cmap="plasma")
        ax3.set_xlabel("Training Time (seconds)")
        ax3.set_ylabel("Accuracy")
        ax3.set_title("Training Efficiency")
        ax3.grid(True, alpha=0.3)
        
        # Add model labels
        for idx, row in self.comparison_df.iterrows():
            ax3.annotate(row["Model_ID"], (row["Total_Train_Time"], row["Accuracy"]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        # 4. Memory Usage
        ax4 = axes[1, 0]
        ax4.bar(range(len(self.comparison_df)), self.comparison_df["Peak_Memory_GB"],
               color=plt.cm.viridis(np.linspace(0, 1, len(self.comparison_df))))
        ax4.set_xlabel("Models")
        ax4.set_ylabel("Peak Memory (GB)")
        ax4.set_title("Memory Usage During Training")
        ax4.set_xticks(range(len(self.comparison_df)))
        ax4.set_xticklabels(self.comparison_df["Model_ID"], rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Comprehensive Metrics Heatmap
        ax5 = axes[1, 1]
        metrics_cols = ["Accuracy", "F1_Score", "AUROC", "AUPRC"]
        heatmap_data = self.comparison_df[metrics_cols].T
        im = ax5.imshow(heatmap_data, cmap="RdYlGn", aspect="auto")
        ax5.set_xticks(range(len(self.comparison_df)))
        ax5.set_xticklabels(self.comparison_df["Model_ID"], rotation=45, ha='right')
        ax5.set_yticks(range(len(metrics_cols)))
        ax5.set_yticklabels(metrics_cols)
        ax5.set_title("Performance Metrics Heatmap")
        
        # Add text annotations
        for i in range(len(metrics_cols)):
            for j in range(len(self.comparison_df)):
                text = ax5.text(j, i, f'{heatmap_data.iloc[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        # Add colorbar
        cbar5 = plt.colorbar(im, ax=ax5)
        cbar5.set_label("Score")
        
        # 6. Efficiency Analysis
        ax6 = axes[1, 2]
        # Create efficiency score: Accuracy / (Parameters + Training_Time)
        efficiency_score = self.comparison_df["Accuracy"] / (
            self.comparison_df["Parameters_M"] + self.comparison_df["Total_Train_Time"] + 1
        )
        
        bars = ax6.bar(range(len(self.comparison_df)), efficiency_score,
                      color=plt.cm.plasma(np.linspace(0, 1, len(self.comparison_df))))
        ax6.set_xlabel("Models")
        ax6.set_ylabel("Efficiency Score")
        ax6.set_title("Overall Efficiency\n(Accuracy / (Params + Time))")
        ax6.set_xticks(range(len(self.comparison_df)))
        ax6.set_xticklabels(self.comparison_df["Model_ID"], rotation=45, ha='right')
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance comparison plot saved to: {save_path}")
        
        return fig
    
    def create_training_curves_plot(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create training curves visualization"""
        if not self.benchmark_data:
            raise ValueError("No benchmark data loaded")
        
        results = self.benchmark_data.get("results", {})
        successful_results = {k: v for k, v in results.items() if "error" not in v}
        
        if not successful_results:
            print("No successful results to plot training curves")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Training Progress Comparison", fontsize=16, fontweight='bold')
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(successful_results)))
        
        # 1. Training Loss
        ax1 = axes[0, 0]
        for (model_name, result), color in zip(successful_results.items(), colors):
            train_metrics = result.get("training_metrics", {})
            losses = train_metrics.get("train_losses", [])
            if losses:
                epochs = range(1, len(losses) + 1)
                ax1.plot(epochs, losses, label=model_name, color=color, marker='o', markersize=4)
        
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training Loss")
        ax1.set_title("Training Loss Curves")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Training Accuracy
        ax2 = axes[0, 1]
        for (model_name, result), color in zip(successful_results.items(), colors):
            train_metrics = result.get("training_metrics", {})
            accuracies = train_metrics.get("train_accuracies", [])
            if accuracies:
                epochs = range(1, len(accuracies) + 1)
                ax2.plot(epochs, accuracies, label=model_name, color=color, marker='s', markersize=4)
        
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Training Accuracy")
        ax2.set_title("Training Accuracy Curves")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Memory Usage Over Time
        ax3 = axes[1, 0]
        for (model_name, result), color in zip(successful_results.items(), colors):
            train_metrics = result.get("training_metrics", {})
            memory_usage = train_metrics.get("memory_usage", [])
            if memory_usage:
                epochs = range(1, len(memory_usage) + 1)
                ax3.plot(epochs, memory_usage, label=model_name, color=color, marker='^', markersize=4)
        
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Memory Usage (GB)")
        ax3.set_title("Memory Usage During Training")
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Epoch Time
        ax4 = axes[1, 1]
        for (model_name, result), color in zip(successful_results.items(), colors):
            train_metrics = result.get("training_metrics", {})
            epoch_times = train_metrics.get("epoch_times", [])
            if epoch_times:
                epochs = range(1, len(epoch_times) + 1)
                ax4.plot(epochs, epoch_times, label=model_name, color=color, marker='d', markersize=4)
        
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Time per Epoch (seconds)")
        ax4.set_title("Training Speed")
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves plot saved to: {save_path}")
        
        return fig
    
    def generate_detailed_report(self, save_path: Optional[str] = None, experiment_type: str = None) -> str:
        """Generate a detailed text report"""
        if self.comparison_df is None:
            self.create_comparison_dataframe(experiment_type)
        
        report_lines = []
        report_lines.append("=" * 100)
        if experiment_type:
            report_lines.append(f"HUGGINGFACE MODEL BENCHMARK DETAILED REPORT - {experiment_type.upper()}")
        else:
            report_lines.append("HUGGINGFACE MODEL BENCHMARK DETAILED REPORT")
        report_lines.append("=" * 100)
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if experiment_type:
            report_lines.append(f"Experiment Type: {experiment_type}")
        report_lines.append("")
        
        # Benchmark Configuration
        if self.benchmark_data:
            config = self.benchmark_data.get("benchmark_config", {})
            device_info = self.benchmark_data.get("device_info", {})
            
            report_lines.append("BENCHMARK CONFIGURATION:")
            report_lines.append("-" * 50)
            for key, value in config.items():
                report_lines.append(f"  {key}: {value}")
            report_lines.append("")
            
            report_lines.append("DEVICE INFORMATION:")
            report_lines.append("-" * 50)
            for key, value in device_info.items():
                report_lines.append(f"  {key}: {value}")
            report_lines.append("")
            
            # Add PyTorch version check
            pytorch_info = check_pytorch_version()
            report_lines.append("PYTORCH VERSION CHECK:")
            report_lines.append("-" * 50)
            report_lines.append(f"  Current Version: {pytorch_info['current_version']}")
            report_lines.append(f"  Security Status: {'✓ Compatible' if pytorch_info['is_compatible'] else '⚠️  Vulnerable (CVE-2025-32434)'}")
            if not pytorch_info["is_compatible"]:
                report_lines.append("  Recommendation: Upgrade with 'pip install torch>=2.6.0'")
            report_lines.append("")
        
        # Overall Statistics
        report_lines.append("OVERALL STATISTICS:")
        report_lines.append("-" * 50)
        successful_models = len(self.comparison_df)
        failed_models = len(self.failed_models)
        total_attempted = successful_models + failed_models
        report_lines.append(f"  Number of models attempted: {total_attempted}")
        report_lines.append(f"  Number of models successful: {successful_models}")
        if failed_models > 0:
            report_lines.append(f"  Number of models failed: {failed_models}")
        
        # Only show performance statistics if we have successful models
        if successful_models > 0:
            report_lines.append(f"  Best accuracy: {self.comparison_df['Accuracy'].max():.4f} ({self.comparison_df.loc[self.comparison_df['Accuracy'].idxmax(), 'Model']})")
            report_lines.append(f"  Worst accuracy: {self.comparison_df['Accuracy'].min():.4f} ({self.comparison_df.loc[self.comparison_df['Accuracy'].idxmin(), 'Model']})")
            report_lines.append(f"  Mean accuracy: {self.comparison_df['Accuracy'].mean():.4f}")
            report_lines.append(f"  Fastest training: {self.comparison_df['Total_Train_Time'].min():.1f}s ({self.comparison_df.loc[self.comparison_df['Total_Train_Time'].idxmin(), 'Model']})")
            report_lines.append(f"  Slowest training: {self.comparison_df['Total_Train_Time'].max():.1f}s ({self.comparison_df.loc[self.comparison_df['Total_Train_Time'].idxmax(), 'Model']})")
            report_lines.append(f"  Lowest memory usage: {self.comparison_df['Peak_Memory_GB'].min():.2f}GB ({self.comparison_df.loc[self.comparison_df['Peak_Memory_GB'].idxmin(), 'Model']})")
            report_lines.append(f"  Highest memory usage: {self.comparison_df['Peak_Memory_GB'].max():.2f}GB ({self.comparison_df.loc[self.comparison_df['Peak_Memory_GB'].idxmax(), 'Model']})")
        else:
            report_lines.append("  No successful models to analyze.")
        report_lines.append("")
        
        # Detailed Model Results
        report_lines.append("DETAILED MODEL RESULTS:")
        report_lines.append("-" * 50)
        
        for idx, row in self.comparison_df.iterrows():
            report_lines.append(f"\n{idx + 1}. {row['Model']} ({row['Model_ID']})")
            report_lines.append("   " + "-" * 60)
            report_lines.append(f"   Architecture: {row['Architecture']}")
            report_lines.append(f"   Parameters: {row['Parameters']:,} ({row['Parameters_M']:.1f}M)")
            report_lines.append("")
            report_lines.append("   Performance Metrics:")
            report_lines.append(f"     • Accuracy: {row['Accuracy']:.4f}")
            report_lines.append(f"     • F1 Score: {row['F1_Score']:.4f}")
            report_lines.append(f"     • Recall: {row['Recall']:.4f}")
            report_lines.append(f"     • AUROC: {row['AUROC']:.4f}")
            report_lines.append(f"     • AUPRC: {row['AUPRC']:.4f}")
            report_lines.append("")
            report_lines.append("   Training Metrics:")
            report_lines.append(f"     • Total Training Time: {row['Total_Train_Time']:.1f}s")
            report_lines.append(f"     • Average Epoch Time: {row['Avg_Epoch_Time']:.1f}s")
            report_lines.append(f"     • Final Training Loss: {row['Final_Train_Loss']:.4f}")
            report_lines.append(f"     • Final Training Accuracy: {row['Final_Train_Accuracy']:.4f}")
            report_lines.append(f"     • Peak Memory Usage: {row['Peak_Memory_GB']:.2f}GB")
            report_lines.append("")
            report_lines.append("   Efficiency Metrics:")
            report_lines.append(f"     • Inference Time: {row['Inference_Time']:.3f}s")
            report_lines.append(f"     • Model Loading Time: {row['Model_Loading_Time']:.2f}s")
            report_lines.append(f"     • Parameters per Accuracy: {row['Params_per_Accuracy']:,.0f}")
            report_lines.append(f"     • Time per Accuracy: {row['Time_per_Accuracy']:.1f}s")
        
        # Failed Models Section
        if self.failed_models:
            report_lines.append("\nFAILED MODELS:")
            report_lines.append("-" * 50)
            
            for model_name, error_info in self.failed_models.items():
                report_lines.append(f"\n❌ {model_name}")
                report_lines.append("   " + "-" * 40)
                error_msg = error_info.get("error", "Unknown error")
                
                # Check for specific error types and provide helpful information
                if "torch.load" in error_msg and "CVE-2025-32434" in error_msg:
                    report_lines.append("   Error Type: PyTorch Security Vulnerability")
                    report_lines.append("   Issue: Requires PyTorch >= v2.6 for security compliance")
                    report_lines.append("   Solution: Upgrade PyTorch with: pip install torch>=2.6")
                    report_lines.append(f"   Original Error: {error_msg[:100]}...")
                elif "error" in error_info:
                    report_lines.append(f"   Error: {error_msg}")
                
                if "timestamp" in error_info:
                    report_lines.append(f"   Failed at: {error_info['timestamp']}")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.append("\n" + "=" * 100)
        report_lines.append("RECOMMENDATIONS:")
        report_lines.append("=" * 100)
        
        if successful_models > 0:
            # Best overall model
            best_model = self.comparison_df.iloc[0]
            report_lines.append(f"🏆 BEST OVERALL: {best_model['Model']} ({best_model['Model_ID']})")
            report_lines.append(f"   Accuracy: {best_model['Accuracy']:.4f}, Training Time: {best_model['Total_Train_Time']:.1f}s")
            report_lines.append("")
            
            # Most efficient lightweight model
            lightweight_models = self.comparison_df[self.comparison_df['Parameters_M'] < 10]
            if not lightweight_models.empty:
                best_lightweight = lightweight_models.iloc[0]
                report_lines.append(f"⚡ BEST LIGHTWEIGHT: {best_lightweight['Model']} ({best_lightweight['Model_ID']})")
                report_lines.append(f"   {best_lightweight['Parameters_M']:.1f}M params, Accuracy: {best_lightweight['Accuracy']:.4f}")
                report_lines.append("")
            
            # Fastest training
            fastest_model = self.comparison_df.loc[self.comparison_df['Total_Train_Time'].idxmin()]
            report_lines.append(f"🚀 FASTEST TRAINING: {fastest_model['Model']} ({fastest_model['Model_ID']})")
            report_lines.append(f"   Training Time: {fastest_model['Total_Train_Time']:.1f}s, Accuracy: {fastest_model['Accuracy']:.4f}")
            report_lines.append("")
            
            # Most memory efficient
            memory_efficient = self.comparison_df.loc[self.comparison_df['Peak_Memory_GB'].idxmin()]
            report_lines.append(f"💾 MOST MEMORY EFFICIENT: {memory_efficient['Model']} ({memory_efficient['Model_ID']})")
            report_lines.append(f"   Peak Memory: {memory_efficient['Peak_Memory_GB']:.2f}GB, Accuracy: {memory_efficient['Accuracy']:.4f}")
        else:
            report_lines.append("❌ No successful models to provide recommendations.")
            report_lines.append("Please address the model loading issues listed above.")
        
        report_lines.append("\n" + "=" * 100)
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Detailed report saved to: {save_path}")
        
        return report_text
    
    def export_results_to_csv(self, save_path: Optional[str] = None, experiment_type: str = None) -> str:
        """Export comparison results to CSV"""
        if self.comparison_df is None:
            self.create_comparison_dataframe(experiment_type)
        
        if save_path is None:
            save_path = os.path.join(self.reports_dir, f"benchmark_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        self.comparison_df.to_csv(save_path, index=False)
        print(f"Results exported to CSV: {save_path}")
        return save_path
    
    def create_full_analysis_report(self, output_prefix: Optional[str] = None):
        """Create a complete analysis with all visualizations and reports"""
        if output_prefix is None:
            output_prefix = os.path.join(self.reports_dir, f"full_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        print("Creating comprehensive benchmark analysis...")
        
        # Check PyTorch version and warn if needed
        pytorch_info = check_pytorch_version()
        if not pytorch_info["is_compatible"]:
            print(f"\n{pytorch_info['recommendation']}")
            print("This may explain why some models failed to load.\n")
        else:
            print(f"{pytorch_info['recommendation']}\n")
        
        # Get available experiment types
        experiment_types = ["regular", "fullimage", "irregular"]
        print(f"Found experiment types: {experiment_types}")
        
        all_generated_files = {}
        
        # Generate separate reports for each experiment type
        for exp_type in experiment_types:
            print(f"\nGenerating analysis for experiment type: {exp_type}")
            
            # Create comparison DataFrame for this experiment type
            self.create_comparison_dataframe(exp_type)
            
            # Create experiment-specific output prefix
            exp_output_prefix = f"{output_prefix}_{exp_type}"
            
            if not self.comparison_df.empty:
                # Generate visualizations for this experiment type
                self.create_performance_comparison_plot(f"{exp_output_prefix}_performance.png")
                self.create_training_curves_plot(f"{exp_output_prefix}_training_curves.png")
            else:
                print(f"  No successful models for {exp_type}, skipping visualizations")
            
            # Generate reports (includes failed models even if no successful ones)
            self.generate_detailed_report(f"{exp_output_prefix}_detailed_report.txt", exp_type)
            
            if not self.comparison_df.empty:
                self.export_results_to_csv(f"{exp_output_prefix}_results.csv")
            
            # Store generated files for this experiment type
            exp_files = {
                "detailed_report": f"{exp_output_prefix}_detailed_report.txt"
            }
            
            if not self.comparison_df.empty:
                exp_files.update({
                    "performance_plot": f"{exp_output_prefix}_performance.png",
                    "training_curves": f"{exp_output_prefix}_training_curves.png",
                    "csv_results": f"{exp_output_prefix}_results.csv"
                })
            
            all_generated_files[exp_type] = exp_files
            
            print(f"  ✓ Analysis completed for {exp_type}")
        
        # If we have multiple experiment types, also create a combined report
        if len(experiment_types) > 1:
            print(f"\nGenerating combined analysis across all experiment types...")
            self.create_comparison_dataframe()  # All experiment types combined
            
            combined_prefix = f"{output_prefix}_combined"
            if not self.comparison_df.empty:
                self.create_performance_comparison_plot(f"{combined_prefix}_performance.png")
                self.create_training_curves_plot(f"{combined_prefix}_training_curves.png")
                self.export_results_to_csv(f"{combined_prefix}_results.csv")
            
            self.generate_detailed_report(f"{combined_prefix}_detailed_report.txt")
            
            combined_files = {
                "detailed_report": f"{combined_prefix}_detailed_report.txt"
            }
            if not self.comparison_df.empty:
                combined_files.update({
                    "performance_plot": f"{combined_prefix}_performance.png",
                    "training_curves": f"{combined_prefix}_training_curves.png",
                    "csv_results": f"{combined_prefix}_results.csv"
                })
            
            all_generated_files["combined"] = combined_files
            print("  ✓ Combined analysis completed")
        
        print(f"\n✓ Full analysis completed! Files generated:")
        for exp_type, files in all_generated_files.items():
            print(f"  {exp_type}:")
            for file_type, file_path in files.items():
                print(f"    - {file_type}: {file_path}")
        
        # Display plots if available
        if any("performance_plot" in files for files in all_generated_files.values()):
            plt.show()
        
        return all_generated_files

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze HuggingFace benchmark results")
    parser.add_argument("--results-dir", default="benchmarks/results", help="Results directory")
    parser.add_argument("--reports-dir", default="benchmarks/reports", help="Reports output directory")
    parser.add_argument("--pattern", default="*benchmark*.json", help="Pattern to match result files (benchmark_suite_*.json or comprehensive_benchmark_*.json)")
    parser.add_argument("--output-prefix", help="Output file prefix")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = BenchmarkAnalyzer(args.results_dir, args.reports_dir)
    
    try:
        # Load results
        analyzer.load_results(args.pattern)
        
        # Create full analysis
        analyzer.create_full_analysis_report(args.output_prefix)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run the benchmark first using huggingface_benchmark.py")

if __name__ == "__main__":
    main()