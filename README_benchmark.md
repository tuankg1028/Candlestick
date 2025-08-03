# HuggingFace Model Benchmarking Framework for Candlestick Analysis

This framework provides comprehensive benchmarking of 9 different HuggingFace models for candlestick image classification tasks.

## Overview

The framework integrates with your existing candlestick analysis pipeline to benchmark the following models:

### Lightweight Models (1M - 8M parameters)
- **timm/edgenext_xx_small.in1k** (1.33M) - Ultra-lightweight CNN for edge devices
- **timm/mobilenetv3_small_100.lamb_in1k** (2.55M) - Mobile-optimized CNN
- **timm/ghostnet_100.in1k** (5.2M) - Efficient CNN with ghost modules
- **google/efficientnet-b0** (5.3M) - Compound scaling CNN
- **facebook/levit-128S** (7.91M) - Hybrid CNN-Transformer

### Performance Models (25M - 87M parameters)
- **microsoft/resnet-50** (25.6M) - Deep residual network
- **microsoft/swin-tiny-patch4-window7-224** (28.3M) - Hierarchical transformer
- **facebook/convnext-tiny-224** (29.0M) - Modernized CNN design
- **google/vit-base-patch16-224** (86.6M) - Pure transformer architecture

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_benchmark.txt
```

2. Ensure you have your candlestick data generated using the existing pipeline:
```bash
python merged_candlestick.py --experiment fullimage
```

## Quick Start

### 1. Run a Quick Benchmark
Test 2 lightweight models on a small dataset:
```bash
python huggingface_benchmark.py --model-set quick_test --max-samples 500 --epochs 3
```

### 2. Run Full Lightweight Benchmark
Test all lightweight models:
```bash
python huggingface_benchmark.py --model-set lightweight --max-samples 1000 --epochs 5
```

### 3. Run Complete Benchmark Suite
Test all 9 models (requires significant computational resources):
```bash
python huggingface_benchmark.py --model-set full_benchmark --epochs 10
```

### 4. Analyze Results
Generate comprehensive analysis and visualizations:
```bash
python results_analyzer.py
```

## Command Line Options

### huggingface_benchmark.py
```bash
python huggingface_benchmark.py [OPTIONS]

Options:
  --model-set {quick_test,lightweight,balanced,full_benchmark,transformers_only,cnns_only}
                        Set of models to benchmark (default: quick_test)
  --coin TEXT           Cryptocurrency to analyze (default: BTCUSDT)
  --period TEXT         Time period for analysis (default: 7days)
  --window-size INT     Window size for candlestick images (default: 5)
  --experiment-type {regular,fullimage,irregular}
                        Experiment type (default: regular)
  --epochs INT          Number of training epochs (default: 5)
  --max-samples INT     Maximum samples to use (0 for all, default: 1000)
  --output-dir TEXT     Output directory (default: benchmarks)
```

### results_analyzer.py
```bash
python results_analyzer.py [OPTIONS]

Options:
  --results-dir TEXT    Results directory (default: benchmarks/results)
  --reports-dir TEXT    Reports output directory (default: benchmarks/reports)
  --pattern TEXT        Pattern to match result files (default: benchmark_suite_*.json)
  --output-prefix TEXT  Output file prefix
```

## Model Sets

- **quick_test**: 2 models for rapid testing
- **lightweight**: 3 models under 8M parameters
- **balanced**: 5 models balancing performance and efficiency
- **full_benchmark**: All 9 models
- **transformers_only**: Transformer-based models only
- **cnns_only**: CNN-based models only

## Output Files

The framework generates several types of output:

### Results Directory (`benchmarks/results/`)
- `benchmark_suite_*.json`: Complete benchmark results
- `{model_name}_results.json`: Individual model results

### Reports Directory (`benchmarks/reports/`)
- `*_performance.png`: Performance comparison plots
- `*_training_curves.png`: Training progress visualization
- `*_detailed_report.txt`: Comprehensive text report
- `*_results.csv`: Results in CSV format

## Understanding the Results

### Performance Metrics
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **AUROC**: Area under ROC curve
- **AUPRC**: Area under precision-recall curve

### Efficiency Metrics
- **Training Time**: Total time for all epochs
- **Memory Usage**: Peak GPU/RAM usage during training
- **Inference Time**: Time for model predictions
- **Parameters**: Total model parameters

### Visualizations
1. **Accuracy vs Parameters**: Model size vs performance trade-offs
2. **Training Curves**: Loss and accuracy progression
3. **Memory Usage**: Resource consumption analysis
4. **Efficiency Analysis**: Combined performance and resource metrics

## Best Practices

### For Quick Testing
```bash
python huggingface_benchmark.py --model-set quick_test --max-samples 200 --epochs 2
```

### For Production Benchmarking
```bash
python huggingface_benchmark.py --model-set full_benchmark --epochs 15 --max-samples 5000
```

### For Memory-Constrained Systems
```bash
python huggingface_benchmark.py --model-set lightweight --epochs 5
```

## Integration with Existing Pipeline

The framework automatically integrates with your existing candlestick data:

1. **Data Loading**: Uses the same image loading functions from `merged_candlestick.py`
2. **Preprocessing**: Adapts images to different model input requirements
3. **Evaluation**: Uses consistent metrics for fair comparison

## Troubleshooting

### CUDA Out of Memory
- Reduce `--max-samples`
- Use `--model-set lightweight`
- Ensure other GPU processes are closed

### Missing Data Files
- Run your candlestick pipeline first:
  ```bash
  python merged_candlestick.py --experiment fullimage
  ```

### Import Errors
- Install all requirements:
  ```bash
  pip install -r requirements_benchmark.txt
  ```

## Example Workflow

1. **Generate candlestick data**:
   ```bash
   python merged_candlestick.py --experiment fullimage
   ```

2. **Run quick benchmark**:
   ```bash
   python huggingface_benchmark.py --model-set quick_test
   ```

3. **Analyze results**:
   ```bash
   python results_analyzer.py
   ```

4. **Run full benchmark** (if satisfied with quick test):
   ```bash
   python huggingface_benchmark.py --model-set full_benchmark --epochs 10
   ```

5. **Generate final report**:
   ```bash
   python results_analyzer.py --output-prefix final_analysis
   ```

## Model Recommendations

Based on typical use cases:

- **Best Overall**: Usually EfficientNet-B0 or ConvNeXt-Tiny
- **Fastest Training**: EdgeNeXT XX Small or MobileNetV3 Small
- **Most Accurate**: Often Vision Transformer Base or Swin Tiny
- **Most Efficient**: EdgeNeXT XX Small or GhostNet
- **Production Use**: EfficientNet-B0 (good balance of all metrics)

The actual best model depends on your specific requirements for accuracy, speed, and resource constraints.