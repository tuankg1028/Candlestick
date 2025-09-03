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

### Option 1: Automated Installation (Recommended)
```bash
# Install all dependencies with automatic fallbacks
python install_requirements.py

# Verify installation
python install_requirements.py --check
```

### Option 2: Manual Installation
```bash
# Install dependencies
pip install -r requirements_benchmark.txt

# If you encounter version conflicts:
pip install -r requirements_benchmark_compatible.txt

# For mplfinance specifically:
pip install mplfinance --pre
```

### Option 3: Test First, Then Install
```bash
# Test what's already working
python test_benchmark.py

# Install only what's missing based on test results
```

## Data Setup

### Generate Candlestick Data (Required)
```bash
# Generate regular data (recommended for first test)
python merged_candlestick.py --experiment regular

# Generate full image data (higher quality)
python merged_candlestick.py --experiment fullimage

# Generate irregular data (with missing candles)
python merged_candlestick.py --experiment irregular
```

### Verify Data Availability
```bash
# List all available datasets
python huggingface_benchmark.py --list-data

# Test data loading
python data_loader.py
```

## Quick Start

### Complete Setup Workflow
```bash
# 1. Install and test
python install_requirements.py
python test_benchmark.py

# 2. Generate data (if needed)
python merged_candlestick.py --experiment regular

# 3. Verify data
python huggingface_benchmark.py --list-data

# 4. Run quick comprehensive test
python huggingface_benchmark.py --comprehensive --model-set quick_test --epochs 2
# Generates: comprehensive_metrics_full_benchmark_YYYYMMDD_HHMMSS.csv

# 5. View results
head -20 comprehensive_metrics_full_benchmark_*.csv
```

### Individual Steps

#### 1. Quick Comprehensive Test (30 minutes)
Test 2 lightweight models across all combinations:
```bash
python huggingface_benchmark.py --comprehensive --model-set quick_test --epochs 2
```

#### 2. Lightweight Comprehensive Benchmark (2-4 hours)
Test lightweight models across all combinations:
```bash
python huggingface_benchmark.py --comprehensive --model-set lightweight --epochs 5
```

#### 3. Full Comprehensive Benchmark (8-24 hours)
Test all 9 models across all combinations (requires significant computational resources):
```bash
python huggingface_benchmark.py --comprehensive --model-set full_benchmark --epochs 10
```

#### 4. Results Analysis
The benchmark automatically generates a CSV report. You can analyze it directly:
```bash
# View the generated CSV file
cat comprehensive_metrics_full_benchmark_*.csv | head -20

# Or import into Python for analysis
python -c "import pandas as pd; df = pd.read_csv('comprehensive_metrics_full_benchmark_*.csv'); print(df.head())"
```

## Command Line Options

### huggingface_benchmark.py
```bash
python huggingface_benchmark.py [OPTIONS]

Options:
  --model-set {quick_test,lightweight,balanced,full_benchmark,transformers_only,cnns_only}
                        Set of models to benchmark (default: full_benchmark)
  --comprehensive       Run comprehensive benchmark across all combinations
  --coins TEXT [TEXT ...]  Coins to test (for comprehensive benchmark)
  --periods TEXT [TEXT ...]  Periods to test (for comprehensive benchmark)  
  --window-sizes INT [INT ...]  Window sizes to test (for comprehensive benchmark)
  --experiment-types {regular,fullimage,irregular} [...]  Experiment types to test
  --coin TEXT           Cryptocurrency to analyze (single benchmark, default: BTCUSDT)
  --period TEXT         Time period for analysis (single benchmark, default: 7days)
  --window-size INT     Window size for candlestick images (single benchmark, default: 5)
  --experiment-type {regular,fullimage,irregular}  Experiment type (single benchmark, default: regular)
  --epochs INT          Number of training epochs (default: 5)
  --max-samples INT     Maximum samples to use (0 for all, default: 0)
  --output-dir TEXT     Output directory (default: benchmarks)
  --list-data           List available candlestick data and exit

Examples:
  # Comprehensive benchmark (recommended) - generates single CSV report
  python huggingface_benchmark.py --comprehensive --model-set full_benchmark
  
  # Quick comprehensive test
  python huggingface_benchmark.py --comprehensive --model-set quick_test --epochs 2
  
  # Single configuration test (no CSV generated)  
  python huggingface_benchmark.py --coin ETHUSDT --period 14days --model-set lightweight
  
  # List available data
  python huggingface_benchmark.py --list-data
```

**Note**: Only `--comprehensive` mode generates the CSV report file. Single benchmarks are for testing only.

## Model Sets

- **quick_test**: 2 models for rapid testing
- **lightweight**: 3 models under 8M parameters
- **balanced**: 5 models balancing performance and efficiency
- **full_benchmark**: All 9 models
- **transformers_only**: Transformer-based models only
- **cnns_only**: CNN-based models only

## Output Files

The framework generates **one streamlined report file**:

### Main Directory
- `comprehensive_metrics_full_benchmark_{timestamp}.csv`: **Single comprehensive report**
  - Format: `Coin,Experiment,Window_Size,Period,Month,Model,Accuracy,F1,Recall,AUROC,AUPRC`
  - Contains: Test dataset results only with model attribution
  - Purpose: Complete performance comparison across all models and combinations

**Note**: All JSON files and multiple reports have been removed to simplify output. Only the essential CSV metrics file is generated.

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

### CSV Data Analysis
The CSV file contains all the data needed for analysis:
1. **Model Comparison**: Compare performance across different models
2. **Combination Analysis**: Identify best model for each configuration
3. **Trend Analysis**: Track performance across different periods and window sizes
4. **Import into Excel/Python**: Easy to analyze with pandas, Excel, or other tools

## Best Practices

### For Quick Testing
```bash
python huggingface_benchmark.py --comprehensive --model-set quick_test --epochs 2
# Generates comprehensive CSV in ~30 minutes
```

### For Production Benchmarking
```bash
python huggingface_benchmark.py --comprehensive --model-set full_benchmark --epochs 10
# Generates comprehensive CSV with all models and combinations
```

### For Memory-Constrained Systems
```bash
python huggingface_benchmark.py --comprehensive --model-set lightweight --epochs 5
# Only lightweight models to reduce memory usage
```

### For Custom Analysis
```bash
# Test specific combinations only
python huggingface_benchmark.py --comprehensive \
    --coins BTCUSDT ETHUSDT \
    --periods 7days 14days \
    --model-set balanced
```

## Integration with Existing Pipeline

The framework automatically integrates with your existing candlestick data:

1. **Data Loading**: Uses the same image loading functions from `merged_candlestick.py`
2. **Preprocessing**: Adapts images to different model input requirements
3. **Evaluation**: Uses consistent metrics for fair comparison

## Troubleshooting

### Quick Diagnosis
```bash
# Run comprehensive test
python test_benchmark.py

# Check available data
python huggingface_benchmark.py --list-data

# Test individual components
python data_loader.py
python model_configs.py
python benchmark_utils.py
```

### Common Issues

#### Import/Loading Errors
```bash
# Error: "Could not import from merged_candlestick.py" or "name 'load_images_parallel' is not defined"
# Solution: Framework now uses standalone data_loader.py
python test_benchmark.py  # Will show if data_loader.py is missing or broken

# If data_loader.py is missing, re-download the framework files
```

#### Missing Data Files
```bash
# Error: "No candlestick data found"
# Solution: Generate data first
python merged_candlestick.py --experiment regular  # Basic data
python merged_candlestick.py --experiment fullimage  # Higher quality

# Check what's available
python huggingface_benchmark.py --list-data
python data_loader.py  # Test data loading directly
```

#### Installation Issues
```bash
# Error: mplfinance version conflicts or other dependency issues
# Solution: Use automated installer
python install_requirements.py

# Or use compatible versions
pip install -r requirements_benchmark_compatible.txt

# For mplfinance specifically
pip install mplfinance --pre
```

#### CUDA Out of Memory
```bash
# Solutions (in order of preference):
python huggingface_benchmark.py --model-set quick_test --max-samples 200 --epochs 2
python huggingface_benchmark.py --model-set lightweight --max-samples 500
# Close other GPU applications
# Restart Python kernel/terminal
```

#### Model Download/Loading Failures
```bash
# Check internet connection and try:
python -c "from transformers import AutoModel; print('HuggingFace access OK')"

# Clear cache if needed:
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/torch/

# Test individual model loading:
python -c "from model_configs import get_model_config; print(get_model_config('efficientnet_b0'))"
```

### Performance Optimization
- **Start small**: `--max-samples 200 --epochs 2` for first test
- **Use appropriate model sets**: `quick_test` → `lightweight` → `balanced` → `full_benchmark`
- **Monitor resources**: Use `htop`, `nvidia-smi`, or Task Manager
- **Data preparation**: Generate all experiment types beforehand
- **Incremental testing**: Test one model at a time first

## Example Workflow

1. **Generate candlestick data**:
   ```bash
   python merged_candlestick.py --experiment fullimage
   ```

2. **Run quick comprehensive test**:
   ```bash
   python huggingface_benchmark.py --comprehensive --model-set quick_test --epochs 2
   # Generates: comprehensive_metrics_full_benchmark_YYYYMMDD_HHMMSS.csv
   ```

3. **Review results**:
   ```bash
   # View top results
   head -20 comprehensive_metrics_full_benchmark_*.csv
   
   # Or import into Python/Excel for analysis
   python -c "import pandas as pd; print(pd.read_csv('comprehensive_metrics_full_benchmark_*.csv').head())"
   ```

4. **Run full comprehensive benchmark** (if satisfied with quick test):
   ```bash
   python huggingface_benchmark.py --comprehensive --model-set full_benchmark --epochs 10
   # Generates final comprehensive CSV report
   ```

## Model Recommendations

Based on typical use cases:

- **Best Overall**: Usually EfficientNet-B0 or ConvNeXt-Tiny
- **Fastest Training**: EdgeNeXT XX Small or MobileNetV3 Small
- **Most Accurate**: Often Vision Transformer Base or Swin Tiny
- **Most Efficient**: EdgeNeXT XX Small or GhostNet
- **Production Use**: EfficientNet-B0 (good balance of all metrics)

The actual best model depends on your specific requirements for accuracy, speed, and resource constraints.