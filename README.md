# Candlestick Image Classification & HuggingFace Model Benchmarking

A comprehensive framework for cryptocurrency candlestick pattern analysis using deep learning, featuring both traditional CNN approaches and state-of-the-art HuggingFace model benchmarking.

## 📋 Project Overview

This project combines:
1. **Traditional Candlestick Analysis** - Custom CNN models for candlestick pattern classification
2. **HuggingFace Model Benchmarking** - Systematic evaluation of 9 pre-trained models from HuggingFace

## 🚀 Quick Start

### Option 1: Traditional Candlestick Analysis
```bash
# Run regular experiments
python merged_candlestick.py --experiment regular

# Run full image experiments  
python merged_candlestick.py --experiment fullimage

# Run irregular data experiments
python merged_candlestick.py --experiment irregular
```

### Option 2: HuggingFace Model Benchmarking
```bash
# 1. Install dependencies (choose one method)
python install_requirements.py  # Automated (recommended)
# OR
pip install -r requirements_benchmark.txt  # Manual

# 2. Test the framework
python test_benchmark.py

# 3. Check available data (optional)
python huggingface_benchmark.py --list-data

# 4. Run quick benchmark
python huggingface_benchmark.py --model-set quick_test --max-samples 500 --epochs 3

# 5. Analyze results
python results_analyzer.py
```

## 📁 Project Structure

```
Candlestick/
├── README.md                           # Main project README
├── README_benchmark.md                 # Detailed benchmarking guide
├── CHANGELOG.md                        # Version history and updates
├── 
├── # Installation & Requirements
├── install_requirements.py            # Automated installation script
├── requirements_benchmark.txt         # Benchmark dependencies
├── requirements_benchmark_compatible.txt # Compatible versions
├── test_benchmark.py                  # Framework test suite
├── 
├── # Original Candlestick Analysis
├── merged_candlestick.py              # Main analysis script (merged from 3 files)
├── irregular.py                       # Original irregular data script
├── last_candle.py                     # Original regular data script  
├── last_candle-2.py                   # Original full image script
├── 
├── # HuggingFace Benchmarking Framework
├── model_configs.py                   # Model specifications
├── benchmark_utils.py                 # Utility functions
├── data_loader.py                     # Standalone data loading functions
├── huggingface_benchmark.py           # Main benchmarking script
├── results_analyzer.py                # Results analysis & visualization
├── 
└── benchmarks/                        # Outputs
    ├── results/                       # JSON benchmark results
    ├── models/                        # Saved model files
    └── reports/                       # Analysis reports & plots
```

## 🎯 Features

### Traditional Candlestick Analysis
- **Multiple Experiment Types**: Regular, full image, and irregular data
- **6 Cryptocurrencies**: BTC, ETH, BNB, XRP, ADA, DOGE
- **Variable Time Windows**: 7, 14, 21, 28 days
- **Multiple Window Sizes**: 5, 15, 30 candles per image
- **GPU Acceleration**: Automatic GPU detection and optimization
- **Parallel Processing**: Multi-threaded execution

### HuggingFace Model Benchmarking
- **9 Pre-trained Models**: From lightweight (1.3M params) to large (86M params)
- **Comprehensive Metrics**: Accuracy, F1, AUROC, AUPRC, training time, memory usage
- **Advanced Visualizations**: Performance plots, training curves, efficiency analysis
- **Flexible Model Sets**: Quick test, lightweight, balanced, full benchmark
- **Automatic Integration**: Uses existing candlestick data pipeline

## 📊 Supported Models

| Model | Parameters | Type | Use Case |
|-------|------------|------|----------|
| EdgeNeXT XX Small | 1.33M | CNN | Edge devices |
| MobileNetV3 Small | 2.55M | CNN | Mobile apps |
| GhostNet 100 | 5.2M | CNN | Efficient inference |
| EfficientNet-B0 | 5.3M | CNN | Balanced performance |
| LeViT-128S | 7.91M | Hybrid | CNN-Transformer |
| ResNet-50 | 25.6M | CNN | Proven architecture |
| Swin Tiny | 28.3M | Transformer | Modern approach |
| ConvNeXt Tiny | 29.0M | CNN | State-of-the-art CNN |
| ViT Base | 86.6M | Transformer | Maximum performance |

## 🛠️ Installation

### Option 1: Automated Installation (Recommended)
```bash
# Run the automated installer
python install_requirements.py

# Check if everything is working
python install_requirements.py --check
```

### Option 2: Manual Installation
```bash
# For complete framework (traditional + benchmarking)
pip install -r requirements_benchmark.txt

# If you encounter version conflicts, try the compatible version:
pip install -r requirements_benchmark_compatible.txt

# For mplfinance specifically (if it fails):
pip install mplfinance --pre
```

### Option 3: Traditional Analysis Only
```bash
# Minimal requirements for candlestick analysis
pip install requests pandas mplfinance matplotlib numpy pillow tensorflow scikit-learn psutil
```

## 📈 Usage Examples

### 1. Getting Started (First Time Setup)
```bash
# Install all dependencies
python install_requirements.py

# Test everything is working
python test_benchmark.py

# Generate candlestick data (if not done already)
python merged_candlestick.py --experiment regular

# Check what data is available
python huggingface_benchmark.py --list-data
```

### 2. Quick Model Comparison
```bash
# Test 2 lightweight models quickly (5 minutes)
python huggingface_benchmark.py --model-set quick_test --max-samples 200 --epochs 2

# View results with visualizations
python results_analyzer.py
```

### 3. Production Benchmarking
```bash
# Test all lightweight models (30 minutes)
python huggingface_benchmark.py --model-set lightweight --epochs 5 --max-samples 1000

# Test balanced models (1-2 hours)
python huggingface_benchmark.py --model-set balanced --epochs 8 --max-samples 2000

# Full benchmark - all models (4-8 hours)
python huggingface_benchmark.py --model-set full_benchmark --epochs 10

# Generate comprehensive analysis report
python results_analyzer.py --output-prefix full_analysis
```

### 4. Custom Configuration Examples
```bash
# Focus on specific cryptocurrency
python huggingface_benchmark.py \
  --coin ETHUSDT \
  --period 14days \
  --window-size 15 \
  --model-set balanced \
  --epochs 8 \
  --max-samples 2000

# Compare transformer models only
python huggingface_benchmark.py \
  --model-set transformers_only \
  --experiment-type fullimage \
  --epochs 10

# Quick test with irregular data
python huggingface_benchmark.py \
  --model-set quick_test \
  --experiment-type irregular \
  --max-samples 500
```

### 5. Troubleshooting and Diagnostics
```bash
# Check if framework is working correctly
python test_benchmark.py

# List all available datasets
python huggingface_benchmark.py --list-data

# Test data loading manually
python data_loader.py

# Check model configurations
python model_configs.py

# Test individual components
python benchmark_utils.py
```

## 📊 Expected Results

### Traditional Analysis Output
- **Raw Data**: CSV files with OHLC data
- **Images**: Candlestick chart images (PNG)
- **Models**: Trained TensorFlow models (.h5)
- **Results**: Performance metrics (TXT files)

### Benchmark Analysis Output
- **Performance Plots**: Accuracy vs parameters, training curves
- **Detailed Reports**: Text summaries with recommendations
- **Data Exports**: CSV files for further analysis
- **Model Rankings**: Best models for different use cases

## 🎯 Use Cases

### For Researchers
- Compare different model architectures systematically
- Analyze trade-offs between accuracy and efficiency
- Generate publication-ready performance comparisons

### For Practitioners  
- Find the best model for production deployment
- Understand resource requirements for different models
- Optimize for specific constraints (memory, speed, accuracy)

### For Educators
- Demonstrate modern deep learning approaches
- Show real-world model comparison methodology
- Provide hands-on experience with multiple architectures

## 🔧 Configuration

### Model Sets
- `quick_test`: Fast testing with 2 models
- `lightweight`: Memory-efficient models < 8M parameters
- `balanced`: Good performance/efficiency trade-off
- `full_benchmark`: All 9 models
- `transformers_only`: Transformer architectures only
- `cnns_only`: CNN architectures only

### Data Options
- **Coins**: BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT, DOGEUSDT
- **Periods**: 7days, 14days, 21days, 28days
- **Window Sizes**: 5, 15, 30 candles
- **Experiment Types**: regular, fullimage, irregular

## 🚨 System Requirements

### Minimum
- **RAM**: 8GB
- **Storage**: 5GB free space
- **Python**: 3.8+

### Recommended
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with 6GB+ VRAM
- **Storage**: 20GB+ free space
- **CPU**: Multi-core processor

### For Full Benchmarking
- **RAM**: 32GB+
- **GPU**: NVIDIA GPU with 12GB+ VRAM
- **Storage**: 50GB+ free space

## 🔍 Troubleshooting

### Quick Diagnostics
```bash
# Run comprehensive test suite
python test_benchmark.py

# Check what data is available
python huggingface_benchmark.py --list-data

# Test data loading specifically
python data_loader.py
```

### Common Issues & Solutions

#### 1. **Import Errors**
```bash
# Error: "Could not import from merged_candlestick.py" or "name 'load_images_parallel' is not defined"
# Solution: Use the test suite to diagnose
python test_benchmark.py

# If data_loader.py is missing, re-download or recreate it
```

#### 2. **Missing Data Files**
```bash
# Error: "No candlestick data found"
# Solution: Generate data first
python merged_candlestick.py --experiment regular
python merged_candlestick.py --experiment fullimage

# Check what's available
python huggingface_benchmark.py --list-data
```

#### 3. **Installation Issues**
```bash
# Error: mplfinance version conflicts
# Solution: Use automated installer
python install_requirements.py

# Or install with pre-release
pip install mplfinance --pre

# Or use compatible requirements
pip install -r requirements_benchmark_compatible.txt
```

#### 4. **CUDA/Memory Issues**
```bash
# Error: "CUDA out of memory"
# Solutions:
python huggingface_benchmark.py --model-set lightweight --max-samples 500
python huggingface_benchmark.py --model-set quick_test --max-samples 200 --epochs 2

# Close other GPU applications
# Reduce batch sizes in model configs
```

#### 5. **Model Loading Errors**
```bash
# Error: Model download or loading failures
# Solutions: Check internet connection, try individual models
python -c "from model_configs import get_model_config; print(get_model_config('efficientnet_b0'))"

# Clear cache and retry
rm -rf ~/.cache/huggingface/
```

### Performance Optimization Tips
- **Start Small**: Use `quick_test` with `--max-samples 200 --epochs 2`
- **Use GPU**: Ensure CUDA is available for faster training
- **Monitor Resources**: Watch memory usage with `htop` or Task Manager
- **Incremental Testing**: Test one model at a time first
- **Data Preparation**: Generate all needed data beforehand

### Debug Mode
```bash
# Enable verbose logging
export TRANSFORMERS_VERBOSITY=info

# Run with detailed error messages
python huggingface_benchmark.py --model-set quick_test --max-samples 100 --epochs 1
```

## 📚 Documentation

- **README_benchmark.md**: Detailed benchmarking framework guide
- **model_configs.py**: Model specifications and configurations  
- **Code Comments**: Extensive inline documentation
- **Example Scripts**: Command examples throughout

## 🤝 Contributing

This project integrates multiple approaches to candlestick analysis:
1. Traditional custom CNN models
2. Modern pre-trained model adaptation
3. Comprehensive benchmarking methodology

Feel free to extend with additional models, datasets, or analysis techniques.

## 📄 License

Please check individual model licenses when using HuggingFace models in production.

---

## 🆕 Recent Updates (Version 2.0)

### ✅ Major Fixes
- **Fixed Import Error**: No more `'load_images_parallel' is not defined` errors
- **Fixed Installation Issues**: Automated installer handles mplfinance and version conflicts
- **Enhanced Reliability**: Standalone data loading module for robust operation

### 🚀 New Features  
- **Test Suite**: `python test_benchmark.py` verifies everything works
- **Data Discovery**: `--list-data` option shows available datasets
- **Automated Installation**: `python install_requirements.py` handles complex dependencies
- **Better Documentation**: Step-by-step workflows with time estimates

### 📋 Migration for Existing Users
```bash
# Quick update - just run the test suite
python test_benchmark.py

# If issues found, use automated installer
python install_requirements.py

# Continue as normal - all commands unchanged
python huggingface_benchmark.py --model-set quick_test --max-samples 200 --epochs 2
```

---

**🎉 Ready to start?** Complete setup workflow:
```bash
# 1. Install and test
python install_requirements.py
python test_benchmark.py

# 2. Generate data (if needed)  
python merged_candlestick.py --experiment regular

# 3. Run benchmark
python huggingface_benchmark.py --model-set quick_test --max-samples 200 --epochs 2

# 4. Analyze results
python results_analyzer.py
```