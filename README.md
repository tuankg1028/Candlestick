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
# Install benchmark dependencies
pip install -r requirements_benchmark.txt

# Quick benchmark test
python huggingface_benchmark.py --model-set quick_test --max-samples 500 --epochs 3

# Analyze results
python results_analyzer.py
```

## 📁 Project Structure

```
Candlestick/
├── README.md                          # This file
├── README_benchmark.md                 # Detailed benchmarking guide
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
├── huggingface_benchmark.py           # Main benchmarking script
├── results_analyzer.py                # Results analysis & visualization
├── requirements_benchmark.txt         # Benchmark dependencies
├── 
└── benchmarks/                        # Outputs
    ├── results/                       # JSON results
    ├── models/                        # Saved models
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

### For Traditional Analysis Only
```bash
# Install basic requirements (infer from merged_candlestick.py imports)
pip install requests pandas mplfinance matplotlib numpy pillow tensorflow scikit-learn psutil
```

### For Complete Framework (Traditional + Benchmarking)
```bash
pip install -r requirements_benchmark.txt
```

## 📈 Usage Examples

### 1. Generate Candlestick Data
```bash
# Generate regular candlestick images
python merged_candlestick.py --experiment regular

# Generate full resolution images  
python merged_candlestick.py --experiment fullimage

# Generate with irregular/missing data
python merged_candlestick.py --experiment irregular
```

### 2. Quick Model Comparison
```bash
# Test 2 lightweight models quickly
python huggingface_benchmark.py --model-set quick_test --max-samples 500 --epochs 3

# View results
python results_analyzer.py
```

### 3. Comprehensive Benchmarking
```bash
# Test all lightweight models
python huggingface_benchmark.py --model-set lightweight --epochs 5

# Test all models (requires significant resources)
python huggingface_benchmark.py --model-set full_benchmark --epochs 10

# Generate detailed analysis
python results_analyzer.py --output-prefix comprehensive_analysis
```

### 4. Custom Configuration
```bash
# Specific coin and parameters
python huggingface_benchmark.py \
  --coin ETHUSDT \
  --period 14days \
  --window-size 15 \
  --model-set balanced \
  --epochs 8 \
  --max-samples 2000
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

### Common Issues
1. **CUDA Out of Memory**: Reduce `--max-samples` or use `--model-set lightweight`
2. **Missing Data**: Run `python merged_candlestick.py --experiment fullimage` first
3. **Import Errors**: Install all requirements with `pip install -r requirements_benchmark.txt`

### Performance Tips
- Use GPU for faster training
- Start with `quick_test` model set
- Limit samples with `--max-samples` for testing
- Close other applications to free memory

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

**🎉 Ready to start?** Begin with a quick test:
```bash
python huggingface_benchmark.py --model-set quick_test --max-samples 200 --epochs 2
```