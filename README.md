# Candlestick: Cryptocurrency Price Prediction using CNN

A machine learning research project that analyzes candlestick chart patterns to predict cryptocurrency price movements. Uses CNN models trained on minute-level data from Binance across multiple coins, timeframes, and pre-trained architectures.

## Features

- **Multi-Cryptocurrency Support**: BTC, ETH, BNB, XRP, ADA, DOGE
- **3 Dataset Variations**: Regular, Fullimage, Irregular (missing data)
- **4 Pre-trained CNN Models**: EdgeNext, MobileNetV3, GhostNet, LeViT
- **GPU Acceleration**: PyTorch with CUDA 12.3 on NVIDIA RTX 5000 Ada
- **Memory Efficient**: Lazy loading reduces memory from ~24GB to ~500MB
- **Parallel Training**: PM2 process manager for multi-job training
- **Comprehensive Metrics**: Accuracy, F1, Recall, AUROC, AUPRC

## Quick Start

```bash
# Clone the repository
git clone https://github.com/tuankg1028/Candlestick.git
cd Candlestick

# Activate virtual environment
source venv/bin/activate

# Train a single model
python src/train_regular_pretrained_gpu.py --model edgenext --coin BTCUSDT --window 5

# Or use PM2 for parallel training (3 jobs at once)
pm2 start ecosystem_parallel.config.js
pm2 logs 0
```

## Project Structure

```
Candlestick/
├── database/                      # All datasets (centralized)
│   ├── BTCUSDT/                   # Per-coin data
│   ├── crypto_research_minute_fullimage/
│   └── crypto_research_minute_irregular/
├── src/
│   ├── train_regular_pretrained_gpu.py      # Regular dataset
│   ├── train_fullimage_pretrained_gpu.py    # Fullimage dataset
│   └── train_irregular_pretrained_gpu.py    # Irregular dataset
├── run_pm2_remaining.sh           # PM2 runner (Regular)
├── run_fullimage_pm2.sh           # PM2 runner (Fullimage)
├── ecosystem_parallel.config.js   # PM2 config
├── venv/                          # Virtual environment
└── CLAUDE.md                      # Detailed documentation
```

## Datasets

| Dataset | Labeling Logic | Description |
|---------|---------------|-------------|
| **Regular** | Last candle only | Baseline prediction approach |
| **Fullimage** | Full window (first open to last close) | Whole image context |
| **Irregular** | Last candle only | Robustness testing with 40%, 20%, 5% data |

## Models

| Model | Parameters | Input Size | Framework |
|-------|------------|------------|-----------|
| EdgeNext | 1.33M | 256×256 | PyTorch (timm) |
| MobileNetV3 | 2.55M | 224×224 | PyTorch (timm) |
| GhostNet | 5.2M | 224×224 | PyTorch (timm) |
| LeViT | 7.91M | 224×224 | PyTorch (timm) |

## Installation

### Using Included Virtual Environment

```bash
source venv/bin/activate
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Manual Installation

```bash
# Core ML libraries
pip install tensorflow torch timm scikit-learn

# Data processing
pip install pandas numpy mplfinance matplotlib pillow

# API and utilities
pip install requests
```

## Usage

### Command-Line Options

```bash
python src/train_*.py \
  --model <edgenext|mobilenetv3|ghostnet|levit> \
  --coin <BTCUSDT|ETHUSDT|BNBUSDT|XRPUSDT|ADAUSDT|DOGEUSDT> \
  --window <5|15|30> \
  --missing <0.6|0.8|0.95>  # Irregular dataset only
```

### PM2 Parallel Training

```bash
# Start training
pm2 start ecosystem_parallel.config.js

# Monitor progress
pm2 list
pm2 logs 0

# Stop training
pm2 stop 0
pm2 delete 0
```

## Hardware Requirements

- **GPU**: NVIDIA RTX 5000 Ada (32GB) or equivalent
- **CUDA**: Version 12.3
- **RAM**: 32GB+ recommended
- **Storage**: ~100GB for all datasets

## Training Progress

| Dataset | Status | Results |
|---------|--------|---------|
| Regular | ✅ Complete | 2,880 |
| Fullimage | 🔄 In Progress | ~2,900+ |
| Irregular | ⏳ Pending | 0 |

## Output Structure

Each training run generates:

```
database/crypto_research_minute/<COIN>/
├── raw_data/          # OHLCV data from Binance
├── images/            # Generated candlestick images
├── models/            # Trained model files (.pth, .h5)
└── results/           # Evaluation metrics (.txt)
```

### Results Format

```
Train Metrics for edgenext - 2024-06 1m 7days w5 :
Accuracy: 0.5123
F1: 0.4987
Recall: 0.5234
Auroc: 0.5012
Auprc: 0.4956
```

## Export Results

```bash
python export_edgenext_results.py   # Export EdgeNext results
python export_regular_results.py    # Export all Regular results
```

## Configuration

Training parameters are defined in each script:

```python
COINS = {
    "BTCUSDT": {"train_month": (2024, 6), "test_months": [...]},
    "ETHUSDT": {"train_month": (2024, 6), "test_months": [...]},
    # ... more coins
}

TIME_LENGTHS = [7, 14, 21, 28]  # Training days
WINDOW_SIZES = [5, 15, 30]      # Candles per image
MISSING_RATIOS = [0.6, 0.8, 0.95]  # For irregular dataset
```

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@software{candlestick_cnn,
  title = {Candlestick: Cryptocurrency Price Prediction using CNN},
  author = {tuankg1028},
  year = {2025},
  url = {https://github.com/tuankg1028/Candlestick}
}
```

## Acknowledgments

- Binance API for cryptocurrency data
- TensorFlow and PyTorch teams for ML frameworks
- timm library for pre-trained models

---

For detailed documentation, see [CLAUDE.md](CLAUDE.md)
