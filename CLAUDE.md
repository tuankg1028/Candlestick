# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a cryptocurrency trading research project that uses machine learning to analyze candlestick chart patterns for price prediction. The project processes minute-level cryptocurrency data from Binance and trains CNN models to predict price movements using multiple approaches and pre-trained models.

## Dependencies

A virtual environment (`venv/`) is included with all required dependencies installed. To use it:

```bash
# Activate the virtual environment
source venv/bin/activate

# Verify GPU is available
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Manual installation** (if needed):

```bash
# Core ML/AI libraries
pip install tensorflow scikit-learn pandas numpy

# Data visualization and image processing
pip install mplfinance matplotlib pillow

# API and utilities
pip install requests

# Optional: PyTorch and timm for additional pre-trained models
pip install torch timm
```

## Project Structure

```
Candlestick/
├── run_gpu.sh                 # GPU runner script (sets CUDA paths)
├── run_pm2_remaining.sh       # PM2 training script for remaining models (Regular dataset)
├── run_fullimage_pm2.sh       # PM2 training script for Fullimage dataset
├── run_irregular_pm2.sh       # PM2 training script for Irregular dataset
├── ecosystem_parallel.config.js  # PM2 configuration for parallel training (Regular)
├── ecosystem_fullimage.config.js  # PM2 configuration for Fullimage training
├── export_all_results.py      # Export all results to CSV format
├── venv/                      # Virtual environment (with GPU support)
├── logs/                      # PM2 logs
├── results_export/            # Exported CSV results (per coin)
├── database/                  # All datasets stored here (centralized location)
│   ├── BTCUSDT/               # Regular dataset results (per coin)
│   ├── ETHUSDT/
│   ├── BNBUSDT/
│   ├── XRPUSDT/
│   ├── ADAUSDT/
│   ├── DOGEUSDT/
│   ├── crypto_research_minute_fullimage/   # Fullimage dataset
│   │   ├── BTCUSDT/, ETHUSDT/, etc.
│   └── crypto_research_minute_irregular/   # Irregular dataset
│       └── BTCUSDT/, ETHUSDT/, etc.
└── src/
    ├── crypto_research_minute -> ../database  # Symlink to database/
    ├── crypto_research_minute_fullimage -> database/crypto_research_minute_fullimage  # Symlink
    ├── crypto_research_minute_irregular -> database/crypto_research_minute_irregular  # Symlink
    ├── full_image.py          # Full candlestick image classification (original)
    ├── irregular.py           # Robustness testing with missing data (original)
    ├── last_candle.py         # Next candle movement prediction (original)
    ├── train_regular_pretrained_gpu.py    # Memory-efficient GPU training (regular)
    ├── train_fullimage_pretrained_gpu.py  # Memory-efficient GPU training (fullimage)
    └── train_irregular_pretrained_gpu.py  # Memory-efficient GPU training (irregular)
```

### Important: Symlink Structure
All dataset directories in `src/` are **symbolic links** to `database/`. This keeps all training data centralized:
- `src/crypto_research_minute` → `database/` (contains per-coin folders)
- `src/crypto_research_minute_fullimage` → `database/crypto_research_minute_fullimage`
- `src/crypto_research_minute_irregular` → `database/crypto_research_minute_irregular`

**When adding new datasets**, create them in `database/` first, then create symlinks in `src/`.

## Training Scripts Overview

### Original Scripts (Custom CNN only)
| Script | Dataset | Labeling Logic | Purpose |
|--------|---------|---------------|---------|
| `full_image.py` | Fullimage | Full window (last close > first open) | Full image classification |
| `irregular.py` | Irregular | Last candle only | Robustness testing with missing data |
| `last_candle.py` | Regular | Last candle only | Original baseline |

### GPU Training Scripts (Memory-Efficient)
| Script | Dataset | Labeling Logic | Feature |
|--------|---------|---------------|---------|
| `train_regular_pretrained_gpu.py` | Regular | Last candle only | Lazy loading, 4 PyTorch models |
| `train_fullimage_pretrained_gpu.py` | Fullimage | Full window | Lazy loading, reuses regular images |
| `train_irregular_pretrained_gpu.py` | Irregular | Last candle only | Lazy loading, missing data ratios |

**Key Improvements:**
- **Memory Efficient**: Reduced from ~24GB to ~500MB per dataset using lazy loading
- **GPU Acceleration**: PyTorch models (edgenext, mobilenetv3, ghostnet, levit) run on GPU
- **Parallel Processing**: PM2 can run 3 jobs in parallel for faster training

## Pre-trained Models

The GPU training scripts support 4 PyTorch models (TensorFlow models have CuDNN compatibility issues):

| Model | Parameters | Framework | Input Size | Status |
|-------|------------|-----------|------------|--------|
| **edgenext** | 1.33M | timm | 256×256 | ✅ Complete (535 results) |
| **mobilenetv3** | 2.55M | timm | 224×224 | 🔄 In Progress |
| **ghostnet** | 5.2M | timm | 224×224 | ⏳ Pending |
| **levit** | 7.91M | timm | 224×224 | ⏳ Pending |
| **efficientnet** | 5.3M | TensorFlow | 224×224 | ❌ CuDNN incompatibility |

## Running the Code

### Recommended: PM2 Parallel Training

The recommended way to train models is using PM2 with parallel processing:

```bash
# Start PM2 training (runs 3 jobs in parallel)
pm2 start ecosystem_parallel.config.js

# Check status
pm2 list

# View logs
pm2 logs 0

# Stop training
pm2 stop 0

# Restart training
pm2 restart 0
```

### Manual Training (Single Job)

You can also run individual training jobs manually:

```bash
source venv/bin/activate

# Train on Regular dataset
python src/train_regular_pretrained_gpu.py --model mobilenetv3 --coin BTCUSDT --window 5

# Train on Fullimage dataset
python src/train_fullimage_pretrained_gpu.py --model ghostnet --coin ETHUSDT --window 15

# Train on Irregular dataset
python src/train_irregular_pretrained_gpu.py --model levit --coin BTCUSDT --window 30 --missing 0.8
```

### Command-Line Options

```bash
--model <type>     # Model type: edgenext, mobilenetv3, ghostnet, levit
--coin <symbol>    # Specific coin: BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT, DOGEUSDT
--window <size>    # Window size: 5, 15, 30
--missing <ratio>  # Missing ratio (irregular only): 0.6, 0.8, 0.95
--list-models      # List all available models
```


## Data Handling

### Centralized Data Storage

All datasets are stored in the `database/` directory and accessed via symlinks from `src/`:

```
database/                              # Actual data location
├── BTCUSDT/                          # Regular dataset (coin folders)
│   ├── raw_data/                     # Downloaded OHLCV data
│   ├── images/                       # Generated candlestick images
│   ├── models/                       # Trained model files
│   └── results/                      # Evaluation metrics
├── crypto_research_minute_fullimage/ # Fullimage dataset
│   └── BTCUSDT/, ETHUSDT/, etc.
└── crypto_research_minute_irregular/ # Irregular dataset
    └── BTCUSDT/, ETHUSDT/, etc.

src/                                   # Symlinks to database/
├── crypto_research_minute -> ../database
├── crypto_research_minute_fullimage -> database/crypto_research_minute_fullimage
└── crypto_research_minute_irregular -> database/crypto_research_minute_irregular
```

**Why this structure?**
- **Single source of truth**: All data in one location (`database/`)
- **Easy backup**: Just backup the `database/` folder
- **Clean separation**: Code in `src/`, data in `database/`
- **Git-friendly**: Can keep `database/` out of version control if needed

### Dataset Characteristics

| Dataset | Labeling | Data | Reuses Images |
|---------|----------|------|---------------|
| **Regular** | Last candle only | 100% | No |
| **Fullimage** | Full window (first open to last close) | 100% | Yes (from Regular) |
| **Irregular** | Last candle only | 40%, 20%, 5% | No |

All scripts have smart caching - they check if data exists before downloading from Binance API.

## Output Structure

After training completes, each script generates outputs in their respective dataset directory:

```
src/crypto_research_minute/                    # Regular
src/crypto_research_minute_fullimage/          # Fullimage
src/crypto_research_minute_irregular/          # Irregular
└── BTCUSDT/ (or ETHUSDT, BNBUSDT, etc.)
    ├── raw_data/          # Downloaded OHLCV data
    │   └── raw_2024-06_1m_7days.csv
    ├── images/            # Generated candlestick images
    │   ├── 2024-06_1m_7days_w5/
    │   │   ├── labels_2024-06_1m_7days_w5_size224.csv
    │   │   └── candle_0.png, candle_1.png, ...
    ├── models/            # Trained model files
    │   ├── model_efficientnet_2024-06_1m_7days_w5.h5
    │   ├── model_edgenext_2024-06_1m_7days_w5.pth
    │   ├── model_mobilenetv3_2024-06_1m_7days_w5.pth
    │   ├── model_ghostnet_2024-06_1m_7days_w5.pth
    │   └── model_levit_2024-06_1m_7days_w5.pth
    └── results/           # Evaluation metrics
        ├── results_efficientnet_train_2024-06_1m_7days_w5.txt
        └── results_efficientnet_test_2024-12_1m_7days_w5.txt
```

### Results Format

Each results file contains 5 metrics:

```
Train Metrics for mobilenetv3 - 2024-06 1m 7days w5 :
Accuracy: 0.5123
F1: 0.4987
Recall: 0.5234
Auroc: 0.5012
Auprc: 0.4956
```

### Exported CSV Files

Results can be exported to per-coin CSV files using `export_all_results.py`:

```bash
python export_all_results.py
```

**Output format**: `{COIN}_exp{N}_results.csv`
- `exp1`: Regular dataset (Experiment I)
- `exp2`: Fullimage dataset (Experiment II)
- `exp3`: Irregular dataset (Experiment III)

**CSV columns**: `Coin, Experiment, Model, Window_Size, Period, Month, Dataset, Accuracy, F1, Recall, AUROC, AUPRC`

**Example files**:
- `BTCUSDT_exp1_results.csv` - Regular dataset results for BTC
- `ETHUSDT_exp2_results.csv` - Fullimage dataset results for ETH
- `ADAUSDT_exp3_results.csv` - Irregular dataset results for ADA

**Sample data**:
```csv
Coin,Experiment,Model,Window_Size,Period,Month,Dataset,Accuracy,F1,Recall,AUROC,AUPRC
ADAUSDT,I,ghostnet,30,21days,2024-12,Test,1.0,1.0,1.0,1.0,1.0
ADAUSDT,I,edgenext,5,14days,2024-11,Test,0.549,0.0,0.0,0.7654,0.6775
ADAUSDT,II,mobilenetv3,5,21days,2024-01,Test,1.0,1.0,1.0,1.0,1.0
```

**Column descriptions**:
- `Coin`: Cryptocurrency pair (BTCUSDT, ETHUSDT, etc.)
- `Experiment`: I (Regular), II (Fullimage), III (Irregular)
- `Model`: edgenext, mobilenetv3, ghostnet, levit, custom
- `Window_Size`: 5, 15, or 30 candles
- `Period`: 7days, 14days, 21days, 28days
- `Month`: YYYY-MM format
- `Dataset`: Train or Test

### Training Experiments

**Experiment I**: Train and test on matching time lengths
- Train on 7, 14, 21, 28 days
- Test on matching lengths across 5 test months

**Experiment II**: Train on 1 week, test on longer periods
- Train on 1 week
- Test on 2, 3, 4 weeks across 5 test months

### Output Summary

```
Per coin × model × window:
- 4 training models (7, 14, 21, 28 days) + 1 model (1 week) = 5 models
- 24 test results (Exp I) + 15 test results (Exp II) = 39 results

Total: 6 coins × 5 models × 3 windows × 44 files ≈ 3,960 files per dataset
3 Datasets × 3,960 ≈ 11,880 total output files
```

## Architecture

### Core Components

1. **Data Collection**: Uses Binance API endpoint `/api/v3/klines` to fetch 1-minute interval OHLCV data
2. **Image Generation**: Creates candlestick charts using `mplfinance` with matplotlib's 'Agg' backend for headless operation
3. **Memory-Efficient Data Loading**:
   - TensorFlow Dataset API with lazy loading (`create_tf_dataset()`)
   - PyTorch DataLoader with `LazyImageDataset` class
   - Images loaded on-demand from disk during training (not all at once)
   - Reduced memory from ~24GB to ~500MB per dataset
4. **Model Architecture**:
   - Pre-trained PyTorch models via timm (EdgeNext, MobileNetV3, GhostNet, LeViT)
   - GPU acceleration with CUDA 12.3
5. **Evaluation**: Multiple metrics including accuracy, F1-score, recall, AUROC, and AUPRC

### Configuration Constants

Each script uses these key configurations:

```python
COINS = {
    "BTCUSDT": {"train_month": (2024, 6), "test_months": [(2024, 12), (2024, 3), ...]},
    "ETHUSDT": {"train_month": (2024, 6), "test_months": [...]},
    "BNBUSDT": {"train_month": (2024, 10), "test_months": [...]},
    "XRPUSDT": {"train_month": (2024, 9), "test_months": [...]},
    "ADAUSDT": {"train_month": (2024, 9), "test_months": [...]},
    "DOGEUSDT": {"train_month": (2024, 9), "test_months": [...]}
}

TIME_LENGTHS = [7, 14, 21, 28]  # Days of data per experiment
WINDOW_SIZES = [5, 15, 30]      # Candles per training image
MISSING_RATIOS = [0.6, 0.8, 0.95]  # For irregular dataset
```

## Key Features

1. **Multi-Cryptocurrency Support**: Research across 6 major trading pairs (BTC, ETH, BNB, XRP, ADA, DOGE)
2. **Multiple Timeframes**: 1-4 week time periods for training/testing
3. **Window-based Analysis**: Processes sequences of 5, 15, or 30 candles per prediction
4. **Robustness Testing**: The irregular dataset tests model performance with 60%, 80%, and 95% missing data
5. **Pre-trained Models**: Support for 5 ImageNet pre-trained models via transfer learning
6. **Binary Classification**: Predicts whether price will go up or down in the next interval
7. **Labeling Variations**: Last candle only vs full window labeling strategies

## Training Progress

### Regular Dataset ✅ COMPLETE

| Model | Status | Results |
|-------|--------|---------|
| edgenext | ✅ Complete | 720 |
| mobilenetv3 | ✅ Complete | 720 |
| ghostnet | ✅ Complete | 720 |
| levit | ✅ Complete | 720 |
| **Total** | | **2,880** |

### Fullimage Dataset 🔄 IN PROGRESS

| Model | Status | Results |
|-------|--------|---------|
| edgenext | 🔄 In Progress | ~480+ |
| mobilenetv3 | 🔄 In Progress | ~480+ |
| ghostnet | 🔄 In Progress | ~480+ |
| levit | ⏳ Pending | 0 |
| **Current** | BTCUSDT complete, training ETHUSDT+ |

### Irregular Dataset ⏳ PENDING

| Model | Status | Results |
|-------|--------|---------|
| edgenext | ⏳ Pending | 0 |
| mobilenetv3 | ⏳ Pending | 0 |
| ghostnet | ⏳ Pending | 0 |
| levit | ⏳ Pending | 0 |

### Remaining Work

- **Fullimage Dataset**: ~2,160 jobs remaining (ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT, DOGEUSDT)
- **Irregular Dataset**: ~6,480 jobs (all 6 coins × 3 missing ratios)

**Total Remaining**: ~8,640 training jobs

## GPU Support

The project uses GPU acceleration via PyTorch with CUDA 12.3. PM2 manages parallel training jobs.

### Hardware
- **GPU**: NVIDIA RTX 5000 Ada Generation (32GB memory)
- **CUDA**: Version 12.3
- **cuDNN**: Version 9.1.0

### PM2 Configuration
```javascript
// ecosystem_parallel.config.js
module.exports = {
  apps: [{
    name: 'train-parallel-remaining',
    script: './run_pm2_remaining.sh',
    max_memory_restart: '35G',
    env: {
      CUDA_VISIBLE_DEVICES: '0',
      PYTHONUNBUFFERED: '1'
    }
  }]
};
```

### Parallel Processing
- **Max Parallel Jobs**: 3 (optimized for 32GB GPU memory)
- **Memory per Job**: ~3-5GB GPU memory
- **Batch Size**: 64 for both TensorFlow and PyTorch

## Development Notes

- All scripts use `plt.switch_backend('Agg')` for non-interactive matplotlib operation
- **Memory-efficient lazy loading**: Images loaded on-demand during training, not all at once
- Smart data caching avoids re-downloading from Binance API
- PyTorch models are saved as `.pth` files in `models/` directories
- Results are saved as `.txt` files with 5 metrics (Accuracy, F1, Recall, AUROC, AUPRC)
- PM2 manages training with automatic restart on memory issues (>35GB)
- Class weighting is implemented to handle imbalanced datasets

## Common Issues

1. **GPU at 0% Utilization**: Normal during data loading. GPU usage increases during actual training epochs.
2. **Memory Exhaustion**: PM2 will automatically restart if memory exceeds 35GB. Reduce parallel jobs if needed.
3. **CuDNN Version Mismatch**: TensorFlow models (efficientnet) have CuDNN compatibility issues. Use PyTorch models only.
4. **PyTorch Import Error**: Install `torch` and `timm` if using edgenext/mobilenetv3/ghostnet/levit models
5. **ImageNet Weights Download**: First run of pre-trained models will download ImageNet weights (~100MB each)
6. **Training Appears Slow**: GPU utilization fluctuates. Low usage during data loading, high during training.
7. **Directory Not Found Error**: If you see `OSError: Cannot save file into a non-existent directory`, it means the coin directories don't exist in `database/`. Create them with:
   ```bash
   for coin in BTCUSDT ETHUSDT BNBUSDT XRPUSDT ADAUSDT DOGEUSDT; do
     mkdir -p database/crypto_research_minute_fullimage/$coin/{raw_data,images,models,results}
   done
   ```
