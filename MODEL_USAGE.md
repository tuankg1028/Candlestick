# Model Usage Guide

## Quick Start

```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Available Models

### ✅ Ready to Use (TensorFlow)

1. **Custom CNN** - Baseline model, trained from scratch
   ```bash
   python3 src/last_candle_pretrained.py --model custom
   ```

2. **EfficientNetB0** - Google's pre-trained model
   ```bash
   python3 src/last_candle_pretrained.py --model efficientnet
   ```

### ⚠️ Requires PyTorch/timm

3. **EdgeNext Small** - Modern architecture
   ```bash
   python3 src/last_candle_pretrained.py --model edgenext
   ```

4. **MobileNetV3 Small** - Mobile-optimized
   ```bash
   python3 src/last_candle_pretrained.py --model mobilenetv3
   ```

5. **GhostNet 100** - Efficient architecture
   ```bash
   python3 src/last_candle_pretrained.py --model ghostnet
   ```

## Installation for PyTorch Models (Optional)

```bash
pip install --user --break-system-packages torch torchvision timm
```
*Note: Downloads ~3GB of packages*

## Usage Examples

### Run on specific coin with window size
```bash
python3 src/last_candle_pretrained.py --model efficientnet --coin BTCUSDT --window 5
```

### List all available models
```bash
python3 src/last_candle_pretrained.py --list-models
```

### View detailed model information
```bash
python3 src/models_info.py
```

### Interactive menu
```bash
python3 src/run_models.py
```

## Output Files

Models are saved in: `database/{COIN}/models/`
- TensorFlow: `model_{model_type}_{date}_1m_{period}_w{window}.h5`
- PyTorch: `model_{model_type}_{date}_1m_{period}_w{window}.pth`

Results are saved in: `database/{COIN}/results/`
- `results_{model_type}_{train/test}_{date}_1m_{period}_w{window}.txt`

## Supported Cryptocurrencies

- BTCUSDT (Bitcoin)
- ETHUSDT (Ethereum)
- BNBUSDT (Binance Coin)
- XRPUSDT (Ripple)
- ADAUSDT (Cardano)
- DOGEUSDT (Dogecoin)

## Training Parameters

- **Epochs**: 10
- **Batch Size**: 32
- **Learning Rate**: 0.001 (Adam optimizer)
- **Input Sizes**:
  - Custom CNN: 64x64
  - EfficientNetB0: 224x224
  - EdgeNext: 256x256
  - MobileNetV3: 224x224
  - GhostNet: 224x224