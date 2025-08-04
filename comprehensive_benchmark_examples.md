# Comprehensive Benchmark Examples

The HuggingFace benchmark framework now supports running all combinations of coins, periods, window sizes, and experiment types in one shot!

## Usage Examples

### 1. Run ALL combinations (default settings)
```bash
python huggingface_benchmark.py --comprehensive
```

This will test:
- All 6 coins: BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT, DOGEUSDT
- All 4 periods: 7days, 14days, 21days, 28days  
- All 3 window sizes: 5, 15, 30
- All 3 experiment types: regular, fullimage, irregular
- **Total: 216 combinations**

### 2. Test specific coins with all other parameters
```bash
python huggingface_benchmark.py --comprehensive --coins BTCUSDT ETHUSDT
```

### 3. Test specific periods and window sizes
```bash
python huggingface_benchmark.py --comprehensive \
    --periods 7days 14days \
    --window-sizes 5 15 \
    --experiment-types regular fullimage
```

### 4. Use a different model set (more models)
```bash
python huggingface_benchmark.py --comprehensive --model-set balanced
```

### 5. Longer training with more samples
```bash
python huggingface_benchmark.py --comprehensive \
    --epochs 10 \
    --max-samples 5000
```

### 6. Quick test on subset
```bash
python huggingface_benchmark.py --comprehensive \
    --coins BTCUSDT \
    --periods 7days \
    --window-sizes 5 \
    --experiment-types regular \
    --model-set quick_test
```

## Available Model Sets
- `quick_test`: 2 models (edgenext_xx_small, efficientnet_b0)
- `lightweight`: All lightweight models 
- `balanced`: Lightweight + balanced models
- `full_benchmark`: All 9 models
- `transformers_only`: Only transformer models
- `cnns_only`: Only CNN models

## Output

The comprehensive benchmark will:

1. **Progress tracking**: Shows `[X/Y]` progress for each combination
2. **Skip missing data**: Automatically skips combinations where data doesn't exist
3. **Comprehensive results**: Saves detailed JSON with all results
4. **Summary tables**: 
   - Top 10 results by accuracy across all combinations
   - Best model per combination
5. **Time tracking**: Shows total time in hours

Results are saved to: `benchmarks/results/comprehensive_benchmark_[model_set]_[timestamp].json`

## Expected Runtime

For reference (with default settings):
- **quick_test** model set: ~2-4 hours for all 216 combinations
- **balanced** model set: ~8-12 hours  
- **full_benchmark** model set: ~20-30 hours

The framework automatically skips combinations where data doesn't exist, so actual runtime will be less.