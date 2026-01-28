#!/bin/bash

# Set up CUDA environment
export CUDA_DIR=/usr/local/cuda-12.3
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.3/bin:$PATH
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-12.3"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

cd /home/nckh/son/Candlestick
source venv/bin/activate

LOG_FILE="training_irregular_$(date +%Y%m%d_%H%M%S).log"
MAX_PARALLEL=3

# All 4 PyTorch models for Irregular dataset
MODELS=("edgenext" "mobilenetv3" "ghostnet" "levit")
COINS=("BTCUSDT" "ETHUSDT" "BNBUSDT" "XRPUSDT" "ADAUSDT" "DOGEUSDT")
WINDOWS=(5 15 30)
MISSING=(0.6 0.8 0.95)  # Missing data ratios

echo "========================================" | tee "$LOG_FILE"
echo "PM2 IRREGULAR TRAINING - $MAX_PARALLEL jobs" | tee -a "$LOG_FILE"
echo "Models: ${MODELS[*]}" | tee -a "$LOG_FILE"
echo "Missing ratios: ${MISSING[*]}" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

TOTAL=$((${#MODELS[@]} * ${#COINS[@]} * ${#WINDOWS[@]} * ${#MISSING[@]}))
CURRENT=0
PIDS=()

for model in "${MODELS[@]}"; do
  for coin in "${COINS[@]}"; do
    for window in "${WINDOWS[@]}"; do
      for missing in "${MISSING[@]}"; do
        CURRENT=$((CURRENT + 1))

        # Wait for slot
        while [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; do
          NEW_PIDS=()
          for pid in "${PIDS[@]}"; do
            if kill -0 $pid 2>/dev/null; then
              NEW_PIDS+=($pid)
            else
              wait $pid 2>/dev/null
            fi
          done
          PIDS=("${NEW_PIDS[@]}")
          [ ${#PIDS[@]} -ge $MAX_PARALLEL ] && sleep 2
        done

        echo "" | tee -a "$LOG_FILE"
        echo "[$CURRENT/$TOTAL] $model - $coin - w$window - m${missing} (${#PIDS[@]}/$MAX_PARALLEL running)" | tee -a "$LOG_FILE"

        python3 src/train_irregular_pretrained_gpu.py \
            --model "$model" \
            --coin "$coin" \
            --window "$window" \
            --missing "$missing" >> "$LOG_FILE" 2>&1 &

        PIDS+=($!)
        sleep 1
      done
    done
  done
done

echo "Waiting for remaining jobs..." | tee -a "$LOG_FILE"
for pid in "${PIDS[@]}"; do
  wait $pid
done

echo "COMPLETED: $(date)" | tee -a "$LOG_FILE"
