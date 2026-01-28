#!/bin/bash

# Set up CUDA environment
export CUDA_DIR=/usr/local/cuda-12.3
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.3/bin:$PATH
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-12.3"
export PYTHONUNBUFFERED=1

cd /home/nckh/son/Candlestick
source venv/bin/activate

LOG_FILE="training_fullimage_$(date +%Y%m%d_%H%M%S).log"

# Multi-GPU configuration (round-robin distribution)
# Available GPUs - modify this list based on your server
GPUS=(0)  # Add more GPUs like: GPUS=(0 1 2 3) for 4 GPUs
NUM_GPUS=${#GPUS[@]}

# Auto-detect GPUs if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
  DETECTED_GPUS=$(nvidia-smi --list-gpus | wc -l)
  if [ $DETECTED_GPUS -gt 1 ]; then
    GPUS=()
    for ((i=0; i<DETECTED_GPUS; i++)); do
      GPUS+=($i)
    done
    NUM_GPUS=$DETECTED_GPUS
    echo "Auto-detected $NUM_GPUS GPU(s)" | tee "$LOG_FILE"
  fi
fi

# Calculate MAX_PARALLEL based on GPU count (3 jobs per GPU)
MAX_PARALLEL=$((NUM_GPUS * 3))

# All 4 PyTorch models for Fullimage dataset
MODELS=("edgenext" "mobilenetv3" "ghostnet" "levit")
COINS=("BTCUSDT" "ETHUSDT" "BNBUSDT" "XRPUSDT" "ADAUSDT" "DOGEUSDT")
WINDOWS=(5 15 30)

echo "========================================" | tee "$LOG_FILE"
echo "PM2 FULLIMAGE TRAINING - Multi-GPU" | tee -a "$LOG_FILE"
echo "GPUs: ${GPUS[*]} ($NUM_GPUS available)" | tee -a "$LOG_FILE"
echo "Max parallel jobs: $MAX_PARALLEL" | tee -a "$LOG_FILE"
echo "Models: ${MODELS[*]}" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

TOTAL=$((${#MODELS[@]} * ${#COINS[@]} * ${#WINDOWS[@]}))
CURRENT=0
PIDS=()

for model in "${MODELS[@]}"; do
  for coin in "${COINS[@]}"; do
    for window in "${WINDOWS[@]}"; do
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

      # Round-robin GPU assignment
      GPU_INDEX=$(( (CURRENT - 1) % NUM_GPUS ))
      GPU_ID=${GPUS[$GPU_INDEX]}

      echo "" | tee -a "$LOG_FILE"
      echo "[$CURRENT/$TOTAL] $model - $coin - w$window | GPU:$GPU_ID (${#PIDS[@]}/$MAX_PARALLEL running)" | tee -a "$LOG_FILE"

      # Run with specific GPU
      CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/train_fullimage_pretrained_gpu.py \
          --model "$model" \
          --coin "$coin" \
          --window "$window" >> "$LOG_FILE" 2>&1 &

      PIDS+=($!)
      sleep 1
    done
  done
done

echo "Waiting for remaining jobs..." | tee -a "$LOG_FILE"
for pid in "${PIDS[@]}"; do
  wait $pid
done

echo "COMPLETED: $(date)" | tee -a "$LOG_FILE"
