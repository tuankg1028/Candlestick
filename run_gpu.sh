#!/bin/bash

# Set up CUDA environment for TensorFlow
export CUDA_DIR=/usr/local/cuda-12.3
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.3/bin:$PATH
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-12.3"

# Activate venv
source venv/bin/activate

# Run the script with all arguments
python3 src/last_candle_pretrained.py "$@"