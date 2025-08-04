#!/bin/bash

# Comprehensive Benchmark Runner Script
# Usage examples for running all combinations of candlestick analysis

echo "🚀 Comprehensive HuggingFace Benchmark Runner"
echo "=============================================="

# Check if script exists
if [ ! -f "huggingface_benchmark.py" ]; then
    echo "❌ Error: huggingface_benchmark.py not found!"
    exit 1
fi

# Function to run benchmark with logging
run_benchmark() {
    local name="$1"
    shift
    echo ""
    echo "📊 Starting: $name"
    echo "Command: python3 huggingface_benchmark.py $@"
    echo "----------------------------------------"
    
    # Create log file
    log_file="logs/comprehensive_${name}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p logs
    
    # Run with output to both console and log
    python3 huggingface_benchmark.py "$@" 2>&1 | tee "$log_file"
    
    if [ $? -eq 0 ]; then
        echo "✅ Completed: $name"
        echo "📄 Log saved to: $log_file"
    else
        echo "❌ Failed: $name"
        echo "📄 Error log saved to: $log_file"
    fi
}

# Parse command line arguments
case "${1:-help}" in
    "quick")
        echo "Running quick test (1 coin, 1 period, 1 window size, 1 experiment type)"
        run_benchmark "quick_test" --comprehensive \
            --coins BTCUSDT \
            --periods 7days \
            --window-sizes 5 \
            --experiment-types regular \
            --model-set quick_test
        ;;
    
    "btc-eth")
        echo "Running BTC and ETH across all parameters"
        run_benchmark "btc_eth_full" --comprehensive \
            --coins BTCUSDT ETHUSDT \
            --model-set quick_test
        ;;
    
    "short-periods")
        echo "Running short periods (7 and 14 days) for all coins"
        run_benchmark "short_periods" --comprehensive \
            --periods 7days 14days \
            --model-set quick_test
        ;;
    
    "regular-only")
        echo "Running regular experiment type only"
        run_benchmark "regular_only" --comprehensive \
            --experiment-types regular \
            --model-set balanced
        ;;
    
    "full")
        echo "⚠️  WARNING: This will run ALL 216 combinations!"
        echo "Expected runtime: 20-30 hours with full_benchmark model set"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            run_benchmark "full_comprehensive" --comprehensive \
                --model-set full_benchmark \
                --epochs 10 \
                --max-samples 2000
        else
            echo "Cancelled."
        fi
        ;;
    
    "list-data")
        echo "Listing available data..."
        python3 huggingface_benchmark.py --list-data
        ;;
    
    "help"|*)
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  quick        - Quick test (1 combination, 2 models)"
        echo "  btc-eth      - Test BTC and ETH across all parameters"
        echo "  short-periods - Test 7 and 14 day periods only"
        echo "  regular-only - Test regular experiment type only"
        echo "  full         - Run complete comprehensive benchmark (⚠️ LONG!)"
        echo "  list-data    - List available candlestick data"
        echo "  help         - Show this help"
        echo ""
        echo "Examples:"
        echo "  ./run_comprehensive_benchmark.sh quick"
        echo "  ./run_comprehensive_benchmark.sh btc-eth"
        echo "  ./run_comprehensive_benchmark.sh list-data"
        echo ""
        echo "Custom usage:"
        echo "  python3 huggingface_benchmark.py --comprehensive --help"
        ;;
esac