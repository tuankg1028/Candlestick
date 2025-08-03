#!/usr/bin/env python3
"""
Test script for HuggingFace benchmarking framework
Checks if all components are working properly
"""

import os
import sys

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    test_modules = [
        ("model_configs", "Model configurations"),
        ("benchmark_utils", "Benchmark utilities"),  
        ("data_loader", "Data loader"),
        ("huggingface_benchmark", "Main benchmark script"),
        ("results_analyzer", "Results analyzer"),
    ]
    
    success_count = 0
    
    for module, description in test_modules:
        try:
            __import__(module)
            print(f"✓ {description}")
            success_count += 1
        except ImportError as e:
            print(f"✗ {description}: {e}")
    
    print(f"\nImport test: {success_count}/{len(test_modules)} successful")
    return success_count == len(test_modules)

def test_model_configs():
    """Test model configurations"""
    print("\n🎯 Testing model configurations...")
    
    try:
        from model_configs import BENCHMARK_MODELS, get_model_config, MODEL_SETS
        
        print(f"✓ Found {len(BENCHMARK_MODELS)} models configured")
        print(f"✓ Found {len(MODEL_SETS)} model sets configured")
        
        # Test a specific model
        test_model = get_model_config("efficientnet_b0")
        print(f"✓ Test model: {test_model.name} ({test_model.parameters:,} parameters)")
        
        return True
    except Exception as e:
        print(f"✗ Model config test failed: {e}")
        return False

def test_data_loader():
    """Test data loading functionality"""
    print("\n📊 Testing data loader...")
    
    try:
        from data_loader import COINS, find_candlestick_data, list_available_data
        
        print(f"✓ Found {len(COINS)} coins configured")
        
        # Test data discovery
        print("✓ Testing data discovery...")
        
        # Look for any available data
        data_found = False
        for coin in ["BTCUSDT", "ETHUSDT"]:
            for period in ["7days", "14days"]:
                for window_size in [5, 15]:
                    for exp_type in ["regular", "fullimage"]:
                        try:
                            data_files = find_candlestick_data(coin, period, window_size, exp_type)
                            if data_files:
                                print(f"✓ Found data: {coin} - {period} - w{window_size} - {exp_type} ({len(data_files)} files)")
                                data_found = True
                                break
                        except:
                            continue
                    if data_found:
                        break
                if data_found:
                    break
            if data_found:
                break
        
        if not data_found:
            print("⚠️  No candlestick data found. You may need to generate data first:")
            print("   python merged_candlestick.py --experiment regular")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loader test failed: {e}")
        return False

def test_benchmark_utils():
    """Test benchmark utilities"""
    print("\n🛠️  Testing benchmark utilities...")
    
    try:
        from benchmark_utils import get_device_info, ModelAdapter
        
        # Test device info
        device_info = get_device_info()
        print(f"✓ Device: {device_info.get('device_type', 'unknown')}")
        
        # Test model adapter (without actually loading a model)
        print("✓ Model adapter class available")
        
        return True
        
    except Exception as e:
        print(f"✗ Benchmark utils test failed: {e}")
        return False

def test_directory_structure():
    """Test directory structure"""
    print("\n📁 Testing directory structure...")
    
    required_dirs = ["benchmarks", "benchmarks/results", "benchmarks/models", "benchmarks/reports"]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"! Creating {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
    
    return True

def main():
    print("🚀 HuggingFace Benchmarking Framework Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Model Configurations", test_model_configs),
        ("Data Loader", test_data_loader),
        ("Benchmark Utils", test_benchmark_utils),
        ("Directory Structure", test_directory_structure),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 All tests passed! The framework is ready to use.")
        print("\nNext steps:")
        print("1. Generate candlestick data (if not done already):")
        print("   python merged_candlestick.py --experiment regular")
        print("\n2. Run a quick benchmark:")
        print("   python huggingface_benchmark.py --model-set quick_test --max-samples 200 --epochs 2")
        print("\n3. List available data:")
        print("   python huggingface_benchmark.py --list-data")
    else:
        print(f"⚠️  {total - passed} tests failed. Please fix the issues before running benchmarks.")
        print("\nCommon fixes:")
        print("- Install requirements: pip install -r requirements_benchmark.txt")
        print("- Generate data: python merged_candlestick.py --experiment regular")
    
    print("=" * 60)

if __name__ == "__main__":
    main()