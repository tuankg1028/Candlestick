#!/usr/bin/env python3
"""
Installation script for HuggingFace Model Benchmarking Framework
Handles version compatibility issues and provides fallback options
"""

import subprocess
import sys
import os

def run_command(command, description=""):
    """Run a command and handle errors gracefully"""
    print(f"{'='*50}")
    if description:
        print(f"Installing: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✓ SUCCESS")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def install_requirements():
    """Install requirements with fallback options"""
    
    print("🚀 Installing HuggingFace Model Benchmarking Framework Requirements")
    print("=" * 80)
    
    success_count = 0
    total_count = 0
    
    # Core requirements that should work
    core_requirements = [
        ("numpy>=1.20.0", "NumPy for numerical computing"),
        ("pandas>=1.2.0", "Pandas for data manipulation"),
        ("pillow>=8.0.0", "PIL for image processing"),
        ("matplotlib>=3.3.0", "Matplotlib for plotting"),
        ("seaborn>=0.11.0", "Seaborn for advanced plotting"),
        ("scikit-learn>=0.24.0", "Scikit-learn for ML metrics"),
        ("psutil>=5.7.0", "PSUtil for system monitoring"),
        ("requests>=2.20.0", "Requests for HTTP operations"),
    ]
    
    # ML libraries (may need specific versions)
    ml_requirements = [
        ("torch>=1.12.0", "PyTorch for deep learning"),
        ("torchvision>=0.13.0", "TorchVision for computer vision"),
        ("transformers>=4.20.0", "HuggingFace Transformers"),
        ("timm>=0.8.0", "TIMM for image models"),
    ]
    
    # Problematic packages with fallbacks
    special_requirements = [
        ("mplfinance", "MPLFinance for candlestick plots"),
    ]
    
    print("Installing core requirements...")
    for req, desc in core_requirements:
        total_count += 1
        if run_command(f"pip install '{req}'", desc):
            success_count += 1
    
    print("\nInstalling ML requirements...")
    for req, desc in ml_requirements:
        total_count += 1
        if run_command(f"pip install '{req}'", desc):
            success_count += 1
    
    print("\nInstalling special requirements...")
    for req, desc in special_requirements:
        total_count += 1
        # Try multiple approaches for mplfinance
        if req == "mplfinance":
            attempts = [
                "pip install mplfinance",
                "pip install mplfinance --pre",
                "pip install mplfinance==0.12.10b0",
                "pip install 'mplfinance>=0.12.9b0'",
            ]
            
            success = False
            for attempt in attempts:
                print(f"\nTrying: {attempt}")
                if run_command(attempt, f"{desc} (attempt)"):
                    success = True
                    break
            
            if success:
                success_count += 1
        else:
            if run_command(f"pip install '{req}'", desc):
                success_count += 1
    
    print("\n" + "=" * 80)
    print("INSTALLATION SUMMARY")
    print("=" * 80)
    print(f"Successfully installed: {success_count}/{total_count} packages")
    
    if success_count == total_count:
        print("🎉 All requirements installed successfully!")
        print("\nYou can now run:")
        print("  python huggingface_benchmark.py --model-set quick_test")
    else:
        print("⚠️  Some packages failed to install.")
        print("\nYou can try manual installation:")
        print("  pip install -r requirements_benchmark_compatible.txt")
        print("\nOr install mplfinance manually:")
        print("  pip install mplfinance --pre")
    
    print("\n" + "=" * 80)

def check_installation():
    """Check if key packages are installed correctly"""
    print("🔍 Checking installation...")
    
    test_imports = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("timm", "TIMM"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("sklearn", "Scikit-learn"),
        ("PIL", "Pillow"),
        ("mplfinance", "MPLFinance"),
    ]
    
    success_count = 0
    
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✓ {name}")
            success_count += 1
        except ImportError as e:
            print(f"✗ {name}: {e}")
    
    print(f"\nImport test: {success_count}/{len(test_imports)} successful")
    
    if success_count == len(test_imports):
        print("🎉 All packages are working correctly!")
        return True
    else:
        print("⚠️  Some packages are missing or not working.")
        return False

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        check_installation()
    else:
        install_requirements()
        print("\nChecking installation...")
        check_installation()

if __name__ == "__main__":
    main()