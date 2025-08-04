"""
Standalone data loading functions for candlestick image classification
Extracted from merged_candlestick.py for use in benchmarking framework
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import psutil

# Load single image function for parallel processing
def load_single_image(args):
    """Load a single image for parallel processing"""
    image_path, images_dir = args
    try:
        img = Image.open(os.path.join(images_dir, image_path)).convert("RGB").resize((64, 64))
        return np.array(img) / 255.0
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return np.zeros((64, 64, 3))

def load_images_parallel(labels_file, images_dir, max_workers=None):
    """Load images in parallel using multiprocessing"""
    if not os.path.exists(labels_file):
        return None, None
    
    labels_df = pd.read_csv(labels_file)
    
    if max_workers is None:
        max_workers = min(psutil.cpu_count(), 8)  # Limit to 8 processes
    
    # Prepare arguments for parallel processing
    image_args = [(row["image_path"], images_dir) for _, row in labels_df.iterrows()]
    
    # Load images in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        X = list(executor.map(load_single_image, image_args))
    
    X = np.array(X)
    y = np.array([1 if label == "UP" else 0 for label in labels_df["label"]])
    return X, y

def load_images(labels_file, images_dir):
    """Load images sequentially (fallback for parallel loading)"""
    if not os.path.exists(labels_file):
        return None, None
    
    labels_df = pd.read_csv(labels_file)
    X = np.array([np.array(Image.open(os.path.join(images_dir, row["image_path"])).convert("RGB").resize((64, 64))) / 255.0 for _, row in labels_df.iterrows()])
    y = np.array([1 if label == "UP" else 0 for label in labels_df["label"]])
    return X, y

# Coin configurations (copied from merged_candlestick.py)
COINS = {
    "BTCUSDT": {"train_month": (2024, 6), "test_months": [(2024, 12), (2024, 3), (2024, 8), (2024, 4), (2024, 1)]},
    "ETHUSDT": {"train_month": (2024, 6), "test_months": [(2024, 8), (2024, 4), (2024, 5), (2024, 3), (2024, 2)]},
    "BNBUSDT": {"train_month": (2024, 10), "test_months": [(2024, 3), (2024, 12), (2024, 8), (2024, 1), (2024, 4)]},
    "XRPUSDT": {"train_month": (2024, 9), "test_months": [(2024, 11), (2024, 12), (2024, 4), (2024, 8), (2024, 1)]},
    "ADAUSDT": {"train_month": (2024, 9), "test_months": [(2024, 4), (2024, 12), (2024, 1), (2024, 3), (2024, 11)]},
    "DOGEUSDT": {"train_month": (2024, 9), "test_months": [(2024, 3), (2024, 4), (2024, 11), (2024, 8), (2024, 12)]}
}

TIME_LENGTHS = [7, 14, 21, 28]  # 1, 2, 3, 4 weeks in days
WINDOW_SIZES = [5, 15, 30]  # Candles per image

def find_candlestick_data(coin, period, window_size, experiment_type="regular", base_path="."):
    """Find available candlestick data files"""
    
    # Determine base directory based on experiment type
    if experiment_type == "regular":
        base_dir = os.path.join(base_path, "crypto_research_minute")
    elif experiment_type == "fullimage":
        base_dir = os.path.join(base_path, "crypto_research_minute_fullimage")
    elif experiment_type == "irregular":
        base_dir = os.path.join(base_path, "crypto_research_minute_irregular")
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    # Construct paths
    coin_dir = os.path.join(base_dir, coin)
    images_dir = os.path.join(coin_dir, "images")
    
    if not os.path.exists(images_dir):
        return []
    
    # Try to find existing data
    potential_paths = []
    for year in [2024, 2023]:  # Try multiple years
        for month in range(1, 13):
            month_str = f"{year}-{month:02d}"
            subdir = f"{month_str}_1m_{period}_w{window_size}"
            
            # Try different label file patterns
            label_patterns = [
                f"labels_{month_str}_1m_{period}_w{window_size}.csv",
                f"labels_{month_str}_1m_{period}_w{window_size}_60pct.csv",
                f"labels_{month_str}_1m_{period}_w{window_size}_80pct.csv",
                f"labels_{month_str}_1m_{period}_w{window_size}_95pct.csv",
            ]
            
            for pattern in label_patterns:
                labels_file = os.path.join(images_dir, subdir, pattern)
                if os.path.exists(labels_file):
                    potential_paths.append({
                        'labels_file': labels_file,
                        'images_dir': os.path.join(images_dir, subdir),
                        'month': month_str,
                        'pattern': pattern
                    })
    
    return potential_paths

def load_candlestick_data(coin, period, window_size, experiment_type="regular", max_samples=None):
    """Load candlestick data with automatic file discovery"""
    
    # Find available data files
    data_files = find_candlestick_data(coin, period, window_size, experiment_type)
    
    if not data_files:
        raise FileNotFoundError(f"No candlestick data found for {coin}, {period}, window size {window_size}, experiment type {experiment_type}")
    
    print(f"Found {len(data_files)} data files for {coin}")
    
    # Load the first available dataset (you could modify this to load multiple or choose specific ones)
    data_file = data_files[0]
    labels_file = data_file['labels_file']
    images_path = data_file['images_dir']
    
    print(f"Loading data from: {labels_file}")
    
    try:
        # Try parallel loading first, fall back to sequential if it fails
        try:
            X, y = load_images_parallel(labels_file, images_path)
        except Exception as e:
            print(f"Parallel loading failed ({e}), trying sequential loading...")
            X, y = load_images(labels_file, images_path)
        
        if X is None or len(X) == 0:
            raise ValueError("No data loaded from files")
        
        # Limit samples if specified
        if max_samples and len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X, y = X[indices], y[indices]
            print(f"Limited to {max_samples} samples")
        
        print(f"Successfully loaded {len(X)} samples with shape {X[0].shape}")
        return X, y
        
    except Exception as e:
        raise RuntimeError(f"Error loading candlestick data: {str(e)}")

def list_available_data(base_path="."):
    """List all available candlestick data"""
    experiment_types = ["regular", "fullimage", "irregular"]
    
    print("Available Candlestick Data:")
    print("=" * 50)
    
    for exp_type in experiment_types:
        print(f"\n{exp_type.upper()} DATA:")
        
        for coin in COINS.keys():
            for period in [f"{days}days" for days in TIME_LENGTHS]:
                for window_size in WINDOW_SIZES:
                    try:
                        data_files = find_candlestick_data(coin, period, window_size, exp_type, base_path)
                        if data_files:
                            print(f"  ✓ {coin} - {period} - w{window_size} ({len(data_files)} files)")
                    except:
                        continue

if __name__ == "__main__":
    # Test the data loading functions
    print("Testing candlestick data loading...")
    
    # List available data
    list_available_data()
    
    # Try to load some data
    try:
        X, y = load_candlestick_data("BTCUSDT", "7days", 5, "regular", max_samples=10)
        print(f"\nTest successful! Loaded {len(X)} samples")
        print(f"Sample shape: {X[0].shape}")
        print(f"Labels: {y}")
    except Exception as e:
        print(f"\nTest failed: {e}")
        print("Make sure you have generated candlestick data first using:")
        print("python merged_candlestick.py --experiment regular")