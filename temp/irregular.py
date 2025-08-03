import requests
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_recall_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import argparse
import gc
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import psutil

# Configure GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Memory growth prevents TF from allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Use mixed precision for faster training
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print(f"GPU acceleration enabled with {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")

# Use non-interactive backend for matplotlib
plt.switch_backend('Agg')

# Coin configurations
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
MISSING_RATIOS = [0.6, 0.8, 0.95]  # 60%, 80%, 95% missing data

# Set BASE_DIR to new folder for irregular data
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crypto_research_minute_irregular")

# Binance API data fetcher with irregular data omission
# def fetch_coin_data(symbol, start_time, end_time, missing_ratio):
#     url = "https://api.binance.com/api/v3/klines"
#     all_data = []
#     current_start = int(start_time.timestamp() * 1000)
#     end_ms = int(end_time.timestamp() * 1000)
    
#     while current_start < end_ms:
#         params = {"symbol": symbol, "interval": "1m", "startTime": current_start, "endTime": end_ms, "limit": 1000}
#         response = requests.get(url, params=params)
#         data = response.json()
#         if not data:
#             break
#         all_data.extend(data)
#         current_start = int(data[-1][0]) + 60000  # 1 minute in milliseconds
    
#     df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
#     df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
#     df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    
#     # Apply irregular data omission
#     if missing_ratio > 0:
#         n_rows = len(df)
#         n_keep = int(n_rows * (1 - missing_ratio))
#         if n_keep < 1:  # Allow at least 1 row
#             print(f"Warning: Not enough data after {missing_ratio*100}% omission for {symbol}, keeping all data")
#             return df[["timestamp", "open", "high", "low", "close"]]
#         keep_indices = np.random.choice(n_rows, size=n_keep, replace=False)
#         df = df.iloc[keep_indices].sort_values("timestamp").reset_index(drop=True)
    
#     return df[["timestamp", "open", "high", "low", "close"]]

# Generate candlestick images and labels with sparse windows
def generate_images(df, window_size, output_dir, period_name, month_str, missing_ratio):
    os.makedirs(output_dir, exist_ok=True)
    labels_file = os.path.join(output_dir, f"labels_{month_str}_1m_{period_name}_w{window_size}_{int(missing_ratio*100)}pct.csv")
    if os.path.exists(labels_file):
        print(f"Labels already exist at {labels_file}, skipping image generation")
        return labels_file
    
    if len(df) < 1:
        print(f"Warning: DataFrame too small ({len(df)} rows) for any window, skipping image generation")
        return None
    
    labels = []
    start_time = time.time()
    # Use index as timestamps since it's set as index
    original_timestamps = pd.date_range(start=df.index[0], end=df.index[-1], freq="1min")
    
    for i in range(len(original_timestamps) - window_size + 1):
        window_start = original_timestamps[i]
        window_end = original_timestamps[i + window_size - 1]
        window_indices = df.index[(df.index >= window_start) & (df.index <= window_end)]
        window_df = df.loc[window_indices]
        
        if len(window_df) > 0:
            first_candle = window_df.iloc[0]
            last_candle = window_df.iloc[-1]
            label = "UP" if last_candle["close"] > first_candle["open"] else "DOWN"
            labels.append(label)
            
            plt.figure(figsize=(2, 2))
            mpf.plot(window_df, type="candle", style="binance", axisoff=True, title="", ylabel="", xlabel="", volume=False, tight_layout=True)
            image_path = os.path.join(output_dir, f"candle_{i}_{int(missing_ratio*100)}pct.png")
            plt.savefig(image_path, bbox_inches="tight", pad_inches=0, dpi=32)
            plt.close('all')
            
            if i % 1000 == 0:
                elapsed = time.time() - start_time
                images_generated = i + 1
                speed = images_generated / elapsed if elapsed > 0 else 0
                print(f"Generated image {i}/{len(original_timestamps) - window_size + 1} for {month_str} 1m {period_name} w{window_size} {missing_ratio*100}% ({speed:.2f} images/sec)")
        else:
            continue
    
    labels_df = pd.DataFrame({"image_path": [f"candle_{i}_{int(missing_ratio*100)}pct.png" for i in range(len(original_timestamps) - window_size + 1) if os.path.exists(os.path.join(output_dir, f"candle_{i}_{int(missing_ratio*100)}pct.png"))], "label": labels})
    labels_df.to_csv(labels_file, index=False)
    print(f"Saved {len(labels_df)} labels to {labels_file}")
    return labels_file

# Load and preprocess images
def load_images(labels_file, images_dir):
    if not os.path.exists(labels_file):
        return None, None
    labels_df = pd.read_csv(labels_file)
    X = np.array([np.array(Image.open(os.path.join(images_dir, row["image_path"])).convert("RGB").resize((64, 64))) / 255.0 for _, row in labels_df.iterrows()])
    y = np.array([1 if label == "UP" else 0 for label in labels_df["label"]])
    return X, y

# Generate candlestick images and labels with sparse windows using batch processing
def generate_images_batch(df, window_size, output_dir, period_name, month_str, missing_ratio, batch_size=200):
    os.makedirs(output_dir, exist_ok=True)
    labels_file = os.path.join(output_dir, f"labels_{month_str}_1m_{period_name}_w{window_size}_{int(missing_ratio*100)}pct.csv")
    if os.path.exists(labels_file):
        print(f"Labels already exist at {labels_file}, skipping image generation")
        return labels_file
    
    if len(df) < 1:
        print(f"Warning: DataFrame too small ({len(df)} rows) for any window, skipping image generation")
        return None
    
    labels = []
    image_paths = []
    start_time = time.time()
    
    # Use index as timestamps since it's set as index
    original_timestamps = pd.date_range(start=df.index[0], end=df.index[-1], freq="1min")
    total_windows = len(original_timestamps) - window_size + 1
    
    # Process in batches to reduce memory pressure
    for batch_start in range(0, total_windows, batch_size):
        batch_end = min(batch_start + batch_size, total_windows)
        batch_labels = []
        batch_paths = []
        
        for i in range(batch_start, batch_end):
            window_start = original_timestamps[i]
            window_end = original_timestamps[i + window_size - 1]
            window_indices = df.index[(df.index >= window_start) & (df.index <= window_end)]
            window_df = df.loc[window_indices]
            
            if len(window_df) > 0:
                first_candle = window_df.iloc[0]
                last_candle = window_df.iloc[-1]
                label = "UP" if last_candle["close"] > first_candle["open"] else "DOWN"
                image_path = f"candle_{i}_{int(missing_ratio*100)}pct.png"
                batch_labels.append(label)
                batch_paths.append(image_path)
                
                # Use a smaller figure size and lower DPI for speed
                fig = plt.figure(figsize=(1.5, 1.5))
                mpf.plot(window_df, type="candle", style="binance", axisoff=True, 
                         title="", ylabel="", xlabel="", volume=False)
                plt.tight_layout(pad=0)
                plt.savefig(os.path.join(output_dir, image_path), 
                            bbox_inches="tight", pad_inches=0, dpi=24)
                plt.close(fig)
            
        labels.extend(batch_labels)
        image_paths.extend(batch_paths)
        
        if batch_end % 500 == 0 or batch_end == total_windows:
            elapsed = time.time() - start_time
            speed = batch_end / elapsed if elapsed > 0 else 0
            print(f"Generated {batch_end}/{total_windows} images for {month_str} 1m {period_name} w{window_size} {missing_ratio*100}% ({speed:.2f} images/sec)")
        
        # Clear memory after each batch
        gc.collect()
    
    # Only save labels for images that were successfully created
    valid_indices = [i for i, path in enumerate(image_paths) 
                     if os.path.exists(os.path.join(output_dir, path))]
    
    if valid_indices:
        valid_paths = [image_paths[i] for i in valid_indices]
        valid_labels = [labels[i] for i in valid_indices]
        
        labels_df = pd.DataFrame({"image_path": valid_paths, "label": valid_labels})
        labels_df.to_csv(labels_file, index=False)
        print(f"Saved {len(labels_df)} labels to {labels_file}")
        return labels_file
    else:
        print(f"No valid images generated for {month_str} 1m {period_name} w{window_size} {missing_ratio*100}%")
        return None

# Parallel image loading with multiprocessing
def load_single_image(args):
    image_path, images_dir = args
    try:
        img = Image.open(os.path.join(images_dir, image_path)).convert("RGB").resize((64, 64))
        return np.array(img) / 255.0
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return np.zeros((64, 64, 3))

def load_images_parallel(labels_file, images_dir, max_workers=None):
    if not os.path.exists(labels_file):
        return None, None
    
    labels_df = pd.read_csv(labels_file)
    
    if max_workers is None:
        max_workers = min(psutil.cpu_count(), 8)  # Limit to 8 processes
    
    # Prepare arguments for parallel processing
    image_args = [(row["image_path"], images_dir) for _, row in labels_df.iterrows()]
    
    # Use parallel processing for image loading
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        X = list(executor.map(load_single_image, image_args))
    
    X = np.array(X)
    y = np.array([1 if label == "UP" else 0 for label in labels_df["label"]])
    return X, y

# Train CNN model
def train_model(X, y, period_name, month_str, window_size, coin_dir, missing_ratio):
    model_path = os.path.join(coin_dir, "models", f"model_{month_str}_1m_{period_name}_w{window_size}_{int(missing_ratio*100)}pct.h5")
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}, loading instead of training")
        return tf.keras.models.load_model(model_path), None
    
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    history = model.fit(X, y, epochs=10, batch_size=32, class_weight=dict(enumerate(class_weights)))
    
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model, history

# Optimized model training with GPU acceleration and early stopping
def train_model_optimized(X, y, period_name, month_str, window_size, coin_dir, missing_ratio):
    model_path = os.path.join(coin_dir, "models", f"model_{month_str}_1m_{period_name}_w{window_size}_{int(missing_ratio*100)}pct.h5")
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}, loading instead of training")
        return tf.keras.models.load_model(model_path), None
    
    # Optimized model architecture with batch normalization
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.GlobalAveragePooling2D(),  # More efficient than Flatten
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])
    
    # Use mixed precision optimizer if GPU is available
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    
    # Add callbacks for better training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
    ]
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    history = model.fit(X, y, epochs=10, batch_size=64, class_weight=dict(enumerate(class_weights)),
                       callbacks=callbacks, verbose=1)
    
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model, history

# Evaluate and save results
def evaluate_and_save(model, X, y, period_name, month_str, window_size, coin_dir, dataset_type="train", exp_suffix="", missing_ratio=0):
    results_file = os.path.join(coin_dir, "results", f"results_{dataset_type}_{month_str}_1m_{period_name}_w{window_size}_{int(missing_ratio*100)}pct{exp_suffix}.txt")
    if os.path.exists(results_file) and exp_suffix != "_exp2":
        print(f"Results already exist at {results_file}, skipping evaluation")
        return None
    
    y_pred_prob = model.predict(X, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "auroc": roc_auc_score(y, y_pred_prob),
        "auprc": auc(*precision_recall_curve(y, y_pred_prob)[1::-1])
    }
    
    with open(results_file, "w") as f:
        f.write(f"{dataset_type.capitalize()} Metrics for {month_str} 1m {period_name} w{window_size} {missing_ratio*100}% {exp_suffix}:\n")
        for k, v in metrics.items():
            f.write(f"{k.capitalize()}: {v:.4f}\n")
    print(f"Results saved to {results_file}")
    return metrics

# Check if all experiments for a window size and missing ratio are complete
def is_window_size_complete(symbol, train_month, test_months, window_size, missing_ratio):
    # coin_dir = os.path.join(BASE_DIR, symbol)
    # train_year, train_month_num = train_month
    # train_month_str = f"{train_year}-{train_month_num:02d}"
    # ratio_str = f"_{int(missing_ratio*100)}pct"
    
    # # Check Experiment I
    # for days in TIME_LENGTHS:
    #     period_name = f"{days}days"
    #     train_result = os.path.join(coin_dir, "results", f"results_train_{train_month_str}_1m_{period_name}_w{window_size}{ratio_str}.txt")
    #     if not os.path.exists(train_result):
    #         return False
    #     for test_year, test_month_num in test_months:
    #         test_month_str = f"{test_year}-{test_month_num:02d}"
    #         test_result = os.path.join(coin_dir, "results", f"results_test_{test_month_str}_1m_{period_name}_w{window_size}{ratio_str}.txt")
    #         if not os.path.exists(test_result):
    #             return False
    
    # # Check Experiment II
    # period_name = "1week"
    # train_result = os.path.join(coin_dir, "results", f"results_train_{train_month_str}_1m_{period_name}_w{window_size}{ratio_str}_exp2.txt")
    # if not os.path.exists(train_result):
    #     return False
    # for test_year, test_month_num in test_months:
    #     test_month_str = f"{test_year}-{test_month_num:02d}"
    #     for days in [14, 21, 28]:
    #         period_name = f"{days}days"
    #         test_result = os.path.join(coin_dir, "results", f"results_test_{test_month_str}_1m_{period_name}_w{window_size}{ratio_str}_exp2.txt")
    #         if not os.path.exists(test_result):
    #             return False
    
    return False

# Single experiment worker function for parallel processing
def run_single_experiment(args):
    symbol, config, window_size, missing_ratio = args
    try:
        run_experiments_for_coin(symbol, config["train_month"], config["test_months"], window_size, missing_ratio)
        return f"Completed {symbol} w{window_size} {missing_ratio*100}%"
    except Exception as e:
        return f"Error in {symbol} w{window_size} {missing_ratio*100}%: {str(e)}"

# Main experiment runner for a single coin, window size, and missing ratio
def run_experiments_for_coin(symbol, train_month, test_months, window_size, missing_ratio):
    if is_window_size_complete(symbol, train_month, test_months, window_size, missing_ratio):
        print(f"All experiments for {symbol} with window size {window_size} and {missing_ratio*100}% missing complete, skipping")
        return
    
    coin_dir = os.path.join(BASE_DIR, symbol)
    RAW_DATA_DIR = os.path.join(coin_dir, "raw_data")
    IMAGES_DIR = os.path.join(coin_dir, "images")
    MODELS_DIR = os.path.join(coin_dir, "models")
    RESULTS_DIR = os.path.join(coin_dir, "results")
    
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    train_year, train_month_num = train_month
    ratio_str = f"_{int(missing_ratio*100)}pct"
    
    # Experiment I: Train and test on matching timelengths
    for days in TIME_LENGTHS:
        period_name = f"{days}days"
        train_start = datetime(train_year, train_month_num, 1)
        train_end = train_start + timedelta(days=days - 1, hours=23, minutes=59)
        train_month_str = f"{train_year}-{train_month_num:02d}"
        
        raw_file = os.path.join(RAW_DATA_DIR, f"raw_{train_month_str}_1m_{period_name}{ratio_str}.csv")
        if not os.path.exists(raw_file):
            print(f"Raw data file {raw_file} does not exist, skipping")
            continue
            
        print(f"Raw data already exists at {raw_file}, skipping fetch")
        df = pd.read_csv(raw_file, index_col="timestamp", parse_dates=["timestamp"])
        df.index = pd.to_datetime(df.index)
        
        images_subdir = os.path.join(IMAGES_DIR, f"{train_month_str}_1m_{period_name}_w{window_size}{ratio_str}")
        labels_file = generate_images_batch(df, window_size, images_subdir, period_name, train_month_str, missing_ratio)
        if labels_file:
            X, y = load_images_parallel(labels_file, images_subdir)
            if X is not None and len(X) > 0:
                model, history = train_model_optimized(X, y, period_name, train_month_str, window_size, coin_dir, missing_ratio)
                evaluate_and_save(model, X, y, period_name, train_month_str, window_size, coin_dir, "train", missing_ratio=missing_ratio)
        
        tf.keras.backend.clear_session()
        gc.collect()
        
        for test_year, test_month_num in test_months:
            test_start = datetime(test_year, test_month_num, 1)
            test_end = test_start + timedelta(days=days - 1, hours=23, minutes=59)
            test_month_str = f"{test_year}-{test_month_num:02d}"
            
            raw_file = os.path.join(RAW_DATA_DIR, f"raw_{test_month_str}_1m_{period_name}{ratio_str}.csv")
            if not os.path.exists(raw_file):
                df = fetch_coin_data(symbol, test_start, test_end, missing_ratio)
                df.set_index("timestamp", inplace=True)
                df.to_csv(raw_file)
                print(f"Raw data saved to {raw_file}")
            else:
                print(f"Raw data already exists at {raw_file}, skipping fetch")
                df = pd.read_csv(raw_file, index_col="timestamp", parse_dates=["timestamp"])
                df.index = pd.to_datetime(df.index)
            
            images_subdir = os.path.join(IMAGES_DIR, f"{test_month_str}_1m_{period_name}_w{window_size}{ratio_str}")
            labels_file = generate_images(df, window_size, images_subdir, period_name, test_month_str, missing_ratio)
            if labels_file:
                X, y = load_images(labels_file, images_subdir)
                if X is not None:
                    evaluate_and_save(model, X, y, period_name, test_month_str, window_size, coin_dir, "test", missing_ratio=missing_ratio)
        
        tf.keras.backend.clear_session()
        gc.collect()
    
    # Experiment II: Train on 1 week, test on 2-3-4 weeks
    exp2_test_lengths = [14, 21, 28]
    train_start = datetime(train_year, train_month_num, 1)
    train_end = train_start + timedelta(days=6, hours=23, minutes=59)
    train_month_str = f"{train_year}-{train_month_num:02d}"
    period_name = "1week"
    
    raw_file = os.path.join(RAW_DATA_DIR, f"raw_{train_month_str}_1m_{period_name}{ratio_str}.csv")
    if not os.path.exists(raw_file):
        df = fetch_coin_data(symbol, train_start, train_end, missing_ratio)
        df.set_index("timestamp", inplace=True)
        df.to_csv(raw_file)
        print(f"Raw data saved to {raw_file}")
    else:
        print(f"Raw data already exists at {raw_file}, skipping fetch")
        df = pd.read_csv(raw_file, index_col="timestamp", parse_dates=["timestamp"])
        df.index = pd.to_datetime(df.index)
    
    images_subdir = os.path.join(IMAGES_DIR, f"{train_month_str}_1m_{period_name}_w{window_size}{ratio_str}")
    labels_file = generate_images(df, window_size, images_subdir, period_name, train_month_str, missing_ratio)
    if labels_file:
        X, y = load_images(labels_file, images_subdir)
        if X is not None:
            model, history = train_model(X, y, period_name, train_month_str, window_size, coin_dir, missing_ratio)
            evaluate_and_save(model, X, y, period_name, train_month_str, window_size, coin_dir, "train", "_exp2", missing_ratio)
    
    tf.keras.backend.clear_session()
    gc.collect()
    
    for test_year, test_month_num in test_months:
        test_month_str = f"{test_year}-{test_month_num:02d}"
        for days in exp2_test_lengths:
            period_name = f"{days}days"
            test_start = datetime(test_year, test_month_num, 1)
            test_end = test_start + timedelta(days=days - 1, hours=23, minutes=59)
            
            raw_file = os.path.join(RAW_DATA_DIR, f"raw_{test_month_str}_1m_{period_name}{ratio_str}.csv")
            if not os.path.exists(raw_file):
                df = fetch_coin_data(symbol, test_start, end_time, missing_ratio)
                df.set_index("timestamp", inplace=True)
                df.to_csv(raw_file)
                print(f"Raw data saved to {raw_file}")
            else:
                print(f"Raw data already exists at {raw_file}, skipping fetch")
                df = pd.read_csv(raw_file, index_col="timestamp", parse_dates=["timestamp"])
                df.index = pd.to_datetime(df.index)
            
            images_subdir = os.path.join(IMAGES_DIR, f"{test_month_str}_1m_{period_name}_w{window_size}{ratio_str}")
            labels_file = generate_images(df, window_size, images_subdir, period_name, test_month_str, missing_ratio)
            if labels_file:
                X, y = load_images(labels_file, images_subdir)
                if X is not None:
                    evaluate_and_save(model, X, y, period_name, test_month_str, window_size, coin_dir, "test", "_exp2", missing_ratio)
        
        tf.keras.backend.clear_session()
        gc.collect()

# Run experiments in parallel across multiple coins/window sizes/ratios
def run_all_experiments_parallel():
    os.makedirs(BASE_DIR, exist_ok=True)
    
    # Create task list for parallel execution
    tasks = []
    for symbol, config in COINS.items():
        for window_size in WINDOW_SIZES:
            for missing_ratio in MISSING_RATIOS:
                tasks.append((symbol, config, window_size, missing_ratio))
    
    # Determine optimal number of workers
    max_workers = min(psutil.cpu_count() // 2, 4)  # Conservative approach
    print(f"Running {len(tasks)} experiments with {max_workers} parallel workers")
    
    # Use ThreadPoolExecutor for parallel processing of experiments
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(run_single_experiment, task): task for task in tasks}
        
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                print(f"Task completed: {result}")
            except Exception as exc:
                print(f"Task {task} generated an exception: {exc}")

# Run experiments for all coins, window sizes, and missing ratios
def run_all_experiments():
    if any(gpus):
        print("Running experiments with GPU acceleration")
        run_all_experiments_parallel()
    else:
        print("Running experiments in parallel CPU mode")
        run_all_experiments_parallel()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Minute-Based Image Classification with Irregular Missing Data and Sparse Windows")
    parser.add_argument("--parallel", action="store_true", help="Run experiments in parallel (default with GPU)")
    parser.add_argument("--no-parallel", action="store_true", help="Run experiments sequentially")
    args = parser.parse_args()
    
    if args.no_parallel:
        # Sequential mode (original code)
        os.makedirs(BASE_DIR, exist_ok=True)
        for symbol, config in COINS.items():
            for window_size in WINDOW_SIZES:
                for missing_ratio in MISSING_RATIOS:
                    print(f"Running experiments for {symbol} with window size {window_size} and {missing_ratio*100}% missing")
                    run_experiments_for_coin(symbol, config["train_month"], config["test_months"], window_size, missing_ratio)
                    print(f"Completed experiments for {symbol} with window size {window_size} and {missing_ratio*100}% missing")
                    tf.keras.backend.clear_session()
                    gc.collect()
    else:
        # Parallel mode (default)
        run_all_experiments()