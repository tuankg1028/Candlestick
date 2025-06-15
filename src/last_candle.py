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

# Set BASE_DIR to absolute path relative to script location
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crypto_research_minute")

# Binance API data fetcher (fixed to 1m interval)
def fetch_coin_data(symbol, start_time, end_time):
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    current_start = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    
    while current_start < end_ms:
        params = {"symbol": symbol, "interval": "1m", "startTime": current_start, "endTime": end_ms, "limit": 1000}
        response = requests.get(url, params=params)
        data = response.json()
        if not data:
            break
        all_data.extend(data)
        current_start = int(data[-1][0]) + 60000  # 1 minute in milliseconds
    
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    return df[["timestamp", "open", "high", "low", "close"]]

# Generate candlestick images and labels with variable window size
def generate_images(df, window_size, output_dir, period_name, month_str):
    os.makedirs(output_dir, exist_ok=True)
    labels_file = os.path.join(output_dir, f"labels_{month_str}_1m_{period_name}_w{window_size}.csv")
    if os.path.exists(labels_file):
        print(f"Labels already exist at {labels_file}, skipping image generation")
        return labels_file
    
    labels = []
    start_time = time.time()
    for i in range(window_size - 1, len(df)):
        window_df = df.iloc[i - (window_size - 1):i + 1]
        last_candle = window_df.iloc[-1]
        label = "UP" if last_candle["close"] > last_candle["open"] else "DOWN"
        labels.append(label)
        
        plt.figure(figsize=(2, 2))
        mpf.plot(window_df, type="candle", style="binance", axisoff=True, title="", ylabel="", xlabel="", volume=False)
        plt.tight_layout(pad=0)
        image_path = os.path.join(output_dir, f"candle_{i}.png")
        plt.savefig(image_path, bbox_inches="tight", pad_inches=0, dpi=32)
        plt.close('all')  # Explicitly close all figures
        
        if i % 1000 == 0:
            elapsed = time.time() - start_time
            images_generated = i - (window_size - 1) + 1
            speed = images_generated / elapsed if elapsed > 0 else 0
            print(f"Generated image {i}/{len(df)} for {month_str} 1m {period_name} w{window_size} ({speed:.2f} images/sec)")
    
    labels_df = pd.DataFrame({"image_path": [f"candle_{i}.png" for i in range(window_size - 1, len(df))], "label": labels})
    labels_df.to_csv(labels_file, index=False)
    print(f"Saved {len(labels_df)} labels to {labels_file}")
    return labels_file

# Load and preprocess images
def load_images(labels_file, images_dir):
    labels_df = pd.read_csv(labels_file)
    X = np.array([np.array(Image.open(os.path.join(images_dir, row["image_path"])).convert("RGB").resize((64, 64))) / 255.0 for _, row in labels_df.iterrows()])
    y = np.array([1 if label == "UP" else 0 for label in labels_df["label"]])
    return X, y

# Train CNN model
def train_model(X, y, period_name, month_str, window_size, coin_dir):
    model_path = os.path.join(coin_dir, "models", f"model_{month_str}_1m_{period_name}_w{window_size}.h5")
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

# Evaluate and save results
def evaluate_and_save(model, X, y, period_name, month_str, window_size, coin_dir, dataset_type="train", exp_suffix=""):
    results_file = os.path.join(coin_dir, "results", f"results_{dataset_type}_{month_str}_1m_{period_name}_w{window_size}{exp_suffix}.txt")
    if os.path.exists(results_file) and exp_suffix != "_exp2":  # Force write for Experiment II
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
        f.write(f"{dataset_type.capitalize()} Metrics for {month_str} 1m {period_name} w{window_size} {exp_suffix}:\n")
        for k, v in metrics.items():
            f.write(f"{k.capitalize()}: {v:.4f}\n")
    print(f"Results saved to {results_file}")
    return metrics

# Check if all experiments for a window size are complete
def is_window_size_complete(symbol, train_month, test_months, window_size):
    coin_dir = os.path.join(BASE_DIR, symbol)
    train_year, train_month_num = train_month
    train_month_str = f"{train_year}-{train_month_num:02d}"
    
    # Check Experiment I
    for days in TIME_LENGTHS:
        period_name = f"{days}days"
        # Train results
        train_result = os.path.join(coin_dir, "results", f"results_train_{train_month_str}_1m_{period_name}_w{window_size}.txt")
        if not os.path.exists(train_result):
            return False
        # Test results for each volatile month
        for test_year, test_month_num in test_months:
            test_month_str = f"{test_year}-{test_month_num:02d}"
            test_result = os.path.join(coin_dir, "results", f"results_test_{test_month_str}_1m_{period_name}_w{window_size}.txt")
            if not os.path.exists(test_result):
                return False
    
    # Check Experiment II
    period_name = "1week"
    train_result = os.path.join(coin_dir, "results", f"results_train_{train_month_str}_1m_{period_name}_w{window_size}_exp2.txt")
    if not os.path.exists(train_result):
        return False
    for test_year, test_month_num in test_months:
        test_month_str = f"{test_year}-{test_month_num:02d}"
        for days in [14, 21, 28]:  # 2, 3, 4 weeks
            period_name = f"{days}days"
            test_result = os.path.join(coin_dir, "results", f"results_test_{test_month_str}_1m_{period_name}_w{window_size}_exp2.txt")
            if not os.path.exists(test_result):
                return False
    
    return True

# Main experiment runner for a single coin and window size
def run_experiments_for_coin(symbol, train_month, test_months, window_size):
    if is_window_size_complete(symbol, train_month, test_months, window_size):
        print(f"All experiments for {symbol} with window size {window_size} are complete, skipping")
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
    
    # Experiment I: Train and test on matching timelengths
    for days in TIME_LENGTHS:
        period_name = f"{days}days"
        train_start = datetime(train_year, train_month_num, 1)
        train_end = train_start + timedelta(days=days - 1, hours=23, minutes=59)
        train_month_str = f"{train_year}-{train_month_num:02d}"
        
        raw_file = os.path.join(RAW_DATA_DIR, f"raw_{train_month_str}_1m_{period_name}.csv")
        if not os.path.exists(raw_file):
            df = fetch_coin_data(symbol, train_start, train_end)
            df.set_index("timestamp", inplace=True)
            df.to_csv(raw_file)
            print(f"Raw data saved to {raw_file}")
        else:
            print(f"Raw data already exists at {raw_file}, skipping fetch")
            df = pd.read_csv(raw_file, index_col="timestamp", parse_dates=["timestamp"])
            df.index = pd.to_datetime(df.index)  # Ensure DatetimeIndex
        
        images_subdir = os.path.join(IMAGES_DIR, f"{train_month_str}_1m_{period_name}_w{window_size}")
        labels_file = generate_images(df, window_size, images_subdir, period_name, train_month_str)
        X, y = load_images(labels_file, images_subdir)
        model, history = train_model(X, y, period_name, train_month_str, window_size, coin_dir)
        evaluate_and_save(model, X, y, period_name, train_month_str, window_size, coin_dir, "train")
        
        # Clear TensorFlow resources
        tf.keras.backend.clear_session()
        gc.collect()
        
        for test_year, test_month_num in test_months:
            test_start = datetime(test_year, test_month_num, 1)
            test_end = test_start + timedelta(days=days - 1, hours=23, minutes=59)
            test_month_str = f"{test_year}-{test_month_num:02d}"
            
            raw_file = os.path.join(RAW_DATA_DIR, f"raw_{test_month_str}_1m_{period_name}.csv")
            if not os.path.exists(raw_file):
                df = fetch_coin_data(symbol, test_start, test_end)
                df.set_index("timestamp", inplace=True)
                df.to_csv(raw_file)
                print(f"Raw data saved to {raw_file}")
            else:
                print(f"Raw data already exists at {raw_file}, skipping fetch")
                df = pd.read_csv(raw_file, index_col="timestamp", parse_dates=["timestamp"])
                df.index = pd.to_datetime(df.index)  # Ensure DatetimeIndex
            
            images_subdir = os.path.join(IMAGES_DIR, f"{test_month_str}_1m_{period_name}_w{window_size}")
            labels_file = generate_images(df, window_size, images_subdir, period_name, test_month_str)
            X, y = load_images(labels_file, images_subdir)
            evaluate_and_save(model, X, y, period_name, test_month_str, window_size, coin_dir, "test")
        
        # Clear TensorFlow resources again
        tf.keras.backend.clear_session()
        gc.collect()
    
    # Experiment II: Train on 1 week, test on 2-3-4 weeks
    exp2_test_lengths = [14, 21, 28]  # 2, 3, 4 weeks
    train_start = datetime(train_year, train_month_num, 1)
    train_end = train_start + timedelta(days=6, hours=23, minutes=59)  # 1 week
    train_month_str = f"{train_year}-{train_month_num:02d}"
    period_name = "1week"
    
    raw_file = os.path.join(RAW_DATA_DIR, f"raw_{train_month_str}_1m_{period_name}.csv")
    if not os.path.exists(raw_file):
        df = fetch_coin_data(symbol, train_start, end_time=train_end)
        df.set_index("timestamp", inplace=True)
        df.to_csv(raw_file)
        print(f"Raw data saved to {raw_file}")
    else:
        print(f"Raw data already exists at {raw_file}, skipping fetch")
        df = pd.read_csv(raw_file, index_col="timestamp", parse_dates=["timestamp"])
        df.index = pd.to_datetime(df.index)  # Ensure DatetimeIndex
    
    images_subdir = os.path.join(IMAGES_DIR, f"{train_month_str}_1m_{period_name}_w{window_size}")
    labels_file = generate_images(df, window_size, images_subdir, period_name, train_month_str)
    X, y = load_images(labels_file, images_subdir)
    model, history = train_model(X, y, period_name, train_month_str, window_size, coin_dir)
    evaluate_and_save(model, X, y, period_name, train_month_str, window_size, coin_dir, "train", "_exp2")
    
    # Clear TensorFlow resources
    tf.keras.backend.clear_session()
    gc.collect()
    
    for test_year, test_month_num in test_months:
        test_month_str = f"{test_year}-{test_month_num:02d}"
        for days in exp2_test_lengths:
            period_name = f"{days}days"
            test_start = datetime(test_year, test_month_num, 1)
            test_end = test_start + timedelta(days=days - 1, hours=23, minutes=59)
            
            raw_file = os.path.join(RAW_DATA_DIR, f"raw_{test_month_str}_1m_{period_name}.csv")
            if not os.path.exists(raw_file):
                df = fetch_coin_data(symbol, test_start, test_end)
                df.set_index("timestamp", inplace=True)
                df.to_csv(raw_file)
                print(f"Raw data saved to {raw_file}")
            else:
                print(f"Raw data already exists at {raw_file}, skipping fetch")
                df = pd.read_csv(raw_file, index_col="timestamp", parse_dates=["timestamp"])
                df.index = pd.to_datetime(df.index)  # Ensure DatetimeIndex
            
            images_subdir = os.path.join(IMAGES_DIR, f"{test_month_str}_1m_{period_name}_w{window_size}")
            labels_file = generate_images(df, window_size, images_subdir, period_name, test_month_str)
            X, y = load_images(labels_file, images_subdir)
            evaluate_and_save(model, X, y, period_name, test_month_str, window_size, coin_dir, "test", "_exp2")
        
        # Clear TensorFlow resources
        tf.keras.backend.clear_session()
        gc.collect()

# Run experiments for all coins and window sizes
def run_all_experiments():
    os.makedirs(BASE_DIR, exist_ok=True)  # Ensure BASE_DIR exists
    for symbol, config in COINS.items():
        for window_size in WINDOW_SIZES:
            print(f"Running experiments for {symbol} with window size {window_size}")
            run_experiments_for_coin(symbol, config["train_month"], config["test_months"], window_size)
            print(f"Completed experiments for {symbol} with window size {window_size}")
            # Clear TensorFlow resources between window sizes
            tf.keras.backend.clear_session()
            gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Minute-Based Image Classification Research for Multiple Coins")
    args = parser.parse_args()
    run_all_experiments()