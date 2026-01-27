#!/usr/bin/env python3
"""
Memory-efficient version of train_fullimage_pretrained.py with GPU support.
Dataset: Fullimage (full window labeling - last close > first open = UP)
Reuses images from regular dataset with different labels.

Key improvements:
- Uses TensorFlow Dataset and PyTorch DataLoader for lazy loading
- Added --time-length flag to process specific time periods
- Batch loading from disk instead of loading all images into RAM
- Reduced memory usage from ~24GB per dataset to ~500MB

Usage:
    # Process only 7-day periods with GPU
    python src/train_regular_pretrained_gpu.py --model mobilenetv3 --coin BTCUSDT --window 5 --time-length 7

    # Process all time lengths (will run sequentially to save memory)
    python src/train_regular_pretrained_gpu.py --model mobilenetv3 --coin BTCUSDT --window 5

    # Process only exp2 (1-week training)
    python src/train_regular_pretrained_gpu.py --model mobilenetv3 --coin BTCUSDT --window 5 --exp2-only
"""

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
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Try to import PyTorch and timm (optional dependencies)
TORCH_AVAILABLE = False
TIMM_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available. Only TensorFlow models will work.")

# Configure GPU memory growth for TensorFlow to avoid memory issues
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory growth setting failed: {e}")

# Auto-detect optimal batch size based on GPU availability
def get_optimal_batch_size(framework='tensorflow'):
    """Determine optimal batch size based on GPU availability"""
    try:
        if framework == 'tensorflow' and tf.config.list_physical_devices('GPU'):
            return 64  # Larger batch for GPU
        elif framework == 'timm' and torch.cuda.is_available():
            return 64
        else:
            return 32  # Default for CPU
    except:
        return 32

BATCH_SIZE_TF = get_optimal_batch_size('tensorflow')
BATCH_SIZE_TORCH = get_optimal_batch_size('timm')

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    print("Warning: timm not available. timm models won't work.")

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

TIME_LENGTHS = [7, 14, 21, 28]
WINDOW_SIZES = [5, 15, 30]

# Pre-trained models configuration
PRETRAINED_MODELS = {
    'edgenext': {
        'framework': 'timm',
        'model_name': 'edgenext_small',
        'input_size': 256,
        'description': 'EdgeNext Small (1.33M params)',
        'requires': ['torch', 'timm']
    },
    'mobilenetv3': {
        'framework': 'timm',
        'model_name': 'mobilenetv3_small_100',
        'input_size': 224,
        'description': 'MobileNetV3 Small (2.55M params)',
        'requires': ['torch', 'timm']
    },
    'ghostnet': {
        'framework': 'timm',
        'model_name': 'ghostnet_100',
        'input_size': 224,
        'description': 'GhostNet 100 (5.2M params)',
        'requires': ['torch', 'timm']
    },
    'efficientnet': {
        'framework': 'tensorflow',
        'model_name': 'EfficientNetB0',
        'input_size': 224,
        'description': 'EfficientNetB0 (5.3M params)'
    },
    'levit': {
        'framework': 'timm',
        'model_name': 'levit_128s',
        'input_size': 224,
        'description': 'LeViT-128S Facebook (7.91M params)',
        'requires': ['torch', 'timm']
    }
}

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crypto_research_minute_fullimage")
OLD_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crypto_research_minute")

# =============================================================================
# MEMORY-EFFICIENT DATA LOADERS
# =============================================================================

class LazyImageDataset(Dataset):
    """PyTorch Dataset that loads images lazily from disk instead of preloading all.
    For fullimage dataset, loads images from regular dataset directory."""
    def __init__(self, labels_df, images_dir, input_size, transform=None, regular_images_dir=None):
        self.labels_df = labels_df
        self.images_dir = images_dir  # Used for checking if images exist
        self.input_size = input_size
        self.transform = transform
        # For fullimage: use images from regular dataset
        self.actual_images_dir = regular_images_dir if regular_images_dir else images_dir

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        # Load image from regular dataset (for fullimage)
        image_path = os.path.join(self.actual_images_dir, row["image_path"])

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.input_size, self.input_size), Image.Resampling.LANCZOS)

        label = 1 if row["label"] == "UP" else 0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


def create_tf_dataset(labels_df, images_dir, input_size, batch_size=BATCH_SIZE_TF, shuffle=True, regular_images_dir=None):
    """Create a TensorFlow Dataset that loads images lazily from disk.
    For fullimage, loads images from regular dataset directory."""
    actual_images_dir = regular_images_dir if regular_images_dir else images_dir
    image_paths = labels_df["image_path"].values
    labels = labels_df["label"].values

    def load_image(image_path, label):
        img = tf.io.read_file(os.path.join(actual_images_dir, image_path))
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [input_size, input_size])
        img = tf.cast(img, tf.float32) / 255.0
        label = 1 if label == "UP" else 0
        return img, label

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(1000, len(labels_df)))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_pytorch_dataloader(labels_df, images_dir, input_size, batch_size=BATCH_SIZE_TORCH, shuffle=True, regular_images_dir=None):
    """Create a PyTorch DataLoader that loads images lazily from disk.
    For fullimage, loads images from regular dataset directory."""
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = LazyImageDataset(labels_df, images_dir, input_size, transform=transform, regular_images_dir=regular_images_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)
    return dataloader


# =============================================================================
# ORIGINAL FUNCTIONS (kept for data fetching and image generation)
# =============================================================================

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
        current_start = int(data[-1][0]) + 60000

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    return df[["timestamp", "open", "high", "low", "close"]]


def generate_images(df, window_size, output_dir, period_name, month_str, input_size):
    """
    Fullimage dataset: reuses images from regular dataset but generates new labels
    using FULL WINDOW labeling (last close > first open = UP)
    """
    os.makedirs(output_dir, exist_ok=True)
    labels_file = os.path.join(output_dir, f"labels_{month_str}_1m_{period_name}_w{window_size}_size{input_size}.csv")
    if os.path.exists(labels_file):
        print(f"Labels already exist at {labels_file}, skipping label generation")
        return labels_file

    # For fullimage: we reuse images from regular dataset but generate different labels
    # Full window labeling: compare last close with first open of the window
    labels = []
    start_time = time.time()

    for i in range(window_size - 1, len(df)):
        window_df = df.iloc[i - (window_size - 1):i + 1]
        first_candle = window_df.iloc[0]
        last_candle = window_df.iloc[-1]
        # FULL WINDOW LABELING: last close > first open = UP
        label = "UP" if last_candle["close"] > first_candle["open"] else "DOWN"
        labels.append(label)

        if i % 1000 == 0:
            elapsed = time.time() - start_time
            images_generated = i - (window_size - 1) + 1
            speed = images_generated / elapsed if elapsed > 0 else 0
            print(f"Generated label {i}/{len(df)} for {month_str} 1m {period_name} w{window_size} ({speed:.2f} labels/sec)")

    labels_df = pd.DataFrame({"image_path": [f"candle_{i}.png" for i in range(window_size - 1, len(df))], "label": labels})
    labels_df.to_csv(labels_file, index=False)
    print(f"Saved {len(labels_df)} labels to {labels_file}")

    # Note: images are loaded from regular dataset at inference time
    return labels_file


# =============================================================================
# MODEL CREATION AND TRAINING (MEMORY-EFFICIENT VERSIONS)
# =============================================================================

def create_tensorflow_model(model_type, input_shape):
    if model_type == 'efficientnet':
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def create_pytorch_model(model_config, num_classes=1):
    if not TIMM_AVAILABLE:
        raise ImportError("timm is required for PyTorch models")

    model = timm.create_model(
        model_config['model_name'],
        pretrained=True,
        num_classes=num_classes,
        global_pool='avg'
    )

    num_features = model.num_features
    model.reset_classifier(num_classes, global_pool='avg')

    return model


def train_tensorflow_model(model, train_dataset, model_type, period_name, month_str, window_size, coin_dir, num_samples):
    model_path = os.path.join(coin_dir, "models", f"model_{model_type}_{month_str}_1m_{period_name}_w{window_size}.h5")

    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}, loading instead of training")
        return tf.keras.models.load_model(model_path), None

    print(f"Training TensorFlow {model_type} model on {num_samples} samples with batch size {BATCH_SIZE_TF}...")

    # Compute class weights from the dataset
    labels = []
    for _, y in train_dataset:
        labels.extend(y.numpy())
    labels = np.array(labels)
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)

    history = model.fit(
        train_dataset,
        epochs=10,
        class_weight=dict(enumerate(class_weights)),
        verbose=1
    )

    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model, history


def train_pytorch_model(model, train_loader, model_type, period_name, month_str, window_size, coin_dir):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for timm models")

    model_path = os.path.join(coin_dir, "models", f"model_{model_type}_{month_str}_1m_{period_name}_w{window_size}.pth")

    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}, loading instead of training")
        model.load_state_dict(torch.load(model_path))
        return model, None

    print(f"Training PyTorch {model_type} model with batch size {BATCH_SIZE_TORCH}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Compute class weights from the dataset
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.FloatTensor(class_weights).to(device)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)

            weights = class_weights[labels.long().squeeze()]
            loss = (criterion(outputs, labels) * weights.unsqueeze(1)).mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/10 - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return model, None


def evaluate_tensorflow_model(model, test_dataset, model_type, period_name, month_str, window_size, coin_dir, dataset_type="train", exp_suffix=""):
    y_pred_prob = model.predict(test_dataset, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Get true labels
    y_true = []
    for _, y in test_dataset:
        y_true.extend(y.numpy())
    y_true = np.array(y_true)

    # Ensure shapes match
    if len(y_pred) > len(y_true):
        y_pred = y_pred[:len(y_true)]
        y_pred_prob = y_pred_prob[:len(y_true)]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_pred_prob),
        "auprc": auc(*precision_recall_curve(y_true, y_pred_prob)[1::-1])
    }

    results_file = os.path.join(coin_dir, "results", f"results_{model_type}_{dataset_type}_{month_str}_1m_{period_name}_w{window_size}{exp_suffix}.txt")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        f.write(f"{dataset_type.capitalize()} Metrics for {model_type} - {month_str} 1m {period_name} w{window_size} {exp_suffix}:\n")
        for k, v in metrics.items():
            f.write(f"{k.capitalize()}: {v:.4f}\n")

    print(f"Results saved to {results_file}")
    return metrics


def evaluate_pytorch_model(model, test_loader, model_type, period_name, month_str, window_size, coin_dir, dataset_type="train", exp_suffix="", input_size=None):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for timm models")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.unsqueeze(1).to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    y_pred_prob = np.array(all_preds)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_true = np.array(all_labels)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_pred_prob),
        "auprc": auc(*precision_recall_curve(y_true, y_pred_prob)[1::-1])
    }

    results_file = os.path.join(coin_dir, "results", f"results_{model_type}_{dataset_type}_{month_str}_1m_{period_name}_w{window_size}{exp_suffix}.txt")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        f.write(f"{dataset_type.capitalize()} Metrics for {model_type} - {month_str} 1m {period_name} w{window_size} {exp_suffix}:\n")
        for k, v in metrics.items():
            f.write(f"{k.capitalize()}: {v:.4f}\n")

    print(f"Results saved to {results_file}")
    return metrics


def train_and_evaluate_model(labels_df, images_dir, model_type, period_name, month_str, window_size, coin_dir, dataset_type="train", exp_suffix="", regular_images_dir=None):
    """Memory-efficient training and evaluation using lazy data loaders."""
    results_file = os.path.join(coin_dir, "results", f"results_{model_type}_{dataset_type}_{month_str}_1m_{period_name}_w{window_size}{exp_suffix}.txt")
    if os.path.exists(results_file):
        print(f"Results already exist at {results_file}, skipping evaluation")
        return None

    model_config = PRETRAINED_MODELS[model_type]
    input_size = model_config['input_size']

    print(f"Processing {dataset_type} data: {len(labels_df)} samples")

    if model_config['framework'] == 'tensorflow':
        # Create TensorFlow Dataset (load images from regular dataset for fullimage)
        dataset = create_tf_dataset(labels_df, images_dir, input_size, batch_size=BATCH_SIZE_TF, shuffle=(dataset_type == "train"), regular_images_dir=regular_images_dir)

        model = create_tensorflow_model(model_type, (input_size, input_size, 3))
        model, history = train_tensorflow_model(model, dataset, model_type, period_name, month_str, window_size, coin_dir, len(labels_df))
        metrics = evaluate_tensorflow_model(model, dataset, model_type, period_name, month_str, window_size, coin_dir, dataset_type, exp_suffix)
        tf.keras.backend.clear_session()

    elif model_config['framework'] == 'timm':
        # Create PyTorch DataLoader (load images from regular dataset for fullimage)
        dataloader = create_pytorch_dataloader(labels_df, images_dir, input_size, batch_size=BATCH_SIZE_TORCH, shuffle=(dataset_type == "train"), regular_images_dir=regular_images_dir)

        model = create_pytorch_model(model_config)
        model, history = train_pytorch_model(model, dataloader, model_type, period_name, month_str, window_size, coin_dir)
        metrics = evaluate_pytorch_model(model, dataloader, model_type, period_name, month_str, window_size, coin_dir, dataset_type, exp_suffix, input_size)

    # Clean up
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def process_coin(symbol, train_month, test_months, window_size, model_type, time_lengths=None, exp2_only=False):
    print(f"\nProcessing {symbol} with {model_type} model - window size {window_size}")
    print(f"Model: {PRETRAINED_MODELS[model_type]['description']}")

    coin_dir = os.path.join(BASE_DIR, symbol)
    IMAGES_DIR = os.path.join(coin_dir, "images")
    RAW_DATA_DIR = os.path.join(coin_dir, "raw_data")
    MODELS_DIR = os.path.join(coin_dir, "models")
    RESULTS_DIR = os.path.join(coin_dir, "results")

    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_year, train_month_num = train_month
    input_size = PRETRAINED_MODELS[model_type]['input_size']

    # Filter time lengths if specified
    if time_lengths:
        TIME_LENGTHS_FILTERED = [t for t in TIME_LENGTHS if t in time_lengths]
    else:
        TIME_LENGTHS_FILTERED = TIME_LENGTHS

    if not exp2_only:
        # Experiment I
        for days in TIME_LENGTHS_FILTERED:
            period_name = f"{days}days"
            train_month_str = f"{train_year}-{train_month_num:02d}"

            # Check if train result already exists
            train_result = os.path.join(RESULTS_DIR, f"results_{model_type}_train_{train_month_str}_1m_{period_name}_w{window_size}.txt")
            if os.path.exists(train_result):
                print(f"Train results already exist at {train_result}, skipping")
            else:
                train_start = datetime(train_year, train_month_num, 1)
                train_end = train_start + timedelta(days=days - 1, hours=23, minutes=59)

                raw_file = os.path.join(RAW_DATA_DIR, f"raw_{train_month_str}_1m_{period_name}.csv")
                if not os.path.exists(raw_file):
                    df = fetch_coin_data(symbol, train_start, train_end)
                    df.set_index("timestamp", inplace=True)
                    df.to_csv(raw_file)
                    print(f"Raw data saved to {raw_file}")
                else:
                    print(f"Raw data already exists at {raw_file}, skipping fetch")
                    df = pd.read_csv(raw_file, index_col="timestamp", parse_dates=["timestamp"])
                    df.index = pd.to_datetime(df.index)

                images_subdir = os.path.join(IMAGES_DIR, f"{train_month_str}_1m_{period_name}_w{window_size}")
                labels_file = generate_images(df, window_size, images_subdir, period_name, train_month_str, input_size)
                labels_df = pd.read_csv(labels_file)
                # For fullimage: load images from regular dataset
                regular_images_subdir = os.path.join(OLD_BASE_DIR, symbol, "images", f"{train_month_str}_1m_{period_name}_w{window_size}")
                train_and_evaluate_model(labels_df, images_subdir, model_type, period_name, train_month_str, window_size, coin_dir, "train", "", regular_images_subdir)
                del labels_df
                gc.collect()

            for test_year, test_month_num in test_months:
                test_month_str = f"{test_year}-{test_month_num:02d}"
                test_result = os.path.join(RESULTS_DIR, f"results_{model_type}_test_{test_month_str}_1m_{period_name}_w{window_size}.txt")

                if os.path.exists(test_result):
                    print(f"Test results already exist at {test_result}, skipping")
                else:
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
                        df.index = pd.to_datetime(df.index)

                    images_subdir = os.path.join(IMAGES_DIR, f"{test_month_str}_1m_{period_name}_w{window_size}")
                    labels_file = generate_images(df, window_size, images_subdir, period_name, test_month_str, input_size)
                    labels_df = pd.read_csv(labels_file)
                    # For fullimage: load images from regular dataset
                    regular_images_subdir = os.path.join(OLD_BASE_DIR, symbol, "images", f"{test_month_str}_1m_{period_name}_w{window_size}")
                    train_and_evaluate_model(labels_df, images_subdir, model_type, period_name, test_month_str, window_size, coin_dir, "test", "", regular_images_subdir)
                    del labels_df
                    gc.collect()

    # Experiment II
    exp2_test_lengths = [14, 21, 28]
    train_month_str = f"{train_year}-{train_month_num:02d}"
    period_name = "1week"

    train_result = os.path.join(RESULTS_DIR, f"results_{model_type}_train_{train_month_str}_1m_{period_name}_w{window_size}_exp2.txt")
    if os.path.exists(train_result):
        print(f"Exp2 train results already exist at {train_result}, skipping")
    else:
        train_start = datetime(train_year, train_month_num, 1)
        train_end = train_start + timedelta(days=6, hours=23, minutes=59)

        raw_file = os.path.join(RAW_DATA_DIR, f"raw_{train_month_str}_1m_{period_name}.csv")
        if not os.path.exists(raw_file):
            df = fetch_coin_data(symbol, train_start, end_time=train_end)
            df.set_index("timestamp", inplace=True)
            df.to_csv(raw_file)
            print(f"Raw data saved to {raw_file}")
        else:
            print(f"Raw data already exists at {raw_file}, skipping fetch")
            df = pd.read_csv(raw_file, index_col="timestamp", parse_dates=["timestamp"])
            df.index = pd.to_datetime(df.index)

        images_subdir = os.path.join(IMAGES_DIR, f"{train_month_str}_1m_{period_name}_w{window_size}")
        labels_file = generate_images(df, window_size, images_subdir, period_name, train_month_str, input_size)
        labels_df = pd.read_csv(labels_file)
        # For fullimage: load images from regular dataset
        regular_images_subdir = os.path.join(OLD_BASE_DIR, symbol, "images", f"{train_month_str}_1m_{period_name}_w{window_size}")
        train_and_evaluate_model(labels_df, images_subdir, model_type, period_name, train_month_str, window_size, coin_dir, "train", "_exp2", regular_images_subdir)
        del labels_df
        gc.collect()

    for test_year, test_month_num in test_months:
        test_month_str = f"{test_year}-{test_month_num:02d}"
        for days in exp2_test_lengths:
            period_name = f"{days}days"
            test_result = os.path.join(RESULTS_DIR, f"results_{model_type}_test_{test_month_str}_1m_{period_name}_w{window_size}_exp2.txt")

            if os.path.exists(test_result):
                print(f"Exp2 test results already exist at {test_result}, skipping")
            else:
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
                    df.index = pd.to_datetime(df.index)

                images_subdir = os.path.join(IMAGES_DIR, f"{test_month_str}_1m_{period_name}_w{window_size}")
                labels_file = generate_images(df, window_size, images_subdir, period_name, test_month_str, input_size)
                labels_df = pd.read_csv(labels_file)
                # For fullimage: load images from regular dataset
                regular_images_subdir = os.path.join(OLD_BASE_DIR, symbol, "images", f"{test_month_str}_1m_{period_name}_w{window_size}")
                train_and_evaluate_model(labels_df, images_subdir, model_type, period_name, test_month_str, window_size, coin_dir, "test", "_exp2", regular_images_subdir)
                del labels_df
                gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Memory-efficient training with GPU support")
    parser.add_argument("--model", nargs="+", choices=list(PRETRAINED_MODELS.keys()), help="Model type(s) - runs all if not specified")
    parser.add_argument("--coin", nargs="+", help="Specific coin(s) - runs all if not specified")
    parser.add_argument("--window", type=int, nargs="+", choices=[5, 15, 30], help="Window size(s) - runs all if not specified")
    parser.add_argument("--time-length", type=int, nargs="+", choices=[7, 14, 21, 28], help="Time length(s) in days (default: all)")
    parser.add_argument("--exp2-only", action="store_true", help="Only run Experiment II (1-week training)")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--force", action="store_true", help="Force re-run even if results exist")

    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable Pre-trained Models:")
        print("=" * 60)
        for model_type, config in PRETRAINED_MODELS.items():
            status = "Ready (GPU)" if (config['framework'] == 'tensorflow' and len(gpus) > 0) or (config['framework'] == 'timm' and TORCH_AVAILABLE and torch.cuda.is_available()) else "Ready (CPU)"
            if 'requires' in config:
                missing = [req for req in config['requires']
                          if (req == 'torch' and not TORCH_AVAILABLE) or
                             (req == 'timm' and not TIMM_AVAILABLE)]
                if missing:
                    status = f"Needs {', '.join(missing)}"
            print(f"\n{model_type:15} - {config['description']}")
            print(f"{'':15}   Framework: {config['framework']}")
            print(f"{'':15}   Status: {status}")
        print("\n" + "=" * 60)
        return

    # Default to all models/coins/windows if not specified
    if args.model:
        models_to_run = args.model
    else:
        models_to_run = []
        for model_type in PRETRAINED_MODELS.keys():
            model_config = PRETRAINED_MODELS[model_type]
            if 'requires' in model_config:
                missing = []
                if 'torch' in model_config['requires'] and not TORCH_AVAILABLE:
                    missing.append('torch')
                if 'timm' in model_config['requires'] and not TIMM_AVAILABLE:
                    missing.append('timm')
                if missing:
                    print(f"Note: Skipping {model_type} - requires {', '.join(missing)}")
                    continue
            models_to_run.append(model_type)
        if not models_to_run:
            print("Error: No models available. Please install torch and timm for timm models.")
            return

    if args.coin:
        coins = {c: COINS.get(c) for c in args.coin}
        for c in args.coin:
            if not coins[c]:
                print(f"Error: Unknown coin {c}")
                return
    else:
        coins = COINS

    window_sizes = args.window if args.window else WINDOW_SIZES
    time_lengths = args.time_length if args.time_length else None

    # Check GPU availability
    if TORCH_AVAILABLE:
        torch_gpu_available = torch.cuda.is_available()
        if torch_gpu_available:
            print(f"PyTorch GPU: {torch.cuda.get_device_name(0)}")
            print(f"PyTorch GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        torch_gpu_available = False

    tf_gpu_available = len(gpus) > 0

    os.makedirs(BASE_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Memory-Efficient Training on REGULAR dataset")
    print(f"Models: {', '.join(models_to_run)}")
    print(f"Coins: {', '.join(coins.keys())}")
    print(f"Windows: {window_sizes}")
    if time_lengths:
        print(f"Time lengths: {time_lengths} days")
    if args.exp2_only:
        print(f"Mode: Experiment II only (1-week training)")
    print(f"GPU: TensorFlow={tf_gpu_available}, PyTorch={torch_gpu_available}")
    print(f"Batch sizes: TF={BATCH_SIZE_TF}, PyTorch={BATCH_SIZE_TORCH}")
    print(f"{'='*60}")

    # Run jobs
    total_jobs = 0
    for symbol in coins.keys():
        for model_type in models_to_run:
            for window_size in window_sizes:
                total_jobs += 1

    print(f"Total jobs: {total_jobs}")
    print(f"\n{'='*60}\n")

    start_time = time.time()
    completed = 0

    for symbol, config in coins.items():
        train_month = config["train_month"]
        test_months = config["test_months"]
        for model_type in models_to_run:
            for window_size in window_sizes:
                completed += 1
                pct = (completed / total_jobs) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / completed) * (total_jobs - completed) if completed > 0 else 0
                print(f"\n[{completed}/{total_jobs}] {pct:5.1f}% | ETA: {eta/60:.1f}m | {symbol} - {model_type} - w{window_size}")
                print("-" * 60)
                try:
                    process_coin(symbol, train_month, test_months, window_size, model_type, time_lengths, args.exp2_only)
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Completed in {total_time/60:.1f} minutes")
    print(f"Average time per job: {total_time/total_jobs:.1f} seconds")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
