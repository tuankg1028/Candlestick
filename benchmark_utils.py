"""
Utility functions for HuggingFace model benchmarking
Handles model loading, preprocessing, and performance monitoring
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import AutoModel, AutoImageProcessor, AutoConfig
import timm
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import psutil
import gc
from contextlib import contextmanager
import logging
from model_configs import ModelConfig, BENCHMARK_MODELS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor performance metrics during model operations"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = None
        self.end_time = None
        self.peak_memory = 0
        self.start_memory = 0
    
    @contextmanager
    def monitor(self):
        """Context manager to monitor performance"""
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
        
        try:
            yield self
        finally:
            self.end_time = time.time()
            self.peak_memory = max(self.peak_memory, self._get_memory_usage())
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        else:
            process = psutil.Process()
            return process.memory_info().rss / (1024**3)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        if self.start_time is None or self.end_time is None:
            return {}
        
        return {
            "execution_time": self.end_time - self.start_time,
            "peak_memory_gb": self.peak_memory,
            "memory_increase_gb": self.peak_memory - self.start_memory
        }

class ModelAdapter:
    """Adapter class to handle different model architectures uniformly"""
    
    def __init__(self, model_name: str, num_classes: int = 2):
        self.model_name = model_name
        self.config = BENCHMARK_MODELS[model_name]
        self.num_classes = num_classes
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self) -> nn.Module:
        """Load and adapt model for binary classification"""
        logger.info(f"Loading model: {self.config.name}")
        
        try:
            if self.config.requires_timm:
                # Load TIMM models
                self.model = timm.create_model(
                    self.config.hf_identifier.split('/')[-1],
                    pretrained=True,
                    num_classes=self.num_classes
                )
            else:
                # Load HuggingFace models
                config = AutoConfig.from_pretrained(self.config.hf_identifier)
                self.model = AutoModel.from_pretrained(self.config.hf_identifier)
                
                # Add classification head
                if hasattr(self.model, 'classifier'):
                    # Models with existing classifier
                    in_features = self.model.classifier.in_features
                    self.model.classifier = nn.Linear(in_features, self.num_classes)
                elif hasattr(self.model, 'heads'):
                    # Vision Transformers
                    in_features = self.model.heads.head.in_features
                    self.model.heads.head = nn.Linear(in_features, self.num_classes)
                elif hasattr(self.model, 'head'):
                    # Some models use 'head'
                    in_features = self.model.head.in_features
                    self.model.head = nn.Linear(in_features, self.num_classes)
                else:
                    # Add custom head for models without classifier
                    self.model = ModelWithHead(self.model, self.num_classes)
            
            self.model = self.model.to(self.device)
            logger.info(f"Model loaded successfully on {self.device}")
            return self.model
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            raise
    
    def get_preprocessor(self) -> transforms.Compose:
        """Get appropriate preprocessing transforms"""
        if self.config.requires_timm:
            # TIMM models often use their own preprocessing
            data_config = timm.data.resolve_model_data_config(self.model)
            transform = timm.data.create_transform(**data_config, is_training=False)
            return transform
        else:
            # Standard ImageNet preprocessing
            return transforms.Compose([
                transforms.Resize(self.config.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def preprocess_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """Preprocess a batch of images"""
        transform = self.get_preprocessor()
        
        # Handle different transform types
        if isinstance(transform, transforms.Compose):
            processed = torch.stack([transform(img) for img in images])
        else:
            # For TIMM transforms that might return different formats
            processed_list = []
            for img in images:
                processed_img = transform(img)
                if isinstance(processed_img, dict):
                    processed_img = processed_img['image']
                processed_list.append(processed_img)
            processed = torch.stack(processed_list)
        
        return processed.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.model.eval()
        with torch.no_grad():
            return self.model(x)
    
    def get_model_info(self) -> Dict:
        """Get detailed model information"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.config.name,
            "architecture_type": self.config.architecture_type,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_size": self.config.input_size,
            "device": str(self.device)
        }

class ModelWithHead(nn.Module):
    """Wrapper for models that need a custom classification head"""
    
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        
        # Try to find the last layer's output size
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            try:
                dummy_output = backbone(dummy_input)
                if hasattr(dummy_output, 'last_hidden_state'):
                    # Transformer models
                    hidden_size = dummy_output.last_hidden_state.shape[-1]
                    self.head = nn.Sequential(
                        nn.AdaptiveAvgPool1d(1),
                        nn.Flatten(),
                        nn.Linear(hidden_size, num_classes)
                    )
                else:
                    # CNN models
                    hidden_size = dummy_output.view(dummy_output.size(0), -1).shape[1]
                    self.head = nn.Linear(hidden_size, num_classes)
            except:
                # Fallback: assume 768 hidden size (common for transformers)
                self.head = nn.Linear(768, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        if hasattr(features, 'last_hidden_state'):
            # Transformer output
            features = features.last_hidden_state.mean(dim=1)  # Global average pooling
        elif len(features.shape) > 2:
            # CNN output that needs flattening
            features = features.view(features.size(0), -1)
        
        return self.head(features)

class BenchmarkDataLoader:
    """Custom data loader for benchmarking with candlestick images"""
    
    def __init__(self, images: np.ndarray, labels: np.ndarray, batch_size: int = 32):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_samples = len(images)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
    
    def __iter__(self):
        for i in range(0, self.num_samples, self.batch_size):
            end_idx = min(i + self.batch_size, self.num_samples)
            batch_images = []
            
            # Convert numpy arrays to PIL Images
            for img_array in self.images[i:end_idx]:
                # Ensure image is in correct format (0-255, uint8)
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                batch_images.append(img)
            
            batch_labels = torch.tensor(self.labels[i:end_idx], dtype=torch.long)
            
            yield batch_images, batch_labels
    
    def __len__(self):
        return self.num_batches

def cleanup_gpu_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def get_device_info() -> Dict[str, Union[str, float]]:
    """Get device information for benchmarking"""
    info = {
        "device_type": "cuda" if torch.cuda.is_available() else "cpu",
        "python_version": f"{torch.__version__}",
    }
    
    if torch.cuda.is_available():
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "gpu_memory_available": (torch.cuda.get_device_properties(0).total_memory - 
                                   torch.cuda.memory_allocated(0)) / (1024**3)
        })
    else:
        info.update({
            "cpu_count": psutil.cpu_count(),
            "ram_total": psutil.virtual_memory().total / (1024**3),
            "ram_available": psutil.virtual_memory().available / (1024**3)
        })
    
    return info

def test_model_loading():
    """Test function to verify model loading works"""
    print("Testing model loading...")
    
    # Test a lightweight model
    try:
        adapter = ModelAdapter("efficientnet_b0", num_classes=2)
        model = adapter.load_model()
        info = adapter.get_model_info()
        
        print(f"✓ Successfully loaded {info['model_name']}")
        print(f"  Parameters: {info['total_parameters']:,}")
        print(f"  Device: {info['device']}")
        
        # Test preprocessing
        dummy_image = Image.new('RGB', (64, 64), color='red')
        processed = adapter.preprocess_batch([dummy_image])
        print(f"  Preprocessed shape: {processed.shape}")
        
        # Test forward pass
        output = adapter.forward(processed)
        print(f"  Output shape: {output.shape}")
        
        cleanup_gpu_memory()
        print("✓ Model test completed successfully")
        
    except Exception as e:
        print(f"✗ Model test failed: {str(e)}")

if __name__ == "__main__":
    print("=" * 60)
    print("Benchmark Utils Test")
    print("=" * 60)
    
    # Print device info
    device_info = get_device_info()
    print("Device Information:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    
    # Test model loading
    test_model_loading()