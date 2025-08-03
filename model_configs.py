"""
Model configurations for HuggingFace benchmarking framework
Contains specifications for all 9 models from the benchmarking table
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch

@dataclass
class ModelConfig:
    """Configuration class for HuggingFace models"""
    name: str
    hf_identifier: str
    parameters: int
    input_size: Tuple[int, int]
    num_classes: int
    requires_timm: bool = False
    architecture_type: str = "cnn"  # cnn, transformer, hybrid
    batch_size_recommendation: int = 32
    memory_efficient: bool = True
    description: str = ""

# Model configurations based on the HuggingFace benchmarking table
BENCHMARK_MODELS = {
    "edgenext_xx_small": ModelConfig(
        name="EdgeNeXT XX Small",
        hf_identifier="timm/edgenext_xx_small.in1k",
        parameters=1_330_000,
        input_size=(224, 224),
        num_classes=1000,
        requires_timm=True,
        architecture_type="cnn",
        batch_size_recommendation=64,
        memory_efficient=True,
        description="Ultra-lightweight CNN optimized for edge devices"
    ),
    
    "mobilenetv3_small": ModelConfig(
        name="MobileNetV3 Small",
        hf_identifier="timm/mobilenetv3_small_100.lamb_in1k",
        parameters=2_550_000,
        input_size=(224, 224),
        num_classes=1000,
        requires_timm=True,
        architecture_type="cnn",
        batch_size_recommendation=64,
        memory_efficient=True,
        description="Mobile-optimized CNN with inverted residuals"
    ),
    
    "ghostnet_100": ModelConfig(
        name="GhostNet 100",
        hf_identifier="timm/ghostnet_100.in1k",
        parameters=5_200_000,
        input_size=(224, 224),
        num_classes=1000,
        requires_timm=True,
        architecture_type="cnn",
        batch_size_recommendation=64,
        memory_efficient=True,
        description="Efficient CNN using ghost modules to reduce parameters"
    ),
    
    "efficientnet_b0": ModelConfig(
        name="EfficientNet B0",
        hf_identifier="google/efficientnet-b0",
        parameters=5_300_000,
        input_size=(224, 224),
        num_classes=1000,
        requires_timm=False,
        architecture_type="cnn",
        batch_size_recommendation=64,
        memory_efficient=True,
        description="Compound scaling CNN optimizing depth, width, and resolution"
    ),
    
    "levit_128s": ModelConfig(
        name="LeViT 128S",
        hf_identifier="facebook/levit-128S",
        parameters=7_910_000,
        input_size=(224, 224),
        num_classes=1000,
        requires_timm=False,
        architecture_type="hybrid",
        batch_size_recommendation=64,
        memory_efficient=True,
        description="Hybrid CNN-Transformer for efficient vision tasks"
    ),
    
    "resnet_50": ModelConfig(
        name="ResNet 50",
        hf_identifier="microsoft/resnet-50",
        parameters=25_600_000,
        input_size=(224, 224),
        num_classes=1000,
        requires_timm=False,
        architecture_type="cnn",
        batch_size_recommendation=32,
        memory_efficient=False,
        description="Deep residual network with skip connections"
    ),
    
    "swin_tiny": ModelConfig(
        name="Swin Tiny",
        hf_identifier="microsoft/swin-tiny-patch4-window7-224",
        parameters=28_300_000,
        input_size=(224, 224),
        num_classes=1000,
        requires_timm=False,
        architecture_type="transformer",
        batch_size_recommendation=32,
        memory_efficient=False,
        description="Hierarchical transformer using shifted windows"
    ),
    
    "convnext_tiny": ModelConfig(
        name="ConvNeXt Tiny",
        hf_identifier="facebook/convnext-tiny-224",
        parameters=29_000_000,
        input_size=(224, 224),
        num_classes=1000,
        requires_timm=False,
        architecture_type="cnn",
        batch_size_recommendation=32,
        memory_efficient=False,
        description="Modernized CNN design inspired by transformers"
    ),
    
    "vit_base": ModelConfig(
        name="Vision Transformer Base",
        hf_identifier="google/vit-base-patch16-224",
        parameters=86_600_000,
        input_size=(224, 224),
        num_classes=1000,
        requires_timm=False,
        architecture_type="transformer",
        batch_size_recommendation=16,
        memory_efficient=False,
        description="Pure transformer architecture for image classification"
    )
}

# Model categories for different use cases
LIGHTWEIGHT_MODELS = ["edgenext_xx_small", "mobilenetv3_small", "ghostnet_100"]
BALANCED_MODELS = ["efficientnet_b0", "levit_128s"]
PERFORMANCE_MODELS = ["resnet_50", "swin_tiny", "convnext_tiny", "vit_base"]

# Recommended model sets for different scenarios
MODEL_SETS = {
    "quick_test": ["edgenext_xx_small", "efficientnet_b0"],
    "lightweight": LIGHTWEIGHT_MODELS,
    "balanced": LIGHTWEIGHT_MODELS + BALANCED_MODELS,
    "full_benchmark": list(BENCHMARK_MODELS.keys()),
    "transformers_only": ["levit_128s", "swin_tiny", "vit_base"],
    "cnns_only": ["edgenext_xx_small", "mobilenetv3_small", "ghostnet_100", 
                  "efficientnet_b0", "resnet_50", "convnext_tiny"]
}

def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a specific model"""
    if model_name not in BENCHMARK_MODELS:
        raise ValueError(f"Model {model_name} not found. Available models: {list(BENCHMARK_MODELS.keys())}")
    return BENCHMARK_MODELS[model_name]

def get_models_by_category(category: str) -> List[str]:
    """Get model names by category (lightweight, balanced, performance)"""
    category_map = {
        "lightweight": LIGHTWEIGHT_MODELS,
        "balanced": BALANCED_MODELS,
        "performance": PERFORMANCE_MODELS
    }
    if category not in category_map:
        raise ValueError(f"Category {category} not found. Available: {list(category_map.keys())}")
    return category_map[category]

def get_model_set(set_name: str) -> List[str]:
    """Get predefined model sets for different benchmarking scenarios"""
    if set_name not in MODEL_SETS:
        raise ValueError(f"Model set {set_name} not found. Available: {list(MODEL_SETS.keys())}")
    return MODEL_SETS[set_name]

def get_memory_requirements() -> Dict[str, Dict[str, float]]:
    """Get estimated memory requirements for each model (in GB)"""
    memory_estimates = {}
    for name, config in BENCHMARK_MODELS.items():
        # Rough estimation based on parameters and architecture
        base_memory = config.parameters * 4 / (1024**3)  # 4 bytes per parameter
        training_multiplier = 4 if config.architecture_type == "transformer" else 3
        
        memory_estimates[name] = {
            "model_size_gb": base_memory,
            "training_memory_gb": base_memory * training_multiplier,
            "inference_memory_gb": base_memory * 1.5
        }
    
    return memory_estimates

def print_model_summary():
    """Print a summary of all available models"""
    print("=" * 80)
    print("HuggingFace Model Benchmarking Suite")
    print("=" * 80)
    
    for category in ["lightweight", "balanced", "performance"]:
        models = get_models_by_category(category) if category != "balanced" else BALANCED_MODELS
        print(f"\n{category.upper()} MODELS:")
        print("-" * 40)
        
        for model_name in models:
            config = BENCHMARK_MODELS[model_name]
            params_m = config.parameters / 1_000_000
            print(f"  {config.name:<25} | {params_m:>6.1f}M params | {config.architecture_type}")
    
    print(f"\nTotal models available: {len(BENCHMARK_MODELS)}")
    print("=" * 80)

if __name__ == "__main__":
    print_model_summary()
    
    # Test model configuration access
    print("\nTesting model configuration access:")
    test_model = get_model_config("efficientnet_b0")
    print(f"Test model: {test_model.name}")
    print(f"Parameters: {test_model.parameters:,}")
    print(f"Input size: {test_model.input_size}")
    
    # Test memory requirements
    memory_reqs = get_memory_requirements()
    print(f"\nEfficientNet-B0 estimated training memory: {memory_reqs['efficientnet_b0']['training_memory_gb']:.2f} GB")