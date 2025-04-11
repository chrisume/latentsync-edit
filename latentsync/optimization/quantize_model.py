import os
import torch
import argparse
import yaml
from omegaconf import OmegaConf
from typing import Dict, Optional, Any

from latentsync.models.unet import HighResUNet
from latentsync.models.phoneme_mapper import EnhancedPhonemeMapper

def parse_args():
    parser = argparse.ArgumentParser(description="Quantize LatentSync model for faster inference")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/high_res_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output quantized model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda, cpu)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="dynamic",
        choices=["dynamic", "static", "qat"],
        help="Quantization type",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        choices=[8, 4],
        help="Number of bits for quantization",
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> Any:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Convert to OmegaConf for easier access
    config = OmegaConf.create(config)
    
    return config

def load_models(config: Any, checkpoint_path: str, device: str) -> Dict[str, torch.nn.Module]:
    """Load models from checkpoint"""
    # Create models
    unet = HighResUNet(config).to(device)
    phoneme_mapper = EnhancedPhonemeMapper(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    unet.load_state_dict(checkpoint["unet"])
    phoneme_mapper.load_state_dict(checkpoint["phoneme_mapper"])
    
    # Set to eval mode
    unet.eval()
    phoneme_mapper.eval()
    
    return {
        "unet": unet,
        "phoneme_mapper": phoneme_mapper
    }

def quantize_models(
    models: Dict[str, torch.nn.Module],
    config: Any,
    device: str,
    quantization_type: str = "dynamic",
    bits: int = 8
) -> Dict[str, torch.nn.Module]:
    """
    Quantize models for faster inference
    
    Args:
        models: Dictionary of models
        config: Configuration
        device: Device to use
        quantization_type: Type of quantization
        bits: Number of bits for quantization
        
    Returns:
        Dictionary of quantized models
    """
    # Check if torch.quantization is available
    if not hasattr(torch, "quantization"):
        print("torch.quantization is not available. Please use PyTorch 1.8 or later.")
        return models
    
    # Import quantization modules
    from torch.quantization import quantize_dynamic, quantize_static, prepare, convert
    
    # Create quantized models
    quantized_models = {}
    
    # Quantize UNet
    print("Quantizing UNet...")
    if quantization_type == "dynamic":
        # Dynamic quantization
        quantized_unet = quantize_dynamic(
            models["unet"],
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8 if bits == 8 else torch.qint4
        )
        quantized_models["unet"] = quantized_unet
    elif quantization_type == "static":
        # Static quantization
        # This requires calibration data, which we don't have here
        # For simplicity, we'll just use dynamic quantization
        print("Static quantization requires calibration data. Using dynamic quantization instead.")
        quantized_unet = quantize_dynamic(
            models["unet"],
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8 if bits == 8 else torch.qint4
        )
        quantized_models["unet"] = quantized_unet
    else:
        # Quantization-aware training (QAT)
        # This requires training, which we don't do here
        # For simplicity, we'll just use dynamic quantization
        print("QAT requires training. Using dynamic quantization instead.")
        quantized_unet = quantize_dynamic(
            models["unet"],
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8 if bits == 8 else torch.qint4
        )
        quantized_models["unet"] = quantized_unet
    
    # Quantize phoneme mapper
    print("Quantizing phoneme mapper...")
    if quantization_type == "dynamic":
        # Dynamic quantization
        quantized_phoneme_mapper = quantize_dynamic(
            models["phoneme_mapper"],
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8 if bits == 8 else torch.qint4
        )
        quantized_models["phoneme_mapper"] = quantized_phoneme_mapper
    elif quantization_type == "static":
        # Static quantization
        # This requires calibration data, which we don't have here
        # For simplicity, we'll just use dynamic quantization
        print("Static quantization requires calibration data. Using dynamic quantization instead.")
        quantized_phoneme_mapper = quantize_dynamic(
            models["phoneme_mapper"],
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8 if bits == 8 else torch.qint4
        )
        quantized_models["phoneme_mapper"] = quantized_phoneme_mapper
    else:
        # Quantization-aware training (QAT)
        # This requires training, which we don't do here
        # For simplicity, we'll just use dynamic quantization
        print("QAT requires training. Using dynamic quantization instead.")
        quantized_phoneme_mapper = quantize_dynamic(
            models["phoneme_mapper"],
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8 if bits == 8 else torch.qint4
        )
        quantized_models["phoneme_mapper"] = quantized_phoneme_mapper
    
    return quantized_models

def save_quantized_models(
    models: Dict[str, torch.nn.Module],
    output_path: str
):
    """
    Save quantized models
    
    Args:
        models: Dictionary of quantized models
        output_path: Path to output file
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save models
    torch.save({
        "unet": models["unet"].state_dict(),
        "phoneme_mapper": models["phoneme_mapper"].state_dict()
    }, output_path)
    
    print(f"Quantized models saved to {output_path}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load models
    models = load_models(
        config=config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Quantize models
    quantized_models = quantize_models(
        models=models,
        config=config,
        device=args.device,
        quantization_type=args.quantization,
        bits=args.bits
    )
    
    # Save quantized models
    save_quantized_models(
        models=quantized_models,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
