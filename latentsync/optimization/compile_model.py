import os
import torch
import argparse
import yaml
from omegaconf import OmegaConf
from typing import Dict, Optional, Any

from latentsync.models.unet import HighResUNet
from latentsync.models.phoneme_mapper import EnhancedPhonemeMapper

def parse_args():
    parser = argparse.ArgumentParser(description="Compile LatentSync model for faster inference")
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
        help="Path to output compiled model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda, cpu)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Precision to use for compilation",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="Compilation mode",
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> Any:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Convert to OmegaConf for easier access
    config = OmegaConf.create(config)
    
    return config

def load_models(config: Any, checkpoint_path: str, device: str, precision: str) -> Dict[str, torch.nn.Module]:
    """Load models from checkpoint"""
    # Create models
    unet = HighResUNet(config).to(device)
    phoneme_mapper = EnhancedPhonemeMapper(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    unet.load_state_dict(checkpoint["unet"])
    phoneme_mapper.load_state_dict(checkpoint["phoneme_mapper"])
    
    # Set precision
    if precision == "float16" and device == "cuda":
        unet = unet.half()
        phoneme_mapper = phoneme_mapper.half()
    elif precision == "bfloat16" and device == "cuda" and torch.cuda.is_bf16_supported():
        unet = unet.to(torch.bfloat16)
        phoneme_mapper = phoneme_mapper.to(torch.bfloat16)
    
    # Set to eval mode
    unet.eval()
    phoneme_mapper.eval()
    
    return {
        "unet": unet,
        "phoneme_mapper": phoneme_mapper
    }

def compile_models(
    models: Dict[str, torch.nn.Module],
    config: Any,
    device: str,
    mode: str = "reduce-overhead"
) -> Dict[str, torch.nn.Module]:
    """
    Compile models for faster inference
    
    Args:
        models: Dictionary of models
        config: Configuration
        device: Device to use
        mode: Compilation mode
        
    Returns:
        Dictionary of compiled models
    """
    # Check if torch.compile is available
    if not hasattr(torch, "compile"):
        print("torch.compile is not available. Please use PyTorch 2.0 or later.")
        return models
    
    # Create example inputs
    batch_size = 1
    resolution = config.model.resolution
    phoneme_dim = config.audio.phoneme_dim
    
    # Example inputs for UNet
    unet_inputs = {
        "x": torch.randn(batch_size, 3, resolution, resolution, device=device),
        "timesteps": torch.zeros(batch_size, device=device),
        "audio_features": torch.randn(batch_size, 1, phoneme_dim, device=device)
    }
    
    # Example inputs for phoneme mapper
    phoneme_mapper_inputs = {
        "phoneme_features": torch.randn(batch_size, 1, phoneme_dim, device=device),
        "audio_features": torch.randn(batch_size, 1, phoneme_dim, device=device)
    }
    
    # Compile UNet
    print("Compiling UNet...")
    compiled_unet = torch.compile(
        models["unet"],
        mode=mode,
        fullgraph=True
    )
    
    # Warmup UNet
    with torch.no_grad():
        _ = compiled_unet(**unet_inputs)
    
    # Compile phoneme mapper
    print("Compiling phoneme mapper...")
    compiled_phoneme_mapper = torch.compile(
        models["phoneme_mapper"],
        mode=mode,
        fullgraph=True
    )
    
    # Warmup phoneme mapper
    with torch.no_grad():
        _ = compiled_phoneme_mapper(**phoneme_mapper_inputs)
    
    return {
        "unet": compiled_unet,
        "phoneme_mapper": compiled_phoneme_mapper
    }

def save_compiled_models(
    models: Dict[str, torch.nn.Module],
    output_path: str
):
    """
    Save compiled models
    
    Args:
        models: Dictionary of compiled models
        output_path: Path to output file
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save models
    torch.save({
        "unet": models["unet"].state_dict(),
        "phoneme_mapper": models["phoneme_mapper"].state_dict()
    }, output_path)
    
    print(f"Compiled models saved to {output_path}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override configuration with command line arguments
    config.hardware.precision = args.precision
    
    # Load models
    models = load_models(
        config=config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        precision=args.precision
    )
    
    # Compile models
    compiled_models = compile_models(
        models=models,
        config=config,
        device=args.device,
        mode=args.mode
    )
    
    # Save compiled models
    save_compiled_models(
        models=compiled_models,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
