import torch
import math
from typing import Dict, Optional, Tuple, Union, Any

def get_optimal_batch_size(
    resolution: int, 
    available_vram_gb: float,
    model_type: str = "unet",
    precision: str = "float16"
) -> int:
    """
    Calculate optimal batch size based on resolution and available VRAM
    
    Args:
        resolution: Image resolution (assuming square images)
        available_vram_gb: Available VRAM in GB
        model_type: Model type (unet, vae, etc.)
        precision: Precision (float32, float16, bfloat16)
        
    Returns:
        Optimal batch size
    """
    # Approximate VRAM usage per sample at different resolutions
    # These are empirical values and may need adjustment
    vram_usage_per_sample = {
        "unet": {
            256: {
                "float32": 1.2,  # GB
                "float16": 0.6,  # GB
                "bfloat16": 0.6  # GB
            },
            512: {
                "float32": 4.8,  # GB
                "float16": 2.4,  # GB
                "bfloat16": 2.4  # GB
            },
            768: {
                "float32": 10.8,  # GB
                "float16": 5.4,  # GB
                "bfloat16": 5.4  # GB
            },
            1024: {
                "float32": 19.2,  # GB
                "float16": 9.6,  # GB
                "bfloat16": 9.6  # GB
            }
        },
        "vae": {
            256: {
                "float32": 0.3,  # GB
                "float16": 0.15,  # GB
                "bfloat16": 0.15  # GB
            },
            512: {
                "float32": 1.2,  # GB
                "float16": 0.6,  # GB
                "bfloat16": 0.6  # GB
            },
            768: {
                "float32": 2.7,  # GB
                "float16": 1.35,  # GB
                "bfloat16": 1.35  # GB
            },
            1024: {
                "float32": 4.8,  # GB
                "float16": 2.4,  # GB
                "bfloat16": 2.4  # GB
            }
        }
    }
    
    # Get closest resolution key
    available_resolutions = list(vram_usage_per_sample[model_type].keys())
    closest_res = min(available_resolutions, key=lambda x: abs(x - resolution))
    
    # Get VRAM usage per sample
    vram_per_sample = vram_usage_per_sample[model_type][closest_res][precision]
    
    # Calculate batch size with 20% buffer for other operations
    optimal_batch_size = max(1, int((available_vram_gb * 0.8) / vram_per_sample))
    
    return optimal_batch_size

def get_vram_info() -> Dict[str, Union[float, str]]:
    """
    Get VRAM information
    
    Returns:
        Dictionary containing VRAM information
    """
    if not torch.cuda.is_available():
        return {
            "available": False,
            "device_name": "CPU",
            "total_vram_gb": 0,
            "free_vram_gb": 0
        }
    
    device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device)
    total_vram = torch.cuda.get_device_properties(device).total_memory
    free_vram = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    
    # Convert to GB
    total_vram_gb = total_vram / (1024 ** 3)
    free_vram_gb = free_vram / (1024 ** 3)
    
    return {
        "available": True,
        "device_name": device_name,
        "total_vram_gb": total_vram_gb,
        "free_vram_gb": free_vram_gb
    }

def auto_configure_batch_size(config: Any) -> int:
    """
    Automatically configure batch size based on available VRAM and model configuration
    
    Args:
        config: Model configuration
        
    Returns:
        Configured batch size
    """
    # Get VRAM information
    vram_info = get_vram_info()
    
    if not vram_info["available"]:
        # If CUDA is not available, use CPU batch size
        return config.training.min_batch_size
    
    # Get available VRAM
    available_vram_gb = vram_info["free_vram_gb"]
    
    # If hardware.vram_gb is specified, use that instead
    if hasattr(config.hardware, "vram_gb"):
        available_vram_gb = config.hardware.vram_gb
    
    # Get optimal batch size
    optimal_batch_size = get_optimal_batch_size(
        resolution=config.model.resolution,
        available_vram_gb=available_vram_gb,
        model_type="unet",
        precision=config.hardware.precision
    )
    
    # Clamp batch size to min/max
    batch_size = max(config.training.min_batch_size, min(optimal_batch_size, config.training.max_batch_size))
    
    return batch_size

def get_tiled_processing_params(
    resolution: int,
    tile_size: int = 256,
    tile_overlap: int = 32
) -> Dict[str, Any]:
    """
    Get parameters for tiled processing of large images
    
    Args:
        resolution: Image resolution (assuming square images)
        tile_size: Tile size
        tile_overlap: Overlap between tiles
        
    Returns:
        Dictionary containing tiled processing parameters
    """
    # Calculate number of tiles
    num_tiles_x = math.ceil(resolution / (tile_size - tile_overlap))
    num_tiles_y = math.ceil(resolution / (tile_size - tile_overlap))
    
    # Calculate effective tile size
    effective_tile_size_x = math.ceil(resolution / num_tiles_x) + tile_overlap
    effective_tile_size_y = math.ceil(resolution / num_tiles_y) + tile_overlap
    
    # Calculate tile positions
    tile_positions = []
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            # Calculate tile position
            pos_x = x * (effective_tile_size_x - tile_overlap)
            pos_y = y * (effective_tile_size_y - tile_overlap)
            
            # Ensure tile doesn't go beyond image bounds
            pos_x = min(pos_x, resolution - effective_tile_size_x)
            pos_y = min(pos_y, resolution - effective_tile_size_y)
            
            # Add tile position
            tile_positions.append((pos_x, pos_y))
    
    return {
        "num_tiles_x": num_tiles_x,
        "num_tiles_y": num_tiles_y,
        "effective_tile_size_x": effective_tile_size_x,
        "effective_tile_size_y": effective_tile_size_y,
        "tile_positions": tile_positions,
        "tile_overlap": tile_overlap
    }
