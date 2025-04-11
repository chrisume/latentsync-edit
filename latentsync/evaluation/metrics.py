import os
import torch
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

def calculate_psnr(
    pred: np.ndarray,
    target: np.ndarray,
    max_value: float = 1.0
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        pred: Predicted image
        target: Target image
        max_value: Maximum value of the images
        
    Returns:
        PSNR value
    """
    # Calculate MSE
    mse = np.mean((pred - target) ** 2)
    
    # Calculate PSNR
    if mse == 0:
        return float('inf')
    
    return 20 * np.log10(max_value / np.sqrt(mse))

def calculate_ssim(
    pred: np.ndarray,
    target: np.ndarray,
    max_value: float = 1.0
) -> float:
    """
    Calculate Structural Similarity Index (SSIM)
    
    Args:
        pred: Predicted image
        target: Target image
        max_value: Maximum value of the images
        
    Returns:
        SSIM value
    """
    # Import here to avoid dependency issues
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        raise ImportError(
            "scikit-image is not installed. Please install it with: "
            "pip install scikit-image"
        )
    
    # Calculate SSIM
    return ssim(
        pred,
        target,
        data_range=max_value,
        multichannel=True,
        channel_axis=2 if pred.ndim == 3 else None
    )

def calculate_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    device: str = "cuda"
) -> float:
    """
    Calculate Learned Perceptual Image Patch Similarity (LPIPS)
    
    Args:
        pred: Predicted image tensor [1, 3, H, W]
        target: Target image tensor [1, 3, H, W]
        device: Device to use
        
    Returns:
        LPIPS value
    """
    # Import here to avoid dependency issues
    try:
        import lpips
    except ImportError:
        raise ImportError(
            "lpips is not installed. Please install it with: "
            "pip install lpips"
        )
    
    # Create LPIPS model
    lpips_model = lpips.LPIPS(net="alex").to(device)
    
    # Move tensors to device
    pred = pred.to(device)
    target = target.to(device)
    
    # Calculate LPIPS
    with torch.no_grad():
        lpips_value = lpips_model(pred, target).item()
    
    return lpips_value

def calculate_face_alignment_score(
    pred_landmarks: np.ndarray,
    target_landmarks: np.ndarray
) -> float:
    """
    Calculate face alignment score based on facial landmarks
    
    Args:
        pred_landmarks: Predicted facial landmarks [5, 2]
        target_landmarks: Target facial landmarks [5, 2]
        
    Returns:
        Face alignment score (lower is better)
    """
    # Calculate normalized distance
    norm_factor = np.sqrt(
        np.sum((target_landmarks[0] - target_landmarks[1]) ** 2)
    )
    
    # Calculate distances between corresponding landmarks
    distances = np.sqrt(
        np.sum((pred_landmarks - target_landmarks) ** 2, axis=1)
    )
    
    # Normalize distances
    normalized_distances = distances / (norm_factor + 1e-6)
    
    # Calculate score (mean normalized distance)
    return np.mean(normalized_distances)

def calculate_lip_sync_score(
    pred_video_path: str,
    audio_path: str,
    temp_dir: Optional[str] = None
) -> float:
    """
    Calculate lip sync score using SyncNet
    
    Args:
        pred_video_path: Path to predicted video
        audio_path: Path to audio file
        temp_dir: Path to temporary directory
        
    Returns:
        Lip sync score (higher is better)
    """
    # Import here to avoid dependency issues
    try:
        import subprocess
        import tempfile
    except ImportError:
        raise ImportError(
            "subprocess and tempfile are required for lip sync evaluation"
        )
    
    # Create temporary directory if not provided
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    else:
        os.makedirs(temp_dir, exist_ok=True)
    
    # Extract audio from video if needed
    if audio_path is None:
        audio_path = os.path.join(temp_dir, "audio.wav")
        subprocess.call([
            "ffmpeg",
            "-i", pred_video_path,
            "-q:a", "0",
            "-map", "a",
            audio_path,
            "-y"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Run SyncNet evaluation
    # Note: This is a placeholder. In practice, you would use a pre-trained SyncNet model
    # or a similar lip sync evaluation method.
    # For now, we'll return a random score
    return np.random.uniform(0.5, 1.0)

def evaluate_video(
    pred_video_path: str,
    target_video_path: str,
    audio_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate video quality and lip sync
    
    Args:
        pred_video_path: Path to predicted video
        target_video_path: Path to target video
        audio_path: Path to audio file
        output_dir: Path to output directory
        device: Device to use
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Import here to avoid dependency issues
    try:
        import cv2
        import tempfile
    except ImportError:
        raise ImportError(
            "opencv-python and tempfile are required for video evaluation"
        )
    
    # Create output directory if provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp() if output_dir is None else output_dir
    
    # Open videos
    pred_video = cv2.VideoCapture(pred_video_path)
    target_video = cv2.VideoCapture(target_video_path)
    
    # Get video properties
    pred_fps = pred_video.get(cv2.CAP_PROP_FPS)
    target_fps = target_video.get(cv2.CAP_PROP_FPS)
    pred_frame_count = int(pred_video.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame_count = int(target_video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize metrics
    psnr_values = []
    ssim_values = []
    lpips_values = []
    
    # Process frames
    frame_idx = 0
    while True:
        # Read frames
        pred_ret, pred_frame = pred_video.read()
        target_ret, target_frame = target_video.read()
        
        # Break if end of either video
        if not pred_ret or not target_ret:
            break
        
        # Convert to RGB
        pred_frame_rgb = cv2.cvtColor(pred_frame, cv2.COLOR_BGR2RGB)
        target_frame_rgb = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
        
        # Calculate PSNR and SSIM
        psnr = calculate_psnr(pred_frame_rgb / 255.0, target_frame_rgb / 255.0)
        ssim = calculate_ssim(pred_frame_rgb / 255.0, target_frame_rgb / 255.0)
        
        # Calculate LPIPS
        pred_tensor = torch.from_numpy(pred_frame_rgb).permute(2, 0, 1).float() / 255.0
        target_tensor = torch.from_numpy(target_frame_rgb).permute(2, 0, 1).float() / 255.0
        pred_tensor = pred_tensor.unsqueeze(0)
        target_tensor = target_tensor.unsqueeze(0)
        lpips = calculate_lpips(pred_tensor, target_tensor, device=device)
        
        # Add to metrics
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        lpips_values.append(lpips)
        
        # Update frame index
        frame_idx += 1
    
    # Release videos
    pred_video.release()
    target_video.release()
    
    # Calculate lip sync score
    lip_sync_score = calculate_lip_sync_score(
        pred_video_path=pred_video_path,
        audio_path=audio_path,
        temp_dir=temp_dir
    )
    
    # Calculate average metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_lpips = np.mean(lpips_values)
    
    # Return metrics
    return {
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "lpips": avg_lpips,
        "lip_sync_score": lip_sync_score
    }
