import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any

class LatentSyncLoss(nn.Module):
    """
    Loss function for LatentSync training
    """
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        
        # Initialize loss weights
        self.reconstruction_weight = 1.0
        self.perceptual_weight = 0.1
        self.viseme_weight = 0.05
        
        # Initialize perceptual loss if enabled
        self.use_perceptual_loss = True
        if self.use_perceptual_loss:
            self.perceptual_loss = PerceptualLoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        phoneme_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate loss
        
        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
            phoneme_output: Output from phoneme mapper
            
        Returns:
            Dictionary of losses
        """
        # Calculate reconstruction loss (L1 + SSIM)
        reconstruction_loss = self._reconstruction_loss(pred, target)
        
        # Initialize total loss
        total_loss = self.reconstruction_weight * reconstruction_loss
        
        # Initialize loss dictionary
        loss_dict = {
            "reconstruction_loss": reconstruction_loss,
            "total_loss": total_loss
        }
        
        # Add perceptual loss if enabled
        if self.use_perceptual_loss:
            perceptual_loss = self.perceptual_loss(pred, target)
            total_loss = total_loss + self.perceptual_weight * perceptual_loss
            loss_dict["perceptual_loss"] = perceptual_loss
            loss_dict["total_loss"] = total_loss
        
        # Add viseme loss if available
        if "viseme_logits" in phoneme_output and "temporal_viseme_logits" in phoneme_output:
            viseme_loss = F.mse_loss(
                phoneme_output["viseme_logits"],
                phoneme_output["temporal_viseme_logits"].detach()
            )
            total_loss = total_loss + self.viseme_weight * viseme_loss
            loss_dict["viseme_loss"] = viseme_loss
            loss_dict["total_loss"] = total_loss
        
        return loss_dict
    
    def _reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate reconstruction loss (L1 + SSIM)
        
        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
            
        Returns:
            Reconstruction loss
        """
        # Calculate L1 loss
        l1_loss = F.l1_loss(pred, target)
        
        # Calculate SSIM loss
        ssim_loss = 1.0 - ssim(pred, target)
        
        # Combine losses
        return l1_loss + ssim_loss

class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features
    """
    def __init__(self, layers: list = None):
        super().__init__()
        
        # Import here to avoid dependency issues
        try:
            from torchvision.models import vgg16
            from torchvision.models.feature_extraction import create_feature_extractor
        except ImportError:
            raise ImportError(
                "torchvision is not installed. Please install it with: "
                "pip install torchvision"
            )
        
        # Define layers to extract features from
        if layers is None:
            layers = ["features.4", "features.9", "features.16", "features.23"]
        
        # Load VGG16 model
        vgg = vgg16(pretrained=True)
        vgg.eval()
        
        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Create feature extractor
        self.feature_extractor = create_feature_extractor(vgg, return_nodes=layers)
        
        # Register buffer for mean and std
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate perceptual loss
        
        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
            
        Returns:
            Perceptual loss
        """
        # Normalize images
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        # Extract features
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        
        # Calculate loss
        loss = 0.0
        for key in pred_features:
            loss = loss + F.l1_loss(pred_features[key], target_features[key])
        
        return loss

def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    size_average: bool = True
) -> torch.Tensor:
    """
    Calculate SSIM
    
    Args:
        pred: Predicted image [B, C, H, W]
        target: Target image [B, C, H, W]
        window_size: Window size for SSIM
        sigma: Sigma for Gaussian window
        size_average: Whether to average over batch
        
    Returns:
        SSIM value
    """
    # Create Gaussian window
    window = _create_window(window_size, sigma, pred.device, pred.dtype, pred.shape[1])
    
    # Calculate SSIM
    return _ssim(pred, target, window, window_size, pred.shape[1], size_average)

def _create_window(
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
    channels: int
) -> torch.Tensor:
    """
    Create Gaussian window for SSIM
    
    Args:
        window_size: Window size
        sigma: Sigma for Gaussian window
        device: Device for window
        dtype: Data type for window
        channels: Number of channels
        
    Returns:
        Gaussian window
    """
    # Create 1D Gaussian window
    window_1d = torch.exp(
        -torch.arange(window_size, device=device, dtype=dtype).pow(2) / (2 * sigma ** 2)
    )
    window_1d = window_1d / window_1d.sum()
    
    # Create 2D Gaussian window
    window_2d = window_1d.unsqueeze(1) * window_1d.unsqueeze(0)
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)
    window_2d = window_2d.expand(channels, 1, window_size, window_size)
    
    return window_2d

def _ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channels: int,
    size_average: bool = True
) -> torch.Tensor:
    """
    Calculate SSIM
    
    Args:
        pred: Predicted image [B, C, H, W]
        target: Target image [B, C, H, W]
        window: Gaussian window
        window_size: Window size
        channels: Number of channels
        size_average: Whether to average over batch
        
    Returns:
        SSIM value
    """
    # Calculate means
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channels)
    
    # Calculate squared means
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu1_mu2
    
    # Constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # Average if needed
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
