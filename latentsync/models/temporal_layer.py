import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any

class TemporalConsistencyLayer(nn.Module):
    """
    Enhanced temporal consistency layer for smoother mouth movements
    in lip sync applications.
    """
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.feature_dim = config.model.temporal.hidden_dim
        self.window_size = config.model.temporal.window_size
        self.num_layers = config.model.temporal.num_layers
        self.bidirectional = config.model.temporal.bidirectional
        
        # Temporal convolution for capturing local motion patterns
        self.temporal_conv = nn.Conv1d(
            in_channels=self.feature_dim,
            out_channels=self.feature_dim,
            kernel_size=self.window_size,
            padding=self.window_size // 2,
            groups=4  # Group convolution for efficiency
        )
        
        # GRU for modeling sequential dependencies
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=self.feature_dim // 2 if self.bidirectional else self.feature_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        
        # Output projection
        self.output_projection = nn.Linear(self.feature_dim * 2, self.feature_dim)
        
        # Adaptive instance normalization for style consistency
        self.instance_norm = nn.InstanceNorm1d(self.feature_dim, affine=True)
        
        # Attention for long-range dependencies
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Viseme classifier for specific mouth shapes
        self.viseme_classifier = nn.Linear(self.feature_dim, 20)  # 20 common visemes
        
        # Smoothing factor (learnable)
        self.smoothing_factor = nn.Parameter(torch.tensor(0.5))
    
    def forward(
        self, 
        x: torch.Tensor,
        prev_states: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Apply temporal consistency to visual features
        
        Args:
            x: Input features [B, T, D]
            prev_states: Previous hidden states for GRU
            
        Returns:
            Dictionary containing processed features and metadata
        """
        batch_size, seq_len, feat_dim = x.shape
        
        # Apply temporal convolution
        x_conv = x.transpose(1, 2)  # [B, D, T]
        x_conv = self.temporal_conv(x_conv)
        x_conv = x_conv.transpose(1, 2)  # [B, T, D]
        
        # Apply GRU
        x_gru, hidden_states = self.gru(x, prev_states)
        
        # Apply self-attention for long-range dependencies
        x_attn, _ = self.self_attention(x, x, x)
        
        # Combine features
        x_combined = torch.cat([x_conv, x_gru], dim=-1)
        x_out = self.output_projection(x_combined)
        
        # Add attention residual
        x_out = x_out + 0.1 * x_attn
        
        # Apply instance normalization for style consistency
        x_out = x_out.transpose(1, 2)  # [B, D, T]
        x_out = self.instance_norm(x_out)
        x_out = x_out.transpose(1, 2)  # [B, T, D]
        
        # Apply smoothing with learnable factor
        smoothing_factor = torch.sigmoid(self.smoothing_factor)  # Constrain to [0, 1]
        x_out = smoothing_factor * x_out + (1 - smoothing_factor) * x
        
        # Classify visemes
        viseme_logits = self.viseme_classifier(x_out)
        
        return {
            "features": x_out,
            "hidden_states": hidden_states,
            "viseme_logits": viseme_logits
        }
    
    def smooth_sequence(self, x: torch.Tensor, window_size: int = 5) -> torch.Tensor:
        """
        Apply additional temporal smoothing to a sequence
        
        Args:
            x: Input sequence [B, T, D]
            window_size: Smoothing window size
            
        Returns:
            Smoothed sequence [B, T, D]
        """
        # Apply moving average smoothing
        batch_size, seq_len, feat_dim = x.shape
        
        # Pad sequence for convolution
        x_padded = F.pad(x, (0, 0, window_size // 2, window_size // 2), mode="replicate")
        
        # Create smoothing kernel
        kernel = torch.ones(1, 1, window_size, device=x.device) / window_size
        
        # Apply smoothing for each feature dimension
        x_smoothed = torch.zeros_like(x)
        for b in range(batch_size):
            for d in range(feat_dim):
                x_slice = x_padded[b, :, d].unsqueeze(0).unsqueeze(0)
                x_smoothed[b, :, d] = F.conv1d(x_slice, kernel).squeeze()
        
        return x_smoothed
