import math
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class HighResUNet(nn.Module):
    """
    Enhanced UNet model for high-resolution image generation in LatentSync.
    Supports resolutions of 512x512 and higher.
    """
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        
        # Model parameters
        self.in_channels = config.model.unet.in_channels
        self.model_channels = config.model.unet.model_channels
        self.out_channels = config.model.unet.in_channels
        self.num_res_blocks = config.model.unet.num_res_blocks
        self.attention_resolutions = config.model.unet.attention_resolutions
        self.dropout = 0.0  # Fixed dropout for stability
        self.channel_mult = config.model.unet.channel_mult
        self.conv_resample = True
        self.num_heads = config.model.unet.num_heads
        self.use_scale_shift_norm = True
        self.resblock_updown = True
        self.use_checkpoint = config.model.unet.use_checkpoint
        self.use_temporal_attention = config.model.temporal.enabled
        
        # Determine attention implementation
        self.attention_type = config.model.attention_type
        
        # Resolution-specific adjustments
        self.resolution = config.model.resolution
        if self.resolution >= 512:
            # Increase model capacity for higher resolutions
            self.model_channels = max(self.model_channels, 320)
            self.num_heads = max(self.num_heads, 16)
            
            # Add additional downsampling for very high resolutions
            if self.resolution >= 768:
                self.channel_mult = list(self.channel_mult) + [4]
        
        # Time embedding
        time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Audio conditioning
        self.audio_dim = config.audio.phoneme_dim
        self.audio_proj = nn.Sequential(
            nn.Linear(self.audio_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Initialize model components
        self._init_layers()
        
        # Apply memory optimizations
        self._apply_optimizations()
    
    def _init_layers(self):
        """Initialize all layers of the UNet model"""
        # Input convolution
        self.input_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.model_channels, kernel_size=3, padding=1)
            )
        ])
        
        # Middle block channels
        self._feature_size = self.model_channels
        input_block_channels = [self.model_channels]
        
        # Downsampling blocks
        ds = 1
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(
                        channels=self._feature_size,
                        emb_channels=self.time_embed.out_features,
                        dropout=self.dropout,
                        out_channels=self.model_channels * mult,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
                self._feature_size = self.model_channels * mult
                
                # Add attention if needed
                if ds in self.attention_resolutions:
                    layers.append(
                        self._make_attention_block(
                            channels=self._feature_size,
                            num_heads=self.num_heads
                        )
                    )
                
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_channels.append(self._feature_size)
            
            # Add downsampling if not the last level
            if level != len(self.channel_mult) - 1:
                self.input_blocks.append(
                    nn.Sequential(
                        ResBlock(
                            channels=self._feature_size,
                            emb_channels=self.time_embed.out_features,
                            dropout=self.dropout,
                            out_channels=self._feature_size,
                            use_checkpoint=self.use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            down=True,
                        )
                        if self.resblock_updown
                        else nn.Conv2d(
                            self._feature_size, self._feature_size, kernel_size=3, stride=2, padding=1
                        )
                    )
                )
                input_block_channels.append(self._feature_size)
                ds *= 2
        
        # Middle block
        self.middle_block = nn.Sequential(
            ResBlock(
                channels=self._feature_size,
                emb_channels=self.time_embed.out_features,
                dropout=self.dropout,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
            ),
            self._make_attention_block(
                channels=self._feature_size,
                num_heads=self.num_heads
            ),
            ResBlock(
                channels=self._feature_size,
                emb_channels=self.time_embed.out_features,
                dropout=self.dropout,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
            ),
        )
        
        # Upsampling blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                layers = [
                    ResBlock(
                        channels=self._feature_size + input_block_channels.pop(),
                        emb_channels=self.time_embed.out_features,
                        dropout=self.dropout,
                        out_channels=self.model_channels * mult,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
                self._feature_size = self.model_channels * mult
                
                # Add attention if needed
                if ds in self.attention_resolutions:
                    layers.append(
                        self._make_attention_block(
                            channels=self._feature_size,
                            num_heads=self.num_heads
                        )
                    )
                
                # Add upsampling if not the last block
                if level != 0 and i == self.num_res_blocks:
                    layers.append(
                        ResBlock(
                            channels=self._feature_size,
                            emb_channels=self.time_embed.out_features,
                            dropout=self.dropout,
                            out_channels=self._feature_size,
                            use_checkpoint=self.use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            up=True,
                        )
                        if self.resblock_updown
                        else nn.Upsample(scale_factor=2, mode="nearest")
                    )
                    ds //= 2
                
                self.output_blocks.append(nn.Sequential(*layers))
        
        # Final normalization and convolution
        self.out = nn.Sequential(
            nn.GroupNorm(32, self._feature_size),
            nn.SiLU(),
            nn.Conv2d(self._feature_size, self.out_channels, kernel_size=3, padding=1),
        )
    
    def _make_attention_block(self, channels: int, num_heads: int) -> nn.Module:
        """
        Create an attention block based on the configured attention type
        
        Args:
            channels: Number of channels
            num_heads: Number of attention heads
            
        Returns:
            Attention module
        """
        if self.attention_type == "xformers" and self._is_xformers_available():
            from xformers.ops import memory_efficient_attention
            
            return MemoryEfficientAttention(
                channels=channels,
                num_heads=num_heads,
                attention_op=memory_efficient_attention
            )
        elif self.attention_type == "flash_attention" and self._is_flash_attention_available():
            return FlashAttention(
                channels=channels,
                num_heads=num_heads
            )
        else:
            # Fallback to vanilla attention
            return AttentionBlock(
                channels=channels,
                num_heads=num_heads
            )
    
    def _is_xformers_available(self) -> bool:
        """Check if xformers is available"""
        try:
            import xformers
            return True
        except ImportError:
            return False
    
    def _is_flash_attention_available(self) -> bool:
        """Check if flash attention is available"""
        return hasattr(F, "scaled_dot_product_attention")
    
    def _apply_optimizations(self):
        """Apply memory and performance optimizations"""
        # Enable gradient checkpointing if configured
        if self.config.model.use_gradient_checkpointing:
            self.enable_gradient_checkpointing()
        
        # Use channels_last memory format for better performance on CUDA
        if self.config.optimization.channels_last and torch.cuda.is_available():
            self = self.to(memory_format=torch.channels_last)
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        # Enable gradient checkpointing for all ResBlocks and Attention blocks
        for module in self.modules():
            if isinstance(module, ResBlock) or isinstance(module, AttentionBlock):
                module.use_checkpoint = True
    
    def forward(
        self, 
        x: torch.Tensor, 
        timesteps: torch.Tensor, 
        audio_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the UNet model
        
        Args:
            x: Input tensor [B, C, H, W]
            timesteps: Timestep embeddings [B]
            audio_features: Audio conditioning features [B, T, D]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        # Ensure x is in the correct memory format
        if self.config.optimization.channels_last and x.device.type == "cuda":
            x = x.to(memory_format=torch.channels_last)
        
        # Time embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        # Add audio conditioning if provided
        if audio_features is not None:
            audio_emb = self.audio_proj(audio_features)
            emb = emb + audio_emb
        
        # Process input blocks
        h = x
        hs = []
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        
        # Process middle block
        h = self.middle_block(h, emb)
        
        # Process output blocks
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        
        # Final output
        return self.out(h)


# Helper modules

class ResBlock(nn.Module):
    """
    Residual block with time embedding and optional up/downsampling
    """
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        use_scale_shift_norm: bool = False,
        down: bool = False,
        up: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # Normalization and activation
        self.in_norm = nn.GroupNorm(32, channels)
        self.in_act = nn.SiLU()
        
        # First convolution
        self.in_conv = nn.Conv2d(
            channels, self.out_channels, kernel_size=3, padding=1
        )
        
        # Upsampling or downsampling
        self.updown = up or down
        if up:
            self.h_upd = nn.Upsample(scale_factor=2, mode="nearest")
            self.x_upd = nn.Upsample(scale_factor=2, mode="nearest")
        elif down:
            self.h_upd = nn.Conv2d(
                self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1
            )
            self.x_upd = nn.Conv2d(
                channels, channels, kernel_size=3, stride=2, padding=1
            )
        
        # Time embedding projection
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )
        
        # Second normalization and activation
        self.out_norm = nn.GroupNorm(32, self.out_channels)
        self.out_act = nn.SiLU()
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Second convolution
        self.out_conv = nn.Conv2d(
            self.out_channels, self.out_channels, kernel_size=3, padding=1
        )
        
        # Skip connection
        if self.out_channels != channels:
            self.skip_connection = nn.Conv2d(
                channels, self.out_channels, kernel_size=1
            )
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResBlock
        
        Args:
            x: Input tensor [B, C, H, W]
            emb: Time embedding [B, D]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        if self.use_checkpoint and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(self._forward, x, emb)
        else:
            return self._forward(x, emb)
    
    def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementation
        
        Args:
            x: Input tensor [B, C, H, W]
            emb: Time embedding [B, D]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        # Skip connection
        skip = self.skip_connection(x)
        
        # Apply first normalization and activation
        h = self.in_norm(x)
        h = self.in_act(h)
        
        # Apply upsampling or downsampling to h if needed
        if self.updown:
            h = self.h_upd(h)
            skip = self.x_upd(skip)
        
        # Apply first convolution
        h = self.in_conv(h)
        
        # Apply time embedding
        emb_out = self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1)
        
        # Apply scale-shift normalization if enabled
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = self.out_norm(h) * (1 + scale) + shift
            h = self.out_act(h)
        else:
            h = h + emb_out
            h = self.out_norm(h)
            h = self.out_act(h)
        
        # Apply dropout
        h = self.dropout_layer(h)
        
        # Apply second convolution
        h = self.out_conv(h)
        
        # Add skip connection
        return h + skip


class AttentionBlock(nn.Module):
    """
    Standard self-attention block
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        
        # Normalization
        self.norm = nn.GroupNorm(32, channels)
        
        # QKV projection
        self.qkv = nn.Conv1d(
            channels, channels * 3, kernel_size=1
        )
        
        # Output projection
        self.proj_out = nn.Conv1d(
            channels, channels, kernel_size=1
        )
        
        # Initialize weights
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
    
    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the AttentionBlock
        
        Args:
            x: Input tensor [B, C, H, W]
            emb: Ignored, included for compatibility with ResBlock
            
        Returns:
            Output tensor [B, C, H, W]
        """
        if self.use_checkpoint and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)
    
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementation
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        b, c, h, w = x.shape
        
        # Apply normalization
        h_norm = self.norm(x)
        
        # Reshape for attention
        h_flat = h_norm.reshape(b, c, -1)  # [B, C, H*W]
        
        # QKV projection
        qkv = self.qkv(h_flat)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # Reshape for multi-head attention
        q = q.reshape(b, self.num_heads, c // self.num_heads, -1)
        k = k.reshape(b, self.num_heads, c // self.num_heads, -1)
        v = v.reshape(b, self.num_heads, c // self.num_heads, -1)
        
        # Compute attention
        scale = (c // self.num_heads) ** -0.5
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        h_attn = torch.matmul(attn, v)
        
        # Reshape back
        h_attn = h_attn.reshape(b, c, -1)
        
        # Output projection
        h_out = self.proj_out(h_attn)
        
        # Reshape to original shape
        h_out = h_out.reshape(b, c, h, w)
        
        # Residual connection
        return x + h_out


class MemoryEfficientAttention(nn.Module):
    """
    Memory-efficient attention using xformers
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        attention_op: Optional[Any] = None,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.attention_op = attention_op
        
        # Normalization
        self.norm = nn.GroupNorm(32, channels)
        
        # QKV projection
        self.qkv = nn.Conv1d(
            channels, channels * 3, kernel_size=1
        )
        
        # Output projection
        self.proj_out = nn.Conv1d(
            channels, channels, kernel_size=1
        )
        
        # Initialize weights
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
    
    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the MemoryEfficientAttention
        
        Args:
            x: Input tensor [B, C, H, W]
            emb: Ignored, included for compatibility with ResBlock
            
        Returns:
            Output tensor [B, C, H, W]
        """
        b, c, h, w = x.shape
        
        # Apply normalization
        h_norm = self.norm(x)
        
        # Reshape for attention
        h_flat = h_norm.reshape(b, c, -1)  # [B, C, H*W]
        
        # QKV projection
        qkv = self.qkv(h_flat)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # Reshape for multi-head attention
        head_dim = c // self.num_heads
        q = q.reshape(b, self.num_heads, head_dim, -1).transpose(2, 3)  # [B, H, L, D]
        k = k.reshape(b, self.num_heads, head_dim, -1).transpose(2, 3)  # [B, H, L, D]
        v = v.reshape(b, self.num_heads, head_dim, -1).transpose(2, 3)  # [B, H, L, D]
        
        # Apply memory-efficient attention
        h_attn = self.attention_op(q, k, v)
        
        # Reshape back
        h_attn = h_attn.transpose(2, 3).reshape(b, c, -1)
        
        # Output projection
        h_out = self.proj_out(h_attn)
        
        # Reshape to original shape
        h_out = h_out.reshape(b, c, h, w)
        
        # Residual connection
        return x + h_out


class FlashAttention(nn.Module):
    """
    Flash attention using PyTorch's scaled_dot_product_attention
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        # Normalization
        self.norm = nn.GroupNorm(32, channels)
        
        # QKV projection
        self.qkv = nn.Conv1d(
            channels, channels * 3, kernel_size=1
        )
        
        # Output projection
        self.proj_out = nn.Conv1d(
            channels, channels, kernel_size=1
        )
        
        # Initialize weights
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
    
    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the FlashAttention
        
        Args:
            x: Input tensor [B, C, H, W]
            emb: Ignored, included for compatibility with ResBlock
            
        Returns:
            Output tensor [B, C, H, W]
        """
        b, c, h, w = x.shape
        
        # Apply normalization
        h_norm = self.norm(x)
        
        # Reshape for attention
        h_flat = h_norm.reshape(b, c, -1)  # [B, C, H*W]
        
        # QKV projection
        qkv = self.qkv(h_flat)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # Reshape for multi-head attention
        head_dim = c // self.num_heads
        q = q.reshape(b, self.num_heads, head_dim, -1).transpose(2, 3)  # [B, H, L, D]
        k = k.reshape(b, self.num_heads, head_dim, -1).transpose(2, 3)  # [B, H, L, D]
        v = v.reshape(b, self.num_heads, head_dim, -1).transpose(2, 3)  # [B, H, L, D]
        
        # Apply flash attention
        h_attn = F.scaled_dot_product_attention(q, k, v)
        
        # Reshape back
        h_attn = h_attn.transpose(2, 3).reshape(b, c, -1)
        
        # Output projection
        h_out = self.proj_out(h_attn)
        
        # Reshape to original shape
        h_out = h_out.reshape(b, c, h, w)
        
        # Residual connection
        return x + h_out


def timestep_embedding(
    timesteps: torch.Tensor, 
    dim: int, 
    max_period: int = 10000
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings
    
    Args:
        timesteps: Timestep tensor [B]
        dim: Embedding dimension
        max_period: Maximum period
        
    Returns:
        Timestep embeddings [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
