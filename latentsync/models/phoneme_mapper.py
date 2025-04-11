import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any

class EnhancedPhonemeMapper(nn.Module):
    """
    Enhanced phoneme mapper for improved mapping between phonemes and visual features
    for better mouth articulation in lip sync applications.
    """
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        
        # Feature dimensions
        self.phoneme_dim = config.audio.phoneme_dim
        self.hidden_dim = config.model.temporal.hidden_dim
        self.visual_dim = config.model.temporal.hidden_dim
        
        # Number of visemes (mouth shapes)
        self.num_visemes = 20  # Common number of visemes
        
        # Enhanced mapping network
        self.mapping_network = nn.Sequential(
            nn.Linear(self.phoneme_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.visual_dim)
        )
        
        # Audio feature projection
        self.audio_projection = nn.Linear(self.phoneme_dim, self.hidden_dim)
        
        # Attention mechanism for better temporal alignment
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Viseme classifier for specific mouth shapes
        self.viseme_classifier = nn.Linear(self.hidden_dim, self.num_visemes)
        
        # Temporal consistency layer
        from latentsync.models.temporal_layer import TemporalConsistencyLayer
        self.temporal_layer = TemporalConsistencyLayer(config)
    
    def forward(
        self, 
        phoneme_features: torch.Tensor, 
        audio_features: Optional[torch.Tensor] = None,
        prev_states: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Map phoneme features to visual features for lip sync
        
        Args:
            phoneme_features: Phoneme features [B, T, D]
            audio_features: Additional audio features [B, T, D]
            prev_states: Previous hidden states for temporal layer
            
        Returns:
            Dictionary containing visual features and metadata
        """
        # Initial mapping
        x = self.mapping_network(phoneme_features)
        
        # Apply attention for temporal context if audio features are provided
        if audio_features is not None:
            # Project audio features to the same dimension
            audio_proj = self.audio_projection(audio_features)
            
            # Apply cross-attention
            attn_output, _ = self.attention(
                query=x,
                key=audio_proj,
                value=audio_proj
            )
            
            # Combine with original features
            x = x + attn_output
        
        # Classify visemes for specific mouth shapes
        viseme_logits = self.viseme_classifier(x)
        
        # Apply temporal consistency
        temporal_output = self.temporal_layer(x, prev_states)
        
        # Get final visual features
        visual_features = temporal_output["features"]
        
        return {
            'visual_features': visual_features,
            'viseme_logits': viseme_logits,
            'hidden_states': temporal_output["hidden_states"],
            'temporal_viseme_logits': temporal_output["viseme_logits"]
        }
    
    def get_viseme_probabilities(self, viseme_logits: torch.Tensor) -> torch.Tensor:
        """
        Get viseme probabilities from logits
        
        Args:
            viseme_logits: Viseme logits [B, T, num_visemes]
            
        Returns:
            Viseme probabilities [B, T, num_visemes]
        """
        return F.softmax(viseme_logits, dim=-1)
    
    def get_dominant_viseme(self, viseme_logits: torch.Tensor) -> torch.Tensor:
        """
        Get dominant viseme for each frame
        
        Args:
            viseme_logits: Viseme logits [B, T, num_visemes]
            
        Returns:
            Dominant viseme indices [B, T]
        """
        return torch.argmax(viseme_logits, dim=-1)
