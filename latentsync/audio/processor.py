import os
import torch
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

from latentsync.audio.whisper_model import EnhancedWhisperModel, PhonemeData

@dataclass
class AudioFeatures:
    """Data class for storing processed audio features"""
    phoneme_features: torch.Tensor  # [num_frames, phoneme_dim]
    wav2vec_features: Optional[torch.Tensor] = None  # [num_frames, wav2vec_dim]
    combined_features: Optional[torch.Tensor] = None  # [num_frames, combined_dim]
    phoneme_data: Optional[List[PhonemeData]] = None
    frame_rate: int = 25  # Frames per second
    duration: float = 0.0  # Duration in seconds

class EnhancedAudioProcessor:
    """
    Enhanced audio processor for extracting rich audio features
    for improved mouth articulation in lip sync applications.
    """
    def __init__(
        self, 
        config: Any,
        device: str = "cuda",
        cache_dir: Optional[str] = None
    ):
        self.config = config
        self.device = device
        self.sample_rate = config.audio.sample_rate
        self.frame_rate = 25  # Standard video frame rate
        self.cache_dir = cache_dir
        
        # Initialize Whisper model for phoneme extraction
        self.whisper_model = EnhancedWhisperModel(
            model_name=config.audio.whisper_model,
            device=device,
            cache_dir=cache_dir,
            use_flash_attention=True
        )
        
        # Initialize Wav2Vec2 model for additional audio features if enabled
        self.wav2vec_model = None
        self.wav2vec_processor = None
        
        if hasattr(config.audio, "use_wav2vec") and config.audio.use_wav2vec:
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            
            self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(
                config.audio.wav2vec_model,
                cache_dir=cache_dir
            )
            
            self.wav2vec_model = Wav2Vec2Model.from_pretrained(
                config.audio.wav2vec_model,
                cache_dir=cache_dir
            ).to(device)
            
            # Use half precision for efficiency
            if torch.cuda.is_available() and device == "cuda":
                if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
                    self.wav2vec_model = self.wav2vec_model.to(torch.bfloat16)
                else:
                    self.wav2vec_model = self.wav2vec_model.to(torch.float16)
        
        # Feature dimensions
        self.phoneme_dim = config.audio.phoneme_dim
        self.wav2vec_dim = 1024  # Default dimension for wav2vec features
        self.combined_dim = self.phoneme_dim + (self.wav2vec_dim if self.wav2vec_model else 0)
        
        # Feature projection layers
        self.phoneme_projection = torch.nn.Linear(
            self.whisper_model.model.config.d_model * 3,  # 3 layers concatenated
            self.phoneme_dim
        ).to(device)
        
        if self.wav2vec_model:
            self.wav2vec_projection = torch.nn.Linear(
                self.wav2vec_dim,
                self.phoneme_dim
            ).to(device)
            
        # Feature combiner
        if self.wav2vec_model:
            self.feature_combiner = torch.nn.Sequential(
                torch.nn.Linear(self.phoneme_dim * 2, self.phoneme_dim),
                torch.nn.LayerNorm(self.phoneme_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(self.phoneme_dim, self.phoneme_dim)
            ).to(device)
    
    def process_audio(
        self, 
        audio_path: str,
        language: str = "en",
        return_all_features: bool = False
    ) -> AudioFeatures:
        """
        Process audio to extract enhanced features for lip sync
        
        Args:
            audio_path: Path to audio file
            language: Language code
            return_all_features: Whether to return all intermediate features
            
        Returns:
            AudioFeatures object containing processed features
        """
        # Extract phonemes using Whisper
        whisper_output = self.whisper_model.extract_phonemes(
            audio_path=audio_path,
            language=language
        )
        
        # Get frame-level phoneme features
        phoneme_features = whisper_output["frame_features"]
        
        # Project phoneme features to desired dimension
        phoneme_features = self.phoneme_projection(phoneme_features)
        
        # Initialize audio features
        audio_features = AudioFeatures(
            phoneme_features=phoneme_features,
            phoneme_data=whisper_output.get("phoneme_data"),
            frame_rate=self.frame_rate,
            duration=len(phoneme_features) / self.frame_rate
        )
        
        # Extract additional audio features using Wav2Vec2 if enabled
        if self.wav2vec_model:
            wav2vec_features = self._extract_wav2vec_features(audio_path)
            
            # Project wav2vec features to match phoneme dimension
            wav2vec_features = self.wav2vec_projection(wav2vec_features)
            
            # Combine features
            combined_features = self._combine_features(
                phoneme_features, 
                wav2vec_features
            )
            
            # Update audio features
            audio_features.wav2vec_features = wav2vec_features
            audio_features.combined_features = combined_features
        else:
            # If wav2vec is not used, combined features are just phoneme features
            audio_features.combined_features = phoneme_features
        
        return audio_features
    
    def _extract_wav2vec_features(self, audio_path: str) -> torch.Tensor:
        """
        Extract features using Wav2Vec2 model
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Frame-level wav2vec features
        """
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, 
                sample_rate, 
                self.sample_rate
            )
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Process with Wav2Vec2
        inputs = self.wav2vec_processor(
            waveform.squeeze().numpy(), 
            sampling_rate=self.sample_rate, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            wav2vec_outputs = self.wav2vec_model(
                inputs.input_values.to(self.device), 
                output_hidden_states=True
            )
        
        # Get hidden states
        hidden_states = wav2vec_outputs.hidden_states
        
        # Concatenate last 3 layers for richer representation
        concat_hidden = torch.cat([
            hidden_states[-1],
            hidden_states[-2],
            hidden_states[-3]
        ], dim=-1)
        
        # Interpolate to match video frame rate
        wav2vec_features = self._interpolate_to_frames(
            concat_hidden.squeeze(0),
            waveform.shape[1] / self.sample_rate,
            self.frame_rate
        )
        
        return wav2vec_features
    
    def _interpolate_to_frames(
        self, 
        features: torch.Tensor, 
        audio_duration: float,
        frame_rate: int
    ) -> torch.Tensor:
        """
        Interpolate features to match video frame rate
        
        Args:
            features: Feature tensor [seq_len, feature_dim]
            audio_duration: Audio duration in seconds
            frame_rate: Video frame rate
            
        Returns:
            Frame-level features [num_frames, feature_dim]
        """
        # Calculate number of frames
        num_frames = int(audio_duration * frame_rate)
        
        # Get feature dimensions
        seq_len, feature_dim = features.shape
        
        # Create empty frame features
        frame_features = torch.zeros(
            (num_frames, feature_dim), 
            device=features.device
        )
        
        # Calculate time per feature and per frame
        time_per_feature = audio_duration / seq_len
        time_per_frame = 1.0 / frame_rate
        
        # Interpolate features to frames
        for i in range(num_frames):
            frame_time = i * time_per_frame
            
            # Find closest features
            feature_idx = int(frame_time / time_per_feature)
            
            # Ensure index is within bounds
            feature_idx = min(feature_idx, seq_len - 1)
            
            # Get feature
            frame_features[i] = features[feature_idx]
        
        return frame_features
    
    def _combine_features(
        self, 
        phoneme_features: torch.Tensor, 
        wav2vec_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine phoneme and wav2vec features
        
        Args:
            phoneme_features: Phoneme features [num_frames, phoneme_dim]
            wav2vec_features: Wav2Vec2 features [num_frames, wav2vec_dim]
            
        Returns:
            Combined features [num_frames, combined_dim]
        """
        # Ensure same number of frames
        min_frames = min(phoneme_features.shape[0], wav2vec_features.shape[0])
        phoneme_features = phoneme_features[:min_frames]
        wav2vec_features = wav2vec_features[:min_frames]
        
        # Concatenate features
        concat_features = torch.cat([
            phoneme_features,
            wav2vec_features
        ], dim=-1)
        
        # Apply feature combiner
        combined_features = self.feature_combiner(concat_features)
        
        return combined_features
