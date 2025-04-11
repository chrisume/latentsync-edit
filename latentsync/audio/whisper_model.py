import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig
from dataclasses import dataclass

@dataclass
class PhonemeData:
    """Data class for storing phoneme information with timing"""
    phoneme: str
    start_time: float
    end_time: float
    confidence: float
    features: torch.Tensor

class EnhancedWhisperModel:
    """
    Enhanced Whisper model for improved phoneme extraction and timing
    for better mouth articulation in lip sync applications.
    """
    def __init__(
        self, 
        model_name: str = "openai/whisper-large-v3", 
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        use_flash_attention: bool = True
    ):
        self.device = device
        self.model_name = model_name
        
        # Configure model with flash attention if available
        config = WhisperConfig.from_pretrained(model_name)
        if use_flash_attention and hasattr(config, "attn_implementation"):
            config.attn_implementation = "flash_attention_2"
        
        # Load processor and model
        self.processor = WhisperProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name, 
            config=config,
            cache_dir=cache_dir
        ).to(device)
        
        # Enable more detailed phoneme extraction
        if hasattr(self.model.config, "return_timestamps"):
            self.model.config.return_timestamps = True
        
        # Use half precision for efficiency on high-end GPUs
        if torch.cuda.is_available() and device == "cuda":
            if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
                self.model = self.model.to(torch.bfloat16)
            else:
                self.model = self.model.to(torch.float16)
    
    def extract_phonemes(
        self, 
        audio_path: str, 
        language: str = "en",
        return_timestamps: bool = True,
        chunk_length_s: int = 30,
        stride_length_s: int = 5
    ) -> Dict[str, Union[List[PhonemeData], torch.Tensor]]:
        """
        Extract detailed phoneme information with timing from audio file
        
        Args:
            audio_path: Path to audio file
            language: Language code
            return_timestamps: Whether to return timestamps
            chunk_length_s: Length of audio chunks in seconds
            stride_length_s: Stride length for overlapping chunks
            
        Returns:
            Dictionary containing phoneme data and features
        """
        # Load and process audio
        audio_input = self.processor.feature_extractor(
            audio_path, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).to(self.device)
        
        # Process in chunks for long audio
        audio_length = audio_input["input_features"].shape[-1] / 16000  # in seconds
        
        if audio_length > chunk_length_s:
            return self._process_long_audio(
                audio_input, 
                language, 
                chunk_length_s, 
                stride_length_s
            )
        
        # Get model outputs with detailed phoneme information
        with torch.no_grad():
            outputs = self.model.generate(
                **audio_input,
                language=language,
                task="transcribe",
                return_timestamps=return_timestamps,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
            
        # Process outputs to get phoneme-level features
        phoneme_data = self._process_outputs(outputs, audio_input)
        return phoneme_data
    
    def _process_outputs(
        self, 
        outputs, 
        audio_input
    ) -> Dict[str, Union[List[PhonemeData], torch.Tensor]]:
        """
        Process model outputs to extract phoneme-level features
        
        Args:
            outputs: Model outputs
            audio_input: Audio input features
            
        Returns:
            Dictionary containing phoneme data and features
        """
        # Extract hidden states from the decoder
        hidden_states = outputs.hidden_states
        
        # Get the last hidden state from each layer
        last_hidden_states = [states[-1] for states in hidden_states]
        
        # Concatenate hidden states from the last few layers for richer representation
        concat_hidden = torch.cat([
            last_hidden_states[-1],
            last_hidden_states[-2],
            last_hidden_states[-3]
        ], dim=-1)
        
        # Get token timestamps from the model
        # This is a simplified approach - in practice, you'd use the timestamps from the tokenizer
        token_timestamps = self._estimate_token_timestamps(outputs, audio_input)
        
        # Map tokens to phonemes (simplified)
        phoneme_data = self._map_tokens_to_phonemes(outputs, concat_hidden, token_timestamps)
        
        # Create frame-level features by interpolating phoneme features
        frame_features = self._create_frame_features(phoneme_data, audio_input)
        
        return {
            "phoneme_data": phoneme_data,
            "frame_features": frame_features
        }
    
    def _estimate_token_timestamps(self, outputs, audio_input) -> List[Tuple[float, float]]:
        """
        Estimate timestamps for each token in the output
        
        Args:
            outputs: Model outputs
            audio_input: Audio input features
            
        Returns:
            List of (start_time, end_time) tuples for each token
        """
        # This is a simplified implementation
        # In practice, you would use the timestamps from the tokenizer or model
        
        # Get sequence length and audio duration
        seq_len = outputs.sequences.shape[1]
        audio_duration = audio_input["input_features"].shape[-1] / 16000  # in seconds
        
        # Estimate timestamps by dividing audio duration by sequence length
        timestamps = []
        for i in range(seq_len):
            start_time = (i / seq_len) * audio_duration
            end_time = ((i + 1) / seq_len) * audio_duration
            timestamps.append((start_time, end_time))
            
        return timestamps
    
    def _map_tokens_to_phonemes(
        self, 
        outputs, 
        hidden_states, 
        timestamps
    ) -> List[PhonemeData]:
        """
        Map tokens to phonemes with timing information
        
        Args:
            outputs: Model outputs
            hidden_states: Hidden states from the model
            timestamps: Token timestamps
            
        Returns:
            List of PhonemeData objects
        """
        # Decode tokens to text
        tokens = outputs.sequences[0].cpu().numpy()
        text = self.processor.decode(tokens)
        
        # Split text into words (simplified)
        words = text.split()
        
        # Map words to phonemes (simplified)
        # In practice, you would use a phoneme dictionary or G2P model
        phoneme_data = []
        
        # This is a very simplified approach
        # In a real implementation, you would use a proper G2P (Grapheme-to-Phoneme) converter
        for i, word in enumerate(words):
            if i < len(timestamps):
                start_time, end_time = timestamps[i]
                
                # Get hidden state for this token
                token_hidden = hidden_states[i]
                
                # Create phoneme data
                phoneme_data.append(
                    PhonemeData(
                        phoneme=word,
                        start_time=start_time,
                        end_time=end_time,
                        confidence=0.9,  # Placeholder
                        features=token_hidden
                    )
                )
        
        return phoneme_data
    
    def _create_frame_features(
        self, 
        phoneme_data: List[PhonemeData], 
        audio_input
    ) -> torch.Tensor:
        """
        Create frame-level features by interpolating phoneme features
        
        Args:
            phoneme_data: List of PhonemeData objects
            audio_input: Audio input features
            
        Returns:
            Frame-level features tensor
        """
        # Calculate audio duration and frame rate
        audio_duration = audio_input["input_features"].shape[-1] / 16000  # in seconds
        frame_rate = 25  # Frames per second for video
        num_frames = int(audio_duration * frame_rate)
        
        # Create empty frame features
        feature_dim = phoneme_data[0].features.shape[-1] if phoneme_data else 1024
        frame_features = torch.zeros((num_frames, feature_dim), device=self.device)
        
        # Interpolate phoneme features to frame features
        for i in range(num_frames):
            frame_time = i / frame_rate
            
            # Find phonemes that overlap with this frame
            overlapping_phonemes = [
                p for p in phoneme_data 
                if p.start_time <= frame_time <= p.end_time
            ]
            
            if overlapping_phonemes:
                # Average features of overlapping phonemes
                frame_features[i] = torch.mean(
                    torch.stack([p.features for p in overlapping_phonemes]), 
                    dim=0
                )
            elif i > 0:
                # If no overlapping phonemes, use previous frame
                frame_features[i] = frame_features[i-1]
        
        return frame_features
    
    def _process_long_audio(
        self, 
        audio_input, 
        language, 
        chunk_length_s, 
        stride_length_s
    ) -> Dict[str, Union[List[PhonemeData], torch.Tensor]]:
        """
        Process long audio by chunking it into smaller pieces
        
        Args:
            audio_input: Audio input features
            language: Language code
            chunk_length_s: Length of audio chunks in seconds
            stride_length_s: Stride length for overlapping chunks
            
        Returns:
            Dictionary containing phoneme data and features
        """
        # This is a placeholder implementation
        # In practice, you would chunk the audio and process each chunk
        
        # For now, just process the first chunk
        chunk_input = {
            "input_features": audio_input["input_features"][:, :, :int(chunk_length_s * 16000)]
        }
        
        with torch.no_grad():
            outputs = self.model.generate(
                **chunk_input,
                language=language,
                task="transcribe",
                return_timestamps=True,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
            
        # Process outputs to get phoneme-level features
        phoneme_data = self._process_outputs(outputs, chunk_input)
        return phoneme_data
