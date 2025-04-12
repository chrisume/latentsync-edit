import torch
import librosa
import numpy as np
from typing import List, Dict, Union, Optional, Any, Tuple
from dataclasses import dataclass

@dataclass
class PhonemeData:
    """Data class for storing phoneme data"""
    phoneme: str
    start_time: float
    end_time: float
    confidence: float
    embedding: Optional[torch.Tensor] = None

class WhisperModel:
    """Base Whisper model class"""
    def __init__(self, model_name: str = "openai/whisper-small", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model, self.processor = self._load_model()

    def _load_model(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        model = WhisperForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        processor = WhisperProcessor.from_pretrained(self.model_name)
        return model, processor

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
        # Load audio file
        try:
            print(f"Loading audio file: {audio_path}")
            audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
            print(f"Audio loaded successfully. Duration: {len(audio_array)/sampling_rate:.2f}s")
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            raise
        
        # Process audio with feature extractor
        audio_input = self.processor.feature_extractor(
            audio_array, 
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
        print(f"Processing long audio in chunks. Total length: {audio_input['input_features'].shape[-1]/16000:.2f}s")
        
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

    def _process_outputs(self, outputs, audio_input) -> Dict[str, Union[List[PhonemeData], torch.Tensor]]:
        # Placeholder for processing the model outputs
        # This function should extract phoneme data and potentially embeddings
        
        # Example: Extracting hidden states (embeddings)
        hidden_states = outputs.decoder_hidden_states
        
        # Create dummy frame features for now
        # In a real implementation, this would be derived from the hidden states
        num_frames = 100  # Placeholder
        feature_dim = self.model.config.d_model
        frame_features = torch.zeros((num_frames, feature_dim * 3), device=self.device)
        
        # Placeholder phoneme data (replace with actual extraction logic)
        phoneme_data_list: List[PhonemeData] = []
        
        # Create dummy phoneme data for demonstration
        phoneme_data_list.append(PhonemeData(
            phoneme="example",
            start_time=0.0,
            end_time=1.0,
            confidence=0.9
        ))
        
        return {
            "phoneme_data": phoneme_data_list,
            "hidden_states": hidden_states[-1] if hidden_states else None,  # Return last layer's hidden states
            "frame_features": frame_features  # Add frame features
        }


class EnhancedWhisperModel(WhisperModel):
    """Enhanced Whisper model with additional features for lip sync"""
    def __init__(
        self, 
        model_name: str = "openai/whisper-large-v3", 
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        use_flash_attention: bool = True
    ):
        self.cache_dir = cache_dir
        self.use_flash_attention = use_flash_attention
        super().__init__(model_name, device)

    def _load_model(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        
        # Load processor
        processor = WhisperProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        # Load model with flash attention if available and requested
        if self.use_flash_attention and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            print("Using flash attention for Whisper model")
            model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                use_flash_attention_2=True
            ).to(self.device)
        else:
            model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self.device)
        
        # Use half precision for efficiency on CUDA devices
        if torch.cuda.is_available() and self.device == "cuda":
            if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
                model = model.to(torch.bfloat16)
            else:
                model = model.to(torch.float16)
        
        return model, processor

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
        # Call the parent method to get basic phoneme data
        result = super().extract_phonemes(
            audio_path=audio_path,
            language=language,
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s
        )
        
        # Add enhanced features
        # This is a placeholder - in a real implementation, you would add more features
        
        return result

    def _process_outputs(self, outputs, audio_input) -> Dict[str, Union[List[PhonemeData], torch.Tensor]]:
        """
        Process model outputs to extract phoneme data and features
        
        Args:
            outputs: Model outputs
            audio_input: Audio input features
            
        Returns:
            Dictionary containing phoneme data and features
        """
        # Get hidden states from all layers
        all_hidden_states = outputs.decoder_hidden_states
        
        # Get the last three layers for richer representation
        last_layers = [
            all_hidden_states[-1],  # Last layer
            all_hidden_states[-2],  # Second to last layer
            all_hidden_states[-3]   # Third to last layer
        ]
        
        # Stack the layers
        stacked_features = torch.cat([layer[0] for layer in last_layers], dim=-1)
        
        # Create frame features
        # In a real implementation, this would involve more sophisticated processing
        # For now, we'll just use the stacked features directly
        frame_features = stacked_features
        
        # Create dummy phoneme data
        # In a real implementation, this would be derived from the model outputs
        phoneme_data_list: List[PhonemeData] = []
        
        # Add a few dummy phonemes
        for i in range(5):
            phoneme_data_list.append(PhonemeData(
                phoneme=f"phoneme_{i}",
                start_time=i * 0.5,
                end_time=(i + 1) * 0.5,
                confidence=0.9,
                embedding=frame_features[i] if i < frame_features.shape[0] else None
            ))
        
        return {
            "phoneme_data": phoneme_data_list,
            "hidden_states": all_hidden_states,
            "frame_features": frame_features
        }
