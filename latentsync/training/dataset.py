import os
import torch
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import Dataset

class LatentSyncDataset(Dataset):
    """
    Dataset for LatentSync training
    """
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        resolution: int = 512,
        max_frames: int = 100,
        frame_interval: int = 1,
        augmentation: bool = True
    ):
        self.data_dir = data_dir
        self.split = split
        self.resolution = resolution
        self.max_frames = max_frames
        self.frame_interval = frame_interval
        self.augmentation = augmentation and split == "train"
        
        # Get video directories
        self.video_dirs = []
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                self.video_dirs.append(item_path)
        
        # Split videos into train/val
        if split == "train":
            self.video_dirs = self.video_dirs[:-max(1, len(self.video_dirs) // 10)]
        else:
            self.video_dirs = self.video_dirs[-max(1, len(self.video_dirs) // 10):]
        
        # Load metadata
        self.metadata = []
        self.samples = []
        
        for video_dir in self.video_dirs:
            # Load metadata
            metadata_path = os.path.join(video_dir, "metadata.json")
            if not os.path.exists(metadata_path):
                continue
            
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            self.metadata.append(metadata)
            
            # Get frames
            frames_dir = os.path.join(video_dir, "frames")
            if not os.path.exists(frames_dir):
                continue
            
            # Get audio features
            features_dir = os.path.join(video_dir, "features")
            audio_features_path = os.path.join(features_dir, f"{Path(video_dir).name}_audio_features.pt")
            if not os.path.exists(audio_features_path):
                continue
            
            # Get frame indices
            frame_indices = sorted([int(k) for k in metadata["frames"].keys()])
            
            # Sample frames
            if len(frame_indices) > self.max_frames:
                if split == "train":
                    # Randomly sample consecutive frames
                    start_idx = np.random.randint(0, len(frame_indices) - self.max_frames)
                    frame_indices = frame_indices[start_idx:start_idx + self.max_frames]
                else:
                    # Sample evenly spaced frames
                    frame_indices = frame_indices[::len(frame_indices) // self.max_frames][:self.max_frames]
            
            # Skip frames according to interval
            frame_indices = frame_indices[::self.frame_interval]
            
            # Add samples
            for frame_idx in frame_indices:
                frame_key = str(frame_idx)
                if frame_key in metadata["frames"]:
                    frame_data = metadata["frames"][frame_key]
                    frame_path = frame_data["face_path"]
                    
                    if os.path.exists(frame_path):
                        self.samples.append({
                            "video_dir": video_dir,
                            "frame_idx": frame_idx,
                            "frame_path": frame_path,
                            "audio_features_path": audio_features_path
                        })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get sample
        sample = self.samples[idx]
        
        # Load frame
        frame = cv2.imread(sample["frame_path"])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if frame.shape[0] != self.resolution or frame.shape[1] != self.resolution:
            frame = cv2.resize(frame, (self.resolution, self.resolution))
        
        # Apply augmentation if enabled
        if self.augmentation:
            frame = self._augment_image(frame)
        
        # Load audio features
        audio_features = torch.load(sample["audio_features_path"])
        
        # Get phoneme features for this frame
        frame_idx = sample["frame_idx"]
        phoneme_features = audio_features.phoneme_features[frame_idx:frame_idx+1]
        
        # Get additional audio features if available
        additional_audio_features = None
        if hasattr(audio_features, "wav2vec_features") and audio_features.wav2vec_features is not None:
            additional_audio_features = audio_features.wav2vec_features[frame_idx:frame_idx+1]
        
        # Create reference and target
        reference = frame.copy()
        target = frame.copy()
        
        # Convert to tensor
        reference = torch.from_numpy(reference).permute(2, 0, 1).float() / 255.0
        target = torch.from_numpy(target).permute(2, 0, 1).float() / 255.0
        
        # Create sample
        result = {
            "reference": reference,
            "target": target,
            "phoneme_features": phoneme_features,
            "frame_idx": torch.tensor([frame_idx], dtype=torch.long)
        }
        
        # Add additional audio features if available
        if additional_audio_features is not None:
            result["audio_features"] = additional_audio_features
        
        return result
    
    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation to image"""
        # Apply color jitter
        image = self._color_jitter(image)
        
        # Apply random horizontal flip
        if np.random.random() < 0.5:
            image = cv2.flip(image, 1)
        
        return image
    
    def _color_jitter(self, image: np.ndarray) -> np.ndarray:
        """Apply color jitter augmentation"""
        # Convert to float
        image = image.astype(np.float32) / 255.0
        
        # Apply brightness, contrast, saturation, and hue jitter
        # Brightness
        brightness_factor = np.random.uniform(0.8, 1.2)
        image = image * brightness_factor
        
        # Contrast
        contrast_factor = np.random.uniform(0.8, 1.2)
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        image = (image - mean) * contrast_factor + mean
        
        # Saturation
        saturation_factor = np.random.uniform(0.8, 1.2)
        gray = np.mean(image, axis=2, keepdims=True)
        image = image * saturation_factor + gray * (1 - saturation_factor)
        
        # Clip values
        image = np.clip(image, 0.0, 1.0)
        
        # Convert back to uint8
        image = (image * 255.0).astype(np.uint8)
        
        return image
