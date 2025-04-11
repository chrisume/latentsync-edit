import os
import cv2
import numpy as np
import argparse
import glob
import json
import tqdm
import torch
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from latentsync.utils.face_detection import get_face_detector
from latentsync.audio.processor import EnhancedAudioProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for LatentSync training")
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Path to directory containing videos",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/high_res_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="retinaface",
        choices=["retinaface", "insightface"],
        help="Face detector to use",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.9,
        help="Confidence threshold for face detection",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=512,
        help="Target size for extracted faces",
    )
    parser.add_argument(
        "--expand_ratio",
        type=float,
        default=1.5,
        help="Ratio to expand the bounding box",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=1,
        help="Extract faces every N frames",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda, cpu)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    
    return parser.parse_args()

def extract_audio(
    video_path: str,
    output_path: str,
    sample_rate: int = 16000
) -> None:
    """
    Extract audio from video
    
    Args:
        video_path: Path to input video
        output_path: Path to output audio
        sample_rate: Sample rate for output audio
    """
    # Import here to avoid dependency issues
    try:
        import ffmpeg
    except ImportError:
        raise ImportError(
            "ffmpeg-python is not installed. Please install it with: "
            "pip install ffmpeg-python"
        )
    
    # Extract audio
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_path, ar=sample_rate, ac=1)
            .run(quiet=True, overwrite_output=True)
        )
    except ffmpeg.Error as e:
        print(f"Error extracting audio from {video_path}: {e}")

def process_video(
    video_path: str,
    output_dir: str,
    config: Any,
    detector_type: str = "retinaface",
    confidence_threshold: float = 0.9,
    target_size: int = 512,
    expand_ratio: float = 1.5,
    frame_interval: int = 1,
    device: str = "cuda"
) -> None:
    """
    Process a video for LatentSync training
    
    Args:
        video_path: Path to input video
        output_dir: Path to output directory
        config: Configuration
        detector_type: Type of face detector
        confidence_threshold: Confidence threshold for face detection
        target_size: Target size for extracted faces
        expand_ratio: Ratio to expand the bounding box
        frame_interval: Extract faces every N frames
        device: Device to use
    """
    # Create output directories
    video_name = Path(video_path).stem
    video_output_dir = os.path.join(output_dir, video_name)
    frames_dir = os.path.join(video_output_dir, "frames")
    audio_dir = os.path.join(video_output_dir, "audio")
    features_dir = os.path.join(video_output_dir, "features")
    
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    
    # Extract audio
    audio_path = os.path.join(audio_dir, f"{video_name}.wav")
    extract_audio(video_path, audio_path, sample_rate=config.audio.sample_rate)
    
    # Initialize face detector
    detector = get_face_detector(
        detector_type=detector_type,
        device=device,
        confidence_threshold=confidence_threshold,
        max_faces=1
    )
    
    # Initialize audio processor
    audio_processor = EnhancedAudioProcessor(config, device=device)
    
    # Process audio
    audio_features = audio_processor.process_audio(
        audio_path=audio_path,
        return_all_features=True
    )
    
    # Save audio features
    torch.save(
        audio_features,
        os.path.join(features_dir, f"{video_name}_audio_features.pt")
    )
    
    # Open video
    video = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create metadata
    metadata = {
        "video_path": video_path,
        "fps": fps,
        "frame_count": frame_count,
        "detector_type": detector_type,
        "confidence_threshold": confidence_threshold,
        "target_size": target_size,
        "expand_ratio": expand_ratio,
        "frame_interval": frame_interval,
        "frames": {}
    }
    
    # Process frames
    frame_idx = 0
    with tqdm.tqdm(total=frame_count) as pbar:
        while True:
            # Read frame
            ret, frame = video.read()
            
            # Break if end of video
            if not ret:
                break
            
            # Skip frames
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                pbar.update(1)
                continue
            
            # Detect faces
            detections = detector.detect(frame)
            
            # Process first detection
            if detections:
                detection = detections[0]
                
                # Align face
                face = detector.align_face(
                    frame, 
                    detection, 
                    target_size=(target_size, target_size),
                    expand_ratio=expand_ratio
                )
                
                # Save face
                face_path = os.path.join(
                    frames_dir, 
                    f"frame_{frame_idx:06d}.png"
                )
                cv2.imwrite(face_path, face)
                
                # Save metadata
                metadata["frames"][str(frame_idx)] = {
                    "frame_idx": frame_idx,
                    "bbox": detection.bbox.tolist(),
                    "landmarks": detection.landmarks.tolist(),
                    "score": detection.score,
                    "face_path": face_path
                }
            
            # Update frame index
            frame_idx += 1
            pbar.update(1)
    
    # Release video
    video.release()
    
    # Save metadata
    metadata_path = os.path.join(video_output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    import yaml
    from omegaconf import OmegaConf
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    config = OmegaConf.create(config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Copy configuration
    shutil.copy(args.config, os.path.join(args.output_dir, "config.yaml"))
    
    # Get video paths
    video_paths = []
    for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
        video_paths.extend(glob.glob(os.path.join(args.video_dir, ext)))
    
    # Process videos
    for video_path in tqdm.tqdm(video_paths, desc="Processing videos"):
        process_video(
            video_path=video_path,
            output_dir=args.output_dir,
            config=config,
            detector_type=args.detector,
            confidence_threshold=args.confidence,
            target_size=args.target_size,
            expand_ratio=args.expand_ratio,
            frame_interval=args.frame_interval,
            device=args.device
        )

if __name__ == "__main__":
    main()
