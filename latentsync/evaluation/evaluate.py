import os
import torch
import argparse
import yaml
import json
import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from omegaconf import OmegaConf

from latentsync.models.unet import HighResUNet
from latentsync.models.phoneme_mapper import EnhancedPhonemeMapper
from latentsync.audio.processor import EnhancedAudioProcessor
from latentsync.inference.face_processor import FaceProcessor
from latentsync.evaluation.metrics import evaluate_video

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LatentSync model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/high_res_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to evaluation data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/evaluation",
        help="Path to output directory",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Evaluation resolution",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda, cpu)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Precision to use for evaluation",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="retinaface",
        choices=["retinaface", "insightface"],
        help="Face detector to use",
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=10,
        help="Number of videos to evaluate",
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> Any:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Convert to OmegaConf for easier access
    config = OmegaConf.create(config)
    
    return config

def load_models(config: Any, checkpoint_path: str, device: str, precision: str) -> Dict[str, torch.nn.Module]:
    """Load models from checkpoint"""
    # Create models
    unet = HighResUNet(config).to(device)
    phoneme_mapper = EnhancedPhonemeMapper(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    unet.load_state_dict(checkpoint["unet"])
    phoneme_mapper.load_state_dict(checkpoint["phoneme_mapper"])
    
    # Set precision
    if precision == "float16" and device == "cuda":
        unet = unet.half()
        phoneme_mapper = phoneme_mapper.half()
    elif precision == "bfloat16" and device == "cuda" and torch.cuda.is_bf16_supported():
        unet = unet.to(torch.bfloat16)
        phoneme_mapper = phoneme_mapper.to(torch.bfloat16)
    
    # Set to eval mode
    unet.eval()
    phoneme_mapper.eval()
    
    # Create audio processor
    audio_processor = EnhancedAudioProcessor(config, device=device)
    
    return {
        "unet": unet,
        "phoneme_mapper": phoneme_mapper,
        "audio_processor": audio_processor
    }

def evaluate_model(
    models: Dict[str, torch.nn.Module],
    data_dir: str,
    output_dir: str,
    config: Any,
    args: argparse.Namespace
) -> Dict[str, float]:
    """
    Evaluate model on test data
    
    Args:
        models: Dictionary of models
        data_dir: Path to evaluation data directory
        output_dir: Path to output directory
        config: Configuration
        args: Command line arguments
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize face processor
    face_processor = FaceProcessor(
        detector_type=args.detector,
        confidence_threshold=0.9,
        target_size=args.resolution,
        expand_ratio=1.5,
        device=args.device
    )
    
    # Get video directories
    video_dirs = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            video_dirs.append(item_path)
    
    # Limit number of videos
    if args.num_videos > 0 and args.num_videos < len(video_dirs):
        video_dirs = video_dirs[:args.num_videos]
    
    # Initialize metrics
    all_metrics = []
    
    # Process each video
    for video_dir in tqdm.tqdm(video_dirs, desc="Evaluating videos"):
        # Get video name
        video_name = Path(video_dir).name
        
        # Create output directory for this video
        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Get audio path
        audio_dir = os.path.join(video_dir, "audio")
        audio_path = os.path.join(audio_dir, f"{video_name}.wav")
        
        # Get reference video path
        reference_video_path = os.path.join(video_dir, f"{video_name}.mp4")
        
        # Skip if audio or reference video doesn't exist
        if not os.path.exists(audio_path) or not os.path.exists(reference_video_path):
            continue
        
        # Generate output video
        output_video_path = os.path.join(video_output_dir, f"{video_name}_output.mp4")
        
        # Run inference
        from inference import process_audio, process_reference, run_inference, save_output
        
        # Process audio
        audio_features = process_audio(
            audio_path=audio_path,
            audio_processor=models["audio_processor"]
        )
        
        # Process reference
        reference_data = process_reference(
            reference_path=reference_video_path,
            face_processor=face_processor,
            device=args.device
        )
        
        # Run inference
        output_frames = run_inference(
            models=models,
            audio_features=audio_features,
            reference_data=reference_data,
            face_processor=face_processor,
            config=config,
            args=args
        )
        
        # Save output
        save_output(
            output_frames=output_frames,
            audio_path=audio_path,
            output_path=output_video_path,
            fps=reference_data.get("fps", 25.0)
        )
        
        # Evaluate video
        metrics = evaluate_video(
            pred_video_path=output_video_path,
            target_video_path=reference_video_path,
            audio_path=audio_path,
            output_dir=video_output_dir,
            device=args.device
        )
        
        # Add video name to metrics
        metrics["video_name"] = video_name
        
        # Add to all metrics
        all_metrics.append(metrics)
        
        # Save metrics for this video
        with open(os.path.join(video_output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Calculate average metrics
    avg_metrics = {}
    for key in ["psnr", "ssim", "lpips", "lip_sync_score"]:
        avg_metrics[key] = np.mean([m[key] for m in all_metrics if key in m])
    
    # Save all metrics
    with open(os.path.join(output_dir, "all_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    # Save average metrics
    with open(os.path.join(output_dir, "avg_metrics.json"), "w") as f:
        json.dump(avg_metrics, f, indent=2)
    
    return avg_metrics

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override configuration with command line arguments
    config.model.resolution = args.resolution
    config.hardware.precision = args.precision
    
    # Load models
    models = load_models(
        config=config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        precision=args.precision
    )
    
    # Evaluate model
    metrics = evaluate_model(
        models=models,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=config,
        args=args
    )
    
    # Print metrics
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()
