import os
import torch
import argparse
import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
from omegaconf import OmegaConf
import tqdm

from latentsync.models.unet import HighResUNet
from latentsync.models.phoneme_mapper import EnhancedPhonemeMapper
from latentsync.audio.processor import EnhancedAudioProcessor
from latentsync.utils.batch_utils import get_tiled_processing_params, auto_configure_batch_size
from latentsync.inference.face_processor import FaceProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with LatentSync high-resolution model")
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
        "--audio",
        type=str,
        required=True,
        help="Path to audio file",
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path to reference image or video",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Path to output video",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Output resolution",
    )
    parser.add_argument(
        "--use_tiled_processing",
        action="store_true",
        help="Use tiled processing for high resolutions",
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
        help="Precision to use for inference",
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
        "--expand_ratio",
        type=float,
        default=1.5,
        help="Ratio to expand the bounding box",
    )
    parser.add_argument(
        "--smooth_boundary",
        action="store_true",
        help="Smooth the boundary when blending faces",
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

def process_audio(audio_path: str, audio_processor: EnhancedAudioProcessor) -> Dict[str, torch.Tensor]:
    """Process audio to extract features"""
    # Process audio
    audio_features = audio_processor.process_audio(
        audio_path=audio_path,
        return_all_features=True
    )
    
    return audio_features

def process_reference(
    reference_path: str, 
    face_processor: FaceProcessor,
    device: str
) -> Dict[str, Any]:
    """Process reference image or video to extract features"""
    # Check if reference is an image or video
    if reference_path.endswith((".jpg", ".jpeg", ".png")):
        # Load image
        image = cv2.imread(reference_path)
        
        # Process image
        result = face_processor.process_frame(image)
        
        if not result["success"]:
            raise ValueError(f"No face detected in reference image: {reference_path}")
        
        # Return result
        return {
            "is_video": False,
            "face_tensor": result["face_tensor"].unsqueeze(0),
            "detection": result["detection"],
            "frame": image
        }
    else:
        # Process video
        result = face_processor.process_video(reference_path)
        
        if not result["success"]:
            raise ValueError(f"No faces detected in reference video: {reference_path}")
        
        # Return result
        return {
            "is_video": True,
            "face_tensors": result["face_tensors"],
            "detections": result["detections"],
            "frames": result["frames"],
            "fps": result["fps"]
        }

def run_inference(
    models: Dict[str, torch.nn.Module],
    audio_features: Dict[str, torch.Tensor],
    reference_data: Dict[str, Any],
    face_processor: FaceProcessor,
    config: Any,
    args: argparse.Namespace
) -> np.ndarray:
    """Run inference with the models"""
    # Unpack models
    unet = models["unet"]
    phoneme_mapper = models["phoneme_mapper"]
    
    # Get device
    device = next(unet.parameters()).device
    
    # Get number of frames
    num_frames = audio_features.phoneme_features.shape[0]
    
    # Process phoneme features
    with torch.no_grad():
        phoneme_output = phoneme_mapper(
            phoneme_features=audio_features.phoneme_features.unsqueeze(0),
            audio_features=audio_features.wav2vec_features.unsqueeze(0) if audio_features.wav2vec_features is not None else None
        )
    
    # Get visual features
    visual_features = phoneme_output["visual_features"].squeeze(0)
    
    # Create output frames
    output_frames = []
    
    # Process each frame
    for i in tqdm.tqdm(range(num_frames), desc="Processing frames"):
        # Get current visual feature
        visual_feature = visual_features[i:i+1]
        
        # Get reference frame and detection
        if reference_data["is_video"]:
            # Use frame from reference video (loop if needed)
            frame_idx = i % len(reference_data["frames"])
            reference_frame = reference_data["frames"][frame_idx]
            reference_detection = reference_data["detections"][frame_idx]
            reference_tensor = reference_data["face_tensors"][frame_idx:frame_idx+1]
        else:
            # Use single reference image
            reference_frame = reference_data["frame"]
            reference_detection = reference_data["detection"]
            reference_tensor = reference_data["face_tensor"]
        
        # Create timestep embedding (placeholder)
        timestep = torch.zeros(1, device=device)
        
        # Process with UNet
        with torch.no_grad():
            if args.use_tiled_processing and args.resolution > 512:
                # Use tiled processing for high resolutions
                output = process_frame_tiled(
                    unet=unet,
                    reference=reference_tensor,
                    visual_feature=visual_feature,
                    timestep=timestep,
                    resolution=args.resolution,
                    config=config
                )
            else:
                # Process the whole frame at once
                output = unet(
                    x=reference_tensor,
                    timesteps=timestep,
                    audio_features=visual_feature
                ).squeeze(0)
        
        # Convert output to numpy
        output_np = output.cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize
        output_np = (output_np * 255).astype(np.uint8)
        
        # Blend face back into original frame
        blended_frame = face_processor.blend_face_back(
            original_frame=reference_frame,
            generated_face=output_np,
            detection=reference_detection,
            smooth_boundary=args.smooth_boundary
        )
        
        # Add to output frames
        output_frames.append(blended_frame)
    
    return output_frames

def process_frame_tiled(
    unet: torch.nn.Module,
    reference: torch.Tensor,
    visual_feature: torch.Tensor,
    timestep: torch.Tensor,
    resolution: int,
    config: Any
) -> torch.Tensor:
    """Process a frame using tiled processing"""
    # Get tiled processing parameters
    tile_params = get_tiled_processing_params(
        resolution=resolution,
        tile_size=config.inference.tile_size,
        tile_overlap=config.inference.tile_overlap
    )
    
    # Create output tensor
    output = torch.zeros(
        (3, resolution, resolution),
        device=reference.device
    )
    
    # Create weight tensor for blending
    weight = torch.zeros(
        (1, resolution, resolution),
        device=reference.device
    )
    
    # Process each tile
    for pos_x, pos_y in tile_params["tile_positions"]:
        # Extract tile
        tile = reference[:, :, pos_y:pos_y+tile_params["effective_tile_size_y"], pos_x:pos_x+tile_params["effective_tile_size_x"]]
        
        # Process tile
        with torch.no_grad():
            tile_output = unet(
                x=tile,
                timesteps=timestep,
                audio_features=visual_feature
            ).squeeze(0)
        
        # Create weight mask for blending
        mask = create_blending_mask(
            tile_params["effective_tile_size_y"],
            tile_params["effective_tile_size_x"],
            tile_params["tile_overlap"]
        ).to(reference.device)
        
        # Add tile to output
        output[:, pos_y:pos_y+tile_params["effective_tile_size_y"], pos_x:pos_x+tile_params["effective_tile_size_x"]] += tile_output * mask
        
        # Add weight
        weight[:, pos_y:pos_y+tile_params["effective_tile_size_y"], pos_x:pos_x+tile_params["effective_tile_size_x"]] += mask
    
    # Normalize output
    output = output / (weight + 1e-8)
    
    return output

def create_blending_mask(height: int, width: int, overlap: int) -> torch.Tensor:
    """Create a blending mask for tiled processing"""
    mask = torch.ones((1, height, width))
    
    # Create linear ramp for blending at edges
    if overlap > 0:
        ramp = torch.linspace(0, 1, overlap)
        
        # Left edge
        mask[:, :, :overlap] *= ramp.view(1, 1, -1)
        
        # Right edge
        mask[:, :, -overlap:] *= ramp.flip(0).view(1, 1, -1)
        
        # Top edge
        mask[:, :overlap, :] *= ramp.view(1, -1, 1)
        
        # Bottom edge
        mask[:, -overlap:, :] *= ramp.flip(0).view(1, -1, 1)
    
    return mask

def save_output(
    output_frames: List[np.ndarray], 
    audio_path: str, 
    output_path: str,
    fps: float = 25.0
) -> None:
    """Save output video with audio"""
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Get frame dimensions
    height, width = output_frames[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for frame in output_frames:
        writer.write(frame)
    
    # Release writer
    writer.release()
    
    # Add audio
    try:
        import ffmpeg
        
        # Create temporary file
        temp_output = output_path + ".temp.mp4"
        os.rename(output_path, temp_output)
        
        # Add audio
        (
            ffmpeg
            .input(temp_output)
            .output(output_path, acodec="aac", map=0)
            .run(quiet=True, overwrite_output=True)
        )
        
        # Remove temporary file
        os.remove(temp_output)
    except Exception as e:
        print(f"Error adding audio to output video: {e}")
        print("Output video saved without audio.")

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override configuration with command line arguments
    config.model.resolution = args.resolution
    config.hardware.precision = args.precision
    
    # Initialize face processor
    face_processor = FaceProcessor(
        detector_type=args.detector,
        confidence_threshold=args.confidence,
        target_size=args.resolution,
        expand_ratio=args.expand_ratio,
        device=args.device
    )
    
    # Load models
    models = load_models(
        config=config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        precision=args.precision
    )
    
    # Process audio
    audio_features = process_audio(
        audio_path=args.audio,
        audio_processor=models["audio_processor"]
    )
    
    # Process reference
    reference_data = process_reference(
        reference_path=args.reference,
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
        audio_path=args.audio,
        output_path=args.output,
        fps=reference_data.get("fps", 25.0)
    )
    
    print(f"Output saved to {args.output}")

if __name__ == "__main__":
    main()
