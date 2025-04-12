#!/bin/bash

# Run inference for LatentSync with MJPG codec (more widely available than libx264)

# Set environment variables
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export CUDA_VISIBLE_DEVICES=0  # Use single GPU for inference

# Configuration
CONFIG_FILE="configs/high_res_config.yaml"
CHECKPOINT="output/latentsync-highres/checkpoint-final.pt"
AUDIO_FILE="$1"  # First argument: audio file
REFERENCE_FILE="$2"  # Second argument: reference image or video
OUTPUT_FILE="$3"  # Third argument: output video
RESOLUTION=512
DETECTOR="insightface"

# Check if arguments are provided
if [ -z "$AUDIO_FILE" ] || [ -z "$REFERENCE_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 <audio_file> <reference_file> <output_file>"
    exit 1
fi

# Install required dependencies
pip install librosa soundfile ffmpeg-python --quiet

# Check if audio file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: Audio file not found: $AUDIO_FILE"
    exit 1
fi

# Check if reference file exists
if [ ! -f "$REFERENCE_FILE" ]; then
    echo "Error: Reference file not found: $REFERENCE_FILE"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Print debug information
echo "Audio file: $AUDIO_FILE"
echo "Reference file: $REFERENCE_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Resolution: $RESOLUTION"
echo "Detector: $DETECTOR"

# Modify the inference.py script to use MJPG codec
cat > mjpg_inference.py << 'EOF'
import os
import cv2
import numpy as np
import torch
from typing import List
from inference import main as original_main
from inference import save_output as original_save_output

# Override the save_output function to use MJPG codec
def mjpg_save_output(
    output_frames: List[np.ndarray], 
    audio_path: str, 
    output_path: str,
    fps: float = 25.0
) -> None:
    """Save output video with audio using MJPG codec"""
    # Create output directory if needed
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving output to directory: {output_dir}")
    
    # Get frame dimensions
    height, width = output_frames[0].shape[:2]
    print(f"Frame dimensions: {width}x{height}, Total frames: {len(output_frames)}")
    
    # Try different codecs
    codecs = [
        ("MJPG", "avi"),
        ("XVID", "avi"),
        ("mp4v", "mp4")
    ]
    
    success = False
    temp_video_path = None
    
    for codec, ext in codecs:
        try:
            # Create temporary path with the right extension
            temp_path = f"{output_path}.temp.{ext}"
            print(f"Trying codec: {codec}, output: {temp_path}")
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
            
            if not writer.isOpened():
                print(f"Failed to open writer with codec {codec}")
                continue
            
            # Write frames
            print(f"Writing {len(output_frames)} frames to video...")
            for i, frame in enumerate(output_frames):
                writer.write(frame)
                if i % 10 == 0:
                    print(f"Wrote frame {i}/{len(output_frames)}")
            
            # Release writer
            writer.release()
            
            # Check if file exists and has content
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1000:
                print(f"Successfully created video with codec {codec}")
                print(f"File size: {os.path.getsize(temp_path) / 1024:.2f} KB")
                temp_video_path = temp_path
                success = True
                break
            else:
                print(f"File creation failed or file is too small with codec {codec}")
        
        except Exception as e:
            print(f"Error with codec {codec}: {e}")
    
    if not success:
        # Save individual frames as images as a fallback
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Saving individual frames to {frames_dir}")
        for i, frame in enumerate(output_frames):
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
            cv2.imwrite(frame_path, frame)
        print(f"Saved {len(output_frames)} frames to {frames_dir}")
        return
    
    # Try to add audio if we have a video
    if temp_video_path:
        try:
            import ffmpeg
            
            # Create final output path
            final_output = output_path
            
            # Add audio
            print(f"Adding audio from {audio_path} to video...")
            try:
                (
                    ffmpeg
                    .input(temp_video_path)
                    .input(audio_path)
                    .output(final_output, acodec="aac", map=0)
                    .run(quiet=False, overwrite_output=True)
                )
                print(f"Final output with audio: {final_output}")
            except Exception as e:
                print(f"Error adding audio: {e}")
                print("Copying video without audio...")
                import shutil
                shutil.copy(temp_video_path, final_output)
            
            # Remove temporary file
            os.remove(temp_video_path)
            print(f"Temporary file removed. Final output: {final_output}")
            print(f"Final file exists: {os.path.exists(final_output)}, Size: {os.path.getsize(final_output) if os.path.exists(final_output) else 'N/A'} bytes")
        except Exception as e:
            print(f"Error processing final output: {e}")
            print(f"Temporary video saved at: {temp_video_path}")

# Override the save_output function in the inference module
import inference
inference.save_output = mjpg_save_output

# Run the original main function
if __name__ == "__main__":
    original_main()
EOF

# Run the modified inference script
python mjpg_inference.py \
    --config $CONFIG_FILE \
    --checkpoint $CHECKPOINT \
    --audio $AUDIO_FILE \
    --reference $REFERENCE_FILE \
    --output $OUTPUT_FILE \
    --resolution $RESOLUTION \
    --detector $DETECTOR \
    --precision "float32" \
    --smooth_boundary

# Check if output file was created
if [ -f "$OUTPUT_FILE" ]; then
    echo "Inference complete! Output saved to $OUTPUT_FILE"
    echo "File size: $(du -h "$OUTPUT_FILE" | cut -f1)"
else
    echo "Error: Output file was not created: $OUTPUT_FILE"
    # Check for temporary files or frame directory
    echo "Checking for alternative outputs..."
    
    # Check for .avi version
    if [ -f "${OUTPUT_FILE}.temp.avi" ]; then
        echo "Found temporary AVI file: ${OUTPUT_FILE}.temp.avi"
        echo "File size: $(du -h "${OUTPUT_FILE}.temp.avi" | cut -f1)"
    fi
    
    # Check for frames directory
    FRAMES_DIR="$(dirname "$OUTPUT_FILE")/frames"
    if [ -d "$FRAMES_DIR" ]; then
        echo "Found frames directory: $FRAMES_DIR"
        echo "Number of frames: $(ls "$FRAMES_DIR" | wc -l)"
    fi
fi
