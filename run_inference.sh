#!/bin/bash

# Run inference for LatentSync with high-resolution support and enhanced Whisper integration

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
DETECTOR="retinaface"

# Check if arguments are provided
if [ -z "$AUDIO_FILE" ] || [ -z "$REFERENCE_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 <audio_file> <reference_file> <output_file>"
    exit 1
fi

# Run inference
python inference.py \
    --config $CONFIG_FILE \
    --checkpoint $CHECKPOINT \
    --audio $AUDIO_FILE \
    --reference $REFERENCE_FILE \
    --output $OUTPUT_FILE \
    --resolution $RESOLUTION \
    --detector $DETECTOR \
    --precision "bfloat16" \
    --smooth_boundary

echo "Inference complete! Output saved to $OUTPUT_FILE"
