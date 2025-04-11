#!/bin/bash

# Run evaluation for LatentSync with high-resolution support and enhanced Whisper integration

# Set environment variables
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export CUDA_VISIBLE_DEVICES=0  # Use single GPU for evaluation

# Configuration
CONFIG_FILE="configs/high_res_config.yaml"
CHECKPOINT="output/latentsync-highres/checkpoint-final.pt"
DATA_DIR="data/test"
OUTPUT_DIR="output/evaluation"
RESOLUTION=512
DETECTOR="retinaface"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run evaluation
python latentsync/evaluation/evaluate.py \
    --config $CONFIG_FILE \
    --checkpoint $CHECKPOINT \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --resolution $RESOLUTION \
    --detector $DETECTOR \
    --precision "bfloat16" \
    --num_videos 10

echo "Evaluation complete!"
