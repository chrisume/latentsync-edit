#!/bin/bash

# Optimize LatentSync model for faster inference

# Set environment variables
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export CUDA_VISIBLE_DEVICES=0  # Use single GPU for optimization

# Configuration
CONFIG_FILE="configs/high_res_config.yaml"
CHECKPOINT="output/latentsync-highres/checkpoint-final.pt"
OUTPUT_DIR="output/optimized"

# Create output directory
mkdir -p $OUTPUT_DIR

# Compile model (requires PyTorch 2.0+)
python latentsync/optimization/compile_model.py \
    --config $CONFIG_FILE \
    --checkpoint $CHECKPOINT \
    --output $OUTPUT_DIR/compiled_model.pt \
    --precision "bfloat16" \
    --mode "reduce-overhead"

# Quantize model
python latentsync/optimization/quantize_model.py \
    --config $CONFIG_FILE \
    --checkpoint $CHECKPOINT \
    --output $OUTPUT_DIR/quantized_model.pt \
    --quantization "dynamic" \
    --bits 8

echo "Model optimization complete!"
