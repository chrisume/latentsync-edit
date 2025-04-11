#!/bin/bash

# Run training for LatentSync with high-resolution support and enhanced Whisper integration

# Set environment variables
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use multiple GPUs if available

# Configuration
CONFIG_FILE="configs/high_res_config.yaml"
OUTPUT_DIR="output/latentsync-highres"
DATA_DIR="data/processed"
NUM_GPUS=4  # Adjust based on available GPUs

# Create output directory
mkdir -p $OUTPUT_DIR

# Install additional dependencies if needed
pip install xformers --upgrade
pip install flash-attn --upgrade
pip install bitsandbytes --upgrade
pip install torch-ema --upgrade

# Train model with distributed data parallel
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
    latentsync/training/train.py \
    --config $CONFIG_FILE \
    --output_dir $OUTPUT_DIR \
    --data_dir $DATA_DIR \
    --mixed_precision "bf16" \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --max_train_steps 100000 \
    --checkpointing_steps 5000 \
    --validation_steps 1000 \
    --seed 42 \
    --enable_xformers_memory_efficient_attention \
    --resolution 512 \
    --use_8bit_adam \
    --use_ema

echo "Training complete!"
