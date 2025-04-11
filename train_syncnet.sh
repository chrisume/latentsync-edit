#!/bin/bash

# Enhanced training script for LatentSync with high-resolution support
# This script trains the SyncNet model with improved Whisper integration

# Set environment variables
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use multiple GPUs if available

# Configuration
CONFIG_FILE="configs/high_res_config.yaml"
OUTPUT_DIR="output/syncnet_highres"
NUM_GPUS=4  # Adjust based on available GPUs

# Create output directory
mkdir -p $OUTPUT_DIR

# Install additional dependencies if needed
pip install xformers --upgrade
pip install flash-attn --upgrade

# Train SyncNet with distributed data parallel
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
    latentsync/train_syncnet.py \
    --config $CONFIG_FILE \
    --output_dir $OUTPUT_DIR \
    --mixed_precision "bf16" \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --lr_scheduler "cosine" \
    --lr_warmup_steps 500 \
    --max_train_steps 100000 \
    --checkpointing_steps 5000 \
    --validation_steps 1000 \
    --report_to "tensorboard" \
    --seed 42 \
    --enable_xformers_memory_efficient_attention \
    --resolution 512 \
    --use_8bit_adam \
    --use_ema

echo "Training complete!"
