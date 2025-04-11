#!/bin/bash

# Install dependencies for LatentSync with RetinaFace

# Basic dependencies
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install transformers omegaconf tqdm pyyaml opencv-python ffmpeg-python

# Install RetinaFace
pip install retina-face

# Install InsightFace (alternative face detector)
pip install insightface onnxruntime-gpu

# Install xformers for memory-efficient attention
pip install xformers --upgrade

# Install flash-attention for faster attention
pip install flash-attn --upgrade

echo "Dependencies installed successfully!"
