# LatentSync: High-Resolution Lip Sync with Enhanced Face Detection

LatentSync is a state-of-the-art lip synchronization system that supports high-resolution (512+) processing and improved mouth articulation using the Whisper Large-v3 model. This implementation includes enhanced face detection using RetinaFace for more accurate face detection and alignment.

## Features

- **High-Resolution Support (512+)**: Process videos at higher resolutions for better quality results
- **Enhanced Face Detection**: Uses RetinaFace for more accurate face detection and alignment
- **Improved Whisper Integration**: Utilizes Whisper Large-v3 for better phoneme detection
- **Enhanced Audio Processing**: Combines Whisper and Wav2Vec2 for better audio features
- **Temporal Consistency**: Ensures smooth and natural mouth movements
- **Optimization Options**: Includes model compilation and quantization for faster inference

## Installation

Install the required dependencies:

\`\`\`bash
bash install_dependencies.sh
\`\`\`

## Dataset Preparation

Extract faces from videos or images:

\`\`\`bash
python latentsync/data/extract_faces.py \
    --input /path/to/videos \
    --output /path/to/output \
    --detector retinaface \
    --confidence 0.9 \
    --target_size 512 \
    --align \
    --save_metadata
\`\`\`

Prepare dataset for training:

\`\`\`bash
python latentsync/data/prepare_dataset.py \
    --video_dir /path/to/videos \
    --output_dir /path/to/dataset \
    --config configs/high_res_config.yaml \
    --detector retinaface \
    --target_size 512
\`\`\`

## Training

Train the model:

\`\`\`bash
bash run_training.sh
\`\`\`

## Evaluation

Evaluate the model:

\`\`\`bash
bash run_evaluation.sh
\`\`\`

## Inference

Run inference:

\`\`\`bash
bash run_inference.sh /path/to/audio.wav /path/to/reference.mp4 output.mp4
\`\`\`

## Optimization

Optimize the model for faster inference:

\`\`\`bash
bash optimize_model.sh
\`\`\`

## Face Detection Options

The implementation supports two face detection methods:

1. **RetinaFace**: A state-of-the-art face detection model that provides accurate face detection and facial landmark localization.

2. **InsightFace**: An alternative face detection method that provides robust face detection and alignment.

You can choose the face detection method using the `--detector` argument in the scripts.

## High-Resolution Support

The implementation supports high-resolution processing (512+) with the following optimizations:

- Memory-efficient attention with xformers
- Flash attention with PyTorch 2.0+
- Tiled processing for very high resolutions
- Mixed precision training and inference
- Gradient checkpointing for memory efficiency

## Whisper Integration

The implementation includes enhanced audio processing with Whisper Large-v3 for better mouth articulation:

- Detailed phoneme extraction with timing information
- Optional Wav2Vec2 integration for additional audio features
- Temporal consistency layer for smoother mouth movements
- Viseme classifier for specific mouth shapes

## License

This project is licensed under the MIT License - see the LICENSE file for details.
