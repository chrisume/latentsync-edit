#!/bin/bash

# Run tests to diagnose issues

echo "Running tests to diagnose issues..."

# Fix line endings in scripts
echo "Creating fix_line_endings.py..."
cat > fix_line_endings.py << 'EOF'
import os
import glob

def fix_line_endings(file_path):
    """Fix Windows line endings (CRLF) to Unix (LF)"""
    print(f"Fixing line endings in: {file_path}")
    
    try:
        # Read the file
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Replace CRLF with LF
        content = content.replace(b'\r\n', b'\n')
        
        # Write the file back
        with open(file_path, 'wb') as f:
            f.write(content)
        
        print(f"Fixed line endings in: {file_path}")
        return True
    except Exception as e:
        print(f"Error fixing line endings in {file_path}: {e}")
        return False

def fix_all_scripts():
    """Fix line endings in all shell scripts"""
    script_files = glob.glob("*.sh") + glob.glob("run_*.sh")
    
    success = True
    for script_file in script_files:
        if not fix_line_endings(script_file):
            success = False
    
    return success

if __name__ == "__main__":
    success = fix_all_scripts()
    print(f"\nLine ending fix {'successful' if success else 'failed'}")
EOF

echo "Running fix_line_endings.py..."
python fix_line_endings.py

# Check available codecs
echo "Creating check_codecs.py..."
cat > check_codecs.py << 'EOF'
import cv2
import numpy as np

def check_available_codecs():
    """Check which video codecs are available in OpenCV"""
    print("Checking available video codecs in OpenCV...")
    
    # List of common codecs to test
    codecs = [
        "MJPG",  # Motion JPEG
        "XVID",  # XVID MPEG-4
        "mp4v",  # MPEG-4
        "avc1",  # H.264/AVC
        "H264",  # H.264
        "DIVX",  # DivX
        "MPEG",  # MPEG-1
        "WMV1",  # Windows Media Video 7
        "WMV2"   # Windows Media Video 8
    ]
    
    # Create a small test frame
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Test each codec
    working_codecs = []
    
    for codec in codecs:
        try:
            # Try to create a VideoWriter with this codec
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out_path = f"test_codec_{codec}.avi"
            writer = cv2.VideoWriter(out_path, fourcc, 30, (100, 100))
            
            if writer.isOpened():
                # Write a frame
                writer.write(frame)
                writer.release()
                
                # Check if file was created
                import os
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    print(f"✓ Codec {codec} works!")
                    working_codecs.append(codec)
                    # Clean up
                    os.remove(out_path)
                else:
                    print(f"✗ Codec {codec} created a file but it's empty")
            else:
                print(f"✗ Codec {codec} is not available")
        except Exception as e:
            print(f"✗ Codec {codec} error: {e}")
    
    print(f"\nWorking codecs: {', '.join(working_codecs)}")
    return working_codecs

if __name__ == "__main__":
    working_codecs = check_available_codecs()
    
    # Suggest the best codec to use
    if working_codecs:
        print(f"\nRecommended codec: {working_codecs[0]}")
        print(f"Use this in your scripts with: cv2.VideoWriter_fourcc(*'{working_codecs[0]}')")
    else:
        print("\nNo working codecs found. Try installing OpenCV with more codec support.")
EOF

echo "Running check_codecs.py..."
python check_codecs.py

# Test audio loading
echo "Creating test_audio_loading.py..."
cat > test_audio_loading.py << 'EOF'
import os
import librosa
import numpy as np
import torch
from transformers import WhisperProcessor

def test_audio_loading(audio_path):
    """Test audio loading with librosa"""
    print(f"Testing audio loading for: {audio_path}")
    
    # Check if file exists
    if not os.path.exists(audio_path):
        print(f"Error: File does not exist: {audio_path}")
        return False
    
    try:
        # Load audio with librosa
        print("Loading with librosa...")
        audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
        print(f"Success! Audio duration: {len(audio_array)/sampling_rate:.2f}s")
        print(f"Audio shape: {audio_array.shape}, dtype: {audio_array.dtype}")
        print(f"Sample values - min: {audio_array.min()}, max: {audio_array.max()}, mean: {audio_array.mean()}")
        
        # Try processing with Whisper feature extractor
        print("\nTesting Whisper feature extraction...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        features = processor.feature_extractor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        print(f"Feature extraction successful!")
        print(f"Features shape: {features['input_features'].shape}")
        
        return True
    except Exception as e:
        print(f"Error processing audio: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = "test_data/sample.wav"
    
    success = test_audio_loading(audio_path)
    print(f"\nTest {'successful' if success else 'failed'}")
EOF

#


### 8. Configuration Files
