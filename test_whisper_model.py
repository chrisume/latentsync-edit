import os
import torch
import argparse
from latentsync.audio.whisper_model import EnhancedWhisperModel

def parse_args():
    parser = argparse.ArgumentParser(description="Test Whisper model")
    parser.add_argument(
        "--audio",
        type=str,
        default="test_data/sample.wav",
        help="Path to audio file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda, cpu)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-small",
        help="Whisper model to use",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        return
    
    print(f"Testing Whisper model with audio: {args.audio}")
    
    # Initialize Whisper model
    print(f"Initializing Whisper model: {args.model}")
    model = EnhancedWhisperModel(
        model_name=args.model,
        device=args.device,
        use_flash_attention=False  # Disable flash attention to avoid data type issues
    )
    
    # Extract phonemes
    print("Extracting phonemes...")
    try:
        result = model.extract_phonemes(
            audio_path=args.audio,
            language="en",
            return_timestamps=True
        )
        
        # Print results
        print("\nExtraction successful!")
        print(f"Frame features shape: {result['frame_features'].shape}")
        print(f"Number of phonemes: {len(result['phoneme_data'])}")
        
        # Print first few phonemes
        print("\nFirst few phonemes:")
        for i, phoneme in enumerate(result['phoneme_data'][:5]):
            print(f"  {i+1}. {phoneme.phoneme} ({phoneme.start_time:.2f}s - {phoneme.end_time:.2f}s)")
        
        return True
    except Exception as e:
        print(f"Error extracting phonemes: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\nTest {'successful' if success else 'failed'}")
