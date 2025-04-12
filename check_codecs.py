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
