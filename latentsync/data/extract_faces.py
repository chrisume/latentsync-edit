import os
import cv2
import numpy as np
import argparse
import glob
import json
import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from latentsync.utils.face_detection import get_face_detector, FaceDetection

def parse_args():
    parser = argparse.ArgumentParser(description="Extract faces from videos or images")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input video or directory of images",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output directory",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="retinaface",
        choices=["retinaface", "insightface"],
        help="Face detector to use",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.9,
        help="Confidence threshold for face detection",
    )
    parser.add_argument(
        "--max_faces",
        type=int,
        default=1,
        help="Maximum number of faces to extract per frame",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=512,
        help="Target size for extracted faces",
    )
    parser.add_argument(
        "--expand_ratio",
        type=float,
        default=1.5,
        help="Ratio to expand the bounding box",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Align faces using landmarks",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=1,
        help="Extract faces every N frames",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda, cpu)",
    )
    parser.add_argument(
        "--save_metadata",
        action="store_true",
        help="Save metadata for each extracted face",
    )
    
    return parser.parse_args()

def extract_faces_from_video(
    video_path: str,
    output_dir: str,
    detector_type: str = "retinaface",
    confidence_threshold: float = 0.9,
    max_faces: int = 1,
    target_size: int = 512,
    expand_ratio: float = 1.5,
    align: bool = True,
    frame_interval: int = 1,
    device: str = "cuda",
    save_metadata: bool = False
) -> None:
    """
    Extract faces from a video
    
    Args:
        video_path: Path to input video
        output_dir: Path to output directory
        detector_type: Type of face detector
        confidence_threshold: Confidence threshold for face detection
        max_faces: Maximum number of faces to extract per frame
        target_size: Target size for extracted faces
        expand_ratio: Ratio to expand the bounding box
        align: Whether to align faces using landmarks
        frame_interval: Extract faces every N frames
        device: Device to use
        save_metadata: Whether to save metadata for each extracted face
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize face detector
    detector = get_face_detector(
        detector_type=detector_type,
        device=device,
        confidence_threshold=confidence_threshold,
        max_faces=max_faces
    )
    
    # Open video
    video = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create metadata
    metadata = {
        "video_path": video_path,
        "fps": fps,
        "frame_count": frame_count,
        "detector_type": detector_type,
        "confidence_threshold": confidence_threshold,
        "max_faces": max_faces,
        "target_size": target_size,
        "expand_ratio": expand_ratio,
        "align": align,
        "frame_interval": frame_interval,
        "frames": {}
    }
    
    # Process frames
    frame_idx = 0
    with tqdm.tqdm(total=frame_count) as pbar:
        while True:
            # Read frame
            ret, frame = video.read()
            
            # Break if end of video
            if not ret:
                break
            
            # Skip frames
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                pbar.update(1)
                continue
            
            # Detect faces
            detections = detector.detect(frame)
            
            # Process each detection
            for i, detection in enumerate(detections):
                # Extract face
                if align:
                    face = detector.align_face(
                        frame, 
                        detection, 
                        target_size=(target_size, target_size),
                        expand_ratio=expand_ratio
                    )
                else:
                    face = detector.extract_face(
                        frame, 
                        detection, 
                        target_size=(target_size, target_size),
                        expand_ratio=expand_ratio
                    )
                
                # Save face
                face_path = os.path.join(
                    output_dir, 
                    f"frame_{frame_idx:06d}_face_{i:02d}.png"
                )
                cv2.imwrite(face_path, face)
                
                # Save metadata
                if save_metadata:
                    metadata["frames"][frame_idx] = {
                        "frame_idx": frame_idx,
                        "face_idx": i,
                        "bbox": detection.bbox.tolist(),
                        "landmarks": detection.landmarks.tolist(),
                        "score": detection.score,
                        "face_path": face_path
                    }
            
            # Update frame index
            frame_idx += 1
            pbar.update(1)
    
    # Release video
    video.release()
    
    # Save metadata
    if save_metadata:
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

def extract_faces_from_images(
    image_dir: str,
    output_dir: str,
    detector_type: str = "retinaface",
    confidence_threshold: float = 0.9,
    max_faces: int = 1,
    target_size: int = 512,
    expand_ratio: float = 1.5,
    align: bool = True,
    device: str = "cuda",
    save_metadata: bool = False
) -> None:
    """
    Extract faces from a directory of images
    
    Args:
        image_dir: Path to input directory
        output_dir: Path to output directory
        detector_type: Type of face detector
        confidence_threshold: Confidence threshold for face detection
        max_faces: Maximum number of faces to extract per image
        target_size: Target size for extracted faces
        expand_ratio: Ratio to expand the bounding box
        align: Whether to align faces using landmarks
        device: Device to use
        save_metadata: Whether to save metadata for each extracted face
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize face detector
    detector = get_face_detector(
        detector_type=detector_type,
        device=device,
        confidence_threshold=confidence_threshold,
        max_faces=max_faces
    )
    
    # Get image paths
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    image_paths += sorted(glob.glob(os.path.join(image_dir, "*.png")))
    
    # Create metadata
    metadata = {
        "image_dir": image_dir,
        "detector_type": detector_type,
        "confidence_threshold": confidence_threshold,
        "max_faces": max_faces,
        "target_size": target_size,
        "expand_ratio": expand_ratio,
        "align": align,
        "images": {}
    }
    
    # Process images
    for image_idx, image_path in enumerate(tqdm.tqdm(image_paths)):
        # Read image
        image = cv2.imread(image_path)
        
        # Skip invalid images
        if image is None:
            continue
        
        # Detect faces
        detections = detector.detect(image)
        
        # Process each detection
        for i, detection in enumerate(detections):
            # Extract face
            if align:
                face = detector.align_face(
                    image, 
                    detection, 
                    target_size=(target_size, target_size),
                    expand_ratio=expand_ratio
                )
            else:
                face = detector.extract_face(
                    image, 
                    detection, 
                    target_size=(target_size, target_size),
                    expand_ratio=expand_ratio
                )
            
            # Save face
            face_path = os.path.join(
                output_dir, 
                f"{Path(image_path).stem}_face_{i:02d}.png"
            )
            cv2.imwrite(face_path, face)
            
            # Save metadata
            if save_metadata:
                metadata["images"][image_path] = {
                    "image_idx": image_idx,
                    "face_idx": i,
                    "bbox": detection.bbox.tolist(),
                    "landmarks": detection.landmarks.tolist(),
                    "score": detection.score,
                    "face_path": face_path
                }
    
    # Save metadata
    if save_metadata:
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

def main():
    # Parse arguments
    args = parse_args()
    
    # Check if input is a video or directory
    if os.path.isfile(args.input):
        # Extract faces from video
        extract_faces_from_video(
            video_path=args.input,
            output_dir=args.output,
            detector_type=args.detector,
            confidence_threshold=args.confidence,
            max_faces=args.max_faces,
            target_size=args.target_size,
            expand_ratio=args.expand_ratio,
            align=args.align,
            frame_interval=args.frame_interval,
            device=args.device,
            save_metadata=args.save_metadata
        )
    elif os.path.isdir(args.input):
        # Extract faces from images
        extract_faces_from_images(
            image_dir=args.input,
            output_dir=args.output,
            detector_type=args.detector,
            confidence_threshold=args.confidence,
            max_faces=args.max_faces,
            target_size=args.target_size,
            expand_ratio=args.expand_ratio,
            align=args.align,
            device=args.device,
            save_metadata=args.save_metadata
        )
    else:
        raise ValueError(f"Input path does not exist: {args.input}")

if __name__ == "__main__":
    main()
