import os
import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any

from latentsync.utils.face_detection import get_face_detector, FaceDetection

class FaceProcessor:
    """
    Face processor for inference
    """
    def __init__(
        self,
        detector_type: str = "retinaface",
        confidence_threshold: float = 0.9,
        target_size: int = 512,
        expand_ratio: float = 1.5,
        device: str = "cuda"
    ):
        self.detector_type = detector_type
        self.confidence_threshold = confidence_threshold
        self.target_size = target_size
        self.expand_ratio = expand_ratio
        self.device = device
        
        # Initialize face detector
        self.detector = get_face_detector(
            detector_type=detector_type,
            device=device,
            confidence_threshold=confidence_threshold,
            max_faces=1
        )
    
    def process_frame(
        self,
        frame: np.ndarray
    ) -> Dict[str, Any]:
        """
        Process a frame for inference
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary containing processed data
        """
        # Detect faces
        detections = self.detector.detect(frame)
        
        # If no faces detected, return None
        if not detections:
            return {
                "success": False,
                "message": "No face detected"
            }
        
        # Get first detection
        detection = detections[0]
        
        # Align face
        aligned_face = self.detector.align_face(
            frame, 
            detection, 
            target_size=(self.target_size, self.target_size),
            expand_ratio=self.expand_ratio
        )
        
        # Convert to tensor
        face_tensor = torch.from_numpy(aligned_face).permute(2, 0, 1).float().div(255.0)
        
        # Move to device
        face_tensor = face_tensor.to(self.device)
        
        return {
            "success": True,
            "detection": detection,
            "aligned_face": aligned_face,
            "face_tensor": face_tensor
        }
    
    def process_video(
        self,
        video_path: str,
        frame_interval: int = 1
    ) -> Dict[str, Any]:
        """
        Process a video for inference
        
        Args:
            video_path: Path to input video
            frame_interval: Process every N frames
            
        Returns:
            Dictionary containing processed data
        """
        # Open video
        video = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize results
        frames = []
        detections = []
        aligned_faces = []
        face_tensors = []
        
        # Process frames
        frame_idx = 0
        while True:
            # Read frame
            ret, frame = video.read()
            
            # Break if end of video
            if not ret:
                break
            
            # Skip frames
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue
            
            # Process frame
            result = self.process_frame(frame)
            
            # Skip if no face detected
            if not result["success"]:
                frame_idx += 1
                continue
            
            # Add to results
            frames.append(frame)
            detections.append(result["detection"])
            aligned_faces.append(result["aligned_face"])
            face_tensors.append(result["face_tensor"])
            
            # Update frame index
            frame_idx += 1
        
        # Release video
        video.release()
        
        # Stack tensors
        if face_tensors:
            face_tensors = torch.stack(face_tensors)
        else:
            face_tensors = torch.zeros((0, 3, self.target_size, self.target_size), device=self.device)
        
        return {
            "success": len(frames) > 0,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "frames": frames,
            "detections": detections,
            "aligned_faces": aligned_faces,
            "face_tensors": face_tensors
        }
    
    def blend_face_back(
        self,
        original_frame: np.ndarray,
        generated_face: np.ndarray,
        detection: FaceDetection,
        blend_mask: Optional[np.ndarray] = None,
        smooth_boundary: bool = True
    ) -> np.ndarray:
        """
        Blend generated face back into the original frame
        
        Args:
            original_frame: Original frame
            generated_face: Generated face
            detection: Face detection
            blend_mask: Optional mask for blending
            smooth_boundary: Whether to smooth the boundary
            
        Returns:
            Blended frame
        """
        # Get bounding box
        x1, y1, x2, y2 = detection.bbox.astype(int)
        
        # Calculate center and size
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = int((x2 - x1) * self.expand_ratio)
        height = int((y2 - y1) * self.expand_ratio)
        
        # Calculate new bounding box
        new_x1 = max(0, center_x - width // 2)
        new_y1 = max(0, center_y - height // 2)
        new_x2 = min(original_frame.shape[1], center_x + width // 2)
        new_y2 = min(original_frame.shape[0], center_y + height // 2)
        
        # Resize generated face
        face_resized = cv2.resize(
            generated_face, 
            (new_x2 - new_x1, new_y2 - new_y1)
        )
        
        # Create mask
        if blend_mask is None:
            # Create elliptical mask
            mask = np.zeros((new_y2 - new_y1, new_x2 - new_x1), dtype=np.float32)
            center = (mask.shape[1] // 2, mask.shape[0] // 2)
            axes = (mask.shape[1] // 2 - 1, mask.shape[0] // 2 - 1)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
            
            # Smooth mask
            if smooth_boundary:
                mask = cv2.GaussianBlur(mask, (19, 19), 5)
        else:
            # Resize mask
            mask = cv2.resize(
                blend_mask, 
                (new_x2 - new_x1, new_y2 - new_y1)
            )
        
        # Expand mask to 3 channels
        mask_3c = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
        # Create output frame
        output_frame = original_frame.copy()
        
        # Blend face
        output_frame[new_y1:new_y2, new_x1:new_x2] = (
            face_resized * mask_3c + 
            output_frame[new_y1:new_y2, new_x1:new_x2] * (1 - mask_3c)
        ).astype(np.uint8)
        
        return output_frame
