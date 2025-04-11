import os
import torch
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

@dataclass
class FaceDetection:
    """Data class for storing face detection results"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    landmarks: np.ndarray  # [5, 2] for 5 facial landmarks
    score: float
    index: int = 0  # Index of the face in the image

class FaceDetector:
    """
    Base class for face detectors
    """
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in an image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of FaceDetection objects
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def extract_face(
        self, 
        image: np.ndarray, 
        detection: FaceDetection,
        target_size: Tuple[int, int] = (256, 256),
        expand_ratio: float = 1.5
    ) -> np.ndarray:
        """
        Extract face from image based on detection
        
        Args:
            image: Input image (BGR format)
            detection: Face detection
            target_size: Target size for extracted face
            expand_ratio: Ratio to expand the bounding box
            
        Returns:
            Extracted face image
        """
        # Get bounding box
        x1, y1, x2, y2 = detection.bbox.astype(int)
        
        # Calculate center and size
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = int((x2 - x1) * expand_ratio)
        height = int((y2 - y1) * expand_ratio)
        
        # Calculate new bounding box
        new_x1 = max(0, center_x - width // 2)
        new_y1 = max(0, center_y - height // 2)
        new_x2 = min(image.shape[1], center_x + width // 2)
        new_y2 = min(image.shape[0], center_y + height // 2)
        
        # Extract face
        face = image[new_y1:new_y2, new_x1:new_x2]
        
        # Resize to target size
        face = cv2.resize(face, target_size)
        
        return face
    
    def align_face(
        self, 
        image: np.ndarray, 
        detection: FaceDetection,
        target_size: Tuple[int, int] = (256, 256),
        expand_ratio: float = 1.5
    ) -> np.ndarray:
        """
        Align face using facial landmarks
        
        Args:
            image: Input image (BGR format)
            detection: Face detection
            target_size: Target size for aligned face
            expand_ratio: Ratio to expand the bounding box
            
        Returns:
            Aligned face image
        """
        # Get landmarks
        landmarks = detection.landmarks
        
        # Define reference landmarks (centered in the target image)
        target_width, target_height = target_size
        reference_landmarks = np.array([
            [0.31 * target_width, 0.315 * target_height],  # Left eye
            [0.69 * target_width, 0.315 * target_height],  # Right eye
            [0.50 * target_width, 0.47 * target_height],   # Nose
            [0.355 * target_width, 0.63 * target_height],  # Left mouth
            [0.645 * target_width, 0.63 * target_height]   # Right mouth
        ])
        
        # Calculate transformation matrix
        transformation_matrix = cv2.estimateAffinePartial2D(
            landmarks, reference_landmarks, method=cv2.LMEDS
        )[0]
        
        # Apply transformation
        aligned_face = cv2.warpAffine(
            image, transformation_matrix, target_size, borderMode=cv2.BORDER_CONSTANT
        )
        
        return aligned_face


class RetinaFaceDetector(FaceDetector):
    """
    Face detector using RetinaFace
    """
    def __init__(
        self, 
        device: str = "cuda",
        model_path: Optional[str] = None,
        network: str = "resnet50",
        confidence_threshold: float = 0.9,
        nms_threshold: float = 0.4,
        max_faces: int = 1
    ):
        super().__init__(device)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_faces = max_faces
        
        # Import here to avoid dependency issues
        try:
            from retinaface.pre_trained_models import get_model
            from retinaface.utils import load_state_dict
        except ImportError:
            raise ImportError(
                "RetinaFace is not installed. Please install it with: "
                "pip install retina-face"
            )
        
        # Load model
        if network == "resnet50":
            self.model = get_model("resnet50_2020-07-20", max_size=2048)
            if model_path is not None and os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path))
        elif network == "mobilenet":
            self.model = get_model("mobilenet_v2_2020-07-20", max_size=2048)
            if model_path is not None and os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path))
        else:
            raise ValueError(f"Unsupported network: {network}")
        
        # Move model to device
        self.model = self.model.to(device)
        self.model.eval()
    
    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in an image using RetinaFace
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of FaceDetection objects
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Detect faces
        with torch.no_grad():
            annotations = self.model(image_tensor)[0]
        
        # Process detections
        detections = []
        for i, (bbox, score, landmarks) in enumerate(zip(
            annotations["boxes"], 
            annotations["scores"], 
            annotations["landmarks"]
        )):
            # Skip low confidence detections
            if score < self.confidence_threshold:
                continue
            
            # Convert to numpy
            bbox_np = bbox.cpu().numpy()
            landmarks_np = landmarks.cpu().numpy().reshape(5, 2)
            score_np = score.cpu().numpy()
            
            # Create detection
            detection = FaceDetection(
                bbox=bbox_np,
                landmarks=landmarks_np,
                score=float(score_np),
                index=i
            )
            
            detections.append(detection)
        
        # Sort by score
        detections = sorted(detections, key=lambda x: x.score, reverse=True)
        
        # Limit number of faces
        if self.max_faces > 0:
            detections = detections[:self.max_faces]
        
        return detections


class InsightFaceDetector(FaceDetector):
    """
    Face detector using InsightFace
    """
    def __init__(
        self, 
        device: str = "cuda",
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.9,
        max_faces: int = 1
    ):
        super().__init__(device)
        self.confidence_threshold = confidence_threshold
        self.max_faces = max_faces
        
        # Import here to avoid dependency issues
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "InsightFace is not installed. Please install it with: "
                "pip install insightface onnxruntime-gpu"
            )
        
        # Initialize face analyzer
        self.app = FaceAnalysis(
            name="buffalo_l",  # Use large model for better accuracy
            providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(640, 640))
    
    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in an image using InsightFace
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of FaceDetection objects
        """
        # Detect faces
        faces = self.app.get(image)
        
        # Process detections
        detections = []
        for i, face in enumerate(faces):
            # Skip low confidence detections
            if face.det_score < self.confidence_threshold:
                continue
            
            # Get bounding box
            bbox = face.bbox
            
            # Get landmarks
            landmarks = face.kps
            
            # Create detection
            detection = FaceDetection(
                bbox=bbox,
                landmarks=landmarks,
                score=float(face.det_score),
                index=i
            )
            
            detections.append(detection)
        
        # Sort by score
        detections = sorted(detections, key=lambda x: x.score, reverse=True)
        
        # Limit number of faces
        if self.max_faces > 0:
            detections = detections[:self.max_faces]
        
        return detections


def get_face_detector(
    detector_type: str = "retinaface",
    device: str = "cuda",
    **kwargs
) -> FaceDetector:
    """
    Get face detector by type
    
    Args:
        detector_type: Type of face detector
        device: Device to use
        **kwargs: Additional arguments for the detector
        
    Returns:
        Face detector
    """
    if detector_type == "retinaface":
        return RetinaFaceDetector(device=device, **kwargs)
    elif detector_type == "insightface":
        return InsightFaceDetector(device=device, **kwargs)
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")
