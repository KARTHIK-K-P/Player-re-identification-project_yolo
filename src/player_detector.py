import torch
import cv2
import numpy as np
from ultralytics import YOLO
import logging

class PlayerDetector:
    """
    Player detection class using YOLO model
    Detects players (person class) in video frames
    """
    
    def __init__(self, model_path="yolov5s.pt", confidence_threshold=0.5, 
                 iou_threshold=0.45, class_filter=[0]):
        """
        Initialize the player detector
        
        Args:
            model_path (str): Path to YOLO model file
            confidence_threshold (float): Minimum confidence for detections
            iou_threshold (float): IoU threshold for NMS
            class_filter (list): List of class IDs to detect (0 = person)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.class_filter = class_filter
        
        self.logger = logging.getLogger(__name__)
        
        # Load YOLO model
        self._load_model()
        
    def _load_model(self):
        """Load the YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            self.logger.info(f"YOLO model loaded successfully from {self.model_path}")
            
            # Set model parameters
            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {str(e)}")
            raise
    
    def detect(self, frame):
        """
        Detect players in a frame
        
        Args:
            frame (np.array): Input frame (BGR format)
            
        Returns:
            list: List of detections [x1, y1, x2, y2, confidence, class_id]
        """
        if frame is None or frame.size == 0:
            return []
        
        try:
            # Run inference
            results = self.model(frame, verbose=False)
            
            detections = []
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    # Extract box coordinates, confidence, and class
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Filter by class (only person class)
                        if class_id in self.class_filter:
                            # Filter by confidence
                            if confidence >= self.confidence_threshold:
                                detections.append([
                                    float(x1), float(y1), float(x2), float(y2),
                                    float(confidence), int(class_id)
                                ])
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection failed: {str(e)}")
            return []
    
    def detect_batch(self, frames):
        """
        Detect players in multiple frames
        
        Args:
            frames (list): List of frames
            
        Returns:
            list: List of detection results for each frame
        """
        batch_results = []
        
        for frame in frames:
            detections = self.detect(frame)
            batch_results.append(detections)
        
        return batch_results
    
    def visualize_detections(self, frame, detections, color=(0, 255, 0), thickness=2):
        """
        Visualize detections on frame
        
        Args:
            frame (np.array): Input frame
            detections (list): List of detections
            color (tuple): Bounding box color (BGR)
            thickness (int): Line thickness
            
        Returns:
            np.array: Frame with visualized detections
        """
        vis_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
            # Draw confidence
            label = f"Person: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(vis_frame, (int(x1), int(y1) - label_size[1] - 10),
                         (int(x1) + label_size[0], int(y1)), color, -1)
            cv2.putText(vis_frame, label, (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return vis_frame
    
    def get_detection_crops(self, frame, detections, padding=10):
        """
        Extract cropped regions from detections
        
        Args:
            frame (np.array): Input frame
            detections (list): List of detections
            padding (int): Padding around bounding box
            
        Returns:
            list: List of cropped images
        """
        crops = []
        h, w = frame.shape[:2]
        
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            
            # Add padding
            x1 = max(0, int(x1) - padding)
            y1 = max(0, int(y1) - padding)
            x2 = min(w, int(x2) + padding)
            y2 = min(h, int(y2) + padding)
            
            # Extract crop
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
            else:
                # Return empty crop if invalid
                crops.append(np.zeros((64, 64, 3), dtype=np.uint8))
        
        return crops
    
    def filter_detections_by_size(self, detections, min_area=1000, max_area=50000):
        """
        Filter detections by bounding box area
        
        Args:
            detections (list): List of detections
            min_area (int): Minimum bounding box area
            max_area (int): Maximum bounding box area
            
        Returns:
            list: Filtered detections
        """
        filtered_detections = []
        
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            area = (x2 - x1) * (y2 - y1)
            
            if min_area <= area <= max_area:
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def filter_detections_by_aspect_ratio(self, detections, min_ratio=0.3, max_ratio=3.0):
        """
        Filter detections by aspect ratio (width/height)
        
        Args:
            detections (list): List of detections
            min_ratio (float): Minimum aspect ratio
            max_ratio (float): Maximum aspect ratio
            
        Returns:
            list: Filtered detections
        """
        filtered_detections = []
        
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            width = x2 - x1
            height = y2 - y1
            
            if height > 0:
                aspect_ratio = width / height
                if min_ratio <= aspect_ratio <= max_ratio:
                    filtered_detections.append(detection)
        
        return filtered_detections
    
    def get_detection_centers(self, detections):
        """
        Get center points of detections
        
        Args:
            detections (list): List of detections
            
        Returns:
            list: List of [x, y] center coordinates
        """
        centers = []
        
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append([center_x, center_y])
        
        return centers
    
    def update_parameters(self, confidence_threshold=None, iou_threshold=None):
        """
        Update detection parameters
        
        Args:
            confidence_threshold (float): New confidence threshold
            iou_threshold (float): New IoU threshold
        """
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
            self.model.conf = confidence_threshold
        
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
            self.model.iou = iou_threshold
        
        self.logger.info(f"Updated parameters: conf={self.confidence_threshold}, iou={self.iou_threshold}")
    
    def __str__(self):
        """String representation of detector"""
        return f"PlayerDetector: model={self.model_path}, conf={self.confidence_threshold}, iou={self.iou_threshold}"