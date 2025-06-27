import os
import sys
import cv2
import numpy as np
import json
import pickle
from datetime import datetime
from tqdm import tqdm
import logging
import colorsys
from scipy.spatial.distance import cosine, euclidean
from collections import OrderedDict
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn as nn

# Configuration dictionary (embedded directly)
CONFIG = {
    'video': {
        'camera1_path': "data/videos/broadcast.mp4",
        'camera2_path': "data/videos/tacticam.mp4", 
        'output_path': "output/videos/"
    },
    'detection': {
        'model_path': "data/models/yolov5s.pt",
        'confidence_threshold': 0.5,
        'iou_threshold': 0.45,
        'class_filter': [0]  # person class in COCO
    },
    'reid': {
        'feature_dim': 512,
        'similarity_threshold': 0.7,
        'max_distance': 2.0
    },
    'tracking': {
        'max_disappeared': 30,
        'max_distance': 100
    },
    'visualization': {
        'fps': 30,
        'font_scale': 0.8,
        'thickness': 2
    }
}

# Utility Functions
def ensure_dir(directory):
    """Ensure directory exists, create if not"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    ensure_dir("logs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/player_reid_{timestamp}.log"
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")
    logger.info(f"Log file: {log_filename}")

def create_color_palette(num_colors):
    """Create a color palette for visualization"""
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    return colors

# Player Detector Class
class PlayerDetector:
    """YOLO-based player detector"""
    
    def __init__(self, model_path, confidence_threshold=0.5, iou_threshold=0.45, class_filter=None):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.class_filter = class_filter or [0]  # person class
        
        try:
            # Try to load YOLOv5 model
            import torch
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.conf = confidence_threshold
            self.model.iou = iou_threshold
            self.use_yolo = True
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}")
            print("Using OpenCV HOG detector as fallback")
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.use_yolo = False
    
    def detect(self, frame):
        """Detect players in frame"""
        if self.use_yolo:
            return self._detect_yolo(frame)
        else:
            return self._detect_hog(frame)
    
    def _detect_yolo(self, frame):
        """Detect using YOLO"""
        results = self.model(frame)
        detections = []
        
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if int(cls) in self.class_filter and conf >= self.confidence_threshold:
                x1, y1, x2, y2 = map(int, box)
                detections.append([x1, y1, x2, y2, conf, int(cls)])
        
        return detections
    
    def _detect_hog(self, frame):
        """Detect using HOG (fallback)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes, weights = self.hog.detectMultiScale(
            gray, 
            winStride=(8, 8),
            padding=(32, 32),
            scale=1.05
        )
        
        detections = []
        for i, (x, y, w, h) in enumerate(boxes):
            if weights[i] >= self.confidence_threshold:
                detections.append([x, y, x + w, y + h, weights[i], 0])
        
        return detections

# Feature Extractor Class
class FeatureExtractor:
    """Extract features from player crops using ResNet"""
    
    def __init__(self, feature_dim=512):
        self.feature_dim = feature_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained ResNet50
        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Identity()  # Remove final classification layer
        self.model.eval()
        self.model.to(self.device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, crop):
        """Extract features from player crop"""
        if crop.size == 0:
            return np.zeros(self.feature_dim)
        
        try:
            # Preprocess image
            if len(crop.shape) == 3:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            else:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            
            input_tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(input_tensor)
                features = features.cpu().numpy().flatten()
            
            # Normalize features
            features = features / (np.linalg.norm(features) + 1e-8)
            
            return features[:self.feature_dim]
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros(self.feature_dim)

# Re-ID Matcher Class
class ReIDMatcher:
    """Match players across camera views"""
    
    def __init__(self, similarity_threshold=0.7, max_distance=2.0):
        self.similarity_threshold = similarity_threshold
        self.max_distance = max_distance
    
    def compute_similarity(self, feat1, feat2):
        """Compute similarity between two feature vectors"""
        try:
            # Cosine similarity
            cosine_sim = 1 - cosine(feat1, feat2)
            return cosine_sim
        except:
            return 0.0
    
    def match_objects(self, cam1_objects, cam2_objects):
        """Match objects between camera views"""
        matches = []
        
        for id1, obj1 in cam1_objects.items():
            if 'avg_feature' not in obj1:
                continue
                
            best_match = None
            best_similarity = 0
            
            for id2, obj2 in cam2_objects.items():
                if 'avg_feature' not in obj2:
                    continue
                
                similarity = self.compute_similarity(obj1['avg_feature'], obj2['avg_feature'])
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = id2
            
            if best_match is not None:
                matches.append({
                    'cam1_id': id1,
                    'cam2_id': best_match,
                    'similarity': best_similarity
                })
        
        return matches

# Player Tracker Class
class PlayerTracker:
    """Track players across frames"""
    
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def register(self, centroid, feature):
        """Register a new object"""
        self.objects[self.next_id] = {
            'centroid': centroid,
            'feature': feature
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def deregister(self, object_id):
        """Deregister an object"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, centroids, features):
        """Update tracker with new detections"""
        if len(centroids) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return []
        
        # If no existing objects, register all as new
        if len(self.objects) == 0:
            for i, centroid in enumerate(centroids):
                feature = features[i] if i < len(features) else np.zeros(512)
                self.register(centroid, feature)
        else:
            # Compute distances between existing objects and new centroids
            object_centroids = [obj['centroid'] for obj in self.objects.values()]
            object_ids = list(self.objects.keys())
            
            # Compute distance matrix
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - 
                             np.array(centroids), axis=2)
            
            # Find minimum values and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            # Keep track of used row and column indices
            used_rows = set()
            used_cols = set()
            
            # Update existing objects
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                if D[row, col] <= self.max_distance:
                    object_id = object_ids[row]
                    self.objects[object_id]['centroid'] = centroids[col]
                    if col < len(features):
                        self.objects[object_id]['feature'] = features[col]
                    self.disappeared[object_id] = 0
                    
                    used_rows.add(row)
                    used_cols.add(col)
            
            # Handle unmatched detections and objects
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            
            if D.shape[0] >= D.shape[1]:
                # More objects than detections
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                # More detections than objects
                for col in unused_cols:
                    feature = features[col] if col < len(features) else np.zeros(512)
                    self.register(centroids[col], feature)
        
        # Return current tracked objects
        tracked_objects = []
        for object_id, obj in self.objects.items():
            tracked_objects.append({
                'id': object_id,
                'centroid': obj['centroid'],
                'feature': obj['feature']
            })
        
        return tracked_objects

# Visualizer Class
class Visualizer:
    """Visualize detections and tracking results"""
    
    def __init__(self, fps=30, font_scale=0.8, thickness=2):
        self.fps = fps
        self.font_scale = font_scale
        self.thickness = thickness
        self.colors = create_color_palette(50)  # Pre-generate colors
    
    def draw_detections_and_ids(self, frame, tracked_objects, camera_id):
        """Draw detections and tracking IDs on frame"""
        vis_frame = frame.copy()
        
        for obj in tracked_objects:
            obj_id = obj['id']
            centroid = obj['centroid']
            
            # Get color for this ID
            color = self.colors[obj_id % len(self.colors)]
            
            # Draw centroid
            cv2.circle(vis_frame, tuple(centroid), 5, color, -1)
            
            # Draw ID text
            text = f"ID: {obj_id}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                      self.font_scale, self.thickness)[0]
            text_x = centroid[0] - text_size[0] // 2
            text_y = centroid[1] - 15
            
            # Draw text background
            cv2.rectangle(vis_frame, 
                         (text_x - 5, text_y - text_size[1] - 5),
                         (text_x + text_size[0] + 5, text_y + 5),
                         color, -1)
            
            # Draw text
            cv2.putText(vis_frame, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                       (255, 255, 255), self.thickness)
        
        # Add camera label
        cv2.putText(vis_frame, f"Camera {camera_id}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        return vis_frame
    
    def create_side_by_side(self, frame1, frame2, target_height):
        """Create side-by-side visualization"""
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        # Resize frames to target height
        if h1 != target_height:
            ratio = target_height / h1
            frame1 = cv2.resize(frame1, (int(w1 * ratio), target_height))
        
        if h2 != target_height:
            ratio = target_height / h2
            frame2 = cv2.resize(frame2, (int(w2 * ratio), target_height))
        
        # Concatenate horizontally
        combined = np.hstack((frame1, frame2))
        return combined
    
    def add_frame_info(self, frame, frame_idx, num_matches):
        """Add frame information to visualization"""
        info_text = f"Frame: {frame_idx} | Matches: {num_matches}"
        cv2.putText(frame, info_text, (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

# Main Player Re-ID System Class
class PlayerReIDSystem:
    """Main Player Re-Identification System"""
    
    def __init__(self):
        self.config = CONFIG
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        self._setup_directories()
        
        # Storage for results
        self.camera1_results = []
        self.camera2_results = []
        self.matched_pairs = []
    
    def _initialize_components(self):
        """Initialize all system components"""
        self.logger.info("Initializing system components...")
        
        self.detector = PlayerDetector(
            model_path=self.config['detection']['model_path'],
            confidence_threshold=self.config['detection']['confidence_threshold'],
            iou_threshold=self.config['detection']['iou_threshold'],
            class_filter=self.config['detection']['class_filter']
        )
        
        self.feature_extractor = FeatureExtractor(
            feature_dim=self.config['reid']['feature_dim']
        )
        
        self.reid_matcher = ReIDMatcher(
            similarity_threshold=self.config['reid']['similarity_threshold'],
            max_distance=self.config['reid']['max_distance']
        )
        
        self.tracker_cam1 = PlayerTracker(
            max_disappeared=self.config['tracking']['max_disappeared'],
            max_distance=self.config['tracking']['max_distance']
        )
        
        self.tracker_cam2 = PlayerTracker(
            max_disappeared=self.config['tracking']['max_disappeared'],
            max_distance=self.config['tracking']['max_distance']
        )
        
        self.visualizer = Visualizer(
            fps=self.config['visualization']['fps'],
            font_scale=self.config['visualization']['font_scale'],
            thickness=self.config['visualization']['thickness']
        )
    
    def _setup_directories(self):
        """Create necessary output directories"""
        directories = [
            "output/videos",
            "output/detections", 
            "output/features",
            "logs"
        ]
        
        for directory in directories:
            ensure_dir(directory)
    
    def load_videos(self):
        """Load video streams from both cameras"""
        self.logger.info("Loading video streams...")
        
        self.cap1 = cv2.VideoCapture(self.config['video']['camera1_path'])
        if not self.cap1.isOpened():
            raise ValueError(f"Cannot open camera 1 video: {self.config['video']['camera1_path']}")
        
        self.cap2 = cv2.VideoCapture(self.config['video']['camera2_path'])
        if not self.cap2.isOpened():
            raise ValueError(f"Cannot open camera 2 video: {self.config['video']['camera2_path']}")
        
        # Get video properties
        self.fps1 = int(self.cap1.get(cv2.CAP_PROP_FPS))
        self.fps2 = int(self.cap2.get(cv2.CAP_PROP_FPS))
        self.frame_count1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Camera 1: {self.frame_count1} frames at {self.fps1} FPS")
        self.logger.info(f"Camera 2: {self.frame_count2} frames at {self.fps2} FPS")
        
        self.total_frames = min(self.frame_count1, self.frame_count2)
    
    def process_frame_pair(self, frame1, frame2, frame_idx):
        """Process a pair of synchronized frames from both cameras"""
        results = {
            'frame_idx': frame_idx,
            'camera1': {'detections': [], 'features': [], 'tracked_objects': []},
            'camera2': {'detections': [], 'features': [], 'tracked_objects': []}
        }
        
        # Process camera 1 frame
        if frame1 is not None:
            detections1 = self.detector.detect(frame1)
            results['camera1']['detections'] = detections1
            
            features1 = []
            for detection in detections1:
                x1, y1, x2, y2, conf, cls = detection
                player_crop = frame1[int(y1):int(y2), int(x1):int(x2)]
                if player_crop.size > 0:
                    feature = self.feature_extractor.extract_features(player_crop)
                    features1.append(feature)
                else:
                    features1.append(np.zeros(self.config['reid']['feature_dim']))
            
            results['camera1']['features'] = features1
            
            centroids1 = []
            for detection in detections1:
                x1, y1, x2, y2, conf, cls = detection
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                centroids1.append([center_x, center_y])
            
            tracked_objects1 = self.tracker_cam1.update(centroids1, features1)
            results['camera1']['tracked_objects'] = tracked_objects1
        
        # Process camera 2 frame
        if frame2 is not None:
            detections2 = self.detector.detect(frame2)
            results['camera2']['detections'] = detections2
            
            features2 = []
            for detection in detections2:
                x1, y1, x2, y2, conf, cls = detection
                player_crop = frame2[int(y1):int(y2), int(x1):int(x2)]
                if player_crop.size > 0:
                    feature = self.feature_extractor.extract_features(player_crop)
                    features2.append(feature)
                else:
                    features2.append(np.zeros(self.config['reid']['feature_dim']))
            
            results['camera2']['features'] = features2
            
            centroids2 = []
            for detection in detections2:
                x1, y1, x2, y2, conf, cls = detection
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                centroids2.append([center_x, center_y])
            
            tracked_objects2 = self.tracker_cam2.update(centroids2, features2)
            results['camera2']['tracked_objects'] = tracked_objects2
        
        return results
    
    def perform_reid_matching(self):
        """Perform re-identification matching between camera views"""
        self.logger.info("Performing re-identification matching...")
        
        cam1_objects = {}
        cam2_objects = {}
        
        # Aggregate features for each tracked ID
        for frame_results in self.camera1_results:
            for obj in frame_results['camera1']['tracked_objects']:
                obj_id = obj['id']
                if obj_id not in cam1_objects:
                    cam1_objects[obj_id] = {
                        'features': [],
                        'detections': [],
                        'frames': []
                    }
                cam1_objects[obj_id]['features'].append(obj['feature'])
                cam1_objects[obj_id]['detections'].append(obj['centroid'])
                cam1_objects[obj_id]['frames'].append(frame_results['frame_idx'])
        
        for frame_results in self.camera2_results:
            for obj in frame_results['camera2']['tracked_objects']:
                obj_id = obj['id']
                if obj_id not in cam2_objects:
                    cam2_objects[obj_id] = {
                        'features': [],
                        'detections': [],
                        'frames': []
                    }
                cam2_objects[obj_id]['features'].append(obj['feature'])
                cam2_objects[obj_id]['detections'].append(obj['centroid'])
                cam2_objects[obj_id]['frames'].append(frame_results['frame_idx'])
        
        # Compute average features for each object
        for obj_id in cam1_objects:
            if cam1_objects[obj_id]['features']:
                features = np.array(cam1_objects[obj_id]['features'])
                cam1_objects[obj_id]['avg_feature'] = np.mean(features, axis=0)
        
        for obj_id in cam2_objects:
            if cam2_objects[obj_id]['features']:
                features = np.array(cam2_objects[obj_id]['features'])
                cam2_objects[obj_id]['avg_feature'] = np.mean(features, axis=0)
        
        # Perform matching
        self.matched_pairs = self.reid_matcher.match_objects(cam1_objects, cam2_objects)
        self.logger.info(f"Found {len(self.matched_pairs)} matched pairs")
    
    def process_videos(self):
        """Main processing loop for both video streams"""
        self.logger.info("Starting video processing...")
        
        pbar = tqdm(total=self.total_frames, desc="Processing frames")
        
        frame_idx = 0
        while frame_idx < self.total_frames:
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()
            
            if not ret1 or not ret2:
                self.logger.warning(f"Failed to read frame {frame_idx}")
                break
            
            results = self.process_frame_pair(frame1, frame2, frame_idx)
            self.camera1_results.append(results)
            self.camera2_results.append(results)
            
            frame_idx += 1
            pbar.update(1)
            
            # Limit processing for testing (uncomment to use)
            # if frame_idx > 100:
            #     break
        
        pbar.close()
        self.logger.info(f"Processed {frame_idx} frames")
    
    def save_results(self):
        """Save detection and feature results to files"""
        self.logger.info("Saving results...")
        
        # Save detections
        detections_file = "output/detections/detections.json"
        with open(detections_file, 'w') as f:
            json.dump({
                'camera1': self.camera1_results,
                'camera2': self.camera2_results
            }, f, indent=2, default=str)
        
        # Save features
        features_file = "output/features/features.pkl"
        with open(features_file, 'wb') as f:
            pickle.dump({
                'camera1': self.camera1_results,
                'camera2': self.camera2_results,
                'matched_pairs': self.matched_pairs
            }, f)
        
        # Save matching results
        matching_file = "output/features/matched_pairs.json"
        with open(matching_file, 'w') as f:
            json.dump(self.matched_pairs, f, indent=2, default=str)
        
        self.logger.info("Results saved successfully")
    
    def generate_output_video(self):
        """Generate synchronized output video with visualizations"""
        self.logger.info("Generating output video...")
        
        # Reset video captures
        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Setup video writer
        output_path = "output/videos/synchronized_output.mp4"
        ensure_dir(os.path.dirname(output_path))
        
        # Get frame dimensions
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()
        
        if not ret1 or not ret2:
            self.logger.error("Cannot read frames for video output setup")
            return
        
        height1, width1 = frame1.shape[:2]
        height2, width2 = frame2.shape[:2]
        
        output_width = width1 + width2
        output_height = max(height1, height2)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.config['visualization']['fps'], 
                            (output_width, output_height))
        
        # Reset captures
        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Process and write frames
        pbar = tqdm(total=len(self.camera1_results), desc="Generating video")
        
        for i, results in enumerate(self.camera1_results):
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()
            
            if not ret1 or not ret2:
                break
            
            # Draw visualizations
            vis_frame1 = self.visualizer.draw_detections_and_ids(
                frame1, results['camera1']['tracked_objects'], camera_id=1
            )
            vis_frame2 = self.visualizer.draw_detections_and_ids(
                frame2, results['camera2']['tracked_objects'], camera_id=2
            )
            
            # Create side-by-side frame
            combined_frame = self.visualizer.create_side_by_side(
                vis_frame1, vis_frame2, output_height
            )
            
            # Add frame information
            combined_frame = self.visualizer.add_frame_info(
                combined_frame, i, len(self.matched_pairs)
            )
            
            out.write(combined_frame)
            pbar.update(1)
        
        pbar.close()
        out.release()
        
        self.logger.info(f"Output video saved to: {output_path}")
    
    def cleanup(self):
        """Release resources"""
        if hasattr(self, 'cap1'):
            self.cap1.release()
        if hasattr(self, 'cap2'):
            self.cap2.release()
        cv2.destroyAllWindows()
    
    def run(self):
        """Main execution method"""
        try:
            self.logger.info("Starting Player Re-Identification System")
            
            self.load_videos()
            self.process_videos()
            self.perform_reid_matching()
            self.save_results()
            self.generate_output_video()
            
            self.logger.info("Player Re-Identification completed successfully!")
            self.print_summary()
            
        except Exception as e:
            self.logger.error(f"Error during processing: {str(e)}")
            raise
        finally:
            self.cleanup()
    
    def print_summary(self):
        """Print processing summary"""
        print("\n" + "="*60)
        print("PLAYER RE-IDENTIFICATION SUMMARY")
        print("="*60)
        print(f"Total frames processed: {len(self.camera1_results)}")
        print(f"Matched player pairs: {len(self.matched_pairs)}")
        print(f"Output video: output/videos/synchronized_output.mp4")
        print(f"Detection data: output/detections/detections.json")
        print(f"Feature data: output/features/features.pkl")
        print("="*60)


def main():
    """Main entry point"""
    print("Player Re-Identification System")
    print("="*50)
    
    # Check if video files exist
    if not os.path.exists(CONFIG['video']['camera1_path']):
        print(f"Error: Camera 1 video not found: {CONFIG['video']['camera1_path']}")
        print("Please ensure the video file exists at the specified path.")
        return
    
    if not os.path.exists(CONFIG['video']['camera2_path']):
        print(f"Error: Camera 2 video not found: {CONFIG['video']['camera2_path']}")
        print("Please ensure the video file exists at the specified path.")
        return
    
    # Create data directories if they don't exist
    ensure_dir("data/videos")
    ensure_dir("data/models")
    
    print("Configuration loaded successfully!")
    print(f"Camera 1: {CONFIG['video']['camera1_path']}")
    print(f"Camera 2: {CONFIG['video']['camera2_path']}")
    print("="*50)
    
    # Initialize and run the system
    try:
        reid_system = PlayerReIDSystem()
        reid_system.run()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check the error logs for more details.")


if __name__ == "__main__":
    main()