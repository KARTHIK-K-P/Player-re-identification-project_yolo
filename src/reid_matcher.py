import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import logging
from collections import defaultdict

class ReIDMatcher:
    """
    Re-identification matcher for matching players across different camera views
    Uses appearance features and spatial-temporal constraints
    """
    
    def __init__(self, similarity_threshold=0.7, max_distance=2.0, 
                 temporal_weight=0.3, spatial_weight=0.2):
        """
        Initialize the ReID matcher
        
        Args:
            similarity_threshold (float): Minimum similarity for matching
            max_distance (float): Maximum feature distance for matching
            temporal_weight (float): Weight for temporal consistency
            spatial_weight (float): Weight for spatial consistency
        """
        self.similarity_threshold = similarity_threshold
        self.max_distance = max_distance
        self.temporal_weight = temporal_weight
        self.spatial_weight = spatial_weight
        
        self.logger = logging.getLogger(__name__)
        
        # Storage for cross-camera matches
        self.global_id_counter = 0
        self.camera_to_global_mapping = {}  # {camera_id: {local_id: global_id}}
        self.global_to_camera_mapping = {}  # {global_id: [(camera_id, local_id)]}
        
    def compute_feature_similarity(self, features1, features2):
        """
        Compute cosine similarity between feature vectors
        
        Args:
            features1, features2 (np.array): Feature vectors
            
        Returns:
            float: Similarity score (0-1)
        """
        if features1 is None or features2 is None:
            return 0.0
        
        # Normalize features
        feat1_norm = features1 / (np.linalg.norm(features1) + 1e-8)
        feat2_norm = features2 / (np.linalg.norm(features2) + 1e-8)
        
        # Compute cosine similarity
        similarity = np.dot(feat1_norm, feat2_norm)
        return max(0.0, similarity)
    
    def compute_feature_distance(self, features1, features2):
        """
        Compute Euclidean distance between feature vectors
        
        Args:
            features1, features2 (np.array): Feature vectors
            
        Returns:
            float: Distance value
        """
        if features1 is None or features2 is None:
            return float('inf')
        
        return np.linalg.norm(features1 - features2)
    
    def match_objects(self, cam1_objects, cam2_objects):
        """
        Match objects between two camera views
        
        Args:
            cam1_objects (dict): Objects from camera 1 {id: {features, detections, frames}}
            cam2_objects (dict): Objects from camera 2 {id: {features, detections, frames}}
            
        Returns:
            list: List of matched pairs [(cam1_id, cam2_id, similarity)]
        """
        if not cam1_objects or not cam2_objects:
            return []
        
        self.logger.info(f"Matching {len(cam1_objects)} objects from cam1 with {len(cam2_objects)} objects from cam2")
        
        # Extract object IDs and features
        cam1_ids = list(cam1_objects.keys())
        cam2_ids = list(cam2_objects.keys())
        
        # Build feature matrices
        cam1_features = []
        cam2_features = []
        
        for obj_id in cam1_ids:
            if 'avg_feature' in cam1_objects[obj_id]:
                cam1_features.append(cam1_objects[obj_id]['avg_feature'])
            else:
                cam1_features.append(np.zeros(512))  # Default feature size
        
        for obj_id in cam2_ids:
            if 'avg_feature' in cam2_objects[obj_id]:
                cam2_features.append(cam2_objects[obj_id]['avg_feature'])
            else:
                cam2_features.append(np.zeros(512))  # Default feature size
        
        cam1_features = np.array(cam1_features)
        cam2_features = np.array(cam2_features)
        
        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(
            cam1_features, cam2_features, cam1_objects, cam2_objects, cam1_ids, cam2_ids
        )
        
        # Find optimal matching using Hungarian algorithm
        # Convert similarity to cost (negative similarity)
        cost_matrix = 1.0 - similarity_matrix
        
        # Apply threshold - set high cost for low similarities
        cost_matrix[similarity_matrix < self.similarity_threshold] = 999.0
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Extract valid matches
        matched_pairs = []
        for i, j in zip(row_ind, col_ind):
            if similarity_matrix[i, j] >= self.similarity_threshold:
                cam1_id = cam1_ids[i]
                cam2_id = cam2_ids[j]
                similarity = similarity_matrix[i, j]
                matched_pairs.append((cam1_id, cam2_id, similarity))
        
        self.logger.info(f"Found {len(matched_pairs)} valid matches")
        return matched_pairs
    
    def _compute_similarity_matrix(self, cam1_features, cam2_features, 
                                  cam1_objects, cam2_objects, cam1_ids, cam2_ids):
        """
        Compute similarity matrix between objects from two cameras
        
        Args:
            cam1_features, cam2_features (np.array): Feature matrices
            cam1_objects, cam2_objects (dict): Object dictionaries
            cam1_ids, cam2_ids (list): Object ID lists
            
        Returns:
            np.array: Similarity matrix
        """
        n_cam1 = len(cam1_ids)
        n_cam2 = len(cam2_ids)
        
        similarity_matrix = np.zeros((n_cam1, n_cam2))
        
        for i, cam1_id in enumerate(cam1_ids):
            for j, cam2_id in enumerate(cam2_ids):
                # Compute feature similarity
                feature_sim = self.compute_feature_similarity(
                    cam1_features[i], cam2_features[j]
                )
                
                # Compute temporal consistency
                temporal_sim = self._compute_temporal_consistency(
                    cam1_objects[cam1_id], cam2_objects[cam2_id]
                )
                
                # Compute spatial consistency (if available)
                spatial_sim = self._compute_spatial_consistency(
                    cam1_objects[cam1_id], cam2_objects[cam2_id]
                )
                
                # Combine similarities
                combined_sim = (
                    (1.0 - self.temporal_weight - self.spatial_weight) * feature_sim +
                    self.temporal_weight * temporal_sim +
                    self.spatial_weight * spatial_sim
                )
                
                similarity_matrix[i, j] = combined_sim
        
        return similarity_matrix
    
    def _compute_temporal_consistency(self, obj1, obj2):
        """
        Compute temporal consistency between two objects
        
        Args:
            obj1, obj2 (dict): Object dictionaries with frame information
            
        Returns:
            float: Temporal consistency score (0-1)
        """
        if 'frames' not in obj1 or 'frames' not in obj2:
            return 0.5  # Neutral score
        
        frames1 = set(obj1['frames'])
        frames2 = set(obj2['frames'])
        
        # Compute overlap
        overlap = len(frames1.intersection(frames2))
        total = len(frames1.union(frames2))
        
        if total == 0:
            return 0.5
        
        return overlap / total
    
    def _compute_spatial_consistency(self, obj1, obj2):
        """
        Compute spatial consistency between two objects
        
        Args:
            obj1, obj2 (dict): Object dictionaries with detection information
            
        Returns:
            float: Spatial consistency score (0-1)
        """
        # Simple implementation - can be enhanced with camera calibration
        if 'detections' not in obj1 or 'detections' not in obj2:
            return 0.5  # Neutral score
        
        # For now, return neutral score
        # This can be enhanced with camera geometry and epipolar constraints
        return 0.5
    
    def create_global_identity_mapping(self, matched_pairs):
        """
        Create global identity mapping from matched pairs
        
        Args:
            matched_pairs (list): List of matched pairs [(cam1_id, cam2_id, similarity)]
            
        Returns:
            dict: Global identity mapping
        """
        global_mapping = {}
        
        for cam1_id, cam2_id, similarity in matched_pairs:
            global_id = self.global_id_counter
            self.global_id_counter += 1
            
            # Update mappings
            if 1 not in self.camera_to_global_mapping:
                self.camera_to_global_mapping[1] = {}
            if 2 not in self.camera_to_global_mapping:
                self.camera_to_global_mapping[2] = {}
            
            self.camera_to_global_mapping[1][cam1_id] = global_id
            self.camera_to_global_mapping[2][cam2_id] = global_id
            
            self.global_to_camera_mapping[global_id] = [(1, cam1_id), (2, cam2_id)]
            
            global_mapping[global_id] = {
                'camera1_id': cam1_id,
                'camera2_id': cam2_id,
                'similarity': similarity
            }
        
        return global_mapping
    
    def get_global_id(self, camera_id, local_id):
        """
        Get global ID for a local camera ID
        
        Args:
            camera_id (int): Camera ID
            local_id (int): Local object ID
            
        Returns:
            int: Global ID or None if not found
        """
        if camera_id in self.camera_to_global_mapping:
            return self.camera_to_global_mapping[camera_id].get(local_id)
        return None
    
    def get_local_ids(self, global_id):
        """
        Get local IDs for a global ID
        
        Args:
            global_id (int): Global ID
            
        Returns:
            list: List of (camera_id, local_id) tuples
        """
        return self.global_to_camera_mapping.get(global_id, [])
    
    def match_frame_detections(self, detections1, features1, detections2, features2):
        """
        Match detections between two frames
        
        Args:
            detections1, detections2 (list): Detection lists
            features1, features2 (list): Feature lists
            
        Returns:
            list: List of matched pairs [(idx1, idx2, similarity)]
        """
        if not detections1 or not detections2:
            return []
        
        # Compute similarity matrix
        n1, n2 = len(detections1), len(detections2)
        similarity_matrix = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                if i < len(features1) and j < len(features2):
                    similarity = self.compute_feature_similarity(features1[i], features2[j])
                    similarity_matrix[i, j] = similarity
        
        # Apply threshold
        cost_matrix = 1.0 - similarity_matrix
        cost_matrix[similarity_matrix < self.similarity_threshold] = 999.0
        
        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Extract matches
        matches = []
        for i, j in zip(row_ind, col_ind):
            if similarity_matrix[i, j] >= self.similarity_threshold:
                matches.append((i, j, similarity_matrix[i, j]))
        
        return matches
    
    def compute_match_statistics(self, matched_pairs):
        """
        Compute statistics for matching results
        
        Args:
            matched_pairs (list): List of matched pairs
            
        Returns:
            dict: Statistics dictionary
        """
        if not matched_pairs:
            return {
                'total_matches': 0,
                'avg_similarity': 0.0,
                'min_similarity': 0.0,
                'max_similarity': 0.0
            }
        
        similarities = [pair[2] for pair in matched_pairs]
        
        stats = {
            'total_matches': len(matched_pairs),
            'avg_similarity': np.mean(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'std_similarity': np.std(similarities)
        }
        
        return stats
    
    def reset(self):
        """Reset matcher state"""
        self.global_id_counter = 0
        self.camera_to_global_mapping.clear()
        self.global_to_camera_mapping.clear()
    
    def __str__(self):
        """String representation of matcher"""
        return f"ReIDMatcher: threshold={self.similarity_threshold}, global_ids={self.global_id_counter}"