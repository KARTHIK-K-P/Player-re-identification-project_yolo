import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
import logging

class PlayerTracker:
    """
    Player tracking class that maintains consistent IDs across frames
    Uses centroid tracking with feature similarity for robust tracking
    """
    
    def __init__(self, max_disappeared=30, max_distance=100):
        """
        Initialize the tracker
        
        Args:
            max_disappeared (int): Maximum frames object can be missing before deletion
            max_distance (float): Maximum distance for object matching
        """
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        self.logger = logging.getLogger(__name__)
        
    def register(self, centroid, feature=None):
        """
        Register a new object with the tracker
        
        Args:
            centroid (list): [x, y] coordinates of object center
            feature (np.array): Feature vector of the object
            
        Returns:
            int: Assigned object ID
        """
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'feature': feature,
            'last_seen': 0
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        
        return self.next_object_id - 1
    
    def deregister(self, object_id):
        """
        Remove an object from tracking
        
        Args:
            object_id (int): ID of object to remove
        """
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
    
    def compute_feature_similarity(self, feature1, feature2):
        """
        Compute similarity between two feature vectors
        
        Args:
            feature1, feature2 (np.array): Feature vectors to compare
            
        Returns:
            float: Similarity score (0-1, higher is more similar)
        """
        if feature1 is None or feature2 is None:
            return 0.0
        
        # Normalize features
        feature1_norm = feature1 / (np.linalg.norm(feature1) + 1e-8)
        feature2_norm = feature2 / (np.linalg.norm(feature2) + 1e-8)
        
        # Compute cosine similarity
        similarity = np.dot(feature1_norm, feature2_norm)
        return max(0.0, similarity)  # Clamp to [0, 1]
    
    def update(self, centroids, features=None):
        """
        Update tracker with new detections
        
        Args:
            centroids (list): List of [x, y] coordinates
            features (list): List of feature vectors (optional)
            
        Returns:
            list: List of tracked objects with IDs and information
        """
        # If no features provided, create None list
        if features is None:
            features = [None] * len(centroids)
        
        # If no input centroids, mark all as disappeared
        if len(centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # Remove objects that have been gone too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return self._get_current_objects()
        
        # If no existing objects, register all as new
        if len(self.objects) == 0:
            for i, (centroid, feature) in enumerate(zip(centroids, features)):
                self.register(centroid, feature)
        else:
            # Match existing objects to new detections
            self._match_objects(centroids, features)
        
        return self._get_current_objects()
    
    def _match_objects(self, centroids, features):
        """
        Match existing objects to new detections
        
        Args:
            centroids (list): New centroid detections
            features (list): New feature vectors
        """
        # Get existing object centroids and features
        object_centroids = []
        object_ids = []
        object_features = []
        
        for object_id, obj_data in self.objects.items():
            object_centroids.append(obj_data['centroid'])
            object_ids.append(object_id)
            object_features.append(obj_data['feature'])
        
        # Compute distance matrix
        D = dist.cdist(np.array(object_centroids), np.array(centroids))
        
        # Compute feature similarity matrix if features available
        if any(f is not None for f in features) and any(f is not None for f in object_features):
            feature_similarity = np.zeros((len(object_features), len(features)))
            for i, obj_feat in enumerate(object_features):
                for j, new_feat in enumerate(features):
                    if obj_feat is not None and new_feat is not None:
                        feature_similarity[i, j] = self.compute_feature_similarity(obj_feat, new_feat)
            
            # Combine distance and feature similarity
            # Convert similarity to distance (1 - similarity)
            feature_distance = 1.0 - feature_similarity
            # Normalize distances to [0, 1]
            D_norm = D / (np.max(D) + 1e-8)
            # Combine with weights (favor feature similarity)
            combined_distance = 0.3 * D_norm + 0.7 * feature_distance
        else:
            combined_distance = D
        
        # Find the minimum values and sort by distance
        rows = combined_distance.min(axis=1).argsort()
        cols = combined_distance.argmin(axis=1)[rows]
        
        # Keep track of used row and column indices
        used_row_indices = set()
        used_col_indices = set()
        
        # Match objects to detections
        for (row, col) in zip(rows, cols):
            # Skip if already used
            if row in used_row_indices or col in used_col_indices:
                continue
            
            # Skip if distance is too large
            if combined_distance[row, col] > self.max_distance:
                continue
            
            # Update object with new centroid and feature
            object_id = object_ids[row]
            self.objects[object_id]['centroid'] = centroids[col]
            if features[col] is not None:
                # Update feature with exponential moving average
                if self.objects[object_id]['feature'] is not None:
                    alpha = 0.3  # Learning rate
                    self.objects[object_id]['feature'] = (
                        alpha * features[col] + 
                        (1 - alpha) * self.objects[object_id]['feature']
                    )
                else:
                    self.objects[object_id]['feature'] = features[col]
            
            # Reset disappeared counter
            self.disappeared[object_id] = 0
            
            # Mark as used
            used_row_indices.add(row)
            used_col_indices.add(col)
        
        # Handle unmatched detections and objects
        unused_rows = set(range(0, combined_distance.shape[0])).difference(used_row_indices)
        unused_cols = set(range(0, combined_distance.shape[1])).difference(used_col_indices)
        
        # If more objects than detections, mark objects as disappeared
        if combined_distance.shape[0] >= combined_distance.shape[1]:
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                # Remove if disappeared too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        
        # Register new objects for unmatched detections
        else:
            for col in unused_cols:
                self.register(centroids[col], features[col])
    
    def _get_current_objects(self):
        """
        Get current tracked objects
        
        Returns:
            list: List of tracked objects with ID and information
        """
        tracked_objects = []
        
        for object_id, obj_data in self.objects.items():
            tracked_objects.append({
                'id': object_id,
                'centroid': obj_data['centroid'],
                'feature': obj_data['feature'],
                'disappeared_frames': self.disappeared.get(object_id, 0)
            })
        
        return tracked_objects
    
    def get_object_count(self):
        """Get current number of tracked objects"""
        return len(self.objects)
    
    def get_object_by_id(self, object_id):
        """
        Get object information by ID
        
        Args:
            object_id (int): Object ID to retrieve
            
        Returns:
            dict: Object information or None if not found
        """
        if object_id in self.objects:
            return {
                'id': object_id,
                'centroid': self.objects[object_id]['centroid'],
                'feature': self.objects[object_id]['feature'],
                'disappeared_frames': self.disappeared.get(object_id, 0)
            }
        return None
    
    def reset(self):
        """Reset tracker state"""
        self.objects.clear()
        self.disappeared.clear()
        self.next_object_id = 0
        
    def __str__(self):
        """String representation of tracker state"""
        return f"PlayerTracker: {len(self.objects)} active objects, next_id={self.next_object_id}"