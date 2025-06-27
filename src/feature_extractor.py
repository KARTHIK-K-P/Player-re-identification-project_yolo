import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import cv2
import numpy as np
import logging

class FeatureExtractor:
    """
    Feature extraction class for player re-identification
    Uses ResNet50 backbone to extract appearance features
    """
    
    def __init__(self, feature_dim=512, device=None):
        """
        Initialize the feature extractor
        
        Args:
            feature_dim (int): Dimension of output features
            device (str): Device to run inference on ('cpu' or 'cuda')
        """
        self.feature_dim = feature_dim
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self._load_model()
        
        # Setup image preprocessing
        self._setup_transforms()
        
    def _load_model(self):
        """Load and setup the feature extraction model"""
        try:
            # Load pre-trained ResNet50
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            
            # Remove final classification layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            
            # Add custom feature projection if needed
            if self.feature_dim != 2048:  # ResNet50 outputs 2048 features
                self.projection = nn.Sequential(
                    nn.Linear(2048, self.feature_dim),
                    nn.ReLU(),
                    nn.Dropout(0.5)
                )
            else:
                self.projection = None
            
            # Move to device
            self.backbone = self.backbone.to(self.device)
            if self.projection:
                self.projection = self.projection.to(self.device)
            
            # Set to evaluation mode
            self.backbone.eval()
            if self.projection:
                self.projection.eval()
                
            self.logger.info(f"Feature extractor loaded on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load feature extraction model: {str(e)}")
            raise
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_image(self, image):
        """
        Preprocess image for feature extraction
        
        Args:
            image (np.array): Input image (BGR format)
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if image is None or image.size == 0:
            # Return zero tensor for invalid images
            return torch.zeros((1, 3, 224, 224)).to(self.device)
        
        # Convert BGR to RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Apply transforms
        try:
            tensor = self.transform(image_rgb)
            tensor = tensor.unsqueeze(0)  # Add batch dimension
            return tensor.to(self.device)
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed: {str(e)}")
            return torch.zeros((1, 3, 224, 224)).to(self.device)
    
    def extract_features(self, image):
        """
        Extract features from a single image
        
        Args:
            image (np.array): Input image (BGR format)
            
        Returns:
            np.array: Feature vector
        """
        if image is None or image.size == 0:
            return np.zeros(self.feature_dim)
        
        try:
            # Preprocess image
            tensor = self.preprocess_image(image)
            
            # Extract features
            with torch.no_grad():
                # Get backbone features
                features = self.backbone(tensor)
                features = features.view(features.size(0), -1)  # Flatten
                
                # Apply projection if available
                if self.projection:
                    features = self.projection(features)
                
                # Convert to numpy
                features = features.cpu().numpy().flatten()
                
                # L2 normalize
                features = self._l2_normalize(features)
                
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            return np.zeros(self.feature_dim)
    
    def extract_features_batch(self, images):
        """
        Extract features from multiple images
        
        Args:
            images (list): List of images (BGR format)
            
        Returns:
            np.array: Array of feature vectors
        """
        if not images:
            return np.array([])
        
        features_list = []
        
        try:
            # Process images in batches for efficiency
            batch_size = 8
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                
                # Preprocess batch
                batch_tensors = []
                for img in batch:
                    tensor = self.preprocess_image(img)
                    batch_tensors.append(tensor)
                
                if batch_tensors:
                    batch_tensor = torch.cat(batch_tensors, dim=0)
                    
                    # Extract features
                    with torch.no_grad():
                        batch_features = self.backbone(batch_tensor)
                        batch_features = batch_features.view(batch_features.size(0), -1)
                        
                        if self.projection:
                            batch_features = self.projection(batch_features)
                        
                        # Convert to numpy and normalize
                        batch_features = batch_features.cpu().numpy()
                        for j in range(batch_features.shape[0]):
                            features = self._l2_normalize(batch_features[j])
                            features_list.append(features)
        
        except Exception as e:
            self.logger.error(f"Batch feature extraction failed: {str(e)}")
            # Return zero features for failed batch
            for _ in range(len(images)):
                features_list.append(np.zeros(self.feature_dim))
        
        return np.array(features_list)
    
    def _l2_normalize(self, features):
        """
        L2 normalize feature vector
        
        Args:
            features (np.array): Input features
            
        Returns:
            np.array: Normalized features
        """
        norm = np.linalg.norm(features)
        if norm > 1e-8:
            return features / norm
        else:
            return features
    
    def compute_similarity(self, features1, features2):
        """
        Compute cosine similarity between two feature vectors
        
        Args:
            features1, features2 (np.array): Feature vectors
            
        Returns:
            float: Similarity score (0-1)
        """
        if features1 is None or features2 is None:
            return 0.0
        
        # Ensure features are normalized
        feat1_norm = self._l2_normalize(features1)
        feat2_norm = self._l2_normalize(features2)
        
        # Compute cosine similarity
        similarity = np.dot(feat1_norm, feat2_norm)
        return max(0.0, similarity)  # Clamp to [0, 1]
    
    def compute_distance(self, features1, features2):
        """
        Compute Euclidean distance between two feature vectors
        
        Args:
            features1, features2 (np.array): Feature vectors
            
        Returns:
            float: Distance value
        """
        if features1 is None or features2 is None:
            return float('inf')
        
        return np.linalg.norm(features1 - features2)
    
    def create_feature_gallery(self, images, labels=None):
        """
        Create a feature gallery from multiple images
        
        Args:
            images (list): List of images
            labels (list): Optional labels for images
            
        Returns:
            dict: Gallery with features and metadata
        """
        features = self.extract_features_batch(images)
        
        gallery = {
            'features': features,
            'count': len(images)
        }
        
        if labels:
            gallery['labels'] = labels
        
        return gallery
    
    def match_against_gallery(self, query_image, gallery, top_k=5):
        """
        Match query image against feature gallery
        
        Args:
            query_image (np.array): Query image
            gallery (dict): Feature gallery
            top_k (int): Number of top matches to return
            
        Returns:
            list: List of (similarity, index) tuples
        """
        query_features = self.extract_features(query_image)
        gallery_features = gallery['features']
        
        if len(gallery_features) == 0:
            return []
        
        # Compute similarities
        similarities = []
        for i, gallery_feat in enumerate(gallery_features):
            sim = self.compute_similarity(query_features, gallery_feat)
            similarities.append((sim, i))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        return similarities[:top_k]
    
    def save_features(self, features, filepath):
        """
        Save features to file
        
        Args:
            features (np.array): Features to save
            filepath (str): Output file path
        """
        try:
            np.save(filepath, features)
            self.logger.info(f"Features saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save features: {str(e)}")
    
    def load_features(self, filepath):
        """
        Load features from file
        
        Args:
            filepath (str): Input file path
            
        Returns:
            np.array: Loaded features
        """
        try:
            features = np.load(filepath)
            self.logger.info(f"Features loaded from {filepath}")
            return features
        except Exception as e:
            self.logger.error(f"Failed to load features: {str(e)}")
            return np.array([])
    
    def __str__(self):
        """String representation of feature extractor"""
        return f"FeatureExtractor: dim={self.feature_dim}, device={self.device}"