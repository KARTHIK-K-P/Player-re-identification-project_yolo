import cv2
import numpy as np
import colorsys
import logging

class Visualizer:
    """
    Visualization class for player re-identification system
    Handles drawing bounding boxes, IDs, and creating output videos
    """
    
    def __init__(self, fps=30, font_scale=0.8, thickness=2):
        """
        Initialize the visualizer
        
        Args:
            fps (int): Output video FPS
            font_scale (float): Font scale for text
            thickness (int): Line thickness for drawing
        """
        self.fps = fps
        self.font_scale = font_scale
        self.thickness = thickness
        
        self.logger = logging.getLogger(__name__)
        
        # Generate color palette for different IDs
        self.colors = self._generate_color_palette(100)
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def _generate_color_palette(self, num_colors):
        """
        Generate distinct colors for visualization
        
        Args:
            num_colors (int): Number of colors to generate
            
        Returns:
            list: List of (B, G, R) color tuples
        """
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        
        return colors
    
    def get_color_for_id(self, object_id):
        """
        Get consistent color for an object ID
        
        Args:
            object_id (int): Object ID
            
        Returns:
            tuple: (B, G, R) color tuple
        """
        if object_id < len(self.colors):
            return self.colors[object_id]
        else:
            # Generate color based on ID hash
            np.random.seed(object_id)
            return tuple(np.random.randint(0, 255, 3).tolist())
    
    def draw_detections_and_ids(self, frame, tracked_objects, camera_id=1):
        """
        Draw bounding boxes and IDs on frame
        
        Args:
            frame (np.array): Input frame
            tracked_objects (list): List of tracked objects
            camera_id (int): Camera ID for labeling
            
        Returns:
            np.array: Frame with visualizations
        """
        vis_frame = frame.copy()
        
        for obj in tracked_objects:
            obj_id = obj['id']
            centroid = obj['centroid']
            
            # Get color for this ID
            color = self.get_color_for_id(obj_id)
            
            # Draw centroid
            center_x, center_y = int(centroid[0]), int(centroid[1])
            cv2.circle(vis_frame, (center_x, center_y), 8, color, -1)
            cv2.circle(vis_frame, (center_x, center_y), 10, (255, 255, 255), 2)
            
            # Draw ID label
            label = f"ID:{obj_id}"
            label_size = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)[0]
            
            # Position label above centroid
            label_x = center_x - label_size[0] // 2
            label_y = center_y - 20
            
            # Draw label background
            cv2.rectangle(vis_frame, 
                         (label_x - 5, label_y - label_size[1] - 5),
                         (label_x + label_size[0] + 5, label_y + 5),
                         color, -1)
            
            # Draw label text
            cv2.putText(vis_frame, label, (label_x, label_y),
                       self.font, self.font_scale, (255, 255, 255), self.thickness)
            
            # Draw camera ID
            cam_label = f"Cam{camera_id}"
            cv2.putText(vis_frame, cam_label, (center_x - 20, center_y + 30),
                       self.font, 0.5, color, 1)
        
        return vis_frame
    
    def draw_bounding_boxes(self, frame, detections, tracked_objects=None):
        """
        Draw bounding boxes for detections
        
        Args:
            frame (np.array): Input frame
            detections (list): List of detections [x1, y1, x2, y2, conf, cls]
            tracked_objects (list): Optional tracked objects for ID mapping
            
        Returns:
            np.array: Frame with bounding boxes
        """
        vis_frame = frame.copy()
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2, conf, cls = detection
            
            # Get color
            if tracked_objects and i < len(tracked_objects):
                obj_id = tracked_objects[i]['id']
                color = self.get_color_for_id(obj_id)
                label = f"ID:{obj_id} ({conf:.2f})"
            else:
                color = (0, 255, 0)  # Default green
                label = f"Person ({conf:.2f})"
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, self.thickness)
            
            # Draw label
            label_size = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)[0]
            cv2.rectangle(vis_frame, 
                         (int(x1), int(y1) - label_size[1] - 10),
                         (int(x1) + label_size[0], int(y1)),
                         color, -1)
            cv2.putText(vis_frame, label, (int(x1), int(y1) - 5),
                       self.font, self.font_scale, (255, 255, 255), self.thickness)
        
        return vis_frame
    
    def create_side_by_side(self, frame1, frame2, output_height=None):
        """
        Create side-by-side visualization of two frames
        
        Args:
            frame1, frame2 (np.array): Input frames
            output_height (int): Target output height
            
        Returns:
            np.array: Combined side-by-side frame
        """
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        # Determine output dimensions
        if output_height is None:
            output_height = max(h1, h2)
        
        # Resize frames to match height
        if h1 != output_height:
            aspect_ratio = w1 / h1
            new_width = int(output_height * aspect_ratio)
            frame1 = cv2.resize(frame1, (new_width, output_height))
        
        if h2 != output_height:
            aspect_ratio = w2 / h2
            new_width = int(output_height * aspect_ratio)
            frame2 = cv2.resize(frame2, (new_width, output_height))
        
        # Combine frames horizontally
        combined = np.hstack([frame1, frame2])
        
        # Add separator line
        separator_x = frame1.shape[1]
        cv2.line(combined, (separator_x, 0), (separator_x, output_height), (255, 255, 255), 2)
        
        return combined
    
    def add_frame_info(self, frame, frame_idx, total_matches):
        """
        Add frame information overlay
        
        Args:
            frame (np.array): Input frame
            frame_idx (int): Current frame index
            total_matches (int): Total number of matches
            
        Returns:
            np.array: Frame with information overlay
        """
        vis_frame = frame.copy()
        
        # Add frame number
        frame_text = f"Frame: {frame_idx}"
        cv2.putText(vis_frame, frame_text, (10, 30),
                   self.font, self.font_scale, (255, 255, 255), self.thickness)
        
        # Add match count
        match_text = f"Matches: {total_matches}"
        cv2.putText(vis_frame, match_text, (10, 60),
                   self.font, self.font_scale, (255, 255, 255), self.thickness)
        
        # Add camera labels
        h, w = vis_frame.shape[:2]
        mid_x = w // 2
        
        cv2.putText(vis_frame, "Camera 1", (10, h - 20),
                   self.font, self.font_scale, (0, 255, 255), self.thickness)
        cv2.putText(vis_frame, "Camera 2", (mid_x + 10, h - 20),
                   self.font, self.font_scale, (0, 255, 255), self.thickness)
        
        return vis_frame
    
    def draw_matches(self, frame1, frame2, matches, tracked_objects1, tracked_objects2):
        """
        Draw matching lines between corresponding objects
        
        Args:
            frame1, frame2 (np.array): Input frames
            matches (list): List of matches [(id1, id2, similarity)]
            tracked_objects1, tracked_objects2 (list): Tracked objects from both cameras
            
        Returns:
            np.array: Combined frame with match lines
        """
        # Create side-by-side frame
        combined = self.create_side_by_side(frame1, frame2)
        h, w = combined.shape[:2]
        
        # Get offset for second camera
        offset_x = frame1.shape[1]
        
        # Create ID to centroid mapping
        id_to_centroid1 = {obj['id']: obj['centroid'] for obj in tracked_objects1}
        id_to_centroid2 = {obj['id']: obj['centroid'] for obj in tracked_objects2}
        
        # Draw match lines
        for match in matches:
            id1, id2, similarity = match
            
            if id1 in id_to_centroid1 and id2 in id_to_centroid2:
                # Get centroids
                cent1 = id_to_centroid1[id1]
                cent2 = id_to_centroid2[id2]
                
                # Calculate points
                pt1 = (int(cent1[0]), int(cent1[1]))
                pt2 = (int(cent2[0]) + offset_x, int(cent2[1]))
                
                # Get color based on similarity
                color_intensity = int(similarity * 255)
                color = (0, color_intensity, 255 - color_intensity)
                
                # Draw line
                cv2.line(combined, pt1, pt2, color, 2)
                
                # Draw similarity score at midpoint
                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2
                sim_text = f"{similarity:.2f}"
                cv2.putText(combined, sim_text, (mid_x, mid_y),
                           self.font, 0.4, (255, 255, 255), 1)
        
        return combined
    
    def create_detection_summary(self, frame, detections, tracked_objects):
        """
        Create summary overlay with detection statistics
        
        Args:
            frame (np.array): Input frame
            detections (list): List of detections
            tracked_objects (list): List of tracked objects
            
        Returns:
            np.array: Frame with summary overlay
        """
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]
        
        # Create semi-transparent overlay area
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (w - 250, 10), (w - 10, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0, vis_frame)
        
        # Add statistics text
        stats = [
            f"Detections: {len(detections)}",
            f"Tracked Objects: {len(tracked_objects)}",
            f"Frame Size: {w}x{h}",
            f"FPS: {self.fps}"
        ]
        
        y_offset = 35
        for stat in stats:
            cv2.putText(vis_frame, stat, (w - 240, y_offset),
                       self.font, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        return vis_frame
    
    def create_output_video(self, output_path, frame_size):
        """
        Create video writer for output
        
        Args:
            output_path (str): Output video path
            frame_size (tuple): (width, height) of output frames
            
        Returns:
            cv2.VideoWriter: Video writer object
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, self.fps, frame_size)
    
    def draw_trajectory(self, frame, trajectory_points, obj_id, max_points=30):
        """
        Draw trajectory path for an object
        
        Args:
            frame (np.array): Input frame
            trajectory_points (list): List of (x, y) points
            obj_id (int): Object ID
            max_points (int): Maximum number of points to display
            
        Returns:
            np.array: Frame with trajectory
        """
        vis_frame = frame.copy()
        
        if len(trajectory_points) < 2:
            return vis_frame
        
        # Limit trajectory length
        points = trajectory_points[-max_points:]
        color = self.get_color_for_id(obj_id)
        
        # Draw trajectory lines
        for i in range(1, len(points)):
            pt1 = (int(points[i-1][0]), int(points[i-1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            
            # Fade older points
            alpha = i / len(points)
            line_color = tuple(int(c * alpha) for c in color)
            
            cv2.line(vis_frame, pt1, pt2, line_color, 2)
        
        # Draw current position
        current_pos = (int(points[-1][0]), int(points[-1][1]))
        cv2.circle(vis_frame, current_pos, 5, color, -1)
        
        return vis_frame
    
    def add_timestamp(self, frame, timestamp):
        """
        Add timestamp overlay to frame
        
        Args:
            frame (np.array): Input frame
            timestamp (str): Timestamp string
            
        Returns:
            np.array: Frame with timestamp
        """
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]
        
        # Add timestamp in top-right corner
        cv2.putText(vis_frame, timestamp, (w - 200, 30),
                   self.font, 0.6, (255, 255, 255), 2)
        
        return vis_frame
    
    def create_multi_view(self, frames, layout=(2, 2)):
        """
        Create multi-view display from multiple frames
        
        Args:
            frames (list): List of frames to combine
            layout (tuple): (rows, cols) layout
            
        Returns:
            np.array: Combined multi-view frame
        """
        rows, cols = layout
        if len(frames) > rows * cols:
            self.logger.warning(f"Too many frames ({len(frames)}) for layout {layout}")
            frames = frames[:rows * cols]
        
        # Determine output size
        if not frames:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        h, w = frames[0].shape[:2]
        output_h = h * rows
        output_w = w * cols
        
        # Create output canvas
        output = np.zeros((output_h, output_w, 3), dtype=np.uint8)
        
        # Place frames
        for i, frame in enumerate(frames):
            row = i // cols
            col = i % cols
            
            start_y = row * h
            end_y = start_y + h
            start_x = col * w
            end_x = start_x + w
            
            # Resize frame if necessary
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            
            output[start_y:end_y, start_x:end_x] = frame
        
        return output
    
    def draw_confidence_bars(self, frame, detections):
        """
        Draw confidence bars for detections
        
        Args:
            frame (np.array): Input frame
            detections (list): List of detections with confidence scores
            
        Returns:
            np.array: Frame with confidence bars
        """
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]
        
        # Draw confidence bars on the right side
        bar_width = 20
        bar_height = 100
        start_x = w - 50
        start_y = 50
        
        for i, detection in enumerate(detections):
            if len(detection) >= 5:  # Has confidence score
                conf = detection[4]
                
                # Calculate bar dimensions
                filled_height = int(bar_height * conf)
                bar_y = start_y + i * (bar_height + 20)
                
                # Draw empty bar
                cv2.rectangle(vis_frame, 
                             (start_x, bar_y), 
                             (start_x + bar_width, bar_y + bar_height),
                             (100, 100, 100), 2)
                
                # Draw filled portion
                cv2.rectangle(vis_frame,
                             (start_x, bar_y + bar_height - filled_height),
                             (start_x + bar_width, bar_y + bar_height),
                             (0, 255, 0), -1)
                
                # Add confidence text
                conf_text = f"{conf:.2f}"
                cv2.putText(vis_frame, conf_text, 
                           (start_x - 40, bar_y + bar_height // 2),
                           self.font, 0.4, (255, 255, 255), 1)
        
        return vis_frame
    
    def cleanup(self):
        """
        Cleanup resources
        """
        self.logger.info("Visualizer cleanup completed")

# Example usage and testing
if __name__ == "__main__":
    # Initialize visualizer
    visualizer = Visualizer(fps=30)
    
    # Create dummy data for testing
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_detections = [[100, 100, 200, 200, 0.85, 0]]
    dummy_tracked_objects = [{'id': 1, 'centroid': (150, 150)}]
    
    # Test visualization functions
    vis_frame = visualizer.draw_detections_and_ids(dummy_frame, dummy_tracked_objects)
    print("Visualizer initialized and tested successfully!")