# New file: calibration_agent.py
import numpy as np
import time

class CalibrationAgent:
    def __init__(self, screen_width=1.0, screen_height=1.0, calibration_points=9):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.calibration_points = calibration_points
        
        # Grid of calibration points (normalized coordinates)
        self.grid_points = []
        for y in np.linspace(0.1, 0.9, 3):
            for x in np.linspace(0.1, 0.9, 3):
                self.grid_points.append((x, y))
        
        # Transformation matrices
        self.transformation_matrix = np.eye(3)  # Identity initially
        self.samples_per_point = 10
        self.current_point_index = 0
        self.current_samples = []
        self.all_samples = []
        self.is_calibrated = False
        
    def get_current_calibration_point(self):
        """Get the current point to be used for calibration"""
        if self.current_point_index < len(self.grid_points):
            return self.grid_points[self.current_point_index]
        return None
        
    def add_sample(self, gaze_data):
        """Add a gaze sample for the current calibration point"""
        if not gaze_data.get("pupils_located", False):
            return False
            
        h_ratio = gaze_data.get("gaze_ratio_horizontal")
        v_ratio = gaze_data.get("gaze_ratio_vertical")
        
        if h_ratio is None or v_ratio is None:
            return False
            
        self.current_samples.append((h_ratio, v_ratio))
        
        if len(self.current_samples) >= self.samples_per_point:
            # Calculate average for this point
            avg_h = sum(s[0] for s in self.current_samples) / len(self.current_samples)
            avg_v = sum(s[1] for s in self.current_samples) / len(self.current_samples)
            
            # Store with the true point
            true_point = self.grid_points[self.current_point_index]
            self.all_samples.append((avg_h, avg_v, true_point[0], true_point[1]))
            
            # Move to next point
            self.current_point_index += 1
            self.current_samples = []
            
            # Check if calibration is complete
            if self.current_point_index >= len(self.grid_points):
                self._calculate_transformation()
                self.is_calibrated = True
                return True
                
        return False
        
    def _calculate_transformation(self):
        """Calculate the transformation matrix based on collected samples"""
        if len(self.all_samples) < 4:  # Need at least 4 points for a good fit
            return
            
        # Extract coordinates
        observed_points = np.array([(s[0], s[1], 1.0) for s in self.all_samples])
        true_points = np.array([(s[2], s[3], 1.0) for s in self.all_samples])
        
        # Solve for the transformation matrix using least squares
        # This is a simplified version - a proper implementation would use SVD
        self.transformation_matrix, _, _, _ = np.linalg.lstsq(observed_points, true_points, rcond=None)
        
    def transform_gaze_point(self, h_ratio, v_ratio):
        """Transform raw gaze point to calibrated coordinates"""
        if not self.is_calibrated:
            return h_ratio, v_ratio
            
        # Apply transformation
        raw_point = np.array([h_ratio, v_ratio, 1.0])
        transformed = np.dot(self.transformation_matrix, raw_point)
        
        # Normalize and clamp to [0, 1]
        result_h = max(0.0, min(1.0, transformed[0]))
        result_v = max(0.0, min(1.0, transformed[1]))
        
        return result_h, result_v