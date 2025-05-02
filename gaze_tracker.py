from gaze_tracking import GazeTracking
import cv2
import time
import numpy as np

class GazeTracker:
    def __init__(self):
        self.gaze = GazeTracking()
        self.webcam = cv2.VideoCapture(0)
        print("Initializing webcam for gaze tracking...")
        
        # Check if webcam opened successfully
        if not self.webcam.isOpened():
            print("Error: Could not open webcam.")
            raise Exception("Failed to open webcam")
        
        # Wait a bit for the camera to initialize
        time.sleep(1)
    
    def get_gaze_data(self):
        """Get current gaze data from webcam"""
        ret, frame = self.webcam.read()
        if not ret:
            print("Warning: Failed to get frame from webcam")
            return None, None
        
        self.gaze.refresh(frame)
        
        gaze_data = {
            "gaze_ratio_horizontal": self.gaze.horizontal_ratio(),
            "gaze_ratio_vertical": self.gaze.vertical_ratio(),
            "looking_left": self.gaze.is_looking_left(),
            "looking_right": self.gaze.is_looking_right(),
            "looking_center": self.gaze.is_looking_center(),
            "timestamp": time.time()
        }
        
        # Create visualization frame
        vis_frame = frame.copy()
        
        # Visualize gaze
        if gaze_data["gaze_ratio_horizontal"] is not None and gaze_data["gaze_ratio_vertical"] is not None:
            height, width = vis_frame.shape[:2]
            x = int(gaze_data["gaze_ratio_horizontal"] * width)
            y = int(gaze_data["gaze_ratio_vertical"] * height)
            cv2.circle(vis_frame, (x, y), 10, (0, 0, 255), -1)
            
            # Add text showing gaze coordinates
            cv2.putText(vis_frame, f"Gaze: ({x}, {y})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw crosshair lines
            cv2.line(vis_frame, (x, 0), (x, height), (0, 255, 0), 1)
            cv2.line(vis_frame, (0, y), (width, y), (0, 255, 0), 1)
        
        # Show pupils
        left_pupil = self.gaze.pupil_left_coords()
        right_pupil = self.gaze.pupil_right_coords()
        
        if left_pupil is not None:
            cv2.circle(vis_frame, left_pupil, 5, (0, 255, 0), -1)
        if right_pupil is not None:
            cv2.circle(vis_frame, right_pupil, 5, (0, 255, 0), -1)
        
        # Show whether looking left, right, or center
        direction = "Unknown"
        if gaze_data["looking_left"]:
            direction = "Left"
        elif gaze_data["looking_right"]:
            direction = "Right"
        elif gaze_data["looking_center"]:
            direction = "Center"
            
        cv2.putText(vis_frame, f"Direction: {direction}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        return gaze_data, vis_frame
    
    def close(self):
        """Release webcam resources"""
        self.webcam.release()