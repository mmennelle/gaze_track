# Improved gaze_tracker.py with filtering and error handling
import time
import cv2
import numpy as np
from collections import deque
import logging
from gaze_tracking import GazeTracking

# Configure logging
logger = logging.getLogger("GazeTracker")

class GazeFilter:
    """
    Smoothing filter for gaze data to reduce jitter and outliers
    """
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.h_buffer = deque(maxlen=window_size)
        self.v_buffer = deque(maxlen=window_size)
        
    def update(self, h_ratio, v_ratio):
        """Update the filter with new gaze data"""
        if h_ratio is not None:
            self.h_buffer.append(h_ratio)
        if v_ratio is not None:
            self.v_buffer.append(v_ratio)
            
    def get_filtered_ratios(self):
        """Return filtered gaze ratios with outlier removal"""
        if len(self.h_buffer) == 0 or len(self.v_buffer) == 0:
            return None, None
            
        # Convert to numpy arrays for faster processing
        h_values = np.array(list(self.h_buffer))
        v_values = np.array(list(self.v_buffer))
        
        # Simple outlier removal (values beyond 2 standard deviations)
        h_mean = np.mean(h_values)
        h_std = np.std(h_values)
        h_filtered = h_values[np.abs(h_values - h_mean) < 2 * h_std]
        
        v_mean = np.mean(v_values)
        v_std = np.std(v_values)
        v_filtered = v_values[np.abs(v_values - v_mean) < 2 * v_std]
        
        # If all values were outliers, return the mean
        if len(h_filtered) == 0 or len(v_filtered) == 0:
            return h_mean, v_mean
            
        return np.mean(h_filtered), np.mean(v_filtered)

# Initialize gaze tracking
logger.info("Initializing webcam and gaze tracking...")
gaze = None
webcam = None
gaze_filter = None

try:
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)
    
    if not webcam.isOpened():
        raise Exception("Webcam could not be opened")
    
    # Set webcam properties for better performance
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    webcam.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize the gaze filter
    gaze_filter = GazeFilter(window_size=5)
    
    logger.info("Webcam and gaze tracking successfully initialized")
except Exception as e:
    logger.error(f"Failed to initialize gaze tracking: {e}")
    if webcam and webcam.isOpened():
        webcam.release()

def get_gaze_data():
    """
    Capture and process webcam frame to extract gaze data
    Returns tuple of (gaze_data_dict, annotated_frame) or (None, None) on failure
    """
    if gaze is None or webcam is None or not webcam.isOpened():
        logger.error("Gaze tracking not properly initialized")
        return None, None
        
    try:
        # Capture frame
        ret, frame = webcam.read()
        if not ret or frame is None:
            logger.warning("Failed to capture frame from webcam")
            return None, None

        # Process with gaze tracking
        gaze.refresh(frame)
        
        # Get raw gaze data
        h_ratio = gaze.horizontal_ratio()
        v_ratio = gaze.vertical_ratio()
        
        # Apply filtering if we have valid gaze data
        if h_ratio is not None and v_ratio is not None:
            gaze_filter.update(h_ratio, v_ratio)
            filtered_h, filtered_v = gaze_filter.get_filtered_ratios()
            
            # Only use filtered values if they're valid
            if filtered_h is not None and filtered_v is not None:
                h_ratio, v_ratio = filtered_h, filtered_v

        # Create frame with annotations
        frame_annotated = gaze.annotated_frame()
        
        # Add custom visualization for filtered gaze point
        if h_ratio is not None and v_ratio is not None:
            height, width = frame_annotated.shape[:2]
            x = int(h_ratio * width)
            y = int(v_ratio * height)
            cv2.circle(frame_annotated, (x, y), 10, (0, 255, 255), 2)
        
        # Package gaze data
        gaze_data = {
            "gaze_ratio_horizontal": h_ratio,
            "gaze_ratio_vertical": v_ratio,
            "looking_left": gaze.is_left(),
            "looking_right": gaze.is_right(),
            "looking_center": gaze.is_center(),
            "pupils_located": gaze.pupils_located,
            "left_pupil": gaze.pupil_left_coords(),
            "right_pupil": gaze.pupil_right_coords(),
            "timestamp": time.time()
        }

        return gaze_data, frame_annotated
        
    except Exception as e:
        logger.error(f"Error in gaze tracking: {e}")
        return None, None

def close():
    """Release resources"""
    logger.info("Closing gaze tracker resources")
    if webcam is not None and webcam.isOpened():
        webcam.release()
    cv2.destroyAllWindows()