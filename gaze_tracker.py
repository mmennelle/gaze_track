# Updated gaze_tracker.py with emphasis on using raw gaze data for calibration
import time
import cv2
import numpy as np
from collections import deque
import logging
from gaze_tracking import GazeTracking

# Configure logging
logger = logging.getLogger("GazeTracker")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class ImprovedGazeFilter:
    """
    Enhanced gaze filter that treats user's gaze as ground truth during calibration
    and applies the learned transformation afterward
    """
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.h_buffer = deque(maxlen=window_size)
        self.v_buffer = deque(maxlen=window_size)
        
        # Calibration data
        self.calibration_samples = []
        self.calibration_model = None
        self.calibrated = False
        self.min_samples_required = 5
        
    def update(self, h_ratio, v_ratio):
        """Update the filter with new raw gaze data"""
        if h_ratio is not None and 0 <= h_ratio <= 1:
            self.h_buffer.append(h_ratio)
        if v_ratio is not None and 0 <= v_ratio <= 1:
            self.v_buffer.append(v_ratio)
            
    def get_filtered_ratios(self):
        """Return filtered gaze ratios with outlier removal"""
        if len(self.h_buffer) < 3 or len(self.v_buffer) < 3:
            return None, None
            
        # Convert to numpy arrays for faster processing
        h_values = np.array(list(self.h_buffer))
        v_values = np.array(list(self.v_buffer))
        
        # Reject outliers using median absolute deviation (MAD)
        # More robust than standard deviation
        h_median = np.median(h_values)
        v_median = np.median(v_values)
        
        h_mad = np.median(np.abs(h_values - h_median))
        v_mad = np.median(np.abs(v_values - v_median))
        
        # Use MAD to identify outliers (threshold of 3 MADs)
        h_threshold = 3 * h_mad if h_mad > 0 else 0.1
        v_threshold = 3 * v_mad if v_mad > 0 else 0.1
        
        h_filtered = h_values[np.abs(h_values - h_median) < h_threshold]
        v_filtered = v_values[np.abs(v_values - v_median) < v_threshold]
        
        # If all values were outliers, return the median
        if len(h_filtered) == 0 or len(v_filtered) == 0:
            return h_median, v_median
        
        # Calculate filtered values (raw, before calibration)
        h_filtered_value = np.mean(h_filtered)
        v_filtered_value = np.mean(v_filtered)
            
        return h_filtered_value, v_filtered_value
    
    def add_calibration_sample(self, gaze_h, gaze_v, target_h, target_v):
        """
        Add a calibration sample mapping raw gaze coordinates to target coordinates
        
        Args:
            gaze_h, gaze_v: Raw gaze coordinates (0-1)
            target_h, target_v: Actual target coordinates (0-1)
        """
        # Add the gaze-to-target mapping
        self.calibration_samples.append({
            "gaze_h": gaze_h,
            "gaze_v": gaze_v,
            "target_h": target_h,
            "target_v": target_v
        })
        
        logger.info(f"Added calibration sample: gaze ({gaze_h:.3f}, {gaze_v:.3f}) -> target ({target_h:.3f}, {target_v:.3f})")
    
    def finalize_calibration(self):
        """Create a calibration model from collected samples"""
        # Check if we have enough calibration samples
        if len(self.calibration_samples) < self.min_samples_required:
            logger.warning(f"Not enough calibration samples: {len(self.calibration_samples)}/{self.min_samples_required}")
            return False
            
        # Extract data arrays
        gaze_h = np.array([sample["gaze_h"] for sample in self.calibration_samples])
        gaze_v = np.array([sample["gaze_v"] for sample in self.calibration_samples])
        target_h = np.array([sample["target_h"] for sample in self.calibration_samples])
        target_v = np.array([sample["target_v"] for sample in self.calibration_samples])
        
        # Create a polynomial calibration model
        # We'll use a polynomial transformation of degree 2 
        # This allows us to correct for non-linear distortions in gaze tracking
        try:
            # For horizontal calibration (x-axis)
            h_coeffs = np.polyfit(gaze_h, target_h, 2)
            
            # For vertical calibration (y-axis)
            v_coeffs = np.polyfit(gaze_v, target_v, 2)
            
            # Store the model
            self.calibration_model = {
                "h_coeffs": h_coeffs,
                "v_coeffs": v_coeffs
            }
            
            # Mark as calibrated
            self.calibrated = True
            
            logger.info(f"Calibration model created from {len(self.calibration_samples)} samples")
            logger.info(f"Horizontal coefficients: {h_coeffs}")
            logger.info(f"Vertical coefficients: {v_coeffs}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating calibration model: {e}")
            return False
    
    def apply_calibration(self, h_ratio, v_ratio):
        """
        Apply calibration transformation to raw gaze coordinates
        
        Args:
            h_ratio, v_ratio: Raw gaze coordinates (0-1)
        
        Returns:
            Calibrated gaze coordinates (0-1)
        """
        if not self.calibrated or self.calibration_model is None:
            return h_ratio, v_ratio
        
        try:
            # Apply polynomial transformation
            h_coeffs = self.calibration_model["h_coeffs"]
            v_coeffs = self.calibration_model["v_coeffs"]
            
            # Calculate calibrated coordinates using polynomial transformation
            h_calibrated = np.polyval(h_coeffs, h_ratio)
            v_calibrated = np.polyval(v_coeffs, v_ratio)
            
            # Ensure results stay in [0,1] range
            h_calibrated = np.clip(h_calibrated, 0.0, 1.0)
            v_calibrated = np.clip(v_calibrated, 0.0, 1.0)
            
            return h_calibrated, v_calibrated
            
        except Exception as e:
            logger.error(f"Error applying calibration: {e}")
            return h_ratio, v_ratio
    
    def get_calibration_stats(self):
        """Return statistics about the calibration"""
        if not self.calibrated:
            return {"calibrated": False, "samples": 0}
        
        return {
            "calibrated": True,
            "samples": len(self.calibration_samples),
            "model_type": "polynomial-2"
        }

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
    
    # Initialize the improved gaze filter
    gaze_filter = ImprovedGazeFilter(window_size=10)
    
    logger.info("Webcam and gaze tracking successfully initialized")
except Exception as e:
    logger.error(f"Failed to initialize gaze tracking: {e}")
    if webcam and webcam.isOpened():
        webcam.release()

def get_gaze_filter():
    """Get access to the gaze filter for calibration"""
    return gaze_filter

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
        raw_h_ratio = gaze.horizontal_ratio()
        raw_v_ratio = gaze.vertical_ratio()
        
        # Apply filtering if we have valid gaze data
        filtered_h_ratio = raw_h_ratio
        filtered_v_ratio = raw_v_ratio
        calibrated_h_ratio = raw_h_ratio
        calibrated_v_ratio = raw_v_ratio
        is_calibrated = False
        
        if raw_h_ratio is not None and raw_v_ratio is not None:
            # First step: Update the filter with raw gaze data and get filtered values
            gaze_filter.update(raw_h_ratio, raw_v_ratio)
            filtered_h, filtered_v = gaze_filter.get_filtered_ratios()
            
            # Only use filtered values if they're valid
            if filtered_h is not None and filtered_v is not None:
                filtered_h_ratio = filtered_h
                filtered_v_ratio = filtered_v
                
                # Second step: Apply calibration if available
                if gaze_filter.calibrated:
                    calibrated_h, calibrated_v = gaze_filter.apply_calibration(filtered_h, filtered_v)
                    calibrated_h_ratio = calibrated_h
                    calibrated_v_ratio = calibrated_v
                    is_calibrated = True

        # Create frame with annotations
        frame_annotated = gaze.annotated_frame()
        
        # Add custom visualization for eye gaze point
        if raw_h_ratio is not None and raw_v_ratio is not None:
            height, width = frame_annotated.shape[:2]
            
            # Draw raw gaze point (small yellow) - this is what we use for calibration
            raw_x = int(raw_h_ratio * width)
            raw_y = int(raw_v_ratio * height)
            cv2.circle(frame_annotated, (raw_x, raw_y), 5, (0, 255, 255), 1)
            
            # Draw filtered gaze point (medium orange)
            filtered_x = int(filtered_h_ratio * width)
            filtered_y = int(filtered_v_ratio * height)
            cv2.circle(frame_annotated, (filtered_x, filtered_y), 8, (0, 165, 255), 1)
            
            # Draw calibrated gaze point (large green) if calibration is active
            if is_calibrated:
                x = int(calibrated_h_ratio * width)
                y = int(calibrated_v_ratio * height)
                cv2.circle(frame_annotated, (x, y), 12, (0, 255, 0), 2)
                
                # Line connecting raw and calibrated points to show the transformation
                cv2.line(frame_annotated, (raw_x, raw_y), (x, y), (0, 200, 200), 1)
                
                # Label the calibrated point
                cv2.putText(frame_annotated, "Calibrated", (x + 10, y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Add calibration status
            status_text = "CALIBRATED" if is_calibrated else "UNCALIBRATED"
            status_color = (0, 255, 0) if is_calibrated else (0, 100, 255)
            cv2.putText(frame_annotated, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Add explanation of the circles
            cv2.putText(frame_annotated, "Yellow: Raw    Orange: Filtered    Green: Calibrated", 
                       (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Package gaze data for use by other components
        gaze_data = {
            # Calibrated values (or filtered if not calibrated)
            "gaze_ratio_horizontal": calibrated_h_ratio,
            "gaze_ratio_vertical": calibrated_v_ratio,
            
            # Raw values - important for calibration
            "raw_ratio_horizontal": raw_h_ratio,
            "raw_ratio_vertical": raw_v_ratio,
            
            # Filtered values (before calibration)
            "filtered_ratio_horizontal": filtered_h_ratio,
            "filtered_ratio_vertical": filtered_v_ratio,
            
            # Other gaze tracking data
            "looking_left": gaze.is_left(),
            "looking_right": gaze.is_right(),
            "looking_center": gaze.is_center(),
            "pupils_located": gaze.pupils_located,
            "left_pupil": gaze.pupil_left_coords(),
            "right_pupil": gaze.pupil_right_coords(),
            "calibrated": is_calibrated,
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