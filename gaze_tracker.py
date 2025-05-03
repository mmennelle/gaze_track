# Procedural gaze_tracker.py wrapper for GazeTracking library
import time
import cv2
from gaze_tracking import GazeTracking

print("[GazeTracker] Initializing webcam...")
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    raise Exception("Webcam could not be opened")
print("[GazeTracker] Webcam successfully opened.")

def get_gaze_data():
    ret, frame = webcam.read()
    if not ret or frame is None:
        return None, None

    gaze.refresh(frame)
    frame_annotated = gaze.annotated_frame()

    gaze_data = {
        "gaze_ratio_horizontal": gaze.horizontal_ratio(),
        "gaze_ratio_vertical": gaze.vertical_ratio(),
        "looking_left": gaze.is_left(),
        "looking_right": gaze.is_right(),
        "looking_center": gaze.is_center(),
        "pupils_located": gaze.pupils_located,
        "timestamp": time.time()
    }

    return gaze_data, frame_annotated

def close():
    webcam.release()
    cv2.destroyAllWindows()