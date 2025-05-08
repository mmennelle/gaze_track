import cv2

print("Scanning for available camera indices...")
for i in range(5):  # Scan indices 0 through 4
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"Camera index {i} is available")
        cap.release()
    else:
        print(f"Camera index {i} NOT available")

# Update first param of line 185 in gaze_tracker to the
# corresponding index