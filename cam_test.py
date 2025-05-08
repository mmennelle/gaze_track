import cv2

print("Scanning for available camera indices...")
for i in range(5):  # Scan indices 0 through 4
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera index {i} is available")
        cap.release()
    else:
        print(f"Camera index {i} NOT available")
