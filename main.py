# main.py - Final update with enhanced calibration
import time
import cv2
import numpy as np
import threading
import sys
import win32gui
import win32con
from zmq_connection import ZMQConnection
from gaze_tracker import get_gaze_data, close as close_gaze_tracker
from robot_controller import RobotController
from q_learning_agent import GazeJoystickAgent
from keyboard_input import KeyboardController
from gaze_duration_tracker import GazeDurationTracker
from calibration_module import CalibrationModule  # Import the updated calibration module

# Shared states
gaze_data = None
frame = None
running = True
frame_lock = threading.Lock()

def set_window_always_on_top(window_name):
    """Set an OpenCV window to be always on top"""
    # Give time for the window to be created
    time.sleep(0.5)
    
    # Find the window by name
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd:
        # Set the window to be topmost (always on top)
        win32gui.SetWindowPos(
            hwnd,                # Handle to the window
            win32con.HWND_TOPMOST,  # Put it on top of all windows
            0, 0, 0, 0,          # Don't change position or size
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE  # Flags
        )
        print(f"Window '{window_name}' set to always on top")
        return True
    
    print(f"Window '{window_name}' not found")
    return False

def gaze_thread():
    global gaze_data, frame, running
    try:
        while running:
            result = get_gaze_data()
            if result:
                new_gaze_data, new_frame = result
                with frame_lock:
                    gaze_data = new_gaze_data
                    frame = new_frame
            time.sleep(0.033)  # 33 fps
    except Exception as e:
        print(f"[Gaze Thread] Exception: {e}")
        running = False

def run_enhanced_calibration(calibration_module):
    """Run the enhanced calibration sequence"""
    global frame, gaze_data
    
    print("Starting enhanced calibration...")
    
    # Create calibration window
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", 800, 600)
    set_window_always_on_top("Calibration")
    
    # Get frame size for overlay creation
    with frame_lock:
        if frame is not None:
            frame_size = frame.shape[:2]
        else:
            frame_size = (480, 640)  # Default size
    
    # Get the gaze filter for calibration
    from gaze_tracker import get_gaze_filter
    gaze_filter = get_gaze_filter()
    if gaze_filter:
        calibration_module.set_gaze_filter(gaze_filter)
        print("Connected gaze filter to calibration module")
    else:
        print("WARNING: Could not get gaze filter - calibration may not work correctly")
    
    # Set up to handle yield_gaze_data() calls from the calibration module
    # We'll monkey patch the function to return our global gaze_data
    from calibration_module import yield_gaze_data
    def get_current_gaze_data():
        with frame_lock:
            return gaze_data
    
    # Monkey patch the yield_gaze_data function
    import types
    calibration_module.yield_gaze_data = types.MethodType(lambda self: get_current_gaze_data(), calibration_module)
    
    # Start the calibration sequence
    calibration_generator = calibration_module.run_calibration(frame_size)
    
    try:
        for target_obj, overlay, progress, wait_for_key in calibration_generator:
            # Get current frame with gaze data
            with frame_lock:
                current_frame = frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            
            # If we have a frame, blend it with the overlay
            if current_frame is not None and overlay is not None:
                # Blend overlay with frame
                alpha = 0.6
                combined_frame = cv2.addWeighted(current_frame, alpha, overlay, 1-alpha, 0)
                
                # Display the calibration frame
                cv2.imshow("Calibration", combined_frame)
            else:
                # Just show the overlay if no frame
                cv2.imshow("Calibration", overlay)
            
            # Process key events
            if wait_for_key:
                cv2.waitKey(0)  # Wait indefinitely for a key press
            else:
                key = cv2.waitKey(33) & 0xFF
                if key == ord('q'):
                    print("Calibration aborted by user")
                    break
            
            time.sleep(0.03)  # ~30fps
                
        # Calibration complete or aborted
        calibration_success = calibration_module.calibration_complete
        cv2.destroyWindow("Calibration")
        
        return calibration_success
        
    except Exception as e:
        print(f"Error during calibration: {e}")
        cv2.destroyWindow("Calibration")
        return False

def get_scene_objects(zmq_connection):
    objects = []
    object_id = 0
    potential_targets = [f"/target[{i}]" for i in range(20)]
    for target_name in potential_targets:
        try:
            response = zmq_connection.get_object_handle(target_name)
            if response.get('returnCode') == 0:
                target_handle = response.get('handle')
                position_response = zmq_connection.get_object_position(target_handle, -1)
                if position_response.get('returnCode') == 0:
                    position = position_response.get('position')
                    objects.append({
                        "id": object_id,
                        "handle": target_handle,
                        "name": target_name,
                        "position": position
                    })
                    object_id += 1
                    print(f"Found target: {target_name} at position {position}")
            else:
                if "[" in target_name and "target[0]" not in target_name:
                    break
        except:
            if "[" in target_name and "target[0]" not in target_name:
                break
            continue
    return objects

def main():
    global running

    print("Starting Eye Gaze Control System...")
    try:
        zmq_connection = ZMQConnection(ip="127.0.0.1", port=23000)
    except Exception as e:
        print(f"Failed to connect to CoppeliaSim: {e}")
        sys.exit("Connection failed")

    try:
        print("Initializing robot controller...")
        robot = RobotController(zmq_connection)

        print("Initializing keyboard controller...")
        keyboard = KeyboardController()

        print("Initializing Q-learning agent...")
        agent = GazeJoystickAgent()

        print("Initializing gaze duration tracker...")
        gaze_tracker = GazeDurationTracker()
        
        print("Initializing calibration module...")
        calibration_module = CalibrationModule(zmq_connection, gaze_tracker, agent, duration_per_target=3.0)

    except Exception as e:
        print(f"Initialization error: {e}")
        zmq_connection.disconnect()
        sys.exit("Initialization failed")

    # Start gaze tracking thread
    gaze_thread_handle = threading.Thread(target=gaze_thread, daemon=True)
    gaze_thread_handle.start()
    
    # Give the gaze tracker time to initialize
    print("Initializing webcam and gaze tracker...")
    time.sleep(2.0)
    
    # Create window and set it to always be on top
    cv2.namedWindow("Gaze Frame", cv2.WINDOW_NORMAL)
    set_window_always_on_top("Gaze Frame")
    
    # Create additional window for keyboard controls and make it always on top
    cv2.namedWindow("Keyboard Controls", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Keyboard Controls", 0, 0)  # Position at top-left
    cv2.resizeWindow("Keyboard Controls", 300, 150)
    
    # Create a small help image for keyboard controls
    help_img = np.zeros((150, 300, 3), dtype=np.uint8)
    cv2.putText(help_img, "Keyboard Controls:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(help_img, "Arrow keys - Move", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(help_img, "Q - Quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(help_img, "C - Recalibrate", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Set the help window to be always on top
    set_window_always_on_top("Keyboard Controls")

    # Offer calibration to the user
    if input("Run enhanced gaze calibration? (y/n): ").strip().lower().startswith("y"):
        # Run the enhanced calibration sequence
        calibration_success = run_enhanced_calibration(calibration_module)
        if calibration_success:
            print("Enhanced calibration completed successfully!")
        else:
            print("Enhanced calibration did not complete. Using default settings.")

    print("System initialized. Starting main loop...")
    selected_obj_id = None
    last_action_time = 0
    action_cooldown = 1.0
    last_object_check = 0
    object_check_interval = 0.5
    objects = []

    while running:
        # Always display the help window
        cv2.imshow("Keyboard Controls", help_img)
        
        with frame_lock:
            current_frame = frame
            current_gaze_data = gaze_data

        if current_frame is not None:
            cv2.imshow("Gaze Frame", current_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
            break
        elif key == ord('c'):
            # Recalibration option
            print("Starting recalibration...")
            run_enhanced_calibration(calibration_module)

        current_time = time.time()
        if current_time - last_object_check > object_check_interval:
            objects = get_scene_objects(zmq_connection)
            last_object_check = current_time
            if not objects:
                print("No objects found in scene")
            else:
                print(f"Found {len(objects)} objects in scene")

        if not objects:
            x_axis, y_axis, keyboard_running = keyboard.update()
            if not keyboard_running:
                running = False
            continue

        if current_gaze_data and current_gaze_data.get("pupils_located", False):
            gazed_object = None
            h_ratio = current_gaze_data.get("gaze_ratio_horizontal")
            v_ratio = current_gaze_data.get("gaze_ratio_vertical")
            
            # Get whether the gaze is calibrated
            is_calibrated = current_gaze_data.get("calibrated", False)
            
            if h_ratio is not None and v_ratio is not None:
                wx = -1.0 + 2.0 * h_ratio
                wy = -1.0 + 2.0 * v_ratio
                min_distance = float('inf')
                for obj in objects:
                    ox, oy, _ = obj["position"]
                    distance = np.sqrt((wx - ox)**2 + (wy - oy)**2)
                    if distance < min_distance:
                        min_distance = distance
                        gazed_object = obj
                
                # If calibrated, we can use a tighter threshold
                distance_threshold = 0.3 if is_calibrated else 0.5
                if min_distance >= distance_threshold:
                    gazed_object = None
                    
            if gazed_object:
                print(f"Looking at: {gazed_object['name']}")
                gaze_tracker.update(gazed_object["id"])
                agent.update_gaze_history(gazed_object["id"], current_time)

        x_axis, y_axis, keyboard_running = keyboard.update()
        if not keyboard_running:
            running = False
            break

        if (abs(x_axis) > 0.2 or abs(y_axis) > 0.2) and (current_time - last_action_time > action_cooldown):
            joystick_direction = agent.get_joystick_direction(x_axis, y_axis)
            gaze_durations = {obj["id"]: gaze_tracker.get_duration(obj["id"]) for obj in objects}
            selected_obj_id = agent.get_action(joystick_direction, current_time, objects, gaze_durations)

            if selected_obj_id is not None and 0 <= selected_obj_id < len(objects):
                target = objects[selected_obj_id]
                print(f"Selected object: {target['name']}")
                success = robot.move_to_object(target["handle"])
                if success:
                    recent_gazes = [g["object_id"] for g in agent.gaze_history[-5:]]
                    reward = 1.0 if selected_obj_id in recent_gazes else -0.2
                    agent.update_q_table(selected_obj_id, joystick_direction, reward)
                    print(f"Updated Q-values with reward: {reward}")
                last_action_time = current_time

    print("Cleaning up resources...")
    keyboard.close()
    running = False
    
    # Wait for gaze thread to finish
    if gaze_thread_handle.is_alive():
        gaze_thread_handle.join(timeout=1.0)
        
    close_gaze_tracker()
    zmq_connection.disconnect()
    cv2.destroyAllWindows()
    print("System shutdown complete")

if __name__ == '__main__':
    print("Starting system. It takes 1 minute.")
    main()