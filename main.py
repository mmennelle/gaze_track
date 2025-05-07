# Refactored main.py with core functionality optimized
import time
import cv2
import numpy as np
import sys
import threading
from zmq_connection import ZMQConnection
from gaze_tracker import get_gaze_data, close as close_gaze_tracker
from robot_controller import RobotController
from q_learning_agent import GazeJoystickAgent
from keyboard_input import KeyboardController

# Shared states
gaze_data = None
frame = None
running = True
frame_lock = threading.Lock()

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
            time.sleep(0.033)  # set frame rate. .033 == 33fps

    except Exception as e:
        print(f"[Gaze Thread] Exception: {e}")
        running = False

def get_scene_objects(zmq_connection):
    objects = []
    object_id = 0
    
    # targets
    potential_targets = []
    
    # Indexed targets /target[0], /target[1], etc.
    for i in range(20):  # Arbitrary limit
        potential_targets.append(f"/target[{i}]")
    
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
                # Break out of loop if we can't find target[i] for i>0
                if "[" in target_name and "target[0]" not in target_name:
                    break
        except:
            if "[" in target_name and "target[0]" not in target_name:
                break
            continue
    
    return objects

def main():
    global running, gaze_data, frame

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

    except Exception as e:
        print(f"Initialization error: {e}")
        zmq_connection.disconnect()
        sys.exit("Initialization failed")

    # Ask user if they want to calibrate
    print("Do you want to run calibration? (y/n)")
    calibrate = input().lower() == 'y'
    
    if calibrate:
        print("Starting calibration. Look at objects and press arrow keys.")
        
        # Define a function to get joystick direction for calibration
        def get_joystick_for_calibration():
            x_axis, y_axis, _ = keyboard.update()
            return agent.get_joystick_direction(x_axis, y_axis)
        
        # Run calibration
        agent.calibrate_for_user(duration=20.0, joystick_fn=get_joystick_for_calibration)

    # Start threads
    gaze_tracking_thread = threading.Thread(target=gaze_thread, daemon=True)
    gaze_tracking_thread.start()

    print("System initialized. Starting main loop...")
    selected_obj_id = None
    last_action_time = 0
    action_cooldown = 1.0
    last_object_check = 0
    object_check_interval = 0.5
    objects = []  # Initialize objects list

    while running:
        # Handle GUI display
        with frame_lock:
            current_frame = frame
            current_gaze_data = gaze_data
        
        if current_frame is not None:
            cv2.imshow("Eye Gaze Control System", current_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

        # Check for objects periodically
        current_time = time.time()
        if current_time - last_object_check > object_check_interval:
            objects = get_scene_objects(zmq_connection)
            last_object_check = current_time
            
            if not objects:
                print("No objects found in scene")
            else:
                print(f"Found {len(objects)} objects in scene:")
                for obj in objects:
                    print(f"  - {obj['name']} at position {obj['position']}")

        # Skip processing if no objects are available
        if not objects:
            x_axis, y_axis, keyboard_running = keyboard.update()
            if not keyboard_running:
                running = False
            continue

        # Process gaze data and update agent
        gazed_object = None
        if current_gaze_data and current_gaze_data.get("pupils_located", False):
            h_ratio = current_gaze_data.get("gaze_ratio_horizontal")
            v_ratio = current_gaze_data.get("gaze_ratio_vertical")

            if h_ratio is not None and v_ratio is not None:
                # Map screen coordinates to world coordinates
                world_x = -1.0 + 2.0 * h_ratio
                world_y = -1.0 + 2.0 * v_ratio

                # Find closest object with weighted distance
                min_distance = float('inf')
                for obj in objects:
                    obj_x, obj_y, _ = obj["position"]
                    
                    # Calculate Euclidean distance
                    distance = np.sqrt((world_x - obj_x)**2 + (world_y - obj_y)**2)
                    
                    # Calculate angle similarity
                    gaze_vector = np.array([world_x, world_y])
                    obj_vector = np.array([obj_x, obj_y])
                    
                    if np.linalg.norm(gaze_vector) > 0 and np.linalg.norm(obj_vector) > 0:
                        gaze_vector_norm = gaze_vector / np.linalg.norm(gaze_vector)
                        obj_vector_norm = obj_vector / np.linalg.norm(obj_vector)
                        dot_product = np.dot(gaze_vector_norm, obj_vector_norm)
                        angle_similarity = 1.0 - max(0, (1.0 - abs(dot_product)) * 0.5)
                    else:
                        angle_similarity = 0.5
                    
                    # Combined score (lower is better)
                    distance_score = distance * (2.0 - angle_similarity)
                    
                    if distance_score < min_distance:
                        min_distance = distance_score
                        gazed_object = obj
                
                # Apply threshold
                if min_distance >= 0.7:
                    gazed_object = None

            # Update agent's gaze history if looking at an object
            if gazed_object:
                agent.update_gaze_history(gazed_object["id"], current_time)
                print(f"Looking at: {gazed_object['name']}")

        # Handle keyboard input
        x_axis, y_axis, keyboard_running = keyboard.update()
        if not keyboard_running:
            running = False
            continue

        # Process joystick input to select and move to objects
        if (abs(x_axis) > 0.2 or abs(y_axis) > 0.2) and (current_time - last_action_time > action_cooldown):
            joystick_direction = agent.get_joystick_direction(x_axis, y_axis)
            if joystick_direction is not None:
                selected_obj_id = agent.get_action(joystick_direction, current_time, objects)
                if selected_obj_id is not None and selected_obj_id < len(objects):
                    selected_object = objects[selected_obj_id]
                    print(f"\nSelected object: {selected_object['name']}")
                    if current_time - last_action_time > action_cooldown:
                        success = robot.move_to_object(selected_object["handle"])
                        last_action_time = current_time
                        if success:
                            # Determine reward based on whether object was recently gazed at
                            recent_gazes = [g["object_id"] for g in agent.gaze_history[-5:]]
                            reward = 1.0 if selected_obj_id in recent_gazes else -0.2
                            agent.update_q_table(selected_obj_id, joystick_direction, reward)
                            print(f"Updated Q-values with reward: {reward}")

    print("Cleaning up resources...")
    keyboard.close()
    running = False
    gaze_tracking_thread.join()
    close_gaze_tracker()
    zmq_connection.disconnect()
    cv2.destroyAllWindows()
    print("System shutdown complete")

if __name__ == "__main__":
    main()