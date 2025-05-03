# Optimized main.py for better performance
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
            # Increased sleep to reduce CPU usage and camera polling
            time.sleep(0.033)  # ~30 FPS instead of 100 FPS

    except Exception as e:
        print(f"[Gaze Thread] Exception: {e}")
        running = False

def get_scene_objects(zmq_connection):
    objects = []
    object_id = 0
    
    # First, try the base target name
    potential_targets = []
    
    # Then try indexed targets /target[0], /target[1], etc.
    for i in range(20):  # Arbitrary limit, increase if needed
        potential_targets.append(f"/target[{i}]")
    
    # Now check each potential target
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
                # If we hit an error with indexed targets, probably no more exist
                if "[" in target_name and "target[0]" not in target_name:
                    break
        except:
            # If we hit an error with indexed targets, probably no more exist
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

        if current_gaze_data and current_gaze_data.get("pupils_located", False):
            gazed_object = None
            h_ratio = current_gaze_data.get("gaze_ratio_horizontal")
            v_ratio = current_gaze_data.get("gaze_ratio_vertical")

            if h_ratio is not None and v_ratio is not None:
                world_x = -1.0 + 2.0 * h_ratio
                world_y = -1.0 + 2.0 * v_ratio

                min_distance = float('inf')
                for obj in objects:
                    obj_x, obj_y, _ = obj["position"]
                    distance = np.sqrt((world_x - obj_x)**2 + (world_y - obj_y)**2)
                    if distance < min_distance:
                        min_distance = distance
                        gazed_object = obj
                if min_distance >= 0.5:
                    gazed_object = None

            if gazed_object:
                agent.update_gaze_history(gazed_object["id"], current_time)
                print(f"Looking at: {gazed_object['name']}")

        x_axis, y_axis, keyboard_running = keyboard.update()
        if not keyboard_running:
            running = False
            continue

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