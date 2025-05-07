# main.py (with integrated gaze calibration pipeline and restored get_scene_objects)
import time
import cv2
import numpy as np
import threading
import sys
from zmq_connection import ZMQConnection
from gaze_tracker import get_gaze_data, close as close_gaze_tracker
from robot_controller import RobotController
from q_learning_agent import GazeJoystickAgent
from keyboard_input import KeyboardController
from gaze_duration_tracker import GazeDurationTracker

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
            time.sleep(0.033)  # 33 fps
    except Exception as e:
        print(f"[Gaze Thread] Exception: {e}")
        running = False

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

    except Exception as e:
        print(f"Initialization error: {e}")
        zmq_connection.disconnect()
        sys.exit("Initialization failed")

    threading.Thread(target=gaze_thread, daemon=True).start()
    time.sleep(1.0)
    cv2.namedWindow("Gaze Frame", cv2.WINDOW_NORMAL)

    if input("Run gaze calibration phase? (y/n): ").strip().lower().startswith("y"):
        print("Starting calibration phase (10 seconds)...")
        start_time = time.time()
        while time.time() - start_time < 10:
            objects = get_scene_objects(zmq_connection)
            current_time = time.time()
            if gaze_data:
                h = gaze_data.get("gaze_ratio_horizontal")
                v = gaze_data.get("gaze_ratio_vertical")
                if h is not None and v is not None:
                    wx = -1.0 + 2.0 * h
                    wy = -1.0 + 2.0 * v
                    min_dist = float('inf')
                    target = None
                    for obj in objects:
                        ox, oy, _ = obj["position"]
                        dist = np.sqrt((wx - ox)**2 + (wy - oy)**2)
                        if dist < min_dist:
                            min_dist = dist
                            target = obj
                    if target:
                        print(f"[Calibration] Looking at: {target['name']}")
                        gaze_tracker.update(target["id"])
                        agent.update_gaze_history(target["id"], current_time)
            if frame is not None:
                cv2.imshow("Gaze Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            time.sleep(0.033)
        print("Calibration complete. Entering live mode.")

    print("System initialized. Starting main loop...")
    selected_obj_id = None
    last_action_time = 0
    action_cooldown = 1.0
    last_object_check = 0
    object_check_interval = 0.5
    objects = []

    while running:
        with frame_lock:
            current_frame = frame
            current_gaze_data = gaze_data

        if current_frame is not None:
            cv2.imshow("Gaze Frame", current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

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
                wx = -1.0 + 2.0 * h_ratio
                wy = -1.0 + 2.0 * v_ratio
                min_distance = float('inf')
                for obj in objects:
                    ox, oy, _ = obj["position"]
                    distance = np.sqrt((wx - ox)**2 + (wy - oy)**2)
                    if distance < min_distance:
                        min_distance = distance
                        gazed_object = obj
                if min_distance >= 0.5:
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
    close_gaze_tracker()
    zmq_connection.disconnect()
    cv2.destroyAllWindows()
    print("System shutdown complete")

if __name__ == '__main__':
    print("Starting system. It takes 1 minute.")
    main()
