# Optimized main.py with gaze calibration and improved gaze duration tracking
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
from calibration_agent import CalibrationAgent
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
            time.sleep(0.033)  # 33fps
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
    global running, gaze_data, frame
    print("Starting Eye Gaze Control System...")

    try:
        zmq_connection = ZMQConnection(ip="127.0.0.1", port=23000)
    except Exception as e:
        print(f"Failed to connect to CoppeliaSim: {e}")
        sys.exit("Connection failed")

    try:
        robot = RobotController(zmq_connection)
        keyboard = KeyboardController()
        agent = GazeJoystickAgent()
        calibration = CalibrationAgent()
        gaze_tracker = GazeDurationTracker()
    except Exception as e:
        print(f"Initialization error: {e}")
        zmq_connection.disconnect()
        sys.exit("Initialization failed")

    gaze_tracking_thread = threading.Thread(target=gaze_thread, daemon=True)
    gaze_tracking_thread.start()

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
            cv2.imshow("Eye Gaze Control System", current_frame)

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
                gaze_tracker.update(gazed_object["id"])
                agent.update_gaze_history(gazed_object["id"], current_time)
                print(f"Looking at: {gazed_object['name']}")

        x_axis, y_axis, keyboard_running = keyboard.update()
        if not keyboard_running:
            running = False
            continue

        if (abs(x_axis) > 0.2 or abs(y_axis) > 0.2) and (current_time - last_action_time > action_cooldown):
            joystick_direction = agent.get_joystick_direction(x_axis, y_axis)
            if joystick_direction is not None:
                # Create list of durations from tracker
                durations = {obj['id']: gaze_tracker.get_duration(obj['id']) for obj in objects}
                selected_obj_id = agent.get_action(joystick_direction, current_time, objects, durations)
                if selected_obj_id is not None and selected_obj_id < len(objects):
                    selected_object = objects[selected_obj_id]
                    print(f"\nSelected object: {selected_object['name']}")
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