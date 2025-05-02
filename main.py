# Refactored `robot_controller.py` and `main.py` to use updated ZMQConnection API
import time
import cv2
import numpy as np
import sys

from zmq_connection import ZMQConnection
from gaze_tracker import GazeTracker
from robot_controller import RobotController
from q_learning_agent import GazeJoystickAgent
from keyboard_input import KeyboardController

def get_scene_objects(zmq_connection):
    objects = []
    response = zmq_connection.get_object_handle('target')
    if response.get('returnCode') == 0:
        target_handle = response.get('handle')
        position_response = zmq_connection.get_object_position(target_handle, -1)
        if position_response.get('returnCode') == 0:
            position = position_response.get('position')
            objects.append({
                "id": 0,
                "handle": target_handle,
                "name": "target",
                "position": position
            })
    return objects

def main():
    print("Starting Eye Gaze Control System...")
    try:
        zmq_connection = ZMQConnection(ip="127.0.0.1", port=23000)
    except Exception as e:
        print(f"Failed to connect to CoppeliaSim: {e}")
        sys.exit("Connection failed")

    try:
        print("Initializing gaze tracker...")
        gaze_tracker = GazeTracker()

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

    print("System initialized. Starting main loop...")
    running = True
    selected_obj_id = None
    last_action_time = 0
    action_cooldown = 1.0

    while running:
        objects = get_scene_objects(zmq_connection)
        if not objects:
            print("No objects found in scene")
            time.sleep(0.5)
            continue

        gaze_result = gaze_tracker.get_gaze_data()
        if gaze_result is not None:
            gaze_data, frame = gaze_result
            gazed_object = None
            h_ratio = gaze_data.get("gaze_ratio_horizontal")
            v_ratio = gaze_data.get("gaze_ratio_vertical")

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
                agent.update_gaze_history(gazed_object["id"], gaze_data["timestamp"])
                cv2.putText(frame, f"Gazing at: {gazed_object['name']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 240
            cv2.putText(frame, "No gaze data available", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        x_axis, y_axis, keyboard_running = keyboard.update()
        if not keyboard_running:
            running = False
            continue

        current_time = time.time()
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

        cv2.imshow("Eye Gaze Control System", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False

    print("Cleaning up resources...")
    keyboard.close()
    gaze_tracker.close()
    zmq_connection.disconnect()
    cv2.destroyAllWindows()
    print("System shutdown complete")

if __name__ == "__main__":
    main()
