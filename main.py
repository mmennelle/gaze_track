# Refactored main.py - Eye Gaze Control System
import time
import cv2
import numpy as np
import sys
import threading
import logging
import traceback
from queue import Queue
from collections import deque

from zmq_connection import ZMQConnection
from gaze_tracker import get_gaze_data, close as close_gaze_tracker
from robot_controller import RobotController
from q_learning_agent import GazeJoystickAgent
from keyboard_input import KeyboardController
from calibration_agent import CalibrationAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gaze_control.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("GazeControl")

# Shared states using thread-safe structures
gaze_queue = Queue(maxsize=10)
processed_gaze_queue = Queue(maxsize=10)
shared_state = {
    "running": True,
    "objects": [],
    "last_object_check": 0,
    "gaze_data": None,
    "frame": None,
    "gazed_object": None,
    "selected_object": None
}
state_lock = threading.Lock()

class ObjectProximityCache:
    """
    Caches nearest object calculations to improve performance
    """
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        
    def get_nearest_object(self, gaze_x, gaze_y, objects, max_distance=0.5):
        # Create cache key with reduced precision to improve cache hits
        key = (round(gaze_x, 2), round(gaze_y, 2))
        
        if key in self.cache and self.cache[key]["timestamp"] > time.time() - 1.0:
            return self.cache[key]["object"]
            
        # Calculate nearest object
        min_distance = float('inf')
        nearest_obj = None
        
        for obj in objects:
            obj_x, obj_y, _ = obj["position"]
            distance = np.sqrt((gaze_x - obj_x)**2 + (gaze_y - obj_y)**2)
            if distance < min_distance:
                min_distance = distance
                nearest_obj = obj
                
        if min_distance >= max_distance:
            nearest_obj = None
            
        # Update cache
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
            
        self.cache[key] = {
            "object": nearest_obj,
            "timestamp": time.time()
        }
        
        return nearest_obj

def gaze_producer_thread():
    """Thread to capture and produce gaze data"""
    logger.info("Starting gaze producer thread")
    
    try:
        while shared_state["running"]:
            # Get new gaze data
            result = get_gaze_data()
            
            if result:
                gaze_data, frame = result
                
                # Put in queue if not full
                if not gaze_queue.full():
                    gaze_queue.put((gaze_data, frame))
                    
            time.sleep(0.033)  # ~30fps
            
    except Exception as e:
        logger.error(f"Gaze producer thread error: {e}")
        logger.error(traceback.format_exc())
        shared_state["running"] = False

def gaze_consumer_thread(calibration_agent, proximity_cache):
    """Thread to process captured gaze data"""
    logger.info("Starting gaze consumer thread")
    
    try:
        while shared_state["running"]:
            if not gaze_queue.empty():
                gaze_data, frame = gaze_queue.get()
                
                # Process gaze data
                processed_data = {
                    "gaze_data": gaze_data,
                    "frame": frame,
                    "gazed_object": None,
                    "timestamp": time.time()
                }
                
                if gaze_data and gaze_data.get("pupils_located", False):
                    h_ratio = gaze_data.get("gaze_ratio_horizontal")
                    v_ratio = gaze_data.get("gaze_ratio_vertical")
                    
                    if h_ratio is not None and v_ratio is not None:
                        # Apply calibration if available
                        if calibration_agent.is_calibrated:
                            h_ratio, v_ratio = calibration_agent.transform_gaze_point(h_ratio, v_ratio)
                            
                        # Map to world coordinates (-1,1 range)
                        world_x = -1.0 + 2.0 * h_ratio
                        world_y = -1.0 + 2.0 * v_ratio
                        
                        # Find gazed object with thread-safe access to objects
                        with state_lock:
                            objects = shared_state["objects"]
                            
                        if objects:
                            gazed_object = proximity_cache.get_nearest_object(world_x, world_y, objects)
                            processed_data["gazed_object"] = gazed_object
                
                # Put processed data in output queue
                if not processed_gaze_queue.full():
                    processed_gaze_queue.put(processed_data)
                
                gaze_queue.task_done()
            else:
                time.sleep(0.01)
                
    except Exception as e:
        logger.error(f"Gaze consumer thread error: {e}")
        logger.error(traceback.format_exc())
        shared_state["running"] = False

def object_detection_thread(zmq_connection):
    """Thread to periodically detect objects in the scene"""
    logger.info("Starting object detection thread")
    
    object_check_interval = 0.5  # Check every 0.5 seconds
    
    try:
        while shared_state["running"]:
            current_time = time.time()
            
            with state_lock:
                last_check = shared_state["last_object_check"]
                
            # Check for objects periodically
            if current_time - last_check > object_check_interval:
                objects = get_scene_objects(zmq_connection)
                
                with state_lock:
                    shared_state["objects"] = objects
                    shared_state["last_object_check"] = current_time
                
                if not objects:
                    logger.warning("No objects found in scene")
                else:
                    logger.info(f"Found {len(objects)} objects in scene")
                    
            time.sleep(0.1)
            
    except Exception as e:
        logger.error(f"Object detection thread error: {e}")
        logger.error(traceback.format_exc())
        shared_state["running"] = False

def get_scene_objects(zmq_connection):
    """
    Get objects from the scene, properly handling multiple objects
    """
    objects = []
    object_id = 0
    
    # Potential targets to check
    potential_targets = [f"/target[{i}]" for i in range(20)]  # Check up to 20 targets
    
    for target_name in potential_targets:
        try:
            # Try to get object handle
            response = zmq_connection.get_object_handle(target_name)
            
            if response.get('returnCode') == 0:
                target_handle = response.get('handle')
                
                # Get position
                position_response = zmq_connection.get_object_position(target_handle, -1)
                
                if position_response.get('returnCode') == 0:
                    position = position_response.get('position')
                    
                    # Add to objects list
                    objects.append({
                        "id": object_id,
                        "handle": target_handle,
                        "name": target_name,
                        "position": position
                    })
                    object_id += 1
                    logger.debug(f"Found target: {target_name} at position {position}")
            else:
                # If we can't find an object and it's not the first one,
                # assume we've reached the end of valid targets
                if target_name != "/target[0]" and not objects:
                    logger.debug(f"Target {target_name} not found, stopping search")
                    break
        except Exception as e:
            logger.error(f"Error getting object {target_name}: {e}")
            # Only break if we've already found some objects (gap in numbering)
            if objects and "target[0]" not in target_name:
                break
    
    return objects

def draw_ui_overlay(frame, gaze_data, gazed_object, objects, agent, calibration_agent):
    """Draw UI elements on the frame to show system state"""
    if frame is None:
        return frame
    
    h, w = frame.shape[:2]
    
    # Calibration mode overlay
    if not calibration_agent.is_calibrated:
        frame = calibration_agent.draw_calibration_ui(frame)
        return frame
    
    # Draw gaze point
    if gaze_data and gaze_data.get("pupils_located", False):
        h_ratio = gaze_data.get("gaze_ratio_horizontal")
        v_ratio = gaze_data.get("gaze_ratio_vertical")
        
        if h_ratio is not None and v_ratio is not None:
            gaze_x = int(h_ratio * w)
            gaze_y = int(v_ratio * h)
            
            # Draw crosshair
            cv2.line(frame, (gaze_x - 10, gaze_y), (gaze_x + 10, gaze_y), (0, 255, 255), 2)
            cv2.line(frame, (gaze_x, gaze_y - 10), (gaze_x, gaze_y + 10), (0, 255, 255), 2)
    
    # Draw object markers
    for i, obj in enumerate(objects):
        obj_x, obj_y, _ = obj["position"]
        
        # Convert from world to screen coordinates
        screen_x = int((obj_x + 1.0) / 2.0 * w)
        screen_y = int((obj_y + 1.0) / 2.0 * h)
        
        # Draw circle for each object
        color = (0, 255, 0) if gazed_object and obj["id"] == gazed_object["id"] else (0, 0, 255)
        cv2.circle(frame, (screen_x, screen_y), 15, color, 2)
        cv2.putText(frame, obj["name"], (screen_x - 20, screen_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw probabilities bar chart
    if hasattr(agent, 'last_probabilities') and agent.last_probabilities:
        probs = agent.last_probabilities
        max_p = max(probs) if probs else 0
        
        # Background for probability display
        cv2.rectangle(frame, (w - 210, h - 20 - len(probs)*30 - 10), 
                     (w - 10, h - 10), (0, 0, 0), -1)
        
        for i, p in enumerate(probs):
            if max_p > 0:
                bar_height = int((p / max_p) * 100)
                cv2.rectangle(frame, (w - 110, h - 20 - i*30), 
                             (w - 110 + bar_height, h - 10 - i*30), 
                             (0, 200, 0), -1)
            cv2.putText(frame, f"Obj {i}: {p:.2f}", (w - 200, h - 15 - i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw help text
    cv2.putText(frame, "Arrow keys to move | C: calibrate | Q: quit", (10, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def main():
    """Main function for the eye gaze control system"""
    logger.info("Starting Eye Gaze Control System...")
    
    # Initialize ZMQ connection
    try:
        zmq_connection = ZMQConnection(ip="127.0.0.1", port=23000)
    except Exception as e:
        logger.error(f"Failed to connect to CoppeliaSim: {e}")
        sys.exit("Connection failed")

    # Initialize components
    try:
        logger.info("Initializing robot controller...")
        robot = RobotController(zmq_connection)

        logger.info("Initializing keyboard controller...")
        keyboard = KeyboardController()  # Using your keyboard_input.py module

        logger.info("Initializing Q-learning agent...")
        agent = GazeJoystickAgent()
        
        logger.info("Initializing calibration agent...")
        calibration_agent = CalibrationAgent()
        
        logger.info("Initializing object proximity cache...")
        proximity_cache = ObjectProximityCache()

    except Exception as e:
        logger.error(f"Initialization error: {e}")
        zmq_connection.disconnect()
        sys.exit("Initialization failed")

    # Start threads
    threads = []
    
    gaze_producer = threading.Thread(target=gaze_producer_thread, daemon=True)
    gaze_consumer = threading.Thread(target=gaze_consumer_thread, args=(calibration_agent, proximity_cache), daemon=True)
    object_detector = threading.Thread(target=object_detection_thread, args=(zmq_connection,), daemon=True)
    
    threads.extend([gaze_producer, gaze_consumer, object_detector])
    
    for thread in threads:
        thread.start()

    logger.info("System initialized. Starting main loop...")
    
    # Main loop variables
    calibration_mode = True  # Start in calibration mode
    selected_obj_id = None
    last_action_time = 0
    action_cooldown = 1.0  # Time between actions in seconds
    window_name = "Eye Gaze Control System"
    
    # Create named window and set properties
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    # Main control loop
    try:
        while shared_state["running"]:
            # Process latest gaze data
            current_data = None
            while not processed_gaze_queue.empty():
                current_data = processed_gaze_queue.get()
                processed_gaze_queue.task_done()
            
            # Get keyboard input using your keyboard_input module
            x_axis, y_axis, keyboard_running = keyboard.update()
            if not keyboard_running:
                shared_state["running"] = False
                break
            
            # Handle keyboard shortcuts using OpenCV key checks
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                shared_state["running"] = False
                break
            elif key == ord('c'):
                calibration_agent.reset()
                calibration_mode = True
                logger.info("Entering calibration mode")
            
            if not current_data:
                # Display a blank frame if no data yet
                if not cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE):
                    # Window closed
                    shared_state["running"] = False
                    break
                    
                time.sleep(0.01)
                continue
                
            # Extract data
            current_time = time.time()
            current_gaze_data = current_data["gaze_data"]
            current_frame = current_data["frame"]
            gazed_object = current_data["gazed_object"]
            
            # Handle calibration mode
            if calibration_mode:
                if current_gaze_data and current_gaze_data.get("pupils_located", False):
                    calibration_complete = calibration_agent.add_sample(current_gaze_data)
                    if calibration_complete:
                        calibration_mode = False
                        logger.info("Calibration completed")
                
                if current_frame is not None:
                    # Draw calibration UI
                    display_frame = draw_ui_overlay(current_frame.copy(), current_gaze_data, 
                                                  gazed_object, shared_state["objects"], 
                                                  agent, calibration_agent)
                    cv2.imshow(window_name, display_frame)
            
            # Normal operation mode
            else:
                # Update agent with gazed object
                if gazed_object:
                    agent.update_gaze_history(gazed_object["id"], current_time)
                    logger.debug(f"Looking at: {gazed_object['name']}")
                
                # Process joystick/keyboard movement for actions
                if (abs(x_axis) > 0.2 or abs(y_axis) > 0.2) and (current_time - last_action_time > action_cooldown):
                    joystick_direction = agent.get_joystick_direction(x_axis, y_axis)
                    
                    if joystick_direction is not None:
                        # Get objects thread-safely
                        with state_lock:
                            objects = shared_state["objects"]
                        
                        selected_obj_id = agent.get_action(joystick_direction, current_time, objects)
                        
                        if selected_obj_id is not None and selected_obj_id < len(objects):
                            selected_object = objects[selected_obj_id]
                            logger.info(f"Selected object: {selected_object['name']}")
                            
                            # Execute robot movement
                            success = robot.move_to_object(selected_object["handle"])
                            last_action_time = current_time
                            
                            if success:
                                # Calculate reward based on recent gazes
                                recent_gazes = [g["object_id"] for g in agent.gaze_history[-5:]]
                                reward = 1.0 if selected_obj_id in recent_gazes else -0.2
                                
                                # Update Q-table
                                agent.update_q_table(selected_obj_id, joystick_direction, reward)
                                logger.info(f"Updated Q-values with reward: {reward}")
                
                # Draw UI
                if current_frame is not None:
                    display_frame = draw_ui_overlay(current_frame.copy(), current_gaze_data, 
                                                  gazed_object, shared_state["objects"], 
                                                  agent, calibration_agent)
                    cv2.imshow(window_name, display_frame)
    
    except Exception as e:
        logger.error(f"Main loop error: {e}")
        logger.error(traceback.format_exc())
    
    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        keyboard.close()
        shared_state["running"] = False
        
        for thread in threads:
            thread.join(timeout=1.0)
        
        # Cleanup remaining resources
        close_gaze_tracker()
        zmq_connection.disconnect()
        cv2.destroyAllWindows()
        logger.info("System shutdown complete")

if __name__ == "__main__":
    main()