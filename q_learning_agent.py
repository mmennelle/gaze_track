# q_learning_agent.py
import numpy as np
import time

class GazeJoystickAgent:
    def __init__(self, max_objects=10, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.max_objects = max_objects
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Q-table: objects x joystick_directions (8 directions)
        self.q_table = np.zeros((max_objects, 8))
        
        # Gaze history
        self.gaze_history = []
        self.max_history_size = 50
        
        # Time window for correlation (in seconds)
        self.time_window = 3.0  # Increased from 2.0 for better correlation
        
        # Parameter for probability formula
        self.k = 1.0
        
        # Store probabilities for visualization
        self.last_probabilities = []
        
        # Minimum gaze time required to consider an object
        self.min_gaze_duration = 0.5
        
    def update_gaze_history(self, object_id, timestamp):
        """Add a gazed object to history"""
        self.gaze_history.append({
            "object_id": object_id,
            "timestamp": timestamp
        })
        
        # Keep history within size limit
        if len(self.gaze_history) > self.max_history_size:
            self.gaze_history.pop(0)
    
    def get_joystick_direction(self, x_axis, y_axis):
        """Convert joystick/keyboard axes to discrete direction"""
        if abs(x_axis) < 0.2 and abs(y_axis) < 0.2:
            return None  # No significant movement
            
        if abs(x_axis) < 0.2:
            if y_axis < 0:
                return 0  # Up
            else:
                return 2  # Down
        elif abs(y_axis) < 0.2:
            if x_axis > 0:
                return 1  # Right
            else:
                return 3  # Left
        else:
            if x_axis > 0 and y_axis < 0:
                return 4  # Up-right
            elif x_axis < 0 and y_axis < 0:
                return 5  # Up-left
            elif x_axis > 0 and y_axis > 0:
                return 6  # Down-right
            else:
                return 7  # Down-left
    
    def probability_model(self, obj_id, joystick_direction, time_diff):
        """Apply custom probability formula: P(x|(e)ye,(j)oystick) = 1/1+e^k(e^tj) where x=e^j"""
        # Using the provided formula
        exponent = self.k * (np.exp(time_diff * joystick_direction))
        probability = 1.0 / (1.0 + np.exp(exponent))
        return probability
    
    def get_gaze_duration(self, obj_id, current_time):
        """Calculate how long an object has been gazed at"""
        gaze_times = []
        for gaze in reversed(self.gaze_history):
            if gaze["object_id"] == obj_id and (current_time - gaze["timestamp"]) <= self.time_window:
                gaze_times.append(gaze["timestamp"])
        
        if not gaze_times:
            return 0
        
        # Calculate continuous gaze duration
        gaze_times.sort()
        duration = 0
        last_time = gaze_times[0]
        
        for time in gaze_times[1:]:
            if time - last_time < 0.25:  # Consider continuous if less than 0.25s gap
                duration += time - last_time
            last_time = time
        
        return duration
    
    def get_action(self, joystick_direction, current_time, objects):
        """Determine which object to manipulate based on gaze history and Q-values"""
        print(f"[Agent] get_action called with joystick_direction={joystick_direction}, num_objects={len(objects)}")
        
        if joystick_direction is None or not objects:
            print(f"[Agent] No action: joystick_direction is None or no objects")
            return None
            
        # Look for recently gazed objects within time window
        recent_objects = {}
        gaze_durations = {}
        
        for gaze_point in reversed(self.gaze_history):
            time_diff = current_time - gaze_point["timestamp"]
            
            if time_diff <= self.time_window:
                obj_id = gaze_point["object_id"]
                if obj_id not in recent_objects or time_diff < recent_objects[obj_id]:
                    recent_objects[obj_id] = time_diff
                    gaze_durations[obj_id] = self.get_gaze_duration(obj_id, current_time)
        
        print(f"[Agent] Recent objects: {recent_objects}")
        print(f"[Agent] Gaze durations: {gaze_durations}")
        
        # Filter out objects that haven't been gazed at long enough
        valid_objects = {obj_id: time_diff for obj_id, time_diff in recent_objects.items() 
                        if gaze_durations.get(obj_id, 0) >= self.min_gaze_duration}
        
        if not valid_objects:
            print(f"[Agent] No objects have been gazed at long enough")
            return None
        
        # Calculate probabilities for each object
        probabilities = []
        
        for obj_id in range(min(len(objects), self.max_objects)):
            q_value = self.q_table[obj_id, joystick_direction]
            
            if obj_id in valid_objects:
                time_diff = valid_objects[obj_id]
                duration = gaze_durations[obj_id]
                
                # Apply probability model
                p = self.probability_model(obj_id, joystick_direction, time_diff)
                
                # Weight by gaze duration
                p *= min(duration / 2.0, 1.0)  # Normalize duration effect
                
                # Combine with Q-value
                p = 0.7 * p + 0.3 * q_value
            else:
                p = 0  # Zero probability for objects not recently gazed at
                
            probabilities.append(p)
        
        print(f"[Agent] Probabilities: {probabilities}")
        
        # Store for visualization
        self.last_probabilities = probabilities
        
        # Select action
        if max(probabilities) == 0:
            print(f"[Agent] All probabilities are zero")
            return None
            
        # Exploration vs exploitation
        if np.random.random() < self.exploration_rate:
            # Only explore among valid objects
            valid_indices = [i for i, p in enumerate(probabilities) if p > 0]
            if valid_indices:
                action = np.random.choice(valid_indices)
                print(f"[Agent] Exploration: randomly selected object {action}")
                return action
            else:
                return None
        else:
            action = np.argmax(probabilities)
            if probabilities[action] > 0:
                print(f"[Agent] Exploitation: selected object {action} with highest probability")
                return action
            else:
                print(f"[Agent] No valid object to select")
                return None
    
    def update_q_table(self, obj_id, joystick_direction, reward):
        """Update Q-value for the selected object and action"""
        if obj_id < self.q_table.shape[0]:
            self.q_table[obj_id, joystick_direction] += self.learning_rate * reward