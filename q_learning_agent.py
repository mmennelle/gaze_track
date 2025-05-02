import numpy as np
import time

class GazeJoystickAgent:
    def __init__(self, max_objects=10, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3):
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
        self.time_window = 2.0
        
        # Parameter for probability formula
        self.k = 1.0
        
        # Store probabilities for visualization
        self.last_probabilities = []
        
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
    
    def get_action(self, joystick_direction, current_time, objects):
        """Determine which object to manipulate based on gaze history and Q-values"""
        if joystick_direction is None or not objects:
            return None
            
        # Look for recently gazed objects within time window
        recent_objects = {}
        
        for gaze_point in reversed(self.gaze_history):
            time_diff = current_time - gaze_point["timestamp"]
            
            if time_diff <= self.time_window:
                obj_id = gaze_point["object_id"]
                if obj_id not in recent_objects or time_diff < recent_objects[obj_id]:
                    recent_objects[obj_id] = time_diff
        
        # Calculate probabilities for each object
        probabilities = []
        
        for obj_id in range(min(len(objects), self.max_objects)):
            q_value = self.q_table[obj_id, joystick_direction]
            
            if obj_id in recent_objects:
                time_diff = recent_objects[obj_id]
                
                # Apply probability model
                p = self.probability_model(obj_id, joystick_direction, time_diff)
                
                # Combine with Q-value
                p = 0.7 * p + 0.3 * q_value
            else:
                p = 0.1 * q_value  # Lower probability for objects not recently gazed at
                
            probabilities.append(p)
        
        # Store for visualization
        self.last_probabilities = probabilities
        
        # Exploration vs exploitation
        if np.random.random() < self.exploration_rate:
            return np.random.randint(0, len(objects))
        else:
            if probabilities:
                return np.argmax(probabilities)
            return None
    
    def update_q_table(self, obj_id, joystick_direction, reward):
        """Update Q-value for the selected object and action"""
        if obj_id < self.q_table.shape[0]:
            self.q_table[obj_id, joystick_direction] += self.learning_rate * reward