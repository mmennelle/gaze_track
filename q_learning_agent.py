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
    
    # Modify q_learning_agent.py
def get_action(self, joystick_direction, current_time, objects):
    """Optimized version that caches calculations and uses vectorized operations"""
    if joystick_direction is None or not objects:
        return None
    
    # Use numpy arrays for faster computation
    obj_ids = np.array([obj["id"] for obj in objects][:self.max_objects])
    
    # Calculate time differences and durations vectorized
    time_diffs = np.ones(len(obj_ids)) * np.inf
    gaze_durations = np.zeros(len(obj_ids))
    
    recent_time = current_time - self.time_window
    recent_gazes = [g for g in self.gaze_history if g["timestamp"] >= recent_time]
    
    # Create a mapping of object_id to gaze points
    obj_gazes = {}
    for gaze in recent_gazes:
        obj_id = gaze["object_id"]
        if obj_id not in obj_gazes:
            obj_gazes[obj_id] = []
        obj_gazes[obj_id].append(gaze["timestamp"])
    
    # Calculate time diffs and durations
    for obj_id, timestamps in obj_gazes.items():
        if obj_id < len(obj_ids):
            idx = np.where(obj_ids == obj_id)[0]
            if len(idx) > 0:
                time_diffs[idx[0]] = current_time - max(timestamps)
                
                # Calculate duration
                if len(timestamps) > 1:
                    timestamps.sort()
                    duration = 0
                    last_time = timestamps[0]
                    
                    for time in timestamps[1:]:
                        if time - last_time < 0.25:
                            duration += time - last_time
                        last_time = time
                    
                    gaze_durations[idx[0]] = duration
    
    # Find valid objects (gazed long enough)
    valid_mask = gaze_durations >= self.min_gaze_duration
    
    if not np.any(valid_mask):
        return None
    
    # Calculate probabilities using vectorized operations
    q_values = self.q_table[:len(obj_ids), joystick_direction]
    
    # Initialize probabilities
    probabilities = np.zeros(len(obj_ids))
    
    # Calculate only for valid objects
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) > 0:
        # Apply probability model
        exponents = self.k * np.exp(time_diffs[valid_indices] * joystick_direction)
        probs = 1.0 / (1.0 + np.exp(exponents))
        
        # Weight by duration
        duration_weights = np.minimum(gaze_durations[valid_indices] / 2.0, 1.0)
        probs *= duration_weights
        
        # Combine with Q-values
        probs = 0.7 * probs + 0.3 * q_values[valid_indices]
        
        # Assign back to full array
        probabilities[valid_indices] = probs
    
    # Store for visualization
    self.last_probabilities = probabilities.tolist()
    
    if np.max(probabilities) == 0:
        return None
    
    # Exploration vs exploitation
    if np.random.random() < self.exploration_rate:
        valid_indices = np.where(probabilities > 0)[0]
        if len(valid_indices) > 0:
            return np.random.choice(valid_indices)
        return None
    else:
        action = np.argmax(probabilities)
        if probabilities[action] > 0:
            return action
        return None
    def update_q_table(self, obj_id, joystick_direction, reward):
        """Update Q-value for the selected object and action"""
        if obj_id < self.q_table.shape[0]:
            self.q_table[obj_id, joystick_direction] += self.learning_rate * reward