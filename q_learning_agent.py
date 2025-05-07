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
        
        # User calibration data
        self.is_calibrated = False
        self.calibration_data = []
        
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
                if obj_id < len(objects):  # Make sure object exists in current scene
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
        probabilities = [0] * len(objects)  # Initialize with zeros for all objects
        
        for obj_id in valid_objects:
            if obj_id < len(objects):  # Safety check
                q_value = self.q_table[obj_id, joystick_direction]
                time_diff = valid_objects[obj_id]
                duration = gaze_durations[obj_id]
                
                # Apply probability model
                p = self.probability_model(obj_id, joystick_direction, time_diff)
                
                # Weight by gaze duration
                p *= min(duration / 2.0, 1.0)  # Normalize duration effect
                
                # Combine with Q-value
                p = 0.7 * p + 0.3 * q_value
                
                probabilities[obj_id] = p
        
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
            # Standard Q-learning update formula: Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
            # Simplified since we don't model state transitions explicitly
            self.q_table[obj_id, joystick_direction] += self.learning_rate * reward
            
            # Optional: normalize Q-values for this object to prevent unbounded growth
            max_q = np.max(self.q_table[obj_id])
            if max_q > 5.0:  # Arbitrary threshold
                self.q_table[obj_id] = self.q_table[obj_id] / max_q * 5.0
    
    def calibrate_for_user(self, duration=30.0, joystick_fn=None):
        """Run a calibration routine for a new user
        
        Args:
            duration: Calibration duration in seconds
            joystick_fn: Function that returns current joystick direction
        """
        self.calibration_data = []
        calibration_start = time.time()
        
        print("Starting user calibration. Please look at objects and press arrow keys.")
        print(f"Calibration will run for {duration} seconds.")
        
        # If no joystick function provided, we'll collect data passively
        if joystick_fn is None:
            time.sleep(duration)
            print("Passive calibration complete.")
            return
        
        # Store calibration data during the routine
        last_action_time = 0
        action_cooldown = 0.5  # More frequent actions during calibration
        
        while time.time() - calibration_start < duration:
            current_time = time.time()
            
            # Get joystick input
            joystick_direction = joystick_fn()
            
            # Record data when joystick is moved
            if joystick_direction is not None and current_time - last_action_time > action_cooldown:
                last_action_time = current_time
                
                # Find most recent gaze
                if len(self.gaze_history) > 0:
                    latest_gaze = self.gaze_history[-1]
                    time_diff = current_time - latest_gaze["timestamp"]
                    
                    # Only use recent gazes
                    if time_diff <= self.time_window:
                        self.calibration_data.append({
                            "gaze_obj_id": latest_gaze["object_id"],
                            "joystick_direction": joystick_direction,
                            "time_diff": time_diff
                        })
                        print(f"Recorded calibration point: object {latest_gaze['object_id']}, "
                              f"direction {joystick_direction}, time diff {time_diff:.2f}s")
            
            # Brief pause to prevent CPU hogging
            time.sleep(0.01)
        
        # Adjust parameters based on collected data
        if self.calibration_data:
            self._adjust_parameters_from_calibration()
            self.is_calibrated = True
        else:
            print("Calibration failed - no usable data collected.")
    
    def _adjust_parameters_from_calibration(self):
        """Adjust agent parameters based on calibration data"""
        if not self.calibration_data:
            return
        
        # Extract data
        time_diffs = [entry["time_diff"] for entry in self.calibration_data]
        directions = [entry["joystick_direction"] for entry in self.calibration_data]
        object_ids = [entry["gaze_obj_id"] for entry in self.calibration_data]
        
        # Calculate metrics
        avg_time_diff = sum(time_diffs) / len(time_diffs)
        max_time_diff = max(time_diffs)
        
        # Calculate standard deviation if we have enough data
        if len(time_diffs) > 1:
            time_diff_std = np.std(time_diffs)
        else:
            time_diff_std = 0.5  # Default value
        
        # Adjust time window based on observed reaction times
        self.time_window = min(5.0, max(1.5, avg_time_diff * 3))
        
        # Adjust minimum gaze duration based on observed patterns
        self.min_gaze_duration = min(1.0, max(0.3, avg_time_diff * 0.4))
        
        # Adjust k parameter for probability model
        # Lower k makes the probability decay slower with time
        self.k = max(0.1, min(2.0, 1.0 / (avg_time_diff + 0.1)))
        
        # Adjust exploration rate based on consistency
        consistency = 1.0 - (time_diff_std / (avg_time_diff + 0.1))
        self.exploration_rate = max(0.05, min(0.5, 0.3 * (1 - consistency)))
        
        print("\nCalibration complete. Adjusted parameters:")
        print(f"  Time window: {self.time_window:.2f}s")
        print(f"  Min gaze duration: {self.min_gaze_duration:.2f}s")
        print(f"  Probability model k: {self.k:.2f}")
        print(f"  Exploration rate: {self.exploration_rate:.2f}")
    
    def adjust_learning_parameters(self, performance_metric):
        """Dynamically adjust learning parameters based on performance
        
        Args:
            performance_metric: A value between 0 and 1 representing system performance
        """
        # performance_metric should be between 0 and 1
        # Higher values mean better performance
        
        # Avoid adjusting if user calibration was recently performed
        if self.is_calibrated and len(self.calibration_data) > 10:
            return
        
        # Adjust exploration rate - less exploration as system performs better
        self.exploration_rate = max(0.05, min(0.5, 0.5 * (1 - performance_metric)))
        
        # Adjust learning rate - more stable as system performs better
        self.learning_rate = max(0.05, min(0.3, 0.3 * (1 - performance_metric) + 0.05))
        
        # Adjust time window - narrower as system performs better
        self.time_window = max(1.5, min(5.0, 5.0 * (1 - performance_metric) + 1.5))
        
        print(f"Adjusted parameters based on performance ({performance_metric:.2f}):")
        print(f"  Exploration rate: {self.exploration_rate:.2f}")
        print(f"  Learning rate: {self.learning_rate:.2f}")
        print(f"  Time window: {self.time_window:.2f}s")
    
    def reset_q_table(self):
        """Reset the Q-table to initial state"""
        self.q_table = np.zeros((self.max_objects, 8))
        print("Q-table has been reset to initial state.")
    
    def save_model(self, filename="gaze_joystick_model.npz"):
        """Save the agent's model to file"""
        np.savez(filename, 
                 q_table=self.q_table, 
                 k=self.k,
                 time_window=self.time_window,
                 min_gaze_duration=self.min_gaze_duration,
                 learning_rate=self.learning_rate,
                 exploration_rate=self.exploration_rate)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename="gaze_joystick_model.npz"):
        """Load the agent's model from file"""
        try:
            data = np.load(filename)
            self.q_table = data['q_table']
            self.k = float(data['k'])
            self.time_window = float(data['time_window'])
            self.min_gaze_duration = float(data['min_gaze_duration'])
            self.learning_rate = float(data['learning_rate'])
            self.exploration_rate = float(data['exploration_rate'])
            print(f"Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
            
    def get_object_stats(self):
        """Get statistics about objects for visualization"""
        object_stats = []
        
        # Count number of gazes per object in recent history
        gaze_counts = {}
        for gaze in self.gaze_history:
            obj_id = gaze["object_id"]
            if obj_id in gaze_counts:
                gaze_counts[obj_id] += 1
            else:
                gaze_counts[obj_id] = 1
        
        # Calculate average Q-value per object
        for obj_id in range(self.max_objects):
            if obj_id in gaze_counts:
                avg_q = np.mean(self.q_table[obj_id])
                max_q = np.max(self.q_table[obj_id])
                preferred_direction = np.argmax(self.q_table[obj_id])
                
                object_stats.append({
                    "id": obj_id,
                    "gaze_count": gaze_counts[obj_id],
                    "avg_q_value": avg_q,
                    "max_q_value": max_q,
                    "preferred_direction": preferred_direction
                })
        
        return object_stats