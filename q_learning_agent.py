import numpy as np
import time

class GazeJoystickAgent:
    def __init__(self, max_objects=10, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.max_objects = max_objects
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        self.q_table = np.zeros((max_objects, 8))
        self.gaze_history = []
        self.max_history_size = 50
        self.time_window = 3.0
        self.k = 1.0
        self.last_probabilities = []
        self.min_gaze_duration = 0.2  # Reduced from 0.5 to allow faster testing

    def update_gaze_history(self, object_id, timestamp):
        self.gaze_history.append({"object_id": object_id, "timestamp": timestamp})
        if len(self.gaze_history) > self.max_history_size:
            self.gaze_history.pop(0)

    def get_joystick_direction(self, x_axis, y_axis):
        if abs(x_axis) < 0.2 and abs(y_axis) < 0.2:
            return None
        if abs(x_axis) < 0.2:
            return 0 if y_axis < 0 else 2
        elif abs(y_axis) < 0.2:
            return 1 if x_axis > 0 else 3
        else:
            if x_axis > 0 and y_axis < 0:
                return 4
            elif x_axis < 0 and y_axis < 0:
                return 5
            elif x_axis > 0 and y_axis > 0:
                return 6
            else:
                return 7

    def probability_model(self, obj_id, joystick_direction, time_diff):
        exponent = self.k * (np.exp(time_diff * joystick_direction))
        probability = 1.0 / (1.0 + np.exp(exponent))
        return probability

    def get_action(self, joystick_direction, current_time, objects, gaze_durations):
        print(f"[Agent] get_action called with joystick_direction={joystick_direction}, num_objects={len(objects)}")
        print("Durations:", {k: round(v, 2) for k, v in gaze_durations.items()})  # Log gaze durations

        if joystick_direction is None or not objects:
            return None

        recent_objects = {}
        for gaze_point in reversed(self.gaze_history):
            time_diff = current_time - gaze_point["timestamp"]
            if time_diff <= self.time_window:
                obj_id = gaze_point["object_id"]
                if obj_id not in recent_objects or time_diff < recent_objects[obj_id]:
                    recent_objects[obj_id] = time_diff

        valid_objects = {obj_id: time_diff for obj_id, time_diff in recent_objects.items()
                         if gaze_durations.get(obj_id, 0) >= self.min_gaze_duration}

        if not valid_objects:
            print("[Agent] No objects have been gazed at long enough")
            return None

        probabilities = []
        for obj_id in range(min(len(objects), self.max_objects)):
            q_value = self.q_table[obj_id, joystick_direction]
            if obj_id in valid_objects:
                time_diff = valid_objects[obj_id]
                duration = gaze_durations.get(obj_id, 0)
                p = self.probability_model(obj_id, joystick_direction, time_diff)
                p *= min(duration / 2.0, 1.0)
                p = 0.7 * p + 0.3 * q_value
            else:
                p = 0
            probabilities.append(p)

        self.last_probabilities = probabilities
        print("Probabilities:", [f"{p:.2f}" for p in probabilities])  # Log probabilities

        if max(probabilities) == 0:
            return None

        if np.random.random() < self.exploration_rate:
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
                return None

    def update_q_table(self, obj_id, joystick_direction, reward):
        if obj_id < self.q_table.shape[0]:
            self.q_table[obj_id, joystick_direction] += self.learning_rate * reward
