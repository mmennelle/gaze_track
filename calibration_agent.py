# calibration_agent.py
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

class CalibrationAgent:
    def __init__(self):
        self.gaze_ratios = []  # (h_ratio, v_ratio)
        self.screen_coords = []  # (x, y)
        self.model_x = LinearRegression()
        self.model_y = LinearRegression()
        self.calibrated = False

    def add_calibration_point(self, h_ratio, v_ratio, screen_x, screen_y):
        self.gaze_ratios.append([h_ratio, v_ratio])
        self.screen_coords.append([screen_x, screen_y])

    def train(self):
        if len(self.gaze_ratios) >= 5:  # Minimum data points
            X = np.array(self.gaze_ratios)
            y = np.array(self.screen_coords)
            self.model_x.fit(X, y[:, 0])
            self.model_y.fit(X, y[:, 1])
            self.calibrated = True
            print("[Calibration] Calibration completed.")
        else:
            print("[Calibration] Not enough data points to calibrate.")

    def predict(self, h_ratio, v_ratio):
        if not self.calibrated:
            raise RuntimeError("Calibration model not trained.")

        input_vec = np.array([[h_ratio, v_ratio]])
        x = self.model_x.predict(input_vec)[0]
        y = self.model_y.predict(input_vec)[0]
        return x, y

    def save(self, path="calibration_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump((self.model_x, self.model_y), f)

    def load(self, path="calibration_model.pkl"):
        with open(path, "rb") as f:
            self.model_x, self.model_y = pickle.load(f)
            self.calibrated = True
            print("[Calibration] Loaded calibration model from file.")
