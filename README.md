# FROM LOOK TO LIFT: Gaze-Based Robot Control System

This project uses eye gaze and keyboard joystick input to interactively select and manipulate objects in a CoppeliaSim scene. Gaze tracking is performed using a standard webcam and the [GazeTracking](https://github.com/antoinelame/GazeTracking) library, enhanced with smoothing, filtering, and a calibration system that learns from the user’s natural eye movements.

## 🔧 Features

- **Webcam-based gaze tracking** (no special hardware)
- **Polynomial calibration** based on user gaze → scene target mapping
- **Joystick + gaze fusion** using Q-learning
- **Real-time object selection and robot manipulation**
- **Scene object detection** via ZMQ interface to CoppeliaSim
- **Custom calibration flow** with visual feedback and overlays
- **Keyboard fallback control**
- **Duration-based gaze filtering** to reduce noise
- **Q-table visualization (via `test_it.py`)

## 📁 Project Structure

```
gaze_track/
├── main.py                   # Main entry point with full system loop
├── gaze_tracker.py          # Gaze tracking + filtering + calibration model
├── calibration_module.py    # Visual and logical calibration manager
├── q_learning_agent.py      # Q-learning agent for fusing joystick/gaze
├── keyboard_input.py        # Keyboard joystick input via pygame
├── gaze_duration_tracker.py # Tracks how long each object is gazed at
├── robot_controller.py      # Interacts with robot arm in CoppeliaSim
├── zmq_connection.py        # Interface to CoppeliaSim over ZMQ
├── test_it.py               # Standalone testing script for Q-learning
├── requirements.txt         # Project dependencies
```

## 🚀 Requirements

- Python 3.8+
- Recommended OS: Windows (uses `win32gui` to control OpenCV windows)

### Install Dependencies

```bash
pip install -r requirements.txt
```

#### `requirements.txt`

```
cbor==1.0.0
cbor2==5.6.5
contourpy==1.3.1
coppeliasim_zmqremoteapi_client==2.0.4
cycler==0.12.1
distlib==0.3.9
filelock==3.18.0
fonttools==4.57.0
joblib==1.4.2
kiwisolver==1.4.8
matplotlib==3.10.1
msgpack==1.1.0
numpy==2.2.4
packaging==24.2
pillow==11.1.0
platformdirs==4.3.7
pyparsing==3.2.3
python-dateutil==2.9.0.post0
pywin32==310
pyzmq==26.4.0
scipy==1.15.2
six==1.17.0
threadpoolctl==3.6.0
virtualenv==20.30.0
```

##Setup

1. Clone the [GazeTracking](https://github.com/antoinelame/GazeTracking) repo and install it or place it in your project directory.
2. Ensure CoppeliaSim is running and the ZMQ Remote API server is listening on port `23000`.
3. Run the main system:

```bash
python main.py
```

Follow the prompt to run gaze calibration. The system will:
- Identify `/target[i]` objects in CoppeliaSim.
- Ask you to look at pulsing markers.
- Train a polynomial calibration model based on your gaze.

## Controls

- `Arrow keys` — Move joystick input direction.
- `C` — Recalibrate gaze tracking.
- `R` — Reset calibration to raw defaults.
- `Q` — Quit the system.

## How It Works

- Gaze is tracked using the webcam with the GazeTracking library.
- Gaze is **filtered** and **smoothed** with outlier rejection.
- A **calibration model** maps the raw gaze to screen-space based on where you actually look.
- Gaze duration is tracked per object.
- A **Q-learning agent** fuses joystick input with gaze duration and recency to select the most likely intended object.
- The robot arm moves to the selected object using a smoothed motion path derived from a CoppeliaSim curve.

## Testing & Debugging

Run `test_it.py` to simulate agent decision-making and visualize the Q-table:

```bash
python test_it.py
```

##  License

This project uses open-source components but does not have a formal license attached yet. Usage is encouraged for research and non-commercial applications.