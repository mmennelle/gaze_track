import time
import random
import numpy as np
import matplotlib.pyplot as plt

from q_learning_agent import GazeJoystickAgent
from gaze_duration_tracker import GazeDurationTracker

PLOT_Q_TABLE = True


objects = [{"id": i, "name": f"target[{i}]", "position": (random.random(), random.random(), 0)} for i in range(50)]

agent = GazeJoystickAgent()
duration_tracker = GazeDurationTracker()

# ep range
for episode in range(20):
    print(f"\n=== Episode {episode + 1} ===")

   
# object 2
    for _ in range(15):
        duration_tracker.update(2)
        agent.update_gaze_history(2, time.time())
        time.sleep(0.05)

    # object 3
    for _ in range(4):
        duration_tracker.update(3)
        agent.update_gaze_history(3, time.time())
        time.sleep(0.05)

    # joystick press right... i think
    x_axis, y_axis = 0.0, 1.0
    joystick_dir = agent.get_joystick_direction(x_axis, y_axis)

    durations = {obj['id']: duration_tracker.get_duration(obj['id']) for obj in objects}
    chosen_id = agent.get_action(joystick_dir, time.time(), objects, durations)

    if chosen_id is not None:
        print(f"Selected object: {objects[chosen_id]['name']}")
        old_q = agent.q_table[chosen_id, joystick_dir]
        print(f"Q-value before update: {old_q:.3f}")

      
        recent_gazes = [g["object_id"] for g in agent.gaze_history[-5:]]
        reward = 1.0 if chosen_id in recent_gazes else -0.2

        agent.update_q_table(chosen_id, joystick_dir, reward)
        new_q = agent.q_table[chosen_id, joystick_dir]
        print(f"Q-value after update:  {new_q:.3f}  (reward: {reward})")
    else:
        print("No object selected.")

    duration_tracker.reset_all()
    time.sleep(1)

if PLOT_Q_TABLE:
    plt.imshow(agent.q_table, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Q-value')
    plt.title('Q-table Heatmap (objects × directions)')
    plt.xlabel('Joystick Direction Index')
    plt.ylabel('Object ID')
    plt.xticks(np.arange(8))
    plt.yticks(np.arange(agent.q_table.shape[0]))
    plt.grid(False)
    plt.tight_layout()
    plt.show()
