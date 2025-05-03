# calibrate_robot.py
import time
import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

def calibrate_robot():
    print("Robot Calibration Tool")
    print("This will help determine the correct link lengths for IK")
    
    try:
        # Connect to CoppeliaSim
        client = RemoteAPIClient('127.0.0.1', 23000)
        sim = client.getObject('sim')
        print("Connected to CoppeliaSim")
        
        # Get joint handles
        joint_names = [
            '/my_cobot/joint2_to_joint1',
            '/my_cobot/joint3_to_joint2',
            '/my_cobot/joint4_to_joint3',
            '/my_cobot/joint5_to_joint4',
            '/my_cobot/joint6_to_joint5'
        ]
        
        joint_handles = []
        for name in joint_names:
            handle = sim.getObject(name)
            joint_handles.append(handle)
        
        # Test configurations to find link lengths
        print("\n=== Calibration Sequence ===")
        
        # Configuration 1: All joints at 0
        print("Setting all joints to 0...")
        for handle in joint_handles:
            sim.setJointPosition(handle, 0)
        time.sleep(1)
        
        # Configuration 2: Only joint 2 at -90 degrees
        print("Setting joint 2 to -90 degrees...")
        sim.setJointPosition(joint_handles[1], -math.pi/2)
        time.sleep(1)
        
        # Configuration 3: Joint 2 at -90, joint 3 at 90
        print("Setting joint 3 to 90 degrees...")
        sim.setJointPosition(joint_handles[2], math.pi/2)
        time.sleep(1)
        
        # Configuration 4: All joints straight out
        print("Setting all joints for horizontal reach...")
        sim.setJointPosition(joint_handles[1], -math.pi/2)
        sim.setJointPosition(joint_handles[2], 0)
        sim.setJointPosition(joint_handles[3], 0)
        time.sleep(1)
        
        print("\nObserve the robot positions and estimate link lengths.")
        print("Adjust L1, L2, L3, L4 values in robot_controller.py accordingly.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    calibrate_robot()