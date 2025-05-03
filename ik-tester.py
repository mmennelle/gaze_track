# robot_ik_test.py
import time
import math
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

def test_robot_kinematics():
    print("Testing Robot Kinematics...")
    
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
            print(f"Got handle for {name}")
        
        # Get target
        target_handle = sim.getObject('/target')
        target_pos = sim.getObjectPosition(target_handle, -1)
        print(f"Target position: {target_pos}")
        
        # Test 1: Move joints to zero position
        print("\n=== Test 1: Zero Position ===")
        zero_positions = [0, 0, 0, 0, 0]
        for i, (handle, pos) in enumerate(zip(joint_handles, zero_positions)):
            sim.setJointPosition(handle, pos)
        time.sleep(2)
        
        # Test 2: Move joints to reach different positions
        print("\n=== Test 2: Testing Different Configurations ===")
        configurations = [
            [0, -1.57, 1.57, 0, 0],      # Configuration 1: Bent arm
            [0.785, -0.785, 0.785, 0, 0], # Configuration 2: 45-degree angles
            [0, -1.0, 1.0, 0, 0],         # Configuration 3: Moderate bend
            [0, -0.5, 0.5, 0, 0],         # Configuration 4: Slight bend
        ]
        
        for i, config in enumerate(configurations):
            print(f"Testing configuration {i+1}: {config}")
            for j, (handle, pos) in enumerate(zip(joint_handles, config)):
                sim.setJointPosition(handle, pos)
            time.sleep(2)
        
        # Test 3: Better IK calculation
        print("\n=== Test 3: Improved IK Calculation ===")
        
        def calculate_ik_improved(target_position):
            x, y, z = target_position
            
            # Base rotation (joint 1)
            joint1 = math.atan2(y, x)
            
            # Distance in XY plane
            xy_distance = math.sqrt(x**2 + y**2)
            
            # Robot arm parameters (adjust these based on your robot)
            L1 = 0.2  # Length from base to joint 2
            L2 = 0.2  # Length from joint 2 to joint 3
            L3 = 0.2  # Length from joint 3 to end effector
            
            # Target height relative to base
            target_height = z - 0.1  # Assuming base is at 0.1m height
            
            # Calculate reach
            reach = math.sqrt(xy_distance**2 + target_height**2)
            
            # Check if target is reachable
            if reach > (L2 + L3):
                print(f"Target out of reach! Reach: {reach}, Max: {L2 + L3}")
                reach = L2 + L3  # Limit to maximum reach
            
            # Calculate joint 2 and 3 using inverse kinematics
            # This is a 2-link planar arm IK solution
            cos_angle3 = (reach**2 - L2**2 - L3**2) / (2 * L2 * L3)
            cos_angle3 = max(-1, min(1, cos_angle3))  # Clamp to valid range
            
            joint3 = math.acos(cos_angle3)
            
            # Calculate joint 2
            alpha = math.atan2(target_height, xy_distance)
            beta = math.atan2(L3 * math.sin(joint3), L2 + L3 * math.cos(joint3))
            joint2 = alpha - beta
            
            # Adjust for robot's coordinate system
            joint2 = -joint2  # Invert if needed
            joint3 = -joint3  # Invert if needed
            
            # Wrist joints (simple defaults)
            joint4 = 0
            joint5 = 0
            
            return [joint1, joint2, joint3, joint4, joint5]
        
        # Test the improved IK
        calculated_joints = calculate_ik_improved(target_pos)
        print(f"Calculated joints for target: {calculated_joints}")
        
        for i, (handle, pos) in enumerate(zip(joint_handles, calculated_joints)):
            sim.setJointPosition(handle, pos)
        time.sleep(2)
        
        # Test 4: Try to understand robot structure
        print("\n=== Test 4: Understanding Robot Structure ===")
        
        # Get positions of each link
        try:
            base_pos = sim.getObjectPosition(sim.getObject('/my_cobot'), -1)
            print(f"Base position: {base_pos}")
        except:
            print("Could not find base")
            
        # Try to find end effector
        possible_tips = ['/my_cobot/connection', '/my_cobot/tip', '/my_cobot/end_effector', '/my_cobot/link6']
        for tip_name in possible_tips:
            try:
                tip_handle = sim.getObject(tip_name)
                tip_pos = sim.getObjectPosition(tip_handle, -1)
                print(f"Found {tip_name} at position: {tip_pos}")
            except:
                pass
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_robot_kinematics()