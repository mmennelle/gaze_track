# test_robot_movement.py
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

def test_robot_movement():
    print("Testing Robot Movement in CoppeliaSim...")
    
    try:
        # Connect to CoppeliaSim
        client = RemoteAPIClient('127.0.0.1', 23000)
        sim = client.getObject('sim')
        print("Connected to CoppeliaSim")
        
        # Check simulation state
        state = sim.getSimulationState()
        print(f"Simulation state: {state} (0=stopped, 1=running, 2=paused)")
        
        # Get handles
        print("\n=== Getting Handles ===")
        joint_names = [
            '/my_cobot/joint2_to_joint1',
            '/my_cobot/joint3_to_joint2',
            '/my_cobot/joint4_to_joint3',
            '/my_cobot/joint5_to_joint4',
            '/my_cobot/joint6_to_joint5'
        ]
        
        joint_handles = []
        for name in joint_names:
            try:
                handle = sim.getObject(name)
                joint_handles.append(handle)
                print(f"Got handle for {name}: {handle}")
            except Exception as e:
                print(f"Error getting {name}: {e}")
        
        # Test 1: Get current joint positions
        print("\n=== Current Joint Positions ===")
        for i, handle in enumerate(joint_handles):
            try:
                position = sim.getJointPosition(handle)
                print(f"Joint {i+1} position: {position}")
            except Exception as e:
                print(f"Error getting position for joint {i+1}: {e}")
        
        # Test 2: Try to move joints
        print("\n=== Testing Joint Movement ===")
        print("Moving joint 1 to position 0.5 radians...")
        try:
            sim.setJointTargetPosition(joint_handles[0], 0.5)
            time.sleep(2)
            new_pos = sim.getJointPosition(joint_handles[0])
            print(f"New position of joint 1: {new_pos}")
        except Exception as e:
            print(f"Error moving joint 1: {e}")
        
        # Test 3: Try different movement approaches
        print("\n=== Testing Different Movement Approaches ===")
        
        # Approach 1: Direct position setting
        print("Approach 1: Direct position setting...")
        try:
            sim.setJointPosition(joint_handles[0], 0.7)
            time.sleep(1)
            pos = sim.getJointPosition(joint_handles[0])
            print(f"Position after direct setting: {pos}")
        except Exception as e:
            print(f"Error with direct position setting: {e}")
        
        # Approach 2: Target velocity
        print("\nApproach 2: Target velocity...")
        try:
            sim.setJointTargetVelocity(joint_handles[0], 0.5)
            time.sleep(1)
            sim.setJointTargetVelocity(joint_handles[0], 0)  # Stop
            pos = sim.getJointPosition(joint_handles[0])
            print(f"Position after velocity control: {pos}")
        except Exception as e:
            print(f"Error with velocity control: {e}")
        
        # Test 4: Check for IK tip or end effector
        print("\n=== Looking for End Effector/Tip ===")
        possible_tip_names = [
            '/my_cobot/tip',
            '/my_cobot/end_effector',
            '/my_cobot/connection',
            '/my_cobot/link6',
            '/my_cobot/gripper'
        ]
        
        for name in possible_tip_names:
            try:
                handle = sim.getObject(name)
                print(f"Found {name}: handle {handle}")
                position = sim.getObjectPosition(handle, -1)
                print(f"  Position: {position}")
            except Exception as e:
                pass  # Silently skip if not found
        
        # Test 5: Get all objects in scene to find the robot structure
        print("\n=== Getting All Objects in Scene ===")
        try:
            # This might not work with all CoppeliaSim versions
            # You might need to manually list objects you know exist
            print("Checking for known robot components...")
            robot_parts = [
                '/my_cobot',
                '/my_cobot/base',
                '/my_cobot/link1',
                '/my_cobot/link2',
                '/my_cobot/link3',
                '/my_cobot/link4',
                '/my_cobot/link5',
                '/my_cobot/link6'
            ]
            
            for part in robot_parts:
                try:
                    handle = sim.getObject(part)
                    print(f"Found {part}")
                except:
                    pass
        except Exception as e:
            print(f"Error listing objects: {e}")
        
        # Test 6: Try moving target instead (what your current code does)
        print("\n=== Testing Target Movement ===")
        try:
            target_handle = sim.getObject('/target')
            current_pos = sim.getObjectPosition(target_handle, -1)
            print(f"Current target position: {current_pos}")
            
            new_pos = current_pos.copy()
            new_pos[2] += 0.1  # Move up by 0.1
            sim.setObjectPosition(target_handle, -1, new_pos)
            time.sleep(1)
            
            final_pos = sim.getObjectPosition(target_handle, -1)
            print(f"New target position: {final_pos}")
        except Exception as e:
            print(f"Error moving target: {e}")
            
    except Exception as e:
        print(f"Connection error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_robot_movement()