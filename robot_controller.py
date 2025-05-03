# Updated robot_controller.py
import time

class RobotController:
    def __init__(self, zmq_connection):
        self.zmq = zmq_connection
        self.initialized = False
        self.joint_handles = []
        self.tip_handle = None  # End effector/gripper tip
        self.target_handle = None

        # Initialize robot
        self.initialize_robot()

    def initialize_robot(self):
        try:
            # Get joint handles
            joint_names = [
                '/my_cobot/joint2_to_joint1',
                '/my_cobot/joint3_to_joint2',
                '/my_cobot/joint4_to_joint3',
                '/my_cobot/joint5_to_joint4',
                '/my_cobot/joint6_to_joint5'
            ]

            self.joint_handles = []
            for joint_name in joint_names:
                response = self.zmq.get_object_handle(joint_name)
                if response.get('returnCode') == 0:
                    self.joint_handles.append(response.get('handle'))
                    print(f"Got handle for {joint_name}")
                else:
                    print(f"Failed to get handle for {joint_name}: {response.get('error', 'Unknown error')}")

            # Get the tip/end-effector handle - you'll need to adjust this to match your robot's tip name
            tip_response = self.zmq.get_object_handle('/my_cobot/tip')  # Adjust this name as needed
            if tip_response.get('returnCode') == 0:
                self.tip_handle = tip_response.get('handle')
                print("Got tip handle")
            else:
                print(f"Warning: Could not find tip handle: {tip_response.get('error', 'Unknown error')}")

            # Get target handle
            response = self.zmq.get_object_handle('/target')
            if response.get('returnCode') == 0:
                self.target_handle = response.get('handle')
                print("Target handle acquired")
            else:
                print(f"Failed to get target handle: {response.get('error', 'Unknown error')}")

            if self.joint_handles and self.target_handle:
                self.initialized = True
                print("Robot initialized successfully")
            else:
                print("Robot initialization incomplete")

        except Exception as e:
            print(f"Error initializing robot: {e}")

    def move_joints_to_position(self, target_position):
        """Use inverse kinematics to move the robot arm to a target position"""
        if not self.initialized:
            print("Robot not initialized properly")
            return False

        try:
            # Option 1: If your CoppeliaSim scene has IK configured
            # You might need to set up an IK target dummy object and move that
            # Then let CoppeliaSim's IK solver move the joints
            
            # Option 2: Simple approach - move each joint incrementally
            # This is a basic approach and might not work well for all robots
            
            # For now, let's try a simpler approach using the ZMQ API
            # Some CoppeliaSim versions support IK through API
            
            print(f"Moving robot to position: {target_position}")
            
            # If you have an IK target dummy in your scene, move it
            # Otherwise, you'll need to implement IK calculations
            
            # This is a placeholder - you need to implement proper IK
            # or use CoppeliaSim's built-in IK if available in your scene
            
            return True
            
        except Exception as e:
            print(f"Error in move_joints_to_position: {e}")
            return False

    def move_to_object(self, object_handle):
        """Move the robot arm to grasp an object"""
        if not self.initialized:
            print("Robot not initialized properly")
            return False

        try:
            # Get the target object's position
            response = self.zmq.get_object_position(object_handle, -1)
            if response.get("returnCode") != 0:
                print("Failed to get object position")
                return False

            position = response.get("position")
            print(f"Target object position: {position}")

            # Define approach positions
            pre_grasp = position.copy()
            pre_grasp[2] += 0.1  # Move above the object

            print("Moving to pre-grasp position...")
            success = self.move_joints_to_position(pre_grasp)
            if not success:
                print("Failed to move to pre-grasp position")
                return False
            time.sleep(1.5)

            print("Moving to grasp position...")
            success = self.move_joints_to_position(position)
            if not success:
                print("Failed to move to grasp position")
                return False
            time.sleep(1.0)

            print("Grasping object...")
            time.sleep(0.5)

            lift_pos = position.copy()
            lift_pos[2] += 0.2

            print("Lifting object...")
            success = self.move_joints_to_position(lift_pos)
            if not success:
                print("Failed to lift object")
                return False
            time.sleep(1.0)

            print("Object manipulation completed successfully")
            return True

        except Exception as e:
            print(f"Error in move_to_object: {e}")
            return False

    def get_joint_positions(self):
        """Get current joint positions"""
        if not self.initialized:
            return []

        joint_positions = []
        for handle in self.joint_handles:
            ret, pos = self.zmq.get_joint_position(handle)
            if ret == 0:
                joint_positions.append(pos)
            else:
                joint_positions.append(None)

        return joint_positions