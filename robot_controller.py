# Refactored `robot_controller.py` to use updated ZMQConnection API
import time

class RobotController:
    def __init__(self, zmq_connection):
        self.zmq = zmq_connection
        self.initialized = False
        self.joint_handles = []
        self.target_handle = None

        # Initialize robot
        self.initialize_robot()

    def initialize_robot(self):
        try:
            joint_names = [
                '/my_cobot/joint1_respondable',
                '/my_cobot/joint2_respondable',
                '/my_cobot/joint3_respondable',
                '/my_cobot/joint4_respondable',
                '/my_cobot/joint5_respondable',
                '/my_cobot/joint6_respondable'
            ]

            self.joint_handles = []
            for joint_name in joint_names:
                response = self.zmq.get_object_handle(joint_name)
                if response.get('returnCode') == 0:
                    self.joint_handles.append(response.get('handle'))
                    print(f"Got handle for {joint_name}")
                else:
                    print(f"Failed to get handle for {joint_name}: {response.get('error', 'Unknown error')}")

            response = self.zmq.get_object_handle('target')
            if response.get('returnCode') == 0:
                self.target_handle = response.get('handle')
                print("Target handle acquired")
            else:
                print(f"Failed to get target handle: {response.get('error', 'Unknown error')}")

            if len(self.joint_handles) == len(joint_names) and self.target_handle is not None:
                self.initialized = True
                print("Robot initialized successfully")
            else:
                print("Robot initialization incomplete")

        except Exception as e:
            print(f"Error initializing robot: {e}")

    def move_to_position(self, position):
        if not self.initialized or self.target_handle is None:
            print("Robot not initialized properly")
            return False

        ret = self.zmq.set_object_position(self.target_handle, -1, position)
        return ret == 0

    def move_to_object(self, object_handle):
        if not self.initialized:
            print("Robot not initialized properly")
            return False

        try:
            response = self.zmq.get_object_position(object_handle, -1)
            if response.get("returnCode") != 0:
                print("Failed to get object position")
                return False

            position = response.get("position")

            pre_grasp = position.copy()
            pre_grasp[2] += 0.1

            print("Moving to pre-grasp position...")
            if not self.move_to_position(pre_grasp):
                print("Failed to move to pre-grasp position")
                return False
            time.sleep(1.5)

            print("Moving to grasp position...")
            if not self.move_to_position(position):
                print("Failed to move to grasp position")
                return False
            time.sleep(1.0)

            print("Grasping object...")
            time.sleep(0.5)

            lift_pos = position.copy()
            lift_pos[2] += 0.2

            print("Lifting object...")
            if not self.move_to_position(lift_pos):
                print("Failed to lift object")
                return False
            time.sleep(1.0)

            print("Object manipulation completed successfully")
            return True

        except Exception as e:
            print(f"Error in move_to_object: {e}")
            return False

    def get_joint_positions(self):
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