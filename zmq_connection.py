# Refactored `zmq_connection.py`
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class ZMQConnection:
    def __init__(self, ip="127.0.0.1", port=23000):
        self.client = RemoteAPIClient(ip, port)
        self.sim = self.client.getObject('sim')

        print(f"Connected to CoppeliaSim ZMQ server at tcp://{ip}:{port}")

    def get_object_handle(self, object_name):
        try:
            handle = self.sim.getObject(object_name)
            return {"returnCode": 0, "handle": handle}
        except Exception as e:
            return {"returnCode": -1, "error": str(e)}

    def get_object_position(self, object_handle, reference_frame=-1):
        try:
            pos = self.sim.getObjectPosition(object_handle, reference_frame)
            return {"returnCode": 0, "position": pos}
        except Exception as e:
            return {"returnCode": -1, "error": str(e)}

    def set_object_position(self, object_handle, reference_frame, position):
        try:
            self.sim.setObjectPosition(object_handle, reference_frame, position)
            return 0
        except Exception as e:
            print(f"Error setting object position: {e}")
            return -1

    def get_joint_position(self, joint_handle):
        try:
            pos = self.sim.getJointPosition(joint_handle)
            return 0, pos
        except Exception as e:
            print(f"Error getting joint position: {e}")
            return -1, None

    def set_joint_position(self, joint_handle, position):
        try:
            self.sim.setJointTargetPosition(joint_handle, position)
            return 0
        except Exception as e:
            print(f"Error setting joint position: {e}")
            return -1

    def set_joint_velocity(self, joint_handle, velocity):
        try:
            self.sim.setJointTargetVelocity(joint_handle, velocity)
            return 0
        except Exception as e:
            print(f"Error setting joint velocity: {e}")
            return -1

    def disconnect(self):
        print("Disconnected from ZMQ server")  # Nothing to disconnect in new client
