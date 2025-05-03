# test_robot_controller.py
from zmq_connection import ZMQConnection
from robot_controller import RobotController

def test_robot_controller():
    print("Testing Robot Controller...")
    
    try:
        print("1. Connecting to CoppeliaSim...")
        zmq_connection = ZMQConnection(ip="127.0.0.1", port=23000)
        print("   Connected!")
        
        print("2. Initializing robot controller...")
        robot = RobotController(zmq_connection)
        print("   Robot initialized!")
        
        print("3. Testing complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_robot_controller()