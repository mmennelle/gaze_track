# test_connection.py
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time

def test_connection():
    print("Testing CoppeliaSim connection...")
    
    try:
        # Connect to CoppeliaSim
        print("Creating client...")
        client = RemoteAPIClient('127.0.0.1', 23000)
        print("Getting sim object...")
        sim = client.getObject('sim')
        print("Connected successfully!")
        
        # Check simulation state
        print("Checking simulation state...")
        state = sim.getSimulationState()
        print(f"Simulation state: {state} (0=stopped, 1=running, 2=paused)")
        
        # Try to find the target
        print("Looking for /target...")
        try:
            target = sim.getObject('/target')
            print(f"Found /target! Handle: {target}")
            
            print("Getting target position...")
            position = sim.getObjectPosition(target, -1)
            print(f"Target position: {position}")
            
        except Exception as e:
            print(f"Error finding target: {e}")
            
        # Test a few iterations of getting position
        print("\nTesting position updates...")
        for i in range(30):
            try:
                position = sim.getObjectPosition(target, -1)
                print(f"Iteration {i}: Position = {position}")
                time.sleep(0.5)
            except Exception as e:
                print(f"Error at iteration {i}: {e}")
                break
                
    except Exception as e:
        print(f"Connection error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_connection()