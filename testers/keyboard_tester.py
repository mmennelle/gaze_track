# test_keyboard.py
from keyboard_input import KeyboardController
import time

def test_keyboard():
    print("Testing Keyboard Controller...")
    
    try:
        print("1. Initializing keyboard controller...")
        keyboard = KeyboardController()
        print("   Keyboard initialized!")
        
        print("2. Testing keyboard input for 5 seconds...")
        print("   Press arrow keys to test")
        
        start_time = time.time()
        while time.time() - start_time < 5:
            x_axis, y_axis, running = keyboard.update()
            if x_axis != 0 or y_axis != 0:
                print(f"   Input detected: x={x_axis}, y={y_axis}")
            if not running:
                break
                
        print("3. Closing keyboard controller...")
        keyboard.close()
        print("   Keyboard closed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_keyboard()