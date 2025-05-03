# robot_controller.py
import time
import math
import numpy as np

class RobotController:
    def __init__(self, zmq_connection):
        self.zmq = zmq_connection
        self.sim = zmq_connection.sim
        self.initialized = False
        self.joint_handles = []
        
        # Movement parameters (from CoppeliaSim script)
        self.max_vel = [math.pi * 5.5 / 180.0] * 5      # 110/20 deg/s to rad/s
        self.max_accel = [math.pi * 2 / 180.0] * 5      # 40/20 deg/s² to rad/s²
        self.max_jerk = [math.pi * 4 / 180.0] * 5       # 80/20 deg/s³ to rad/s³
        
        # Initialize robot
        self.initialize_robot()
        
        # Pre-defined level curve from CoppeliaSim script
        self.level_curve = self.get_level_curve()

    def get_level_curve(self):
        """Get the pre-defined level curve for different radii"""
        radii = [0.16, 0.19, 0.225, 0.25, 0.275]
        poses = np.deg2rad([
            [-20.0, 0.0, -135.0, 48.0, 0],
            [-20.0, -10.0, -118.0, 44.0, 0],
            [-20.0, -35.0, -83.0, 31.0, 0],
            [-20.0, -47.5, -57.0, 21.0, 0],
            [-20.0, -75.0, 0.0, -7.5, 0]
        ])
        return dict(zip(radii, poses.tolist()))

    def angle_mod(self, x, zero_2_2pi=False, degree=False):
        """Modulates angles to standard ranges: [-π,π) or [0,2π)"""
        is_float = isinstance(x, float)
        x = np.asarray(x).flatten()
        
        if degree:
            x = np.deg2rad(x)
            
        mod_angle = x % (2 * np.pi) if zero_2_2pi else (x + np.pi) % (2 * np.pi) - np.pi
        
        if degree:
            mod_angle = np.rad2deg(mod_angle)
            
        return mod_angle.item() if is_float else mod_angle

    def initialize_robot(self):
        try:
            # Get joint handles in the correct order
            joint_names = []
            for i in range(1, 6):
                joint_name = f'/my_cobot/joint{i+1}_to_joint{i}'
                joint_names.append(joint_name)
            
            self.joint_handles = []
            for joint_name in joint_names:
                try:
                    handle = self.sim.getObject(joint_name)
                    self.joint_handles.append(handle)
                    print(f"Got handle for {joint_name}")
                except Exception as e:
                    print(f"Failed to get handle for {joint_name}: {e}")
            
            # Get base handle
            try:
                self.base_handle = self.sim.getObject('/my_cobot')
                print("Got base handle")
            except:
                self.base_handle = None
                print("Could not get base handle")

            if self.joint_handles:
                self.initialized = True
                print("Robot initialized successfully")
            else:
                print("Robot initialization failed")

        except Exception as e:
            print(f"Error initializing robot: {e}")

    def calculate_joint_positions(self, target_position, target_handle=None):
        """Calculate joint positions using the level curve approach"""
        # Get target position relative to base if target_handle is provided
        if self.base_handle is not None and target_handle is not None:
            try:
                rel_pos = self.sim.getObjectPosition(target_handle, self.base_handle)
            except:
                rel_pos = target_position
        else:
            rel_pos = target_position
        
        # Calculate distance from base
        dist = math.hypot(rel_pos[0], rel_pos[1])
        
        # Find the closest radius in the level curve
        best_radius, target_poses = min(self.level_curve.items(), 
                                       key=lambda x: abs(dist - x[0]))
        
        print(f"Distance: {dist}, Best radius: {best_radius}")
        
        # Calculate the angle for the base joint
        theta = self.angle_mod(math.atan2(rel_pos[1], rel_pos[0]) + math.pi/2 + 0.27)
        
        # Create the target joint positions
        target_joint_positions = target_poses.copy()
        target_joint_positions[0] = theta
        
        return target_joint_positions

    def move_to_config_smooth(self, target_config, duration=2.0):
        """Smoothly move joints to target configuration over a duration"""
        if not self.initialized:
            return False
        
        try:
            # Get current joint positions
            current_positions = self.get_joint_positions()
            if None in current_positions:
                print("Failed to get current joint positions")
                return False
            
            # Calculate the trajectory
            steps = int(duration * 50)  # 50 Hz update rate
            time_step = duration / steps
            
            for i in range(steps + 1):
                t = i / steps  # Progress from 0 to 1
                
                # Smooth interpolation using cosine
                smooth_t = (1 - math.cos(t * math.pi)) / 2
                
                # Interpolate joint positions
                interpolated_positions = []
                for j in range(len(current_positions)):
                    pos = current_positions[j] + (target_config[j] - current_positions[j]) * smooth_t
                    interpolated_positions.append(pos)
                
                # Set joint positions
                for handle, pos in zip(self.joint_handles, interpolated_positions):
                    self.sim.setJointPosition(handle, pos)
                
                time.sleep(time_step)
            
            return True
            
        except Exception as e:
            print(f"Error in smooth movement: {e}")
            return False

    def move_to_object(self, object_handle):
        """Move the robot arm to an object using smooth motion"""
        if not self.initialized:
            print("Robot not initialized properly")
            return False

        try:
            # Get the target object's position
            target_position = self.sim.getObjectPosition(object_handle, -1)
            print(f"Target object position: {target_position}")
            
            # Get position relative to base
            if self.base_handle is not None:
                rel_position = self.sim.getObjectPosition(object_handle, self.base_handle)
                print(f"Target position relative to base: {rel_position}")
            else:
                rel_position = target_position
            
            # Calculate joint positions using level curve with the object handle
            joint_positions = self.calculate_joint_positions(rel_position, object_handle)
            print(f"Calculated joint positions: {joint_positions}")
            
            # Move smoothly to the target
            print("Moving smoothly to target position...")
            success = self.move_to_config_smooth(joint_positions, duration=1.5)
            
            if success:
                print("Successfully moved to target")
            else:
                print("Failed to move to target")
            
            return success

        except Exception as e:
            print(f"Error in move_to_object: {e}")
            return False

    def get_joint_positions(self):
        """Get current joint positions"""
        if not self.initialized:
            return []

        joint_positions = []
        for handle in self.joint_handles:
            try:
                pos = self.sim.getJointPosition(handle)
                joint_positions.append(pos)
            except:
                joint_positions.append(None)

        return joint_positions