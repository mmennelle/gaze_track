# calibration_module.py - Complete rewrite with user gaze as source of truth
import time
import cv2
import numpy as np
import logging

# Configure logging
logger = logging.getLogger("CalibrationModule")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class CalibrationModule:
    def __init__(self, zmq_connection, gaze_tracker, agent, duration_per_target=3.0):
        self.zmq = zmq_connection
        self.gaze_tracker = gaze_tracker
        self.agent = agent
        self.duration_per_target = duration_per_target
        self.calibration_complete = False
        self.calibration_sequence = []
        self.current_target_index = 0
        self.min_targets_for_calibration = 5
        
        # Reference to the gaze filter (will be set later)
        self.gaze_filter = None
        
    def set_gaze_filter(self, gaze_filter):
        """Set the gaze filter reference for calibration"""
        self.gaze_filter = gaze_filter
        
    def get_calibration_objects(self):
        """Get objects from the scene for calibration"""
        objects = []
        object_id = 0
        potential_targets = [f"/target[{i}]" for i in range(20)]
        
        for target_name in potential_targets:
            try:
                response = self.zmq.get_object_handle(target_name)
                if response.get('returnCode') == 0:
                    target_handle = response.get('handle')
                    position_response = self.zmq.get_object_position(target_handle, -1)
                    if position_response.get('returnCode') == 0:
                        position = position_response.get('position')
                        objects.append({
                            "id": object_id,
                            "handle": target_handle,
                            "name": target_name,
                            "position": position
                        })
                        object_id += 1
                        logger.info(f"Found calibration target: {target_name} at position {position}")
                else:
                    if "[" in target_name and "target[0]" not in target_name:
                        break
            except Exception as e:
                logger.error(f"Error accessing target {target_name}: {e}")
                if "[" in target_name and "target[0]" not in target_name:
                    break
                continue
        
        # Sort objects to create an effective calibration pattern
        if objects:
            # Create a sequence that prioritizes corners and center
            corners_and_center = []
            
            # Find objects closest to corners and center
            corners = [
                (-1, -1),  # top-left
                (1, -1),   # top-right
                (-1, 1),   # bottom-left
                (1, 1),    # bottom-right
                (0, 0)     # center
            ]
            
            for corner_x, corner_y in corners:
                closest_obj = min(objects, 
                                  key=lambda obj: (obj["position"][0] - corner_x)**2 + 
                                                 (obj["position"][1] - corner_y)**2)
                if closest_obj not in corners_and_center:
                    corners_and_center.append(closest_obj)
            
            # Add remaining objects
            remaining = [obj for obj in objects if obj not in corners_and_center]
            
            # Create final sequence
            self.calibration_sequence = corners_and_center + remaining
        
        return objects
    
    def create_target_display(self, target_obj, all_objects, frame_size):
        """Create an overlay image showing the calibration target"""
        height, width = frame_size
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw all objects as small circles
        for obj in all_objects:
            x, y, _ = obj["position"]
            # Convert 3D position to screen coordinates (simplified)
            screen_x = int(width * (x + 1) / 2)
            screen_y = int(height * (y + 1) / 2)
            
            # Draw small circle for all objects
            cv2.circle(overlay, (screen_x, screen_y), 5, (50, 50, 50), -1)
            
        # Draw current target as a larger, brighter circle
        x, y, _ = target_obj["position"]
        screen_x = int(width * (x + 1) / 2)
        screen_y = int(height * (y + 1) / 2)
        
        # Draw pulsing target
        pulse = abs(np.sin(time.time() * 3)) * 0.5 + 0.5  # Pulsing effect
        radius = int(20 + pulse * 10)
        color = (0, int(100 + pulse * 155), int(100 + pulse * 155))
        
        # Draw concentric circles for the target
        cv2.circle(overlay, (screen_x, screen_y), radius, color, 3)
        cv2.circle(overlay, (screen_x, screen_y), radius - 5, color, 2)
        
        # Display target name and instructions
        instructions = "Please focus your gaze on the pulsing circle"
        cv2.putText(overlay, instructions, (width // 2 - 200, height - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(overlay, f"Target: {target_obj['name']}", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        # Progress for current target
        cv2.putText(overlay, f"Target {self.current_target_index + 1} of {len(self.calibration_sequence)}", 
                   (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (200, 200, 200), 2)
                   
        return overlay, (screen_x, screen_y)
    
    def run_calibration(self, frame_size):
        """
        Run the calibration sequence with improved logic that fully trusts the user's gaze
        as the source of truth for where they are looking at targets
        """
        logger.info("Starting guided calibration sequence...")
        
        # Get all calibration objects and create sequence
        objects = self.get_calibration_objects()
        if not objects:
            logger.error("No objects found in scene for calibration!")
            yield None, np.zeros((*frame_size, 3), dtype=np.uint8), 0, False
            return
        
        if not self.calibration_sequence:
            self.calibration_sequence = objects
        
        logger.info(f"Created calibration sequence with {len(self.calibration_sequence)} targets")
        
        # Show initial instruction screen with clear instructions
        instruction_overlay = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)
        cv2.putText(instruction_overlay, "Eye Gaze Calibration", 
                   (frame_size[1] // 2 - 150, frame_size[0] // 2 - 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Make it very clear that the system adapts to the user, not vice versa
        cv2.putText(instruction_overlay, "This system adapts to YOUR gaze - not the other way around!", 
                   (frame_size[1] // 2 - 320, frame_size[0] // 2 - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                   
        cv2.putText(instruction_overlay, "You will be asked to look at several targets", 
                   (frame_size[1] // 2 - 250, frame_size[0] // 2 + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(instruction_overlay, "For each target:", 
                   (frame_size[1] // 2 - 100, frame_size[0] // 2 + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(instruction_overlay, "1. Look directly at the pulsing circle", 
                   (frame_size[1] // 2 - 220, frame_size[0] // 2 + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(instruction_overlay, "2. Keep looking until the progress bar completes", 
                   (frame_size[1] // 2 - 260, frame_size[0] // 2 + 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(instruction_overlay, "3. The system will calibrate to your natural gaze pattern", 
                   (frame_size[1] // 2 - 290, frame_size[0] // 2 + 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        cv2.putText(instruction_overlay, "Press any key to begin calibration", 
                   (frame_size[1] // 2 - 180, frame_size[0] // 2 + 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        yield None, instruction_overlay, 0, True  # Signal to wait for key press
        
        # Reset calibration state
        self.current_target_index = 0
        self.calibration_complete = False
        
        # If we have a gaze filter, reset its calibration data
        if self.gaze_filter:
            self.gaze_filter.calibration_samples = []
            self.gaze_filter.calibrated = False
        
        # Countdown before starting
        for countdown in range(3, 0, -1):
            countdown_overlay = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)
            cv2.putText(countdown_overlay, f"Starting in {countdown}...", 
                       (frame_size[1] // 2 - 120, frame_size[0] // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            yield None, countdown_overlay, 0, False
            time.sleep(1.0)
        
        # Run through each target in the sequence
        for i, target_obj in enumerate(self.calibration_sequence):
            self.current_target_index = i
            
            # Parameters for collecting samples
            target_calibration_samples = 0
            required_samples = 20  # Number of gaze samples to collect per target
            sample_interval = 0.1  # Time between samples in seconds
            max_duration = 15.0    # Maximum seconds per target before moving on
            start_time = time.time()
            success_for_target = False
            last_sample_time = 0
            
            # List to store the collected gaze points for this target
            gaze_samples = []
            
            # Introduction to target
            target_intro_overlay = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)
            x, y, _ = target_obj["position"]
            screen_x = int(frame_size[1] * (x + 1) / 2)
            screen_y = int(frame_size[0] * (y + 1) / 2)
            
            # Draw target with animation
            radius = 30
            cv2.circle(target_intro_overlay, (screen_x, screen_y), radius, (0, 255, 255), 3)
            cv2.line(target_intro_overlay, (screen_x - 50, screen_y), (screen_x - 20, screen_y), (0, 255, 255), 2)
            cv2.line(target_intro_overlay, (screen_x + 20, screen_y), (screen_x + 50, screen_y), (0, 255, 255), 2)
            cv2.line(target_intro_overlay, (screen_x, screen_y - 50), (screen_x, screen_y - 20), (0, 255, 255), 2)
            cv2.line(target_intro_overlay, (screen_x, screen_y + 20), (screen_x, screen_y + 50), (0, 255, 255), 2)
            
            cv2.putText(target_intro_overlay, f"Target {i+1}: Look at this point", 
                       (frame_size[1] // 2 - 150, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            yield None, target_intro_overlay, 0, False
            time.sleep(1.0)  # Show target intro briefly
            
            # Main calibration loop for this target
            while target_calibration_samples < required_samples:
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Create target display overlay
                overlay, target_screen_pos = self.create_target_display(target_obj, objects, frame_size)
                
                # Add progress bar for target calibration
                progress = min(target_calibration_samples / required_samples, 1.0)
                bar_width = int(frame_size[1] * 0.6)
                filled_width = int(bar_width * progress)
                start_x = (frame_size[1] - bar_width) // 2
                
                # Draw progress bar background
                cv2.rectangle(overlay, (start_x, 80), (start_x + bar_width, 100), (50, 50, 50), -1)
                
                # Draw filled portion of progress bar
                if progress > 0:
                    cv2.rectangle(overlay, (start_x, 80), (start_x + filled_width, 100), (0, 255, 0), -1)
                
                # Add instructional text emphasizing that user's gaze is correct
                cv2.putText(overlay, 
                           f"Just keep looking at the target naturally ({target_calibration_samples}/{required_samples})", 
                           (frame_size[1] // 2 - 250, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Send current frame with overlay to main thread
                yield target_obj, overlay, target_calibration_samples / required_samples * self.duration_per_target, False
                
                # Check if we need to timeout
                if elapsed > max_duration:
                    logger.warning(f"Timeout on target {target_obj['name']} after {elapsed:.1f} seconds")
                    break
                
                # Get the current gaze data if enough time has passed since the last sample
                if current_time - last_sample_time >= sample_interval:
                    current_gaze = yield_gaze_data()
                    
                    # If we have valid gaze data, collect a sample
                    if current_gaze and current_gaze.get("pupils_located", False):
                        # Use the RAW gaze data (before any calibration) as this is what we're trying to map
                        h_ratio = current_gaze.get("raw_ratio_horizontal")
                        v_ratio = current_gaze.get("raw_ratio_vertical")
                        
                        if h_ratio is not None and v_ratio is not None:
                            # Convert target position to normalized coordinates (0-1)
                            target_h = (target_obj["position"][0] + 1.0) / 2.0
                            target_v = (target_obj["position"][1] + 1.0) / 2.0
                            
                            # Record mapping: "This gaze position (h_ratio, v_ratio) corresponds to
                            # looking at target position (target_h, target_v)"
                            gaze_samples.append((h_ratio, v_ratio, target_h, target_v))
                            target_calibration_samples += 1
                            last_sample_time = current_time
                            
                            logger.info(f"Calibration sample {target_calibration_samples}/{required_samples} " +
                                      f"for target {target_obj['name']}: Gaze ({h_ratio:.3f}, {v_ratio:.3f}) -> " +
                                      f"Target ({target_h:.3f}, {target_v:.3f})")
                
                # Small delay to prevent loop running too fast
                time.sleep(0.03)
            
            # Process all collected samples for this target
            if len(gaze_samples) >= required_samples * 0.7:  # At least 70% of required samples
                # Calculate average gaze position for this target
                avg_h_ratio = sum(sample[0] for sample in gaze_samples) / len(gaze_samples)
                avg_v_ratio = sum(sample[1] for sample in gaze_samples) / len(gaze_samples)
                target_h = gaze_samples[0][2]  # All samples have the same target
                target_v = gaze_samples[0][3]
                
                # Store the mapping in the calibration system
                if self.gaze_filter:
                    self.gaze_filter.add_calibration_sample(avg_h_ratio, avg_v_ratio, target_h, target_v)
                
                logger.info(f"Stored calibration mapping: gaze ({avg_h_ratio:.3f}, {avg_v_ratio:.3f}) -> target ({target_h:.3f}, {target_v:.3f})")
                success_for_target = True
            
            # Add completion indicator for this target
            if success_for_target:
                completed_overlay = overlay.copy()
                cv2.putText(completed_overlay, "Target Calibrated!", 
                           (frame_size[1] // 2 - 100, frame_size[0] // 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                yield target_obj, completed_overlay, self.duration_per_target, False
                time.sleep(1.0)  # Show completion message briefly
        
        # Finalize calibration
        if self.gaze_filter:
            self.calibration_complete = self.gaze_filter.finalize_calibration()
        else:
            self.calibration_complete = False
        
        # Show completion overlay
        if self.calibration_complete:
            logger.info("Calibration sequence successfully completed!")
            
            # Show completion overlay
            completion_overlay = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)
            cv2.putText(completion_overlay, "Calibration Complete!", 
                       (frame_size[1] // 2 - 150, frame_size[0] // 2 - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(completion_overlay, f"Calibrated with {len(self.calibration_sequence)} targets", 
                       (frame_size[1] // 2 - 180, frame_size[0] // 2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(completion_overlay, "Your eye gaze tracking should now be more accurate", 
                       (frame_size[1] // 2 - 250, frame_size[0] // 2 + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(completion_overlay, "Press any key to continue", 
                       (frame_size[1] // 2 - 130, frame_size[0] // 2 + 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            yield None, completion_overlay, 0, True  # Signal to wait for key press
        else:
            logger.warning("Not enough targets calibrated successfully")
            
            # Show incomplete calibration message
            incomplete_overlay = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)
            cv2.putText(incomplete_overlay, "Calibration Incomplete", 
                       (frame_size[1] // 2 - 150, frame_size[0] // 2 - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 2)
            cv2.putText(incomplete_overlay, "Not enough targets were successfully calibrated", 
                       (frame_size[1] // 2 - 250, frame_size[0] // 2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(incomplete_overlay, "The system will use default settings", 
                       (frame_size[1] // 2 - 180, frame_size[0] // 2 + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(incomplete_overlay, "Press any key to continue", 
                       (frame_size[1] // 2 - 130, frame_size[0] // 2 + 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            yield None, incomplete_overlay, 0, True  # Signal to wait for key press
    
    def is_calibrated(self):
        """Check if calibration is complete"""
        return self.calibration_complete

# Function for the main module to provide gaze data to the calibration module
def yield_gaze_data():
    """Utility function to get latest gaze data from global state"""
    # This is a placeholder - in the actual implementation, this will be replaced
    # with a function that returns the current global gaze_data
    return None