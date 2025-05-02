import pygame
import time

class KeyboardController:
    def __init__(self):
        pygame.init()
        # Create a small window to capture keyboard events
        self.screen = pygame.display.set_mode((100, 100))
        pygame.display.set_caption("Keyboard Control")
        
        # Initialize key states
        self.keys = {
            pygame.K_UP: False,
            pygame.K_DOWN: False,
            pygame.K_LEFT: False,
            pygame.K_RIGHT: False
        }
        
    def update(self):
        """Update key states and return x, y axes values"""
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, None, False
            
            if event.type == pygame.KEYDOWN:
                if event.key in self.keys:
                    self.keys[event.key] = True
            
            if event.type == pygame.KEYUP:
                if event.key in self.keys:
                    self.keys[event.key] = False
        
        # Calculate x, y axes similar to joystick
        x_axis = 0
        y_axis = 0
        
        if self.keys[pygame.K_RIGHT]:
            x_axis = 1.0
        if self.keys[pygame.K_LEFT]:
            x_axis = -1.0
        if self.keys[pygame.K_UP]:
            y_axis = -1.0  # Negative because y-axis is inverted in pygame
        if self.keys[pygame.K_DOWN]:
            y_axis = 1.0
        
        return x_axis, y_axis, True
    
    def close(self):
        """Close pygame"""
        pygame.quit()