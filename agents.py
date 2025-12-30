import numpy as np

class Car:
    """ Car agent following Nagel-Schreckenberg traffic rules:
    - Discrete time and space
    - Velocity bounded by v_max
    - Stochastic braking (models human imperfection)
    - Local collision avoidance """

    def __init__(self, position, v_max=5, acceleration=1, brake_prob=0.3):
        """ Args:
            position: Initial position on road
            v_max: Maximum desired speed
            acceleration: Acceleration rate (typically 1)
            brake_prob: Probability of random braking (human imperfection)"""
        self.position = position
        self.velocity = 0  # Start from rest
        self.v_max = v_max
        self.acceleration = acceleration
        self.brake_prob = brake_prob
        
        # Track history for analysis
        self.velocity_history = []
        self.position_history = []
    
    def update(self, distance_to_next, road_length):
        """Update car state following Nagel-Schreckenberg rules.

        Steps:
        1. Accelerate toward v_max
        2. Avoid collision (brake if needed)
        3. Random braking (human error)
        4. Move forward
        
        Args:
            distance_to_next: Gap to next car
            road_length: Total road length (for periodic boundary)"""
        
        # Step 1: Accelerate
        self.velocity = min(self.velocity + self.acceleration, self.v_max)
        
        # Step 2: Collision avoidance
        # Safe distance = current velocity (1-second following rule)
        safe_distance = self.velocity
        if distance_to_next <= safe_distance:
            self.velocity = max(distance_to_next - 1, 0)
        
        # Step 3: Random braking (models human imperfection)
        if np.random.rand() < self.brake_prob:
            self.velocity = max(self.velocity - 1, 0)
        
        # Step 4: Move
        self.position = (self.position + self.velocity) % road_length
        
        # Record state
        self.velocity_history.append(self.velocity)
        self.position_history.append(self.position)
    
    def reset_history(self):
        # Clear tracking history.
        self.velocity_history = []
        self.position_history = []