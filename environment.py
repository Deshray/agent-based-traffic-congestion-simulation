import numpy as np
from agents import Car

class Road:
    # Circular road with periodic boundary conditions.
    def __init__(self, length=1000, n_cars=50, v_max=5, brake_prob=0.3):
        """
        Args:
            length: Road length in units
            n_cars: Number of cars
            v_max: Maximum speed for all cars
            brake_prob: Random braking probability
        """
        self.length = length
        self.n_cars = n_cars
        self.time = 0
        
        # Initialize cars with random spacing
        positions = sorted(np.random.choice(range(length), size=n_cars, replace=False))
        self.cars = [Car(pos, v_max=v_max, brake_prob=brake_prob) for pos in positions]
        
        # Metrics tracking
        self.avg_velocity_history = []
        self.density_history = []
        self.flow_history = []
        
    def step(self):
        # Execute one time step of simulation.
        # Apply accident effects if active
        self._apply_accident_effects()
        
        # Sort cars by position
        self.cars.sort(key=lambda c: c.position)
        
        # Calculate distances to next car
        distances = []
        for i in range(len(self.cars)):
            next_idx = (i + 1) % len(self.cars)
            distance = (self.cars[next_idx].position - self.cars[i].position) % self.length
            distances.append(distance)
        
        # Update all cars
        for car, dist in zip(self.cars, distances):
            car.update(dist, self.length)
        
        self.time += 1
        self._record_metrics()
    
    def _record_metrics(self):
        # Track system-level metrics.
        velocities = [car.velocity for car in self.cars]
        avg_v = np.mean(velocities)
        
        density = self.n_cars / self.length
        flow = density * avg_v  # vehicles per time unit
        
        self.avg_velocity_history.append(avg_v)
        self.density_history.append(density)
        self.flow_history.append(flow)
    
    def get_state(self):
        # Return current state snapshot.
        return {
            'time': self.time,
            'positions': [car.position for car in self.cars],
            'velocities': [car.velocity for car in self.cars],
            'avg_velocity': np.mean([car.velocity for car in self.cars]),
            'density': self.n_cars / self.length,
            'flow': self.flow_history[-1] if self.flow_history else 0
        }
    
    def introduce_accident(self, location, duration):
        """
        Block a road segment to simulate accident.
        
        Creates a bottleneck by forcing cars in the accident zone to stop.
        Enables incident response and shockwave propagation analysis.
        
        Args:
            location: Position of accident on road
            duration: How many timesteps the blockage lasts
        """
        self.accident_location = location
        self.accident_duration = duration
        self.accident_timer = 0
    
    def _apply_accident_effects(self):
        # Check if accident is active and force affected cars to stop.
        if not hasattr(self, 'accident_timer'):
            return
        
        if self.accident_timer < self.accident_duration:
            # Find cars in accident zone (within 20 units)
            for car in self.cars:
                distance_to_accident = abs(car.position - self.accident_location)
                # Handle periodic boundary
                distance_to_accident = min(distance_to_accident, 
                                          self.length - distance_to_accident)
                
                if distance_to_accident < 20:
                    car.velocity = 0  # Force stop
            
            self.accident_timer += 1