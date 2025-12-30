import numpy as np
from environment import Road

def run_simulation(n_steps=500, road_length=1000, n_cars=50, 
                   v_max=5, brake_prob=0.3, seed=None):
    """
    Run single traffic simulation.
    
    Args:
        n_steps: Number of timesteps to simulate
        road_length: Length of circular road
        n_cars: Number of cars
        v_max: Maximum speed
        brake_prob: Probability of random braking
        seed: Random seed for reproducibility
    
    Returns:
        road: Road object with complete history
    """
    if seed is not None:
        np.random.seed(seed)
    
    road = Road(length=road_length, n_cars=n_cars, 
                v_max=v_max, brake_prob=brake_prob)
    
    for _ in range(n_steps):
        road.step()
    
    return road

def run_monte_carlo(n_runs=100, n_steps=500, **kwargs):
    """
    Run multiple simulations with randomized initial conditions.
    
    Args:
        n_runs: Number of independent simulations
        n_steps: Steps per simulation
        **kwargs: Parameters passed to run_simulation
    
    Returns:
        results: Dict containing aggregated metrics
    """
    all_avg_velocities = []
    all_flows = []
    final_velocities = []
    
    for run in range(n_runs):
        road = run_simulation(n_steps=n_steps, seed=run, **kwargs)
        
        all_avg_velocities.append(road.avg_velocity_history)
        all_flows.append(road.flow_history)
        final_velocities.append(road.avg_velocity_history[-1])
    
    return {
        'avg_velocities': np.array(all_avg_velocities),
        'flows': np.array(all_flows),
        'final_velocities': np.array(final_velocities),
        'n_runs': n_runs
    }

def experiment_density_scan(densities, n_steps=500, road_length=1000, n_runs=10, burn_in=100):
    """
    Scan across different traffic densities.
    
    Args:
        densities: List of density values (cars per unit length)
        n_steps: Simulation length
        road_length: Road length
        n_runs: Repetitions per density
        burn_in: Timesteps to discard (transient behavior)
    
    Returns:
        results: Dict mapping density -> metrics
    """
    results = {}
    
    for density in densities:
        n_cars = int(density * road_length)
        
        avg_velocities = []
        avg_flows = []
        
        for run in range(n_runs):
            road = run_simulation(n_steps=n_steps, road_length=road_length,
                                n_cars=n_cars, seed=run)
            
            # Use steady-state values (after burn-in)
            steady_state_velocities = road.avg_velocity_history[burn_in:]
            steady_state_flows = road.flow_history[burn_in:]
            
            avg_velocities.append(np.mean(steady_state_velocities))
            avg_flows.append(np.mean(steady_state_flows))
        
        results[density] = {
            'avg_velocity': np.mean(avg_velocities),
            'std_velocity': np.std(avg_velocities),
            'avg_flow': np.mean(avg_flows),
            'std_flow': np.std(avg_flows)
        }
    
    return results

def run_accident_experiment(n_steps=800, accident_start=200, accident_duration=100,
                           road_length=1000, n_cars=80, **kwargs):
    """
    Simulate accident scenario and measure recovery dynamics.
    
    Args:
        n_steps: Total simulation length
        accident_start: When accident occurs
        accident_duration: How long accident blocks road
        road_length: Road length
        n_cars: Number of cars
        **kwargs: Additional parameters for simulation
    
    Returns:
        road: Road object with accident history
    """
    np.random.seed(42)
    
    road = Road(length=road_length, n_cars=n_cars, **kwargs)
    
    # Run until accident
    for _ in range(accident_start):
        road.step()
    
    # Introduce accident
    accident_location = road_length // 2
    road.introduce_accident(accident_location, accident_duration)
    
    # Continue simulation
    for _ in range(n_steps - accident_start):
        road.step()
    
    return road