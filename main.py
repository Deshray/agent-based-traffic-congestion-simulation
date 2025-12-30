"""
Traffic Congestion Agent-Based Model
Main execution script for baseline simulation and experiments.

Toggle experiments by commenting/uncommenting sections in main().
"""

import numpy as np
import matplotlib.pyplot as plt
from simulation import (run_simulation, run_monte_carlo, 
                       experiment_density_scan, run_accident_experiment)
from analysis import (plot_time_series, plot_spacetime_diagram, 
                     plot_velocity_distribution, plot_fundamental_diagram,
                     plot_monte_carlo_results, analyze_congestion_metrics,
                     detect_congestion_onset, create_traffic_animation)

def baseline_simulation():
    # STEP 3: Run baseline to show emergent congestion.
    # Moderate density (should show congestion)
    road = run_simulation(
        n_steps=500,
        road_length=1000,
        n_cars=80,  # 0.08 density
        v_max=5,
        brake_prob=0.3,
        seed=42
    )
    
    analyze_congestion_metrics(road, burn_in=100)
    
    # Detect when congestion emerges
    detect_congestion_onset(road)
    
    plot_time_series(road)
    plot_spacetime_diagram(road)
    plot_velocity_distribution(road)
    
    # Optional: create animation (commented out by default - takes time)
    create_traffic_animation(road)
    
    return road

def validate_behavior():
    # STEP 4: Validate that model shows correct qualitative behavior.    
    scenarios = {
        'Low Density (Free Flow)': {'n_cars': 20},
        'Medium Density (Congestion)': {'n_cars': 80},
        'High Density (Jams)': {'n_cars': 150}
    }
    
    for name, params in scenarios.items():
        print(f"\n{name}:")
        road = run_simulation(n_steps=500, road_length=1000, 
                            seed=42, **params)
        analyze_congestion_metrics(road)

def fundamental_diagram_experiment():
    # STEP 5 Foundation: Generate fundamental diagram.
    # Shows relationship between density and flow.
    densities = [0.01, 0.03, 0.05, 0.08, 0.10, 0.13, 0.15, 0.18, 0.20]
    results = experiment_density_scan(
        densities=densities,
        n_steps=500,
        road_length=1000,
        n_runs=10
    )
    
    plot_fundamental_diagram(results)
    
    # Find optimal density
    max_flow_density = max(results.items(), key=lambda x: x[1]['avg_flow'])[0]
    print(f"\nOptimal density (maximum flow): {max_flow_density:.3f}")

def monte_carlo_experiment():
    # STEP 6: Monte Carlo simulation for uncertainty quantification.
    mc_results = run_monte_carlo(
        n_runs=100,
        n_steps=500,
        road_length=1000,
        n_cars=80,
        v_max=5,
        brake_prob=0.3
    )
    
    plot_monte_carlo_results(mc_results)
    
    final_v = mc_results['final_velocities']
    print(f"\nFinal velocity statistics:")
    print(f"  Mean: {final_v.mean():.3f}")
    print(f"  Std:  {final_v.std():.3f}")
    print(f"  Range: [{final_v.min():.3f}, {final_v.max():.3f}]")

def accident_experiment():
    # STEP 5 - Policy Experiment: Accident response and recovery.    
    print("Simulating traffic accident at t=200 for 100 timesteps...")
    road = run_accident_experiment(
        n_steps=800,
        accident_start=200,
        accident_duration=100,
        road_length=1000,
        n_cars=80
    )
    
    # Plot with accident highlighted
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(road.avg_velocity_history, linewidth=1.5)
    ax.axvspan(200, 300, alpha=0.3, color='red', label='Accident Period')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Average Velocity')
    ax.set_title('Traffic Response to Accident')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.savefig('outputs/accident_response.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Calculate recovery time
    post_accident_velocities = road.avg_velocity_history[300:]
    pre_accident_avg = np.mean(road.avg_velocity_history[100:200])
    
    # Find when velocity recovers to 90% of pre-accident level
    recovery_threshold = 0.9 * pre_accident_avg
    recovered = False
    for i, v in enumerate(post_accident_velocities):
        if v >= recovery_threshold:
            recovery_time = i
            recovered = True
            break
    
    if recovered:
        print(f"\nRecovery time: {recovery_time} timesteps after accident cleared")
    else:
        print("\nTraffic did not fully recover within simulation period")
    
    return road

if __name__ == "__main__":    
    # CORE EXPERIMENTS (recommended to run all)
    baseline_simulation()
    validate_behavior()
    fundamental_diagram_experiment()
    monte_carlo_experiment()
    
    # POLICY EXPERIMENTS (comment out if not needed)
    accident_experiment()