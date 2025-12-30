import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Create output directory for saved plots
os.makedirs('outputs', exist_ok=True)

def plot_time_series(road, save_path='outputs/velocity_timeseries.png'):
    # Plot average velocity over time.
    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.plot(road.avg_velocity_history, linewidth=1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Average Velocity')
    ax.set_title('System-Level Traffic Flow Over Time')
    ax.grid(alpha=0.3)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_spacetime_diagram(road, save_path='outputs/spacetime_diagram.png'):
    # Space-time diagram showing car trajectories.
    # Reveals stop-and-go waves.
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for car in road.cars:
        ax.plot(car.position_history, linewidth=0.5, alpha=0.7)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Position on Road')
    ax.set_title('Space-Time Diagram (Traffic Waves)')
    ax.grid(alpha=0.3)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_velocity_distribution(road, save_path='outputs/velocity_distribution.png'):
    # Histogram of final velocities.
    velocities = [car.velocity for car in road.cars]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(velocities, bins=range(0, max(velocities)+2), 
            edgecolor='black', alpha=0.7)
    ax.set_xlabel('Velocity')
    ax.set_ylabel('Number of Cars')
    ax.set_title('Final Velocity Distribution')
    ax.grid(alpha=0.3, axis='y')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_fundamental_diagram(results, save_path='outputs/fundamental_diagram.png'):
    # Flow vs Density plot (fundamental diagram of traffic).
    # Shows capacity and congestion regime.
    densities = sorted(results.keys())
    flows = [results[d]['avg_flow'] for d in densities]
    flow_stds = [results[d]['std_flow'] for d in densities]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(densities, flows, yerr=flow_stds, 
                marker='o', capsize=5, linewidth=2)
    ax.set_xlabel('Density (cars per unit length)')
    ax.set_ylabel('Flow (cars per time)')
    ax.set_title('Fundamental Diagram: Flow vs Density')
    ax.grid(alpha=0.3)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_monte_carlo_results(mc_results, save_path='outputs/monte_carlo.png'):
    # Visualize Monte Carlo simulation results.
    final_velocities = mc_results['final_velocities']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time series with confidence bands
    avg_velocities = mc_results['avg_velocities']
    mean_v = np.mean(avg_velocities, axis=0)
    std_v = np.std(avg_velocities, axis=0)
    
    axes[0].plot(mean_v, linewidth=2, label='Mean')
    axes[0].fill_between(range(len(mean_v)), 
                          mean_v - std_v, mean_v + std_v,
                          alpha=0.3, label='±1 std')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Average Velocity')
    axes[0].set_title(f'Monte Carlo Trajectories (n={mc_results["n_runs"]})')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Distribution of final states
    axes[1].hist(final_velocities, bins=20, edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(final_velocities), color='red', 
                    linestyle='--', linewidth=2, label='Mean')
    axes[1].set_xlabel('Final Average Velocity')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Steady-State Velocities')
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def analyze_congestion_metrics(road, burn_in=100):
    """
    Calculate key congestion metrics using steady-state data.
    
    Args:
        road: Road object after simulation
        burn_in: Number of initial timesteps to ignore
    """
    steady_state_velocities = road.avg_velocity_history[burn_in:]
    avg_v = np.mean(steady_state_velocities)
    v_max = road.cars[0].v_max
    
    # Congestion index: how much slower than free flow
    congestion_index = 1 - (avg_v / v_max)
    
    # Variance (measure of stop-and-go behavior)
    velocity_variance = np.var(steady_state_velocities)
    
    print(f"CONGESTION METRICS (after {burn_in}-step burn-in)")
    print(f"Average velocity: {avg_v:.2f} / {v_max} (max)")
    print(f"Congestion index: {congestion_index:.2%}")
    print(f"Velocity variance: {velocity_variance:.3f}")
    print(f"Density: {road.n_cars / road.length:.3f}")
    print(f"Flow: {np.mean(road.flow_history[burn_in:]):.3f}")
    
    return {
        'avg_velocity': avg_v,
        'congestion_index': congestion_index,
        'variance': velocity_variance
    }

def detect_congestion_onset(road, threshold=0.3, window=20):
    """
    Detect when congestion emerges by identifying sharp velocity drops.
    
    Args:
        road: Road object
        threshold: Fraction of v_max below which we consider congested
        window: Rolling window for smoothing
    
    Returns:
        onset_time: Timestep when congestion begins (or None)
    """
    velocities = np.array(road.avg_velocity_history)
    v_max = road.cars[0].v_max
    
    # Smooth with rolling average
    if len(velocities) < window:
        return None
    
    smoothed = np.convolve(velocities, np.ones(window)/window, mode='valid')
    
    # Find first time velocity drops below threshold
    congested_threshold = threshold * v_max
    congested_indices = np.where(smoothed < congested_threshold)[0]
    
    if len(congested_indices) > 0:
        onset_time = congested_indices[0] + window // 2  # Adjust for window
        print(f"Congestion onset detected at t={onset_time}")
        return onset_time
    
    return None

def create_traffic_animation(road, save_path='outputs/traffic_animation.gif', fps=10):
    """
    Create animated visualization of traffic flow.
    Shows car positions over time as a simple 1D animation.
    
    Args:
        road: Road object with simulation history
        save_path: Where to save GIF
        fps: Frames per second
    """
    
    fig, ax = plt.subplots(figsize=(14, 3))
    
    # Get max time
    max_time = len(road.cars[0].position_history)
    
    def update(frame):
        ax.clear()
        
        # Get positions at this timestep
        positions = [car.position_history[frame] for car in road.cars]
        velocities = [car.velocity_history[frame] for car in road.cars]
        
        # Color by velocity (red = stopped, green = fast)
        colors = plt.cm.RdYlGn([v / road.cars[0].v_max for v in velocities])
        
        # Plot cars
        ax.scatter(positions, [0]*len(positions), c=colors, s=100, marker='s')
        
        ax.set_xlim(0, road.length)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('Position on Road')
        ax.set_title(f'Traffic Flow Animation (t={frame})')
        ax.set_yticks([])
        ax.grid(alpha=0.3, axis='x')
        
        # Add colorbar reference
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, 
                                   norm=plt.Normalize(0, road.cars[0].v_max))
        sm.set_array([])
        
        return ax,
    
    # Sample frames to keep file size reasonable
    sample_rate = max(1, max_time // 200)  # Max 200 frames
    frames = range(0, max_time, sample_rate)
    
    anim = FuncAnimation(fig, update, frames=frames, 
                        interval=1000/fps, blit=False)
    
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()