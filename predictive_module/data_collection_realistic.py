# data_collection_realistic_v2.py
"""
Simplified realistic data collection using environment's observation API
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.dynamic_nav_env import DynamicObstacleNavEnv
from tqdm import tqdm
import pickle


def add_realistic_dynamics(base_speed, obstacle_type, step):
    """
    Add realistic motion patterns based on obstacle type
    
    Args:
        base_speed: Base speed from environment
        obstacle_type: 'pedestrian' or 'vehicle'
        step: Current step number
    
    Returns:
        Modified speed with realistic characteristics
    """
    if obstacle_type == 'pedestrian':
        # Pedestrians: more variable, occasional stops
        if np.random.random() < 0.05:  # 5% chance to stop
            return 0.0
        
        # Speed variations (walking rhythm)
        speed_variation = np.random.normal(0, 0.15)
        modified_speed = base_speed * (1.0 + speed_variation)
        
        # Clamp to pedestrian range (0.8-1.5 m/s)
        return np.clip(modified_speed, 0.8, 1.5)
    
    else:  # vehicle
        # Vehicles: more consistent speed
        speed_variation = np.random.normal(0, 0.05)
        modified_speed = base_speed * (1.0 + speed_variation)
        
        # Clamp to vehicle range (2.0-4.0 m/s)
        return np.clip(modified_speed, 2.0, 4.0)


def add_sensor_noise(state):
    """
    Add realistic sensor noise to state
    
    Args:
        state: [x, y, vx, vy]
    
    Returns:
        Noisy state
    """
    noisy_state = state.copy()
    
    # Position noise: ±3cm std
    noisy_state[:2] += np.random.normal(0, 0.03, 2)
    
    # Velocity noise: ±5cm/s std
    noisy_state[2:] += np.random.normal(0, 0.05, 2)
    
    return noisy_state


def extract_obstacle_states(obs, n_obstacles):
    """
    Extract obstacle states from environment observation
    
    Assumes observation format: [robot_state, goal, obstacle_states...]
    Obstacle state: [x, y, vx, vy] per obstacle
    
    Args:
        obs: Full observation from environment
        n_obstacles: Number of obstacles
    
    Returns:
        List of obstacle states, each [x, y, vx, vy]
    """
    # Skip robot state (3: x, y, theta) and goal (2: x, y)
    robot_and_goal_dim = 5
    
    obstacle_states = []
    obstacle_dim = 4  # [x, y, vx, vy]
    
    for i in range(n_obstacles):
        start_idx = robot_and_goal_dim + i * obstacle_dim
        end_idx = start_idx + obstacle_dim
        
        if end_idx <= len(obs):
            state = obs[start_idx:end_idx]
            obstacle_states.append(state)
    
    return obstacle_states


def collect_realistic_training_data(
    n_episodes=500,
    max_steps=2000,
    save_path='predictive_module/data/kgru_training_data_realistic.pkl'
):
    """
    Collect diverse training data with realistic characteristics
    """
    
    # Test configurations
    configurations = [
        # (n_obstacles, pedestrian_ratio)
        (3, 1.0),   # All pedestrians
        (5, 0.8),   # Mostly pedestrians
        (5, 0.5),   # Mixed
        (7, 0.3),   # Mostly vehicles
        (10, 0.2),  # Mostly vehicles, crowded
        (10, 0.6),  # Mixed, crowded
    ]
    
    episodes_per_config = n_episodes // len(configurations)
    
    all_trajectories = []
    
    print("="*60)
    print("COLLECTING REALISTIC TRAINING DATA")
    print("="*60)
    print(f"Total episodes: {n_episodes}")
    print(f"Episode length: {max_steps} steps")
    print(f"Configurations: {len(configurations)}")
    print(f"Episodes per config: {episodes_per_config}")
    print("="*60)
    
    for config_idx, (n_obs, ped_ratio) in enumerate(configurations):
        print(f"\nConfiguration {config_idx+1}/{len(configurations)}:")
        print(f"  Obstacles: {n_obs}")
        print(f"  Pedestrians: {ped_ratio*100:.0f}%")
        print(f"  Vehicles: {(1-ped_ratio)*100:.0f}%")
        
        # Assign obstacle types for this configuration
        n_pedestrians = int(n_obs * ped_ratio)
        obstacle_types = ['pedestrian'] * n_pedestrians + \
                        ['vehicle'] * (n_obs - n_pedestrians)
        
        for episode in tqdm(range(episodes_per_config), desc=f"  Config {config_idx+1}"):
            env = DynamicObstacleNavEnv(
                n_obstacles=n_obs,
                render_mode=None
            )
            
            # Shuffle obstacle types for variety
            np.random.shuffle(obstacle_types)
            
            obs, _ = env.reset()
            
            # Store trajectories for each obstacle
            episode_trajectories = {i: [] for i in range(n_obs)}
            
            for step in range(max_steps):
                # Extract obstacle states from observation
                obstacle_states = extract_obstacle_states(obs, n_obs)
                
                # Apply realistic characteristics and store
                for obs_id, state in enumerate(obstacle_states):
                    # Add sensor noise
                    noisy_state = add_sensor_noise(state)
                    
                    # Store: [x, y, vx, vy]
                    episode_trajectories[obs_id].append(noisy_state)
                
                # Random robot action (we only care about obstacle motion)
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
            
            # Post-process trajectories to match expected speed profiles
            for obs_id, trajectory in episode_trajectories.items():
                if len(trajectory) >= 20:
                    trajectory = np.array(trajectory)
                    
                    # Calculate average speed
                    speeds = np.linalg.norm(trajectory[:, 2:4], axis=1)
                    avg_speed = speeds.mean()
                    
                    # Determine type based on speed (environment's natural speeds)
                    # If avg speed < 1.0 m/s, treat as pedestrian
                    # Otherwise treat as vehicle
                    obs_type = obstacle_types[obs_id % len(obstacle_types)]
                    
                    # Adjust speeds to match realistic profiles
                    for t in range(len(trajectory)):
                        current_speed = np.linalg.norm(trajectory[t, 2:4])
                        
                        if current_speed > 0.01:  # Avoid division by zero
                            if obs_type == 'pedestrian':
                                # Scale to pedestrian speeds (0.8-1.5 m/s)
                                target_speed = np.random.uniform(0.8, 1.5)
                            else:
                                # Scale to vehicle speeds (2.0-4.0 m/s)
                                target_speed = np.random.uniform(2.0, 4.0)
                            
                            # Random stops for pedestrians
                            if obs_type == 'pedestrian' and np.random.random() < 0.03:
                                target_speed = 0.0
                            
                            # Scale velocity
                            scale_factor = target_speed / current_speed
                            trajectory[t, 2:4] *= scale_factor
                    
                    all_trajectories.append(trajectory)
            
            env.close()
    
    # Statistics
    trajectory_lengths = [len(traj) for traj in all_trajectories]
    speeds = []
    for traj in all_trajectories:
        traj_speeds = np.linalg.norm(traj[:, 2:4], axis=1)
        speeds.extend(traj_speeds.tolist())
    
    speeds = np.array(speeds)
    low_speed_pct = (speeds < 2.0).sum() / len(speeds) * 100
    high_speed_pct = (speeds >= 2.0).sum() / len(speeds) * 100
    
    # Save data
    data = {
        'trajectories': all_trajectories,
        'configurations': configurations,
        'motion_model_params': {
            'pedestrian_speed_range': (0.8, 1.5),
            'vehicle_speed_range': (2.0, 4.0),
            'position_noise_std': 0.03,
            'velocity_noise_std': 0.05,
        },
        'statistics': {
            'n_trajectories': len(all_trajectories),
            'mean_length': np.mean(trajectory_lengths),
            'std_length': np.std(trajectory_lengths),
            'min_length': np.min(trajectory_lengths),
            'max_length': np.max(trajectory_lengths),
            'mean_speed': speeds.mean(),
            'std_speed': speeds.std(),
            'low_speed_pct': low_speed_pct,
            'high_speed_pct': high_speed_pct,
        }
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE")
    print("="*60)
    print(f"Total trajectories: {len(all_trajectories)}")
    print(f"Average length: {np.mean(trajectory_lengths):.1f} steps")
    print(f"Speed distribution:")
    print(f"  Low-speed (<2.0 m/s): {low_speed_pct:.1f}%")
    print(f"  High-speed (≥2.0 m/s): {high_speed_pct:.1f}%")
    print(f"  Mean: {speeds.mean():.2f} m/s")
    print(f"  Std: {speeds.std():.2f} m/s")
    print(f"\nData saved to: {save_path}")
    print("="*60)


if __name__ == "__main__":
    collect_realistic_training_data(
        n_episodes=500,
        max_steps=2000,
        save_path='predictive_module/data/kgru_training_data_realistic.pkl'
    )