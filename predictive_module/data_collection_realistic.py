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


def add_sensor_noise(state):
    noisy_state = state.copy()
    
    # Position noise: ±3cm std
    noisy_state[:2] += np.random.normal(0, 0.03, 2)
    
    # Velocity noise: ±5cm/s std
    noisy_state[2:] += np.random.normal(0, 0.05, 2)
    
    return noisy_state


def extract_obstacle_states(obs, n_obstacles):

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


def downsample_trajectories(trajectories, target_dt=0.1, source_dt=0.002):
    factor = int(target_dt / source_dt)  # 50
    
    print(f"\nDownsampling trajectories:")
    print(f"  From: {source_dt}s ({1/source_dt:.0f} Hz)")
    print(f"  To:   {target_dt}s ({1/target_dt:.0f} Hz)")
    print(f"  Factor: {factor}")
    
    downsampled = []
    
    for traj in tqdm(trajectories, desc="  Downsampling"):
        # Check BEFORE downsampling
        if len(traj) < factor * 10:  # Need at least 10 frames after downsampling
            continue
        
        # Downsample
        ds_traj = traj[::factor].copy()
        
        # Recalculate velocities
        for i in range(len(ds_traj) - 1):
            dx = ds_traj[i+1, 0] - ds_traj[i, 0]
            dy = ds_traj[i+1, 1] - ds_traj[i, 1]
            ds_traj[i, 2] = dx / target_dt
            ds_traj[i, 3] = dy / target_dt
        
        ds_traj[-1, 2:4] = ds_traj[-2, 2:4]
        
        # Should have at least 10 frames now
        if len(ds_traj) >= 10:  # ← 10 frames = 1 second
            downsampled.append(ds_traj)
    
    print(f"  Kept {len(downsampled)}/{len(trajectories)} trajectories")
    
    return downsampled


def collect_realistic_training_data(
    n_episodes=500,
    max_steps=10000,
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
            termination_reasons = []  # Add before the for episode loop
            
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
                    reason = f"Step {step}: terminated={terminated}, truncated={truncated}"
                    if 'TimeLimit.truncated' in info:
                        reason += " (TimeLimit)"
                    termination_reasons.append(reason)
                    break
            
            # Collect trajectories (minimum length check)
            for obs_id, trajectory in episode_trajectories.items():
                if len(trajectory) >= 100:  # At 500 Hz, need 100 samples minimum
                    trajectory = np.array(trajectory)
                    all_trajectories.append(trajectory)
            
            env.close()
            if episode == 0 and config_idx == 0:  # Print first episode only
                print(f"\nExample terminations:")
                for r in termination_reasons[:5]:
                    print(f"  {r}")
    
    print(f"\nCollected {len(all_trajectories)} raw trajectories at 500 Hz")
    
    # CRITICAL FIX: Downsample to 10 Hz before saving
    all_trajectories = downsample_trajectories(
        all_trajectories, 
        target_dt=0.1, 
        source_dt=0.002
    )
    
    # Calculate statistics on downsampled data
    trajectory_lengths = [len(traj) for traj in all_trajectories]
    speeds = []
    for traj in all_trajectories:
        traj_speeds = np.linalg.norm(traj[:, 2:4], axis=1)
        speeds.extend(traj_speeds.tolist())
    
    speeds = np.array(speeds)
    low_speed_pct = (speeds < 2.0).sum() / len(speeds) * 100
    high_speed_pct = (speeds >= 2.0).sum() / len(speeds) * 100
    
    # Save data with correct dt
    data = {
        'trajectories': all_trajectories,
        'dt': 0.1,  # CRITICAL: Store the timestep!
        'source_frequency': '10 Hz (downsampled from 500 Hz)',
        'configurations': configurations,
        'motion_model_params': {
            'pedestrian_speed_range': (0.8, 1.5),
            'vehicle_speed_range': (2.0, 4.0),
            'position_noise_std': 0.03,
            'velocity_noise_std': 0.05,
            'timestep': 0.1,
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
    print(f"Frequency: 10 Hz (dt = 0.1s)")
    print(f"Total trajectories: {len(all_trajectories)}")
    print(f"Average length: {np.mean(trajectory_lengths):.1f} steps ({np.mean(trajectory_lengths) * 0.1:.1f} seconds)")
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
        max_steps=10000,
        save_path='predictive_module/data/kgru_training_data_realistic.pkl'
    )