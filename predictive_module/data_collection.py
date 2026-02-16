# data_collection.py
"""
Diverse data collection for K-GRU training
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.dynamic_nav_env import DynamicObstacleNavEnv
from tqdm import tqdm
import pickle

def collect_training_data(
    n_episodes=500,
    save_path='kgru_training_data.pkl'
):
    """
    Collect diverse obstacle trajectory data
    
    Variations:
    - Obstacle count: 3-10 obstacles
    - Speed distributions: different low/high ratios
    - Episode lengths: ensure obstacles move enough
    """
    
    all_trajectories = []
    metadata = {
        'episodes': [],
        'obstacle_counts': [],
        'speed_distributions': []
    }
    
    print("Collecting training data...")
    print("="*60)
    
    # Configuration variations
    obstacle_counts = [3, 5, 7, 10]
    low_speed_ratios = [0.3, 0.5, 0.7]
    
    episodes_per_config = n_episodes // (len(obstacle_counts) * len(low_speed_ratios))
    episode_num = 0
    
    for n_obs in obstacle_counts:
        for low_ratio in low_speed_ratios:
            print(f"\nConfig: {n_obs} obstacles, {int(low_ratio*100)}% low-speed")
            
            # Create environment with this configuration
            env = DynamicObstacleNavEnv(
                n_obstacles=n_obs,
                low_speed_ratio=low_ratio,
                max_episode_steps=2000,
                render_mode=None
            )
            
            config_trajectories = []
            
            for ep in tqdm(range(episodes_per_config), desc=f"  Episodes"):
                obs, info = env.reset()
                
                # Track each obstacle separately
                obstacle_data = {i: [] for i in range(n_obs)}
                
                # Run episode
                for step in range(2000):
                    # Random action (we only care about obstacle motion)
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    # Record obstacle states with IDs
                    for i, obs_info in enumerate(info['obstacles']):
                        state = np.array([
                            obs_info['pos'][0],
                            obs_info['pos'][1],
                            obs_info['vel'][0],
                            obs_info['vel'][1]
                        ])
                        obstacle_data[i].append(state)
                    
                    if terminated or truncated:
                        break
                
                # Store trajectories (only if sufficient length)
                for obs_id, traj in obstacle_data.items():
                    if len(traj) >= 20:  # Minimum 20 timesteps
                        traj_array = np.array(traj)
                        config_trajectories.append(traj_array)
                        all_trajectories.append(traj_array)
                
                episode_num += 1
            
            # Store metadata
            metadata['episodes'].append(episodes_per_config)
            metadata['obstacle_counts'].append(n_obs)
            metadata['speed_distributions'].append(low_ratio)
            
            print(f"  Collected {len(config_trajectories)} trajectories from this config")
            
            env.close()
    
    # Analyze collected data
    print("\n" + "="*60)
    print("DATA COLLECTION SUMMARY")
    print("="*60)
    print(f"Total episodes: {episode_num}")
    print(f"Total trajectories: {len(all_trajectories)}")
    
    # Analyze trajectory lengths
    lengths = [len(traj) for traj in all_trajectories]
    print(f"\nTrajectory lengths:")
    print(f"  Min: {min(lengths)} timesteps")
    print(f"  Max: {max(lengths)} timesteps")
    print(f"  Mean: {np.mean(lengths):.1f} timesteps")
    print(f"  Median: {np.median(lengths):.1f} timesteps")
    
    # Analyze speed distributions
    all_speeds = []
    for traj in all_trajectories:
        speeds = np.linalg.norm(traj[:, 2:4], axis=1)  # vx, vy magnitude
        all_speeds.extend(speeds)
    
    all_speeds = np.array(all_speeds)
    print(f"\nSpeed distribution:")
    print(f"  Min: {all_speeds.min():.3f} m/s")
    print(f"  Max: {all_speeds.max():.3f} m/s")
    print(f"  Mean: {all_speeds.mean():.3f} m/s")
    print(f"  Std: {all_speeds.std():.3f} m/s")
    print(f"  Low-speed (<0.3 m/s): {(all_speeds < 0.3).sum() / len(all_speeds) * 100:.1f}%")
    print(f"  High-speed (>0.5 m/s): {(all_speeds > 0.5).sum() / len(all_speeds) * 100:.1f}%")
    
    # Save data
    data = {
        'trajectories': all_trajectories,
        'metadata': metadata,
        'statistics': {
            'n_trajectories': len(all_trajectories),
            'trajectory_lengths': lengths,
            'speed_distribution': all_speeds
        }
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nData saved to {save_path}")
    print("="*60)
    
    return all_trajectories, metadata


if __name__ == "__main__":
    trajectories, metadata = collect_training_data(
        n_episodes=500,
        save_path='kgru_training_data.pkl'
    )