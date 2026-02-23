# data_collection_realistic.py

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.dynamic_nav_env import DynamicObstacleNavEnv
from tqdm import tqdm
import pickle


def add_sensor_noise(state):
    """Add realistic sensor noise"""
    noisy_state = state.copy()
    noisy_state[:2] += np.random.normal(0, 0.03, 2)
    noisy_state[2:] += np.random.normal(0, 0.05, 2)
    return noisy_state


def extract_obstacle_states(obs, n_obstacles):
    """Extract obstacle states from observation vector"""
    robot_and_goal_dim = 10
    
    obstacle_states = []
    obstacle_dim = 5
    
    for i in range(n_obstacles):
        start_idx = robot_and_goal_dim + i * obstacle_dim
        end_idx = start_idx + obstacle_dim
        
        if end_idx <= len(obs):
            rel_state = obs[start_idx:end_idx]
            robot_pos = obs[0:2]
            abs_pos = rel_state[0:2] + robot_pos
            vel = rel_state[2:4]
            state = np.concatenate([abs_pos, vel])
            obstacle_states.append(state)
    
    return obstacle_states


def smooth_velocities(trajectory, window=3):
    """Apply moving average to velocities"""
    if len(trajectory) < window:
        return trajectory
    
    smoothed = trajectory.copy()
    
    for i in range(window-1, len(trajectory)):
        start_idx = i - window + 1
        end_idx = i + 1
        smoothed[i, 2:4] = trajectory[start_idx:end_idx, 2:4].mean(axis=0)
    
    return smoothed


def downsample_and_validate(trajectories, target_dt=0.1, source_dt=0.002):
    """
    Downsample from 500 Hz to 10 Hz, THEN validate
    Wall bounces are smoothed out at 10 Hz
    """
    factor = int(target_dt / source_dt)
    
    print(f"\nDownsampling and validating:")
    print(f"  From: {source_dt}s ({1/source_dt:.0f} Hz)")
    print(f"  To:   {target_dt}s ({1/target_dt:.0f} Hz)")
    print(f"  Factor: {factor}")
    
    valid_trajectories = []
    too_short = 0
    bad_speed = 0
    
    for traj in tqdm(trajectories, desc="  Processing"):
        # Need enough data to downsample
        if len(traj) < factor * 20:
            too_short += 1
            continue
        
        # Downsample
        ds_traj = traj[::factor].copy()
        
        # Recalculate velocities from positions (smooths wall bounces!)
        for i in range(len(ds_traj) - 1):
            dx = ds_traj[i+1, 0] - ds_traj[i, 0]
            dy = ds_traj[i+1, 1] - ds_traj[i, 1]
            ds_traj[i, 2] = dx / target_dt
            ds_traj[i, 3] = dy / target_dt
        
        ds_traj[-1, 2:4] = ds_traj[-2, 2:4]
        
        # NOW validate (at 10 Hz, wall bounces are smooth)
        speeds = np.linalg.norm(ds_traj[:, 2:4], axis=1)
        max_speed = speeds.max()
        
        if max_speed > 5.0:
            bad_speed += 1
            continue
        
        # Smooth velocities
        ds_traj = smooth_velocities(ds_traj, window=3)
        
        # Final check
        if len(ds_traj) >= 20:
            valid_trajectories.append(ds_traj)
        else:
            too_short += 1
    
    print(f"\n  ✅ Valid: {len(valid_trajectories)}")
    print(f"  ❌ Too short: {too_short}")
    print(f"  ❌ Bad speed (>{5.0} m/s): {bad_speed}")
    
    return valid_trajectories


def collect_realistic_training_data(
    n_episodes=600,
    max_steps=5000,
    save_path='predictive_module/data/synthetic_mixed_traffic.pkl'
):
    """Collect training data"""
    
    configurations = [
        (5, 1.0),   # All pedestrians
        (5, 0.7),   # Mostly pedestrians
        (8, 0.5),   # Mixed
        (8, 0.3),   # Mostly vehicles
        (10, 0.2),  # Mostly vehicles
    ]
    
    episodes_per_config = n_episodes // len(configurations)
    
    all_trajectories = []
    
    print("="*60)
    print("COLLECTING TRAINING DATA")
    print("="*60)
    print(f"Total episodes: {n_episodes}")
    print(f"Episode length: {max_steps} steps ({max_steps * 0.002:.1f}s)")
    print("="*60)
    
    for config_idx, (n_obs, low_ratio) in enumerate(configurations):
        print(f"\nConfig {config_idx+1}/{len(configurations)}: {n_obs} obstacles, {low_ratio*100:.0f}% low-speed")
        
        for episode in tqdm(range(episodes_per_config), desc=f"  Collecting"):
            env = DynamicObstacleNavEnv(
                n_obstacles=n_obs,
                low_speed_ratio=low_ratio,
                render_mode=None,
            )
            
            obs, _ = env.reset()
            episode_trajectories = {i: [] for i in range(n_obs)}
            
            for step in range(max_steps):
                obstacle_states = extract_obstacle_states(obs, n_obs)
                
                for obs_id, state in enumerate(obstacle_states):
                    noisy_state = add_sensor_noise(state)
                    episode_trajectories[obs_id].append(noisy_state)
                
                action = np.array([0.0, 0.0, 0.0])
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
            
            # Collect raw trajectories (no validation yet!)
            for obs_id, trajectory in episode_trajectories.items():
                if len(trajectory) >= 100:  # Minimum raw length
                    all_trajectories.append(np.array(trajectory))
            
            env.close()
    
    print(f"\n✅ Collected {len(all_trajectories)} raw trajectories")
    
    # Downsample AND validate (in one step)
    all_trajectories = downsample_and_validate(all_trajectories, target_dt=0.1, source_dt=0.002)
    
    if len(all_trajectories) == 0:
        print("\n❌ NO VALID TRAJECTORIES!")
        print("   Check environment settings or validation criteria")
        return
    
    # Statistics
    trajectory_lengths = [len(traj) for traj in all_trajectories]
    speeds = []
    for traj in all_trajectories:
        traj_speeds = np.linalg.norm(traj[:, 2:4], axis=1)
        speeds.extend(traj_speeds.tolist())
    
    speeds = np.array(speeds)
    
    # K-means preview
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    traj_speeds = [np.linalg.norm(traj[:, 2:4], axis=1).mean() for traj in all_trajectories]
    labels = kmeans.fit_predict(np.array(traj_speeds).reshape(-1, 1))
    centers = kmeans.cluster_centers_.flatten()
    boundary = (centers.min() + centers.max()) / 2
    low_center = centers.min()
    high_center = centers.max()
    
    low_cluster = 0 if centers[0] < centers[1] else 1
    n_low = np.sum(labels == low_cluster)
    n_high = np.sum(labels == 1 - low_cluster)
    
    # Save
    data = {
        'trajectories': all_trajectories,
        'dt': 0.1,
        'source_frequency': '10 Hz (downsampled from 500 Hz)',
        'statistics': {
            'n_trajectories': len(all_trajectories),
            'mean_length': np.mean(trajectory_lengths),
            'mean_speed': speeds.mean(),
            'std_speed': speeds.std(),
            'min_speed': speeds.min(),
            'max_speed': speeds.max(),
            'kmeans_boundary': boundary,
            'kmeans_low_center': low_center,
            'kmeans_high_center': high_center,
            'kmeans_low_count': int(n_low),
            'kmeans_high_count': int(n_high),
        }
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    print("\n" + "="*60)
    print("COLLECTION COMPLETE")
    print("="*60)
    print(f"✅ Trajectories: {len(all_trajectories)}")
    print(f"✅ Length: {np.mean(trajectory_lengths):.1f} steps ({np.mean(trajectory_lengths)*0.1:.1f}s)")
    print(f"✅ Speed: {speeds.min():.2f} - {speeds.max():.2f} m/s")
    print(f"   Mean: {speeds.mean():.2f} ± {speeds.std():.2f} m/s")
    print(f"\n🎯 K-means Clustering:")
    print(f"   Boundary: {boundary:.2f} m/s")
    print(f"   Low-speed: {low_center:.2f} m/s (n={n_low}, {n_low/len(all_trajectories)*100:.1f}%)")
    print(f"   High-speed: {high_center:.2f} m/s (n={n_high}, {n_high/len(all_trajectories)*100:.1f}%)")
    print(f"\n💾 Saved: {save_path}")
    print("="*60)


if __name__ == "__main__":
    collect_realistic_training_data(
        n_episodes=600,
        max_steps=5000,
        save_path='predictive_module/data/synthetic_mixed_traffic.pkl'
    )