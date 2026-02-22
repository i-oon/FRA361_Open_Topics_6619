# merge_datasets_balanced.py
"""
Merge ETH/UCY pedestrians with synthetic vehicles
WITH BALANCED SAMPLING AND 10 Hz STANDARDIZATION
"""

import numpy as np
import pickle
import sys
import os
from scipy import interpolate
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resample_to_10hz(traj, source_dt=0.4, target_dt=0.1):
    """
    Resample trajectory from 2.5 Hz (ETH/UCY) to 10 Hz
    
    Args:
        traj: Trajectory array [n, 4] at 2.5 Hz
        source_dt: Original timestep (0.4s = 2.5 Hz for ETH/UCY)
        target_dt: Target timestep (0.1s = 10 Hz)
    
    Returns:
        Resampled trajectory at 10 Hz
    """
    if len(traj) < 2:
        return traj
    
    # Time arrays
    t_orig = np.arange(len(traj)) * source_dt
    t_new = np.arange(0, t_orig[-1], target_dt)
    
    if len(t_new) < 2:
        return traj
    
    # Interpolate positions
    f_x = interpolate.interp1d(t_orig, traj[:, 0], kind='cubic', fill_value='extrapolate')
    f_y = interpolate.interp1d(t_orig, traj[:, 1], kind='cubic', fill_value='extrapolate')
    
    new_positions = np.column_stack([f_x(t_new), f_y(t_new)])
    
    # Calculate velocities from position changes
    new_velocities = np.zeros_like(new_positions)
    for i in range(len(new_positions) - 1):
        dx = new_positions[i+1, 0] - new_positions[i, 0]
        dy = new_positions[i+1, 1] - new_positions[i, 1]
        new_velocities[i, 0] = dx / target_dt
        new_velocities[i, 1] = dy / target_dt
    
    new_velocities[-1] = new_velocities[-2]
    
    return np.column_stack([new_positions, new_velocities])


def load_eth_ucy_data(filepath='predictive_module/data/eth_ucy_processed.pkl'):
    """Load ETH/UCY pedestrian data and resample to 10 Hz"""
    print("Loading ETH/UCY pedestrian data...")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    trajectories = data['trajectories']
    print(f"  ✓ Loaded {len(trajectories)} pedestrian trajectories (2.5 Hz)")
    
    # Resample to 10 Hz
    print("  Resampling from 2.5 Hz to 10 Hz...")
    resampled = []
    
    for traj in tqdm(trajectories, desc="  Resampling ETH/UCY"):
        rs_traj = resample_to_10hz(traj, source_dt=0.4, target_dt=0.1)
        if len(rs_traj) >= 20:
            resampled.append(rs_traj)
    
    trajectories = resampled
    
    # Calculate statistics
    lengths = [len(t) for t in trajectories]
    speeds = []
    for traj in trajectories:
        traj_speeds = np.linalg.norm(traj[:, 2:4], axis=1)
        speeds.extend(traj_speeds.tolist())
    
    speeds = np.array(speeds)
    
    print(f"  ✓ Resampled to {len(trajectories)} trajectories (10 Hz)")
    print(f"  ✓ Avg length: {np.mean(lengths):.1f} frames ({np.mean(lengths)*0.1:.1f} sec)")
    print(f"  ✓ Avg speed: {speeds.mean():.2f} m/s")
    
    stats = {
        'mean_length': np.mean(lengths),
        'mean_speed': speeds.mean(),
        'std_speed': speeds.std()
    }
    
    return trajectories, stats


def load_and_truncate_vehicles(
    filepath='predictive_module/data/kgru_training_data_realistic.pkl',
    target_length=180,
    min_length=20
):
    """Load synthetic vehicles at 10 Hz and truncate to reasonable length"""
    print("\nLoading synthetic vehicles...")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    trajectories = data['trajectories']
    dt = data.get('dt', None)
    
    if dt is None:
        print(f"  ⚠️  WARNING: dt not found in synthetic data!")
        print(f"  Assuming dt=0.1 (check if correct)")
        dt = 0.1
    elif dt != 0.1:
        print(f"  ⚠️  WARNING: Expected dt=0.1, got dt={dt}")
        print(f"  Data may not be at 10 Hz! Recollect with fixed script.")
        return []
    
    print(f"  ✓ Data timestep: {dt}s ({1/dt:.0f} Hz)")
    
    # Filter and truncate vehicles
    vehicle_trajectories = []
    
    for traj in trajectories:
        speeds = np.linalg.norm(traj[:, 2:4], axis=1)
        avg_speed = speeds.mean()
        
        if avg_speed >= 2.0:  # Vehicle threshold
            if len(traj) > target_length:
                max_start = len(traj) - target_length
                start_idx = np.random.randint(0, max_start + 1)
                truncated = traj[start_idx:start_idx + target_length]
                vehicle_trajectories.append(truncated)
            elif len(traj) >= min_length:
                vehicle_trajectories.append(traj)
    
    print(f"  ✓ High-speed vehicles extracted: {len(vehicle_trajectories)}")
    print(f"  ✓ Truncated to max {target_length} frames ({target_length*0.1:.1f} sec)")
    
    if len(vehicle_trajectories) > 0:
        lengths = [len(v) for v in vehicle_trajectories]
        speeds = []
        for traj in vehicle_trajectories:
            traj_speeds = np.linalg.norm(traj[:, 2:4], axis=1)
            speeds.extend(traj_speeds.tolist())
        
        speeds = np.array(speeds)
        print(f"  ✓ Avg length: {np.mean(lengths):.1f} frames ({np.mean(lengths)*0.1:.1f} sec)")
        print(f"  ✓ Avg speed: {speeds.mean():.2f} m/s")
    
    return vehicle_trajectories


def create_balanced_hybrid_dataset(
    pedestrian_ratio=0.7,
    vehicle_target_length=180,
    save_path='predictive_module/data/kgru_training_data_hybrid.pkl'
):
    """Create BALANCED hybrid dataset at 10 Hz"""
    
    print("="*60)
    print("CREATING BALANCED HYBRID DATASET (10 Hz)")
    print("="*60)
    print(f"Target: {pedestrian_ratio*100:.0f}% ped / {(1-pedestrian_ratio)*100:.0f}% veh")
    print(f"Frequency: 10 Hz (dt = 0.1s)")
    print()
    
    # Load pedestrians (resampled to 10 Hz)
    pedestrians, ped_stats = load_eth_ucy_data()
    
    # Load vehicles (already at 10 Hz)
    vehicles = load_and_truncate_vehicles(
        target_length=vehicle_target_length,
        min_length=20
    )
    
    if len(vehicles) == 0:
        print("\n⚠️ No vehicles available!")
        print("Run: python data_collection_realistic.py")
        return None, None
    
    # Calculate target counts
    n_peds = len(pedestrians)
    n_vehs_target = int(n_peds * (1 - pedestrian_ratio) / pedestrian_ratio)
    
    print(f"\nTarget composition:")
    print(f"  Pedestrians: {n_peds}")
    print(f"  Vehicles: {n_vehs_target}")
    
    # Sample vehicles
    if n_vehs_target > len(vehicles):
        print(f"  Oversampling vehicles ({len(vehicles)} → {n_vehs_target})")
        indices = np.random.choice(len(vehicles), n_vehs_target, replace=True)
    else:
        indices = np.random.choice(len(vehicles), n_vehs_target, replace=False)
    
    sampled_vehicles = [vehicles[i] for i in indices]
    
    # Combine
    all_trajectories = pedestrians + sampled_vehicles
    np.random.shuffle(all_trajectories)
    
    composition = {
        'pedestrians': n_peds,
        'vehicles': len(sampled_vehicles)
    }
    
    # Calculate statistics
    print("\n" + "="*60)
    print("BALANCED DATASET STATISTICS (10 Hz)")
    print("="*60)
    
    lengths = [len(traj) for traj in all_trajectories]
    all_speeds = []
    for traj in all_trajectories:
        speeds = np.linalg.norm(traj[:, 2:4], axis=1)
        all_speeds.extend(speeds.tolist())
    
    all_speeds = np.array(all_speeds)
    
    n_total = len(all_trajectories)
    n_peds = composition['pedestrians']
    n_vehs = composition['vehicles']
    
    print(f"\nTrajectory Composition:")
    print(f"  Total: {n_total}")
    print(f"  Pedestrians: {n_peds} ({n_peds/n_total*100:.1f}%)")
    print(f"  Vehicles: {n_vehs} ({n_vehs/n_total*100:.1f}%)")
    
    print(f"\nTrajectory Length (10 Hz):")
    print(f"  Mean: {np.mean(lengths):.1f} frames ({np.mean(lengths)*0.1:.1f} sec)")
    print(f"  Range: {np.min(lengths)} - {np.max(lengths)} frames")
    
    # Sample distribution
    n_low = (all_speeds < 2.0).sum()
    n_high = (all_speeds >= 2.0).sum()
    
    print(f"\nSample Distribution:")
    print(f"  Total samples: {len(all_speeds):,}")
    print(f"  Low-speed (<2.0 m/s): {n_low:,} ({n_low/len(all_speeds)*100:.1f}%)")
    print(f"  High-speed (≥2.0 m/s): {n_high:,} ({n_high/len(all_speeds)*100:.1f}%)")
    
    print(f"\nSpeed Statistics:")
    print(f"  Mean: {all_speeds.mean():.2f} m/s")
    print(f"  Std: {all_speeds.std():.2f} m/s")
    
    # Check balance
    sample_ratio = n_low / len(all_speeds)
    traj_ratio = n_peds / n_total
    
    print(f"\nBalance Check:")
    print(f"  Trajectory ratio (ped): {traj_ratio*100:.1f}%")
    print(f"  Sample ratio (low-speed): {sample_ratio*100:.1f}%")
    
    if abs(traj_ratio - sample_ratio) < 0.15:
        print(f"  ✅ BALANCED!")
    else:
        print(f"  ⚠️ Imbalanced (diff {abs(traj_ratio-sample_ratio)*100:.1f}%)")
    
    # Save
    data = {
        'trajectories': all_trajectories,
        'dt': 0.1,  # CRITICAL!
        'frequency': '10 Hz',
        'source': 'Balanced Hybrid: ETH/UCY (resampled to 10 Hz) + Synthetic (10 Hz)',
        'composition': composition,
        'statistics': {
            'n_trajectories': n_total,
            'mean_length': np.mean(lengths),
            'mean_speed': all_speeds.mean(),
            'std_speed': all_speeds.std(),
            'low_speed_pct': sample_ratio * 100,
            'high_speed_pct': (1 - sample_ratio) * 100,
        }
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    size_mb = os.path.getsize(save_path) / 1024 / 1024
    
    print(f"\n✅ Balanced 10 Hz dataset saved!")
    print(f"   Path: {save_path}")
    print(f"   Size: {size_mb:.1f} MB")
    
    return all_trajectories, data['statistics']


if __name__ == "__main__":
    trajectories, stats = create_balanced_hybrid_dataset(
        pedestrian_ratio=0.7,
        vehicle_target_length=180,
        save_path='predictive_module/data/kgru_training_data_hybrid.pkl'
    )
    
    if trajectories is not None:
        print("\n" + "="*60)
        print("✅ READY FOR TRAINING AT 10 Hz!")
        print("="*60)
        print("\nNext: python train_kgru.py")