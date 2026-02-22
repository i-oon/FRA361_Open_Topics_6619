# merge_datasets_balanced.py
"""
Merge ETH/UCY pedestrians with synthetic vehicles
WITH BALANCED SAMPLING - truncates long trajectories
"""

import numpy as np
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_eth_ucy_data(filepath='predictive_module/data/eth_ucy_processed.pkl'):
    """Load ETH/UCY pedestrian data"""
    print("Loading ETH/UCY pedestrian data...")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    trajectories = data['trajectories']
    stats = data.get('statistics', {})
    
    print(f"  ✓ Loaded {len(trajectories)} pedestrian trajectories")
    print(f"  ✓ Avg length: {stats.get('mean_length', 0):.1f} frames")
    print(f"  ✓ Avg speed: {stats.get('mean_speed', 0):.2f} m/s")
    
    return trajectories, stats


def load_and_truncate_vehicles(
    filepath='predictive_module/data/kgru_training_data_realistic.pkl',
    target_length=60,
    min_length=20
):
    """
    Load synthetic vehicles and truncate to reasonable length
    
    Args:
        target_length: Target trajectory length (frames)
        min_length: Minimum acceptable length
    """
    print("\nLoading and truncating synthetic vehicles...")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    trajectories = data['trajectories']
    
    # Filter and truncate vehicles
    vehicle_trajectories = []
    
    for traj in trajectories:
        # Check if high-speed (vehicle)
        speeds = np.linalg.norm(traj[:, 2:4], axis=1)
        avg_speed = speeds.mean()
        
        if avg_speed >= 2.0:  # Vehicle threshold
            # Truncate if too long
            if len(traj) > target_length:
                # Random crop to avoid bias
                max_start = len(traj) - target_length
                start_idx = np.random.randint(0, max_start + 1)
                truncated = traj[start_idx:start_idx + target_length]
                vehicle_trajectories.append(truncated)
            elif len(traj) >= min_length:
                # Keep as is if already reasonable length
                vehicle_trajectories.append(traj)
    
    print(f"  ✓ Total synthetic trajectories: {len(trajectories)}")
    print(f"  ✓ High-speed vehicles extracted: {len(vehicle_trajectories)}")
    print(f"  ✓ Truncated to max {target_length} frames")
    
    # Calculate statistics
    if len(vehicle_trajectories) > 0:
        lengths = [len(v) for v in vehicle_trajectories]
        speeds = []
        for traj in vehicle_trajectories:
            traj_speeds = np.linalg.norm(traj[:, 2:4], axis=1)
            speeds.extend(traj_speeds.tolist())
        
        speeds = np.array(speeds)
        print(f"  ✓ Avg length: {np.mean(lengths):.1f} frames")
        print(f"  ✓ Avg speed: {speeds.mean():.2f} m/s")
    
    return vehicle_trajectories


def create_balanced_hybrid_dataset(
    pedestrian_ratio=0.7,
    vehicle_target_length=60,
    save_path='predictive_module/data/kgru_training_data_hybrid.pkl'
):
    """
    Create BALANCED hybrid dataset
    
    Key improvement: Truncates vehicle trajectories to match pedestrian length
    """
    
    print("="*60)
    print("CREATING BALANCED HYBRID DATASET")
    print("="*60)
    print(f"Target: {pedestrian_ratio*100:.0f}% ped / {(1-pedestrian_ratio)*100:.0f}% veh (by trajectory)")
    print(f"Vehicle max length: {vehicle_target_length} frames")
    print()
    
    # Load pedestrians
    pedestrians, ped_stats = load_eth_ucy_data()
    
    # Load and truncate vehicles
    vehicles = load_and_truncate_vehicles(
        target_length=vehicle_target_length,
        min_length=20
    )
    
    if len(vehicles) == 0:
        print("\n⚠️ No vehicles available, using pedestrians only")
        all_trajectories = pedestrians
        composition = {'pedestrians': len(pedestrians), 'vehicles': 0}
    else:
        # Calculate target counts
        n_peds = len(pedestrians)
        n_vehs_target = int(n_peds * (1 - pedestrian_ratio) / pedestrian_ratio)
        
        print(f"\nTarget composition:")
        print(f"  Pedestrians: {n_peds}")
        print(f"  Vehicles: {n_vehs_target}")
        
        # Sample vehicles
        if n_vehs_target > len(vehicles):
            print(f"  ⚠️ Oversampling vehicles ({len(vehicles)} → {n_vehs_target})")
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
    print("BALANCED DATASET STATISTICS")
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
    
    print(f"\nTrajectory Length:")
    print(f"  Mean: {np.mean(lengths):.1f} frames ({np.mean(lengths)*0.4:.1f} sec)")
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
    print(f"  Range: {all_speeds.min():.2f} - {all_speeds.max():.2f} m/s")
    
    # Check balance
    sample_ratio = n_low / len(all_speeds)
    traj_ratio = n_peds / n_total
    
    print(f"\nBalance Check:")
    print(f"  Trajectory ratio (ped): {traj_ratio*100:.1f}%")
    print(f"  Sample ratio (low-speed): {sample_ratio*100:.1f}%")
    
    if abs(traj_ratio - sample_ratio) < 0.15:
        print(f"  ✅ BALANCED! (difference < 15%)")
    else:
        print(f"  ⚠️ Imbalanced (difference {abs(traj_ratio-sample_ratio)*100:.1f}%)")
    
    # Save
    data = {
        'trajectories': all_trajectories,
        'source': 'Balanced Hybrid: ETH/UCY + Synthetic (truncated)',
        'composition': composition,
        'vehicle_truncation': vehicle_target_length,
        'statistics': {
            'n_trajectories': n_total,
            'mean_length': np.mean(lengths),
            'mean_speed': all_speeds.mean(),
            'std_speed': all_speeds.std(),
            'low_speed_pct': sample_ratio * 100,
            'high_speed_pct': (1 - sample_ratio) * 100,
            'balance_score': 1 - abs(traj_ratio - sample_ratio)
        }
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    size_mb = os.path.getsize(save_path) / 1024 / 1024
    
    print(f"\n✅ Balanced dataset saved!")
    print(f"   Path: {save_path}")
    print(f"   Size: {size_mb:.1f} MB")
    
    return all_trajectories, data['statistics']


if __name__ == "__main__":
    trajectories, stats = create_balanced_hybrid_dataset(
        pedestrian_ratio=0.7,
        vehicle_target_length=60,  # Match pedestrian avg length
        save_path='predictive_module/data/kgru_training_data_hybrid.pkl'
    )
    
    if len(trajectories) > 0:
        print("\n" + "="*60)
        print("✅ READY FOR TRAINING!")
        print("="*60)
        print("\nNext: python3 predictive_module/train_kgru.py")