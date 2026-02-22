# preprocess_eth_ucy_final.py
"""
Preprocess ETH/UCY pedestrian datasets from Social-GAN
Uses raw/all_data folder for complete datasets
"""

import numpy as np
import pickle
import os
from collections import defaultdict
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_dataset_file(filepath, delim='\t'):
    """
    Load ETH/UCY format file
    Format: frame_id  pedestrian_id  x  y (tab or space separated)
    """
    data = []
    print(f"  Loading {os.path.basename(filepath)}...")
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Try tab delimiter first, then space
            parts = line.split(delim) if '\t' in line else line.split()
            
            if len(parts) >= 4:
                try:
                    frame_id = int(float(parts[0]))
                    ped_id = int(float(parts[1]))
                    x = float(parts[2])
                    y = float(parts[3])
                    data.append([frame_id, ped_id, x, y])
                except (ValueError, IndexError):
                    continue
    
    print(f"    → {len(data)} frames loaded")
    return np.array(data) if data else np.array([]).reshape(0, 4)


def extract_trajectories(data, fps=2.5, min_length=20):
    """
    Extract pedestrian trajectories and compute velocities
    
    Args:
        data: Array of [frame_id, ped_id, x, y]
        fps: Frames per second (2.5 for ETH/UCY)
        min_length: Minimum trajectory length in frames
    
    Returns:
        List of trajectories [x, y, vx, vy] × timesteps
    """
    
    if len(data) == 0:
        return []
    
    # Group by pedestrian ID
    ped_data = defaultdict(list)
    for row in data:
        frame_id, ped_id, x, y = row
        ped_data[ped_id].append([int(frame_id), x, y])
    
    trajectories = []
    dt = 1.0 / fps  # 0.4 seconds
    
    for ped_id, frames in ped_data.items():
        if len(frames) < min_length:
            continue
        
        # Sort by frame
        frames = sorted(frames, key=lambda x: x[0])
        
        trajectory = []
        for i in range(len(frames)):
            frame_id, x, y = frames[i]
            
            # Compute velocity
            if i == 0:
                # Forward difference
                if len(frames) > 1:
                    _, x_next, y_next = frames[i + 1]
                    vx = (x_next - x) / dt
                    vy = (y_next - y) / dt
                else:
                    vx, vy = 0.0, 0.0
            elif i == len(frames) - 1:
                # Backward difference
                _, x_prev, y_prev = frames[i - 1]
                vx = (x - x_prev) / dt
                vy = (y - y_prev) / dt
            else:
                # Central difference
                _, x_prev, y_prev = frames[i - 1]
                _, x_next, y_next = frames[i + 1]
                vx = (x_next - x_prev) / (2 * dt)
                vy = (y_next - y_prev) / (2 * dt)
            
            trajectory.append([x, y, vx, vy])
        
        trajectories.append(np.array(trajectory))
    
    return trajectories


def process_all_datasets():
    """Process all ETH/UCY datasets from raw/all_data folder"""
    
    # Path to raw data
    base_path = 'predictive_module/data/sgan/scripts/datasets/raw/all_data'
    
    # All dataset files
    dataset_files = [
        'biwi_eth.txt',         # ETH
        'biwi_hotel.txt',       # Hotel
        'crowds_zara01.txt',    # Zara1
        'crowds_zara02.txt',    # Zara2
        'crowds_zara03.txt',    # Zara3
        'students001.txt',      # UCY students 1
        'students003.txt',      # UCY students 3
        'uni_examples.txt'      # UCY university
    ]
    
    all_trajectories = []
    
    print("="*60)
    print("PREPROCESSING ETH/UCY DATASETS")
    print("="*60)
    print(f"Data path: {base_path}")
    print(f"Datasets: {len(dataset_files)}")
    print()
    
    for dataset_file in dataset_files:
        filepath = os.path.join(base_path, dataset_file)
        
        if not os.path.exists(filepath):
            print(f"⚠️  Skipping {dataset_file} (not found)")
            continue
        
        print(f"Processing {dataset_file}...")
        
        # Load data
        data = load_dataset_file(filepath)
        if len(data) == 0:
            print(f"  ⚠️  No valid data\n")
            continue
        
        # Extract trajectories
        trajectories = extract_trajectories(data, fps=2.5, min_length=20)
        print(f"  ✓ Extracted {len(trajectories)} trajectories")
        
        if len(trajectories) > 0:
            # Statistics
            lengths = [len(traj) for traj in trajectories]
            speeds = []
            for traj in trajectories:
                traj_speeds = np.linalg.norm(traj[:, 2:4], axis=1)
                speeds.extend(traj_speeds.tolist())
            
            print(f"  ✓ Avg length: {np.mean(lengths):.1f} frames")
            print(f"  ✓ Avg speed: {np.mean(speeds):.2f} m/s\n")
            
            all_trajectories.extend(trajectories)
    
    if len(all_trajectories) == 0:
        print("\n❌ ERROR: No trajectories extracted!")
        return [], {}
    
    print("="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total trajectories: {len(all_trajectories)}")
    
    # Overall statistics
    all_lengths = [len(traj) for traj in all_trajectories]
    all_speeds = []
    for traj in all_trajectories:
        traj_speeds = np.linalg.norm(traj[:, 2:4], axis=1)
        all_speeds.extend(traj_speeds.tolist())
    
    all_speeds = np.array(all_speeds)
    
    print(f"\nOverall Statistics:")
    print(f"  Trajectories: {len(all_trajectories)}")
    print(f"  Avg length: {np.mean(all_lengths):.1f} frames ({np.mean(all_lengths)*0.4:.1f} seconds)")
    print(f"  Speed: {np.mean(all_speeds):.2f} ± {np.std(all_speeds):.2f} m/s")
    print(f"  Speed range: {np.min(all_speeds):.2f} - {np.max(all_speeds):.2f} m/s")
    print(f"  Low-speed (<2.0 m/s): {(all_speeds < 2.0).sum() / len(all_speeds) * 100:.1f}%")
    print(f"  High-speed (≥2.0 m/s): {(all_speeds >= 2.0).sum() / len(all_speeds) * 100:.1f}%")
    
    return all_trajectories, {
        'n_trajectories': len(all_trajectories),
        'mean_length': np.mean(all_lengths),
        'mean_speed': np.mean(all_speeds),
        'std_speed': np.std(all_speeds),
        'speed_range': (np.min(all_speeds), np.max(all_speeds)),
        'low_speed_pct': (all_speeds < 2.0).sum() / len(all_speeds) * 100
    }


def save_processed_data(trajectories, stats, save_path='predictive_module/data/eth_ucy_processed.pkl'):
    """Save processed data"""
    
    data = {
        'trajectories': trajectories,
        'source': 'ETH/UCY pedestrian datasets',
        'datasets': [
            'biwi_eth', 'biwi_hotel', 
            'crowds_zara01', 'crowds_zara02', 'crowds_zara03',
            'students001', 'students003', 'uni_examples'
        ],
        'statistics': stats,
        'motion_model_params': {
            'type': 'real_pedestrian',
            'fps': 2.5,
            'min_trajectory_length': 20
        }
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n✅ Data saved to: {save_path}")
    print(f"   File size: {os.path.getsize(save_path) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    # Process all datasets
    trajectories, stats = process_all_datasets()
    
    if len(trajectories) > 0:
        # Save
        save_processed_data(trajectories, stats)
        
        print("\n" + "="*60)
        print("✅ ETH/UCY DATA READY!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Merge with synthetic vehicles:")
        print("     python3 predictive_module/merge_datasets.py")
        print("\n  2. Train K-GRU with hybrid data:")
        print("     python3 predictive_module/train_kgru.py")
    else:
        print("\n❌ FAILED - No data extracted")
        print("Check directory structure:")
        print("  ls -la predictive_module/data/sgan/scripts/datasets/raw/all_data/")