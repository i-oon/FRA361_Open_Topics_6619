"""
Preprocess inD at NATIVE 25 Hz WITH class labels
ALL BUGS FIXED:
- Velocities recalculated at dt=0.04s (NOT using CSV velocities)
- Speed validation after recalculation
- Native 25 Hz (no unnecessary downsampling)
- Class one-hot encoding
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans


def extract_trajectories_with_class(
    tracks, 
    meta, 
    fps, 
    min_length=50,  # 50 frames @ 25Hz = 2 seconds
    max_speed=25.0  # Reasonable max for intersection (90 km/h)
):
    """
    Extract trajectories at NATIVE 25 Hz with class labels
    
    CRITICAL FIXES:
    - Recalculate velocities at dt=0.04s (25 Hz)
    - Validate speeds after recalculation
    - No downsampling (preserve all data)
    """
    
    class_names = ['car', 'pedestrian', 'truck_bus', 'bicycle']
    
    trajectories = []
    rejected_speed = 0
    rejected_length = 0
    
    dt = 1.0 / fps  # 0.04s for 25 Hz
    
    for track_id in tqdm(tracks['trackId'].unique(), desc="  Processing", leave=False):
        track_data = tracks[tracks['trackId'] == track_id].sort_values('frame')
        track_meta = meta[meta['trackId'] == track_id]
        
        if len(track_meta) == 0:
            continue
        
        # Get class and one-hot encode
        track_class = track_meta['class'].values[0].strip('-')
        class_onehot = np.zeros(4)
        
        if track_class in class_names:
            class_idx = class_names.index(track_class)
            class_onehot[class_idx] = 1.0
        else:
            class_onehot[0] = 1.0  # Default to car
        
        # ✅ FIX 1: Extract POSITIONS only (ignore CSV velocities)
        positions = track_data[['xCenter', 'yCenter']].values
        
        # Check length BEFORE velocity calculation
        if len(positions) < min_length:
            rejected_length += 1
            continue
        
        # ✅ FIX 2: Recalculate velocities at correct dt
        velocities = np.zeros_like(positions)
        for i in range(len(positions) - 1):
            dx = positions[i+1, 0] - positions[i, 0]
            dy = positions[i+1, 1] - positions[i, 1]
            velocities[i, 0] = dx / dt
            velocities[i, 1] = dy / dt
        
        # Last velocity = copy previous
        velocities[-1] = velocities[-2] if len(velocities) > 1 else velocities[0]
        
        # ✅ FIX 3: Validate speeds AFTER recalculation
        speeds = np.linalg.norm(velocities, axis=1)
        if speeds.max() > max_speed:
            rejected_speed += 1
            continue
        
        # Combine: [x, y, vx, vy] (4D motion)
        traj_motion = np.concatenate([positions, velocities], axis=1)
        
        # Append class to each timestep: [x, y, vx, vy, c1, c2, c3, c4] (8D)
        traj_with_class = np.concatenate([
            traj_motion,
            np.tile(class_onehot, (len(traj_motion), 1))
        ], axis=1)
        
        trajectories.append(traj_with_class)
    
    if rejected_speed > 0 or rejected_length > 0:
        print(f"    Rejected: {rejected_length} too short, {rejected_speed} too fast (>{max_speed} m/s)")
    
    return trajectories


def process_ind(
    data_dir='predictive_module/data/inD/data',
    output_path='predictive_module/data/ind_with_class.pkl',
    recording_ids=None
):
    """Process inD at NATIVE 25 Hz with class labels"""
    
    if recording_ids is None:
        recording_ids = range(33)  # All 33 recordings
    
    print("="*70)
    print("PROCESSING inD AT NATIVE 25 Hz WITH CLASS LABELS")
    print("="*70)
    print("Features:")
    print("  ✅ Native 25 Hz (NO downsampling)")
    print("  ✅ Velocities recalculated at dt=0.04s")
    print("  ✅ Speed validation (max 25 m/s)")
    print("  ✅ Class one-hot encoding (8D input)")
    print("="*70)
    
    all_trajectories = []
    class_counts = {}
    
    for rec_id in recording_ids:
        print(f"\nRecording {rec_id:02d}:")
        
        try:
            tracks_file = Path(data_dir) / f'{rec_id:02d}_tracks.csv'
            meta_file = Path(data_dir) / f'{rec_id:02d}_tracksMeta.csv'
            recording_meta_file = Path(data_dir) / f'{rec_id:02d}_recordingMeta.csv'
            
            tracks = pd.read_csv(tracks_file)
            meta = pd.read_csv(meta_file)
            recording_meta = pd.read_csv(recording_meta_file)
            fps = recording_meta['frameRate'].values[0]
            
            print(f"  Source FPS: {fps} Hz")
            print(f"  Total tracks: {len(meta)}")
            
            # Extract at NATIVE frequency
            trajs = extract_trajectories_with_class(tracks, meta, fps)
            
            # Count classes
            for traj in trajs:
                # Decode class from one-hot
                class_vec = traj[0, 4:8]  # First timestep's class
                class_idx = np.argmax(class_vec)
                class_name = ['car', 'pedestrian', 'truck_bus', 'bicycle'][class_idx]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            all_trajectories.extend(trajs)
            print(f"  ✅ Extracted: {len(trajs)} valid trajectories")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_trajectories) == 0:
        print("\n❌ NO TRAJECTORIES EXTRACTED!")
        return
    
    # Calculate statistics
    all_speeds = []
    all_lengths = []
    
    for traj in all_trajectories:
        velocities = traj[:, 2:4]  # vx, vy
        speeds = np.linalg.norm(velocities, axis=1)
        all_speeds.append(speeds.mean())
        all_lengths.append(len(traj))
    
    all_speeds = np.array(all_speeds)
    all_lengths = np.array(all_lengths)
    
    # K-means preview
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(all_speeds.reshape(-1, 1))
    centers = kmeans.cluster_centers_.flatten()
    boundary = (centers.min() + centers.max()) / 2
    
    low_cluster = 0 if centers[0] < centers[1] else 1
    n_low = np.sum(labels == low_cluster)
    n_high = np.sum(labels == 1 - low_cluster)
    
    # Save
    data = {
        'trajectories': all_trajectories,
        'dt': 1.0 / 25.0,  # 0.04s
        'frequency': 25.0,
        'source': 'inD Dataset - Native 25 Hz (no downsampling)',
        'input_format': '[x, y, vx, vy, is_car, is_ped, is_truck, is_bicycle]',
        'feature_dim': 8,
        'class_names': ['car', 'pedestrian', 'truck_bus', 'bicycle'],
        'velocities_recalculated': True,
        'class_distribution': class_counts,
        'statistics': {
            'n_trajectories': len(all_trajectories),
            'mean_speed': all_speeds.mean(),
            'median_speed': np.median(all_speeds),
            'max_speed': all_speeds.max(),
            'min_speed': all_speeds.min(),
            'std_speed': all_speeds.std(),
            'mean_length': all_lengths.mean(),
            'median_length': np.median(all_lengths),
            'kmeans_boundary': boundary,
            'kmeans_low_center': centers.min(),
            'kmeans_high_center': centers.max(),
            'kmeans_balance': min(n_low, n_high) / max(n_low, n_high),
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    # Summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"✅ Total trajectories: {len(all_trajectories)}")
    
    print(f"\n📦 Class distribution:")
    for cls, count in sorted(class_counts.items()):
        pct = count / len(all_trajectories) * 100
        print(f"   {cls:12s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\n📊 Speed statistics:")
    print(f"   Mean:   {all_speeds.mean():.2f} m/s ({all_speeds.mean()*3.6:.1f} km/h)")
    print(f"   Median: {np.median(all_speeds):.2f} m/s ({np.median(all_speeds)*3.6:.1f} km/h)")
    print(f"   Range:  {all_speeds.min():.2f} - {all_speeds.max():.2f} m/s")
    print(f"   Std:    {all_speeds.std():.2f} m/s")
    
    print(f"\n📏 Trajectory length (@ 25 Hz):")
    print(f"   Mean:   {all_lengths.mean():.1f} frames ({all_lengths.mean()/25:.1f}s)")
    print(f"   Median: {np.median(all_lengths):.0f} frames ({np.median(all_lengths)/25:.1f}s)")
    print(f"   Range:  {all_lengths.min():.0f} - {all_lengths.max():.0f} frames")
    
    print(f"\n🎯 K-means clustering:")
    print(f"   Boundary: {boundary:.2f} m/s")
    print(f"   Low-speed:  {centers.min():.2f} m/s (n={n_low}, {n_low/len(all_speeds)*100:.1f}%)")
    print(f"   High-speed: {centers.max():.2f} m/s (n={n_high}, {n_high/len(all_speeds)*100:.1f}%)")
    print(f"   Balance: {min(n_low, n_high) / max(n_low, n_high):.2%}")
    
    print(f"\n💾 Saved to: {output_path}")
    print(f"   Format: 8D [motion(4) + class(4)]")
    print(f"   Temporal resolution: 25 Hz (dt=0.04s)")
    print("="*70)


if __name__ == "__main__":
    # Process all 33 recordings
    process_ind()
    
    # Or test with subset first:
    # process_ind(recording_ids=range(5))