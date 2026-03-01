"""
Verify dataset quality for K-GRU training
Checks: speed range, K-means balance, trajectory length, data integrity
"""

import numpy as np
import pickle
from sklearn.cluster import KMeans
import os


def verify_dataset(filepath, expected_name):
    """Check dataset properties and quality"""
    
    print(f"\n{'='*70}")
    print(f"Verifying: {expected_name}")
    print(f"File: {filepath}")
    print(f"{'='*70}")
    
    # Check file exists
    if not os.path.exists(filepath):
        print(f"❌ FILE NOT FOUND: {filepath}")
        return False
    
    try:
        # Load data
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        trajectories = data['trajectories']
        dt = data.get('dt', 0.1)
        
        print(f"✅ Loaded successfully")
        print(f"   Trajectories: {len(trajectories)}")
        print(f"   Temporal resolution: {dt}s ({1/dt:.1f} Hz)")
        
        # Extract all speeds
        all_speeds = []
        trajectory_lengths = []
        
        for traj in trajectories:
            speeds = np.linalg.norm(traj[:, 2:4], axis=1)
            all_speeds.append(speeds.mean())
            trajectory_lengths.append(len(traj))
        
        all_speeds = np.array(all_speeds)
        trajectory_lengths = np.array(trajectory_lengths)
        
        # Speed statistics
        print(f"\n📊 Speed Statistics:")
        print(f"   Min:    {all_speeds.min():.2f} m/s")
        print(f"   Max:    {all_speeds.max():.2f} m/s")
        print(f"   Mean:   {all_speeds.mean():.2f} m/s")
        print(f"   Median: {np.median(all_speeds):.2f} m/s")
        print(f"   Std:    {all_speeds.std():.2f} m/s")
        
        # Check for outliers
        if all_speeds.max() > 15.0:
            print(f"   ⚠️ WARNING: Extreme speed detected (>{15.0} m/s)")
            print(f"      Max speed: {all_speeds.max():.2f} m/s")
            print(f"      Check for data quality issues!")
        elif all_speeds.max() > 5.0:
            print(f"   ⚠️ High speeds detected (>{5.0} m/s)")
            print(f"      Expected for vehicle data")
        else:
            print(f"   ✅ Speeds in reasonable range (< 5.0 m/s)")
        
        # Trajectory length statistics
        print(f"\n📏 Trajectory Length Statistics:")
        print(f"   Min:    {trajectory_lengths.min()} steps ({trajectory_lengths.min() * dt:.1f}s)")
        print(f"   Max:    {trajectory_lengths.max()} steps ({trajectory_lengths.max() * dt:.1f}s)")
        print(f"   Mean:   {trajectory_lengths.mean():.1f} steps ({trajectory_lengths.mean() * dt:.1f}s)")
        print(f"   Median: {np.median(trajectory_lengths):.0f} steps ({np.median(trajectory_lengths) * dt:.1f}s)")
        
        # Check minimum length
        min_required = 20  # Need at least 20 frames for 10-step prediction
        n_too_short = (trajectory_lengths < min_required).sum()
        if n_too_short > 0:
            print(f"   ⚠️ {n_too_short} trajectories < {min_required} steps")
            print(f"      These will be filtered during training")
        else:
            print(f"   ✅ All trajectories ≥ {min_required} steps")
        
        # K-means clustering
        print(f"\n🎯 K-means Clustering (K=2):")
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(all_speeds.reshape(-1, 1))
        
        centers = kmeans.cluster_centers_.flatten()
        low_center = centers.min()
        high_center = centers.max()
        boundary = (low_center + high_center) / 2
        
        low_cluster = 0 if centers[0] < centers[1] else 1
        high_cluster = 1 - low_cluster
        
        n_low = np.sum(labels == low_cluster)
        n_high = np.sum(labels == high_cluster)
        balance = min(n_low, n_high) / max(n_low, n_high)
        
        print(f"   Discovered boundary: {boundary:.2f} m/s")
        print(f"   Low-speed cluster:")
        print(f"     Center: {low_center:.2f} m/s")
        print(f"     Count:  {n_low} ({n_low/len(labels)*100:.1f}%)")
        print(f"     Range:  {all_speeds[labels==low_cluster].min():.2f} - {all_speeds[labels==low_cluster].max():.2f} m/s")
        print(f"   High-speed cluster:")
        print(f"     Center: {high_center:.2f} m/s")
        print(f"     Count:  {n_high} ({n_high/len(labels)*100:.1f}%)")
        print(f"     Range:  {all_speeds[labels==high_cluster].min():.2f} - {all_speeds[labels==high_cluster].max():.2f} m/s")
        print(f"   Balance ratio: {balance:.2%}")
        
        # Assessment
        print(f"\n🔍 Assessment:")
        
        issues = []
        
        # Check balance
        if balance < 0.10:
            print(f"   ❌ SEVERE IMBALANCE ({n_low} vs {n_high})")
            print(f"      K-means clustering not meaningful")
            print(f"      Data collection may need adjustment")
            issues.append("severe_imbalance")
        elif balance < 0.30:
            print(f"   ⚠️ IMBALANCED ({n_low} vs {n_high})")
            print(f"      K-means will work but suboptimal")
            issues.append("imbalance")
        else:
            print(f"   ✅ Well-balanced clusters")
        
        # Check speed range
        if all_speeds.max() > 10.0:
            print(f"   ⚠️ Very high speeds detected")
            print(f"      Ensure this matches expected data (vehicles)")
        
        # Check data integrity
        for i, traj in enumerate(trajectories[:10]):  # Check first 10
            if np.isnan(traj).any():
                print(f"   ❌ NaN values detected in trajectory {i}")
                issues.append("nan_values")
                break
            if np.isinf(traj).any():
                print(f"   ❌ Inf values detected in trajectory {i}")
                issues.append("inf_values")
                break
        
        if not issues:
            print(f"   ✅ Data quality: GOOD")
        
        # Class distribution (if available)
        if 'class_distribution' in data:
            print(f"\n📦 Class Distribution:")
            for cls, count in sorted(data['class_distribution'].items()):
                pct = count / len(trajectories) * 100
                print(f"   {cls}: {count} ({pct:.1f}%)")
        
        print(f"{'='*70}")
        
        return len(issues) == 0
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("DATASET VERIFICATION")
    print("="*70)
    print("Checking dataset quality for K-GRU training...")
    
    # Define datasets to check
    datasets = [
        ('predictive_module/data/synthetic_mixed_traffic.pkl', 
         'Synthetic Mixed Traffic'),
        ('predictive_module/data/eth_ucy_real_pedestrians.pkl',
         'ETH/UCY Real Pedestrians'),
        ('predictive_module/data/ind_mixed_traffic.pkl',
         'inD Real Mixed Traffic'),
    ]
    
    results = {}
    
    for filepath, name in datasets:
        is_good = verify_dataset(filepath, name)
        results[name] = is_good
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    for name, is_good in results.items():
        if is_good is None:
            status = "⚠️ NOT FOUND"
        elif is_good:
            status = "✅ GOOD"
        else:
            status = "❌ ISSUES DETECTED"
        print(f"{name}: {status}")
    
    print("="*70)
    
    # Recommendations
    print("\n💡 Recommendations:")
    
    all_good = all(v for v in results.values() if v is not None)
    
    if all_good:
        print("   ✅ All datasets verified!")
        print("   Ready for training with train_kgru.py")
    else:
        print("   ⚠️ Some datasets have issues")
        print("   Review warnings above before training")
        if results.get('inD Real Mixed Traffic') is None:
            print("   → inD dataset not processed yet (run preprocess_ind.py)")
    
    print()