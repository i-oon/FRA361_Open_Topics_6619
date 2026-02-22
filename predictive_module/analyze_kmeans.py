"""
K-means Clustering Analysis
Validates that K-means discovers natural speed groupings
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def analyze_kmeans_clustering(data_path):
    """
    Analyze K-means clustering on trajectory data
    Shows that K-means discovers natural boundaries
    """
    
    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    trajectories = data['trajectories']
    
    print("="*60)
    print("K-MEANS CLUSTERING ANALYSIS")
    print("="*60)
    print(f"Dataset: {data_path}")
    print(f"Total trajectories: {len(trajectories)}")
    
    # Extract speeds
    all_speeds = []
    for traj in trajectories:
        speeds = np.linalg.norm(traj[:, 2:4], axis=1)
        avg_speed = speeds.mean()
        all_speeds.append(avg_speed)
    
    all_speeds = np.array(all_speeds)
    
    print(f"\nSpeed Statistics:")
    print(f"  Mean: {all_speeds.mean():.2f} m/s")
    print(f"  Std: {all_speeds.std():.2f} m/s")
    print(f"  Min: {all_speeds.min():.2f} m/s")
    print(f"  Max: {all_speeds.max():.2f} m/s")
    
    # K-means clustering
    print(f"\n🔍 Running K-means (K=2)...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(all_speeds.reshape(-1, 1))
    
    # Analyze clusters
    centers = kmeans.cluster_centers_.flatten()
    low_center = centers.min()
    high_center = centers.max()
    boundary = (low_center + high_center) / 2
    
    low_cluster = 0 if centers[0] < centers[1] else 1
    high_cluster = 1 - low_cluster
    
    low_speeds = all_speeds[labels == low_cluster]
    high_speeds = all_speeds[labels == high_cluster]
    
    print(f"\n✅ K-means Results:")
    print(f"  Low-speed cluster:")
    print(f"    Center: {low_center:.2f} m/s")
    print(f"    Count: {len(low_speeds)} ({len(low_speeds)/len(all_speeds)*100:.1f}%)")
    print(f"    Range: {low_speeds.min():.2f} - {low_speeds.max():.2f} m/s")
    print(f"  High-speed cluster:")
    print(f"    Center: {high_center:.2f} m/s")
    print(f"    Count: {len(high_speeds)} ({len(high_speeds)/len(all_speeds)*100:.1f}%)")
    print(f"    Range: {high_speeds.min():.2f} - {high_speeds.max():.2f} m/s")
    print(f"  Discovered boundary: {boundary:.2f} m/s")
    
    # Compare to manual threshold
    manual_threshold = 2.0
    print(f"\n📊 Comparison to Manual Threshold (2.0 m/s):")
    print(f"  K-means boundary: {boundary:.2f} m/s")
    print(f"  Manual threshold: {manual_threshold:.2f} m/s")
    print(f"  Difference: {abs(boundary - manual_threshold):.3f} m/s")
    
    if abs(boundary - manual_threshold) < 0.3:
        print(f"  ✅ K-means validates manual threshold!")
    else:
        print(f"  ⚠️ K-means suggests different boundary")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram with K-means clusters
    axes[0].hist(low_speeds, bins=30, alpha=0.7, label=f'Low-speed (n={len(low_speeds)})', color='blue')
    axes[0].hist(high_speeds, bins=30, alpha=0.7, label=f'High-speed (n={len(high_speeds)})', color='red')
    axes[0].axvline(boundary, color='green', linestyle='--', linewidth=2, 
                   label=f'K-means boundary: {boundary:.2f} m/s')
    axes[0].axvline(manual_threshold, color='orange', linestyle=':', linewidth=2,
                   label=f'Manual threshold: {manual_threshold:.2f} m/s')
    axes[0].set_xlabel('Speed (m/s)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('K-means Discovered Speed Clusters', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot
    indices = np.arange(len(all_speeds))
    colors = ['blue' if l == low_cluster else 'red' for l in labels]
    axes[1].scatter(indices, all_speeds, c=colors, alpha=0.5, s=10)
    axes[1].axhline(boundary, color='green', linestyle='--', linewidth=2,
                   label=f'K-means boundary: {boundary:.2f} m/s')
    axes[1].axhline(manual_threshold, color='orange', linestyle=':', linewidth=2,
                   label=f'Manual: {manual_threshold:.2f} m/s')
    axes[1].set_xlabel('Trajectory Index', fontsize=12)
    axes[1].set_ylabel('Average Speed (m/s)', fontsize=12)
    axes[1].set_title('Speed Distribution with K-means Labels', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictive_module/plot/kmeans_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: predictive_module/plot/kmeans_analysis.png")
    print("="*60)
    
    plt.show()
    
    return boundary, low_speeds, high_speeds

if __name__ == "__main__":
    # Analyze current dataset
    data_path = 'predictive_module/data/kgru_training_data_realistic.pkl'
    
    boundary, low_speeds, high_speeds = analyze_kmeans_clustering(data_path)
    
    print("\n📝 Summary for Thesis:")
    print(f"  'K-means clustering discovered a natural speed boundary at")
    print(f"   {boundary:.2f} m/s, separating {len(low_speeds)} low-speed")
    print(f"   trajectories (μ={low_speeds.mean():.2f} m/s) from {len(high_speeds)}")
    print(f"   high-speed trajectories (μ={high_speeds.mean():.2f} m/s).'")