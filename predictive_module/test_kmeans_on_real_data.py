"""
Test K-means clustering on REAL ETH/UCY pedestrian data
TRUE validation: Does K-means find natural speed groups in real data?
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
from torch.utils.data import DataLoader
from k_gru_predictor import TrajectoryGRU
from train_kgru import TrajectoryDataset, evaluate_predictions

def analyze_eth_ucy_speeds():
    """
    Analyze speed distribution in ETH/UCY to see if natural clusters exist
    """
    print("="*60)
    print("K-MEANS ON REAL ETH/UCY DATA")
    print("="*60)
    
    # Load ETH/UCY data
    print("\n1. Loading ETH/UCY pedestrian data...")
    with open('predictive_module/data/eth_ucy_processed.pkl', 'rb') as f:
        data = pickle.load(f)
    
    trajectories = data['trajectories']
    print(f"   Total trajectories: {len(trajectories)}")
    
    # Extract speeds
    print("\n2. Extracting pedestrian speeds...")
    all_speeds = []
    for traj in trajectories:
        speeds = np.linalg.norm(traj[:, 2:4], axis=1)
        avg_speed = speeds.mean()
        all_speeds.append(avg_speed)
    
    all_speeds = np.array(all_speeds)
    
    print(f"\n   Speed Statistics:")
    print(f"   Min: {all_speeds.min():.2f} m/s")
    print(f"   Max: {all_speeds.max():.2f} m/s")
    print(f"   Mean: {all_speeds.mean():.2f} m/s")
    print(f"   Std: {all_speeds.std():.2f} m/s")
    print(f"   Median: {np.median(all_speeds):.2f} m/s")
    
    # Test different K values
    print("\n3. Testing K-means with different K values...")
    inertias = []
    K_range = range(1, 6)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(all_speeds.reshape(-1, 1))
        inertias.append(kmeans.inertia_)
        print(f"   K={k}: Inertia={kmeans.inertia_:.2f}")
    
    # Run K=2 (as in Liu et al.)
    print("\n4. Running K-means with K=2...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(all_speeds.reshape(-1, 1))
    
    centers = kmeans.cluster_centers_.flatten()
    low_center = centers.min()
    high_center = centers.max()
    discovered_boundary = (low_center + high_center) / 2
    
    low_cluster = 0 if centers[0] < centers[1] else 1
    high_cluster = 1 - low_cluster
    
    n_low = np.sum(labels == low_cluster)
    n_high = np.sum(labels == high_cluster)
    
    print(f"\n   ✅ K-means Results on REAL Pedestrian Data:")
    print(f"   Low-speed cluster:")
    print(f"     Center: {low_center:.2f} m/s")
    print(f"     Count: {n_low} ({n_low/len(labels)*100:.1f}%)")
    print(f"     Range: {all_speeds[labels==low_cluster].min():.2f} - {all_speeds[labels==low_cluster].max():.2f} m/s")
    print(f"\n   High-speed cluster:")
    print(f"     Center: {high_center:.2f} m/s")
    print(f"     Count: {n_high} ({n_high/len(labels)*100:.1f}%)")
    print(f"     Range: {all_speeds[labels==high_cluster].min():.2f} - {all_speeds[labels==high_cluster].max():.2f} m/s")
    print(f"\n   Discovered boundary: {discovered_boundary:.2f} m/s")
    
    # Check if clusters are meaningful
    balance_ratio = min(n_low, n_high) / max(n_low, n_high)
    print(f"\n   Cluster balance: {balance_ratio:.2%}")
    
    if balance_ratio < 0.1:
        print(f"   ⚠️ SEVERE IMBALANCE - Most pedestrians have similar speeds")
        print(f"      K-means may not be finding meaningful natural groups")
    elif balance_ratio < 0.3:
        print(f"   ⚠️ IMBALANCED - Some natural variation but dominated by one speed")
    else:
        print(f"   ✅ BALANCED - Clear natural speed groupings exist!")
    
    # Visualize
    visualize_real_data_clustering(all_speeds, labels, low_cluster, high_cluster,
                                   discovered_boundary, inertias, K_range)
    
    return trajectories, labels, low_cluster, high_cluster, discovered_boundary


def visualize_real_data_clustering(speeds, labels, low_cluster, high_cluster,
                                   boundary, inertias, K_range):
    """Visualize K-means results on real data"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Histogram with clusters
    low_speeds = speeds[labels == low_cluster]
    high_speeds = speeds[labels == high_cluster]
    
    axes[0].hist(low_speeds, bins=30, alpha=0.7, 
                label=f'Low-speed (n={len(low_speeds)})', color='blue')
    axes[0].hist(high_speeds, bins=30, alpha=0.7,
                label=f'High-speed (n={len(high_speeds)})', color='red')
    axes[0].axvline(boundary, color='green', linestyle='--', linewidth=2,
                   label=f'K-means boundary: {boundary:.2f} m/s')
    axes[0].set_xlabel('Speed (m/s)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('K-means on REAL ETH/UCY Pedestrians', 
                     fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Elbow plot
    axes[1].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Inertia', fontsize=12, fontweight='bold')
    axes[1].set_title('Elbow Method: Optimal K?', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(2, color='red', linestyle='--', alpha=0.5, label='K=2 (Liu et al.)')
    axes[1].legend()
    
    # 3. Speed scatter with clusters
    indices = np.arange(len(speeds))
    colors = ['blue' if l == low_cluster else 'red' for l in labels]
    axes[2].scatter(indices, speeds, c=colors, alpha=0.6, s=20)
    axes[2].axhline(boundary, color='green', linestyle='--', linewidth=2,
                   label=f'Boundary: {boundary:.2f} m/s')
    axes[2].set_xlabel('Trajectory Index', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Average Speed (m/s)', fontsize=12, fontweight='bold')
    axes[2].set_title('Real Pedestrian Speed Distribution', 
                     fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictive_module/plot/kmeans_real_data_validation.png', 
               dpi=150, bbox_inches='tight')
    print(f"\n   ✅ Visualization saved: predictive_module/plot/kmeans_real_data_validation.png")
    plt.show()


def train_and_evaluate_with_real_clusters(trajectories, labels, low_cluster, 
                                          high_cluster, discovered_boundary):
    """
    Train model and evaluate using K-means discovered real data clusters
    TRUE test: Does speed differentiation help on REAL data?
    """
    print("\n" + "="*60)
    print("TRAINING WITH K-MEANS DISCOVERED REAL CLUSTERS")
    print("="*60)
    
    # Split data by K-means clusters
    low_speed_trajs = [t for t, l in zip(trajectories, labels) if l == low_cluster]
    high_speed_trajs = [t for t, l in zip(trajectories, labels) if l == high_cluster]
    
    print(f"\n   Low-speed trajectories: {len(low_speed_trajs)}")
    print(f"   High-speed trajectories: {len(high_speed_trajs)}")
    
    # Check if we have enough data in both groups
    if min(len(low_speed_trajs), len(high_speed_trajs)) < 50:
        print(f"\n   ⚠️ Insufficient data in one cluster for meaningful comparison")
        print(f"      Need at least 50 trajectories per group")
        return
    
    # Split each group: 70% train, 15% val, 15% test
    def split_data(trajs):
        n = len(trajs)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        return trajs[:n_train], trajs[n_train:n_train+n_val], trajs[n_train+n_val:]
    
    low_train, low_val, low_test = split_data(low_speed_trajs)
    high_train, high_val, high_test = split_data(high_speed_trajs)
    
    # Combine for training
    train_data = low_train + high_train
    val_data = low_val + high_val
    
    print(f"\n   Training set: {len(train_data)} ({len(low_train)} low + {len(high_train)} high)")
    print(f"   Validation set: {len(val_data)} ({len(low_val)} low + {len(high_val)} high)")
    print(f"   Test low-speed: {len(low_test)}")
    print(f"   Test high-speed: {len(high_test)}")
    
    # Create datasets
    from train_kgru import TrajectoryDataset
    
    train_dataset = TrajectoryDataset(train_data, sequence_length=10, augment=True)
    val_dataset = TrajectoryDataset(val_data, sequence_length=10, augment=False)
    low_test_dataset = TrajectoryDataset(low_test, sequence_length=10, augment=False)
    high_test_dataset = TrajectoryDataset(high_test, sequence_length=10, augment=False)
    
    # Train model (simplified - you can use full training if desired)
    print(f"\n   Note: For full training, use train_kgru.py with this split")
    print(f"   Here we'll just evaluate with existing model as proof-of-concept")
    
    # Load existing model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TrajectoryGRU(input_size=4, hidden_size=128, num_layers=3, output_size=4).to(device)
    
    # Try to load trained model (if exists)
    try:
        model.load_state_dict(torch.load('predictive_module/model/kgru_model.pth'))
        print(f"\n   ✅ Loaded existing model for evaluation")
        
        # Evaluate on both clusters
        from torch.utils.data import DataLoader
        
        low_loader = DataLoader(low_test_dataset, batch_size=256, shuffle=False)
        high_loader = DataLoader(high_test_dataset, batch_size=256, shuffle=False)
        
        low_ade, _, _ = evaluate_predictions(model, low_loader, device, save_plots=False)
        high_ade, _, _ = evaluate_predictions(model, high_loader, device, save_plots=False)
        
        print(f"\n" + "="*60)
        print("REAL DATA K-MEANS EVALUATION")
        print("="*60)
        print(f"\n   K-means discovered boundary: {discovered_boundary:.2f} m/s")
        print(f"\n   Low-speed cluster ADE: {low_ade:.4f}m")
        print(f"   High-speed cluster ADE: {high_ade:.4f}m")
        
        if low_ade < high_ade:
            benefit = ((high_ade - low_ade) / high_ade) * 100
            print(f"\n   ✅ Low-speed {benefit:.1f}% better (traditional pattern)")
        else:
            benefit = ((low_ade - high_ade) / low_ade) * 100
            print(f"\n   ⚠️ High-speed {benefit:.1f}% better (inverted pattern)")
            print(f"      Real human complexity may dominate speed factor")
        
        print("="*60)
        
    except FileNotFoundError:
        print(f"\n   ⚠️ No trained model found")
        print(f"      Train model first, then re-run this analysis")


if __name__ == "__main__":
    # Analyze real ETH/UCY data with K-means
    trajectories, labels, low_cluster, high_cluster, boundary = analyze_eth_ucy_speeds()
    
    # Evaluate with discovered clusters
    train_and_evaluate_with_real_clusters(trajectories, labels, low_cluster, 
                                         high_cluster, boundary)
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("\nThis analysis shows whether K-means discovers NATURAL speed")
    print("groupings in REAL pedestrian data (not artificially created).")
    print("\nIf K-means finds meaningful clusters:")
    print("  ✅ Validates Liu et al.'s methodology")
    print("  ✅ Shows natural speed diversity exists")
    print("  ✅ Proves K-means works on real-world data")
    print("\nIf K-means finds severe imbalance:")
    print("  ⚠️ Real pedestrians may have uniform speeds")
    print("  ⚠️ Speed clustering may not apply to pedestrian-only scenarios")
    print("="*60)