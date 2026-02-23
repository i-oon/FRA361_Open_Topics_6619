"""
Analyze K-means clustering on any trajectory dataset
Works for both synthetic and real-world data
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
from torch.utils.data import DataLoader
from k_gru_predictor import TrajectoryGRU
from train_kgru import TrajectoryDataset, evaluate_predictions


def analyze_dataset_speeds(dataset_path, dataset_name, model_path=None):
    """
    Analyze speed distribution to see if natural clusters exist
    
    Args:
        dataset_path: Path to .pkl file
        dataset_name: Name for plots (e.g., "Synthetic Mixed Traffic", "ETH/UCY Real")
        model_path: Optional path to trained model for evaluation
    """
    print("="*60)
    print(f"K-MEANS ANALYSIS: {dataset_name}")
    print("="*60)
    
    # Load data
    print(f"\n1. Loading {dataset_name} data...")
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    trajectories = data['trajectories']
    dt = data.get('dt', 0.1)
    print(f"   Total trajectories: {len(trajectories)}")
    print(f"   Temporal resolution: {dt}s ({1/dt:.1f} Hz)")
    
    # Extract speeds
    print("\n2. Extracting average speeds...")
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
    
    # Run K=2
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
    
    print(f"\n   ✅ K-means Results on {dataset_name}:")
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
        print(f"   ⚠️ SEVERE IMBALANCE - Clustering may not be meaningful")
    elif balance_ratio < 0.3:
        print(f"   ⚠️ IMBALANCED - Dominated by one speed group")
    else:
        print(f"   ✅ BALANCED - Clear natural speed groupings!")
    
    # Visualize
    output_name = dataset_name.lower().replace(' ', '_').replace('/', '_')
    visualize_clustering(all_speeds, labels, low_cluster, high_cluster,
                        discovered_boundary, inertias, K_range, 
                        dataset_name, output_name)
    
    # Evaluate if model provided
    if model_path is not None:
        evaluate_clusters(trajectories, labels, low_cluster, high_cluster,
                         discovered_boundary, model_path, dataset_name)
    
    return trajectories, labels, low_cluster, high_cluster, discovered_boundary


def visualize_clustering(speeds, labels, low_cluster, high_cluster,
                        boundary, inertias, K_range, dataset_name, output_name):
    """Visualize K-means results"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Histogram with clusters
    low_speeds = speeds[labels == low_cluster]
    high_speeds = speeds[labels == high_cluster]
    
    axes[0].hist(low_speeds, bins=30, alpha=0.7, 
                label=f'Low-speed (n={len(low_speeds)})', color='blue')
    axes[0].hist(high_speeds, bins=30, alpha=0.7,
                label=f'High-speed (n={len(high_speeds)})', color='red')
    axes[0].axvline(boundary, color='green', linestyle='--', linewidth=2,
                   label=f'Boundary: {boundary:.2f} m/s')
    axes[0].set_xlabel('Speed (m/s)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title(f'K-means on {dataset_name}', 
                     fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Elbow plot
    axes[1].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Inertia', fontsize=12, fontweight='bold')
    axes[1].set_title('Elbow Method: Optimal K?', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(2, color='red', linestyle='--', alpha=0.5, label='K=2')
    axes[1].legend()
    
    # 3. Speed scatter
    indices = np.arange(len(speeds))
    colors = ['blue' if l == low_cluster else 'red' for l in labels]
    axes[2].scatter(indices, speeds, c=colors, alpha=0.6, s=20)
    axes[2].axhline(boundary, color='green', linestyle='--', linewidth=2,
                   label=f'Boundary: {boundary:.2f} m/s')
    axes[2].set_xlabel('Trajectory Index', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Average Speed (m/s)', fontsize=12, fontweight='bold')
    axes[2].set_title('Speed Distribution', fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'predictive_module/plot/kmeans_{output_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n   ✅ Visualization saved: {save_path}")
    plt.show()


def evaluate_clusters(trajectories, labels, low_cluster, high_cluster,
                     discovered_boundary, model_path, dataset_name):
    """
    Evaluate model performance on each cluster
    Shows if speed differentiation helps prediction
    """
    print("\n" + "="*60)
    print(f"EVALUATING K-MEANS CLUSTERS: {dataset_name}")
    print("="*60)
    
    # Split by clusters
    low_speed_trajs = [t for t, l in zip(trajectories, labels) if l == low_cluster]
    high_speed_trajs = [t for t, l in zip(trajectories, labels) if l == high_cluster]
    
    print(f"\n   Low-speed trajectories: {len(low_speed_trajs)}")
    print(f"   High-speed trajectories: {len(high_speed_trajs)}")
    
    # Check sufficient data
    if min(len(low_speed_trajs), len(high_speed_trajs)) < 50:
        print(f"\n   ⚠️ Insufficient data (need 50+ per cluster)")
        return
    
    # Use test split (last 15%)
    n_train_val = int(0.85 * len(trajectories))
    
    # Get test samples from each cluster
    low_test = [t for t, l in zip(trajectories[n_train_val:], labels[n_train_val:]) 
                if l == low_cluster]
    high_test = [t for t, l in zip(trajectories[n_train_val:], labels[n_train_val:])
                 if l == high_cluster]
    
    print(f"\n   Test low-speed: {len(low_test)}")
    print(f"   Test high-speed: {len(high_test)}")
    
    # Create datasets
    low_test_dataset = TrajectoryDataset(low_test, sequence_length=10, augment=False)
    high_test_dataset = TrajectoryDataset(high_test, sequence_length=10, augment=False)
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TrajectoryGRU(input_size=4, hidden_size=128, num_layers=3, output_size=4).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"\n   ✅ Loaded model: {model_path}")
        
        # Evaluate on both clusters
        low_loader = DataLoader(low_test_dataset, batch_size=256, shuffle=False)
        high_loader = DataLoader(high_test_dataset, batch_size=256, shuffle=False)
        
        low_ade, _, _ = evaluate_predictions(model, low_loader, device, save_plots=False)
        high_ade, _, _ = evaluate_predictions(model, high_loader, device, save_plots=False)
        
        print(f"\n" + "="*60)
        print(f"K-MEANS PERFORMANCE COMPARISON: {dataset_name}")
        print("="*60)
        print(f"\n   Discovered boundary: {discovered_boundary:.2f} m/s")
        print(f"\n   Low-speed cluster ADE: {low_ade:.4f}m")
        print(f"   High-speed cluster ADE: {high_ade:.4f}m")
        
        if low_ade < high_ade:
            benefit = ((high_ade - low_ade) / high_ade) * 100
            print(f"\n   ✅ Low-speed {benefit:.1f}% better")
            print(f"      Traditional pattern: Low-speed easier to predict")
        else:
            benefit = ((low_ade - high_ade) / low_ade) * 100
            print(f"\n   ⚠️ High-speed {benefit:.1f}% better")
            print(f"      Inverted pattern or minimal speed effect")
        
        print("="*60)
        
    except FileNotFoundError:
        print(f"\n   ❌ Model not found: {model_path}")
        print(f"      Train model first, then re-run")


if __name__ == "__main__":
    import sys
    
    # Configuration
    datasets = [
        {
            'path': 'predictive_module/data/synthetic_mixed_traffic.pkl',
            'name': 'Synthetic Mixed Traffic',
            'model': 'predictive_module/model/kgru_synthetic.pth'
        },
        {
            'path': 'predictive_module/data/eth_ucy_real_pedestrians.pkl',
            'name': 'ETH/UCY Real Pedestrians',
            'model': 'predictive_module/model/kgru_eth_ucy.pth'
        },
    ]
    
    # Run analysis on all datasets
    print("\n" + "="*70)
    print("K-MEANS CLUSTERING ANALYSIS ON ALL DATASETS")
    print("="*70)
    
    results = {}
    
    for dataset in datasets:
        print(f"\n{'='*70}")
        
        trajectories, labels, low_cluster, high_cluster, boundary = analyze_dataset_speeds(
            dataset_path=dataset['path'],
            dataset_name=dataset['name'],
            model_path=dataset['model']
        )
        
        results[dataset['name']] = {
            'boundary': boundary,
            'low_count': np.sum(labels == low_cluster),
            'high_count': np.sum(labels == high_cluster),
            'balance': min(np.sum(labels == low_cluster), np.sum(labels == high_cluster)) / 
                      max(np.sum(labels == low_cluster), np.sum(labels == high_cluster))
        }
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY: K-MEANS ACROSS DATASETS")
    print("="*70)
    
    print(f"\n{'Dataset':<30} {'Boundary':<12} {'Balance':<12} {'Low/High'}")
    print("-"*70)
    
    for name, res in results.items():
        print(f"{name:<30} {res['boundary']:.2f} m/s    {res['balance']:.2%}      "
              f"{res['low_count']}/{res['high_count']}")
    
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    print("\n1. K-means Methodology:")
    print("   ✅ Discovers natural boundaries from data")
    print("   ✅ No manual threshold needed")
    print("   ✅ Adapts to dataset characteristics")
    
    print("\n2. Dataset Characteristics:")
    for name, res in results.items():
        if 'Synthetic' in name:
            print(f"\n   {name}:")
            print(f"   - Boundary: {res['boundary']:.2f} m/s (designed separation)")
            print(f"   - Balance: {res['balance']:.1%} (diverse motion types)")
            print(f"   - Expected: High K-means benefit (50-70%)")
        else:
            print(f"\n   {name}:")
            print(f"   - Boundary: {res['boundary']:.2f} m/s (natural variation)")
            print(f"   - Balance: {res['balance']:.1%} (within-group variation)")
            print(f"   - Expected: Modest K-means benefit (5-15%)")
    
    print("\n3. Context-Dependency:")
    print("   ✅ Mixed motion types → High benefit")
    print("   ⚠️ Homogeneous motion → Limited benefit")
    print("="*70)