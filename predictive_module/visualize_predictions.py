# visualize_predictions.py
"""
Visualize K-GRU predictions vs ground truth trajectories
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictive_module.k_gru_predictor import TrajectoryGRU

def visualize_trajectory_predictions(
    model,
    trajectories,
    n_samples=6,
    sequence_length=10,
    prediction_horizon=10,
    device='cuda'
):
    """
    MUCH better visualization with clear error indication
    """
    
    model.eval()
    
    # Select diverse trajectories
    selected_indices = np.linspace(0, len(trajectories)-1, n_samples, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, traj_idx in enumerate(selected_indices):
        trajectory = trajectories[traj_idx]
        
        if len(trajectory) < sequence_length + prediction_horizon:
            continue
        
        start_idx = np.random.randint(0, len(trajectory) - sequence_length - prediction_horizon)
        
        input_seq = trajectory[start_idx:start_idx+sequence_length]
        ground_truth = trajectory[start_idx+sequence_length:start_idx+sequence_length+prediction_horizon]
        
        # Predict
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
            predictions = model.predict_sequence(input_tensor, prediction_horizon)
            predictions = predictions.cpu().numpy()[0]
        
        avg_speed = np.mean(np.linalg.norm(input_seq[:, 2:4], axis=1))
        
        ax = axes[idx]
        
        # 1. Plot OBSERVATION (with arrows to show direction)
        for i in range(len(input_seq)-1):
            ax.arrow(input_seq[i, 0], input_seq[i, 1],
                    input_seq[i+1, 0] - input_seq[i, 0],
                    input_seq[i+1, 1] - input_seq[i, 1],
                    head_width=0.02, head_length=0.03, 
                    fc='gray', ec='gray', alpha=0.6, linewidth=2)
        
        # 2. Plot GROUND TRUTH (green line with markers)
        ax.plot(ground_truth[:, 0], ground_truth[:, 1],
               'o-', color='#2ecc71', linewidth=3, markersize=6,
               label='Ground Truth', zorder=3)
        
        # 3. Plot PREDICTION (red dashed line)
        ax.plot(predictions[:, 0], predictions[:, 1],
               's--', color='#e74c3c', linewidth=3, markersize=6,
               label='Predicted', alpha=0.8, zorder=2)
        
        # 4. Draw ERROR LINES (connecting predicted to truth)
        for i in range(len(predictions)):
            ax.plot([predictions[i, 0], ground_truth[i, 0]],
                   [predictions[i, 1], ground_truth[i, 1]],
                   ':', color='orange', linewidth=1, alpha=0.5, zorder=1)
        
        # 5. Mark START clearly
        ax.plot(input_seq[0, 0], input_seq[0, 1], 
               'D', color='blue', markersize=12, label='Start', zorder=4)
        
        # 6. Mark END points
        ax.plot(ground_truth[-1, 0], ground_truth[-1, 1],
               '*', color='gold', markersize=18, 
               markeredgecolor='black', markeredgewidth=1.5,
               label='True End', zorder=5)
        ax.plot(predictions[-1, 0], predictions[-1, 1],
               'X', color='red', markersize=12,
               markeredgecolor='darkred', markeredgewidth=1.5,
               label='Pred End', zorder=5)
        
        # Calculate metrics
        position_errors = np.linalg.norm(predictions[:, :2] - ground_truth[:, :2], axis=1)
        ade = position_errors.mean()
        fde = position_errors[-1]
        
        # Speed category
        category = "HIGH-SPEED" if avg_speed >= 2.0 else "LOW-SPEED"
        color = '#3498db' if avg_speed >= 2.0 else '#9b59b6'
        
        ax.set_xlabel('X Position (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y Position (m)', fontsize=11, fontweight='bold')
        ax.set_title(
            f'{category}\n'
            f'Speed: {avg_speed:.2f} m/s | ADE: {ade:.3f}m | FDE: {fde:.3f}m',
            fontsize=12, fontweight='bold', color=color
        )
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
        
        # Add background color based on category
        ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig('predictive_module/plot/trajectory_predictions_improved.png', dpi=200, bbox_inches='tight')
    print("✅ Improved visualization saved!")
    plt.show()


def visualize_error_over_time(
    model,
    trajectories,
    n_trajectories=50,
    sequence_length=10,
    prediction_horizon=20,
    device='cuda'
):
    """
    Show how prediction error grows over prediction horizon
    """
    
    model.eval()
    
    all_position_errors = []
    all_velocity_errors = []
    
    for traj_idx in range(min(n_trajectories, len(trajectories))):
        trajectory = trajectories[traj_idx]
        
        if len(trajectory) < sequence_length + prediction_horizon:
            continue
        
        # Random start
        start_idx = np.random.randint(0, len(trajectory) - sequence_length - prediction_horizon)
        
        input_seq = trajectory[start_idx:start_idx+sequence_length]
        ground_truth = trajectory[start_idx+sequence_length:start_idx+sequence_length+prediction_horizon]
        
        # Predict
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
            predictions = model.predict_sequence(input_tensor, prediction_horizon)
            predictions = predictions.cpu().numpy()[0]
        
        # Calculate errors at each timestep
        position_errors = np.linalg.norm(predictions[:, :2] - ground_truth[:, :2], axis=1)
        velocity_errors = np.linalg.norm(predictions[:, 2:4] - ground_truth[:, 2:4], axis=1)
        
        all_position_errors.append(position_errors)
        all_velocity_errors.append(velocity_errors)
    
    # Convert to arrays
    all_position_errors = np.array(all_position_errors)
    all_velocity_errors = np.array(all_velocity_errors)
    
    # Calculate statistics
    mean_pos_error = all_position_errors.mean(axis=0)
    std_pos_error = all_position_errors.std(axis=0)
    
    mean_vel_error = all_velocity_errors.mean(axis=0)
    std_vel_error = all_velocity_errors.std(axis=0)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    timesteps = np.arange(1, prediction_horizon+1)
    
    # Position error over time
    axes[0].plot(timesteps, mean_pos_error, 'b-', linewidth=2, label='Mean Error')
    axes[0].fill_between(timesteps, 
                         mean_pos_error - std_pos_error,
                         mean_pos_error + std_pos_error,
                         alpha=0.3, label='±1 Std Dev')
    axes[0].set_xlabel('Prediction Step', fontsize=12)
    axes[0].set_ylabel('Position Error (m)', fontsize=12)
    axes[0].set_title('Position Error Growth Over Time', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Velocity error over time
    axes[1].plot(timesteps, mean_vel_error, 'r-', linewidth=2, label='Mean Error')
    axes[1].fill_between(timesteps,
                         mean_vel_error - std_vel_error,
                         mean_vel_error + std_vel_error,
                         alpha=0.3, label='±1 Std Dev')
    axes[1].set_xlabel('Prediction Step', fontsize=12)
    axes[1].set_ylabel('Velocity Error (m/s)', fontsize=12)
    axes[1].set_title('Velocity Error Growth Over Time', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictive_module/plot/error_over_time.png', dpi=150)
    print("Error growth visualization saved to predictive_module/plot/error_over_time.png")
    plt.show()


def visualize_speed_comparison(
    model,
    trajectories,
    sequence_length=10,
    prediction_horizon=10,
    device='cuda'
):
    """
    Compare prediction accuracy using K-means discovered speed groups
    TRUE K-means clustering - discovers boundary from data!
    """
    
    model.eval()
    
    # Step 1: Extract all speeds from trajectories
    print("\n" + "="*60)
    print("K-MEANS CLUSTERING ANALYSIS")
    print("="*60)
    
    all_speeds = []
    trajectory_speeds = []
    
    for trajectory in trajectories:
        if len(trajectory) < sequence_length + prediction_horizon + 1:
            continue
        
        speeds = np.linalg.norm(trajectory[:, 2:4], axis=1)
        avg_speed = speeds.mean()
        all_speeds.append(avg_speed)
        trajectory_speeds.append((trajectory, avg_speed))
    
    all_speeds = np.array(all_speeds).reshape(-1, 1)
    
    # Step 2: K-means discovers natural grouping
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(all_speeds)
    
    # Get cluster centers
    centers = kmeans.cluster_centers_.flatten()
    low_center = centers.min()
    high_center = centers.max()
    discovered_boundary = (low_center + high_center) / 2
    
    # Determine which cluster is low vs high
    low_cluster = 0 if centers[0] < centers[1] else 1
    high_cluster = 1 - low_cluster
    
    print(f"\n✅ K-means Discovered Speed Groups:")
    print(f"  Low-speed cluster center: {low_center:.2f} m/s (n={np.sum(labels==low_cluster)})")
    print(f"  High-speed cluster center: {high_center:.2f} m/s (n={np.sum(labels==high_cluster)})")
    print(f"  Discovered boundary: {discovered_boundary:.2f} m/s")
    print(f"\n📊 Compare to manual threshold (2.0 m/s):")
    print(f"  Difference: {abs(discovered_boundary - 2.0):.3f} m/s")
    if abs(discovered_boundary - 2.0) < 0.3:
        print(f"  ✅ K-means validates 2.0 m/s assumption!")
    else:
        print(f"  ⚠️ K-means suggests different boundary: {discovered_boundary:.2f} m/s")
    print("="*60)
    
    # Step 3: Evaluate predictions for each K-means cluster
    low_speed_errors = []
    high_speed_errors = []
    
    for (trajectory, avg_speed), label in zip(trajectory_speeds, labels):
        if len(trajectory) < sequence_length + prediction_horizon + 1:
            continue
        
        # Random start
        max_start = len(trajectory) - sequence_length - prediction_horizon
        if max_start <= 0:
            continue
        start_idx = np.random.randint(0, max_start)
        
        input_seq = trajectory[start_idx:start_idx+sequence_length]
        ground_truth = trajectory[start_idx+sequence_length:start_idx+sequence_length+prediction_horizon]
        
        # Predict
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
            predictions = model.predict_sequence(input_tensor, prediction_horizon)
            predictions = predictions.cpu().numpy()[0]
        
        # Error
        position_error = np.linalg.norm(predictions[:, :2] - ground_truth[:, :2], axis=1).mean()
        
        # Classify by K-means label (NOT manual threshold!)
        if label == low_cluster:
            low_speed_errors.append(position_error)
        else:
            high_speed_errors.append(position_error)
    
    # Step 4: Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    positions = [1, 2]
    data = [low_speed_errors, high_speed_errors]
    labels_text = [
        f'Low-Speed Cluster\n(μ={low_center:.2f} m/s)\nn={len(low_speed_errors)}',
        f'High-Speed Cluster\n(μ={high_center:.2f} m/s)\nn={len(high_speed_errors)}'
    ]
    
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                    labels=labels_text,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    showfliers=True)
    
    ax.set_ylabel('Position Error (m)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Prediction Accuracy: K-means Discovered Clusters\n'
        f'Boundary: {discovered_boundary:.2f} m/s (vs manual 2.0 m/s)',
        fontsize=13, fontweight='bold'
    )
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    low_mean = np.mean(low_speed_errors)
    high_mean = np.mean(high_speed_errors)
    
    ax.text(1, low_mean, f'Mean: {low_mean:.4f}m', 
           ha='center', va='bottom', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(2, high_mean, f'Mean: {high_mean:.4f}m',
           ha='center', va='bottom', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add K-means benefit analysis
    if low_mean > high_mean:
        benefit_pct = ((low_mean - high_mean) / low_mean) * 100
        winner = "High-speed"
        print(f"\n✅ K-means Benefit: High-speed predictions {benefit_pct:.1f}% better")
    else:
        benefit_pct = ((high_mean - low_mean) / high_mean) * 100
        winner = "Low-speed"
        print(f"\n⚠️ Inverted: Low-speed predictions {benefit_pct:.1f}% better")
    
    ax.text(0.5, 0.95, 
           f'{winner} cluster has {benefit_pct:.1f}% lower error\n'
           f'K-means boundary: {discovered_boundary:.2f} m/s',
           transform=ax.transAxes, fontsize=11,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('predictive_module/plot/speed_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Speed comparison saved to predictive_module/plot/speed_comparison.png")
    print(f"   Low-speed cluster: {low_mean:.4f}m (n={len(low_speed_errors)})")
    print(f"   High-speed cluster: {high_mean:.4f}m (n={len(high_speed_errors)})")
    print(f"   K-means discovered boundary: {discovered_boundary:.2f} m/s")
    print("="*60)
    
    plt.show()


if __name__ == "__main__":
    # Load test data
    print("Loading test data...")
    with open('predictive_module/data/kgru_training_data_realistic.pkl', 'rb') as f:
        data = pickle.load(f)
    
    import random
    trajectories = data['trajectories']
    random.shuffle(trajectories)  # ← Add this!
    n_train = int(0.7 * len(trajectories))
    n_val = int(0.15 * len(trajectories))
    test_trajectories = trajectories[n_train+n_val:]
    
    print(f"Using {len(test_trajectories)} test trajectories")
    
    # Load trained model
    print("Loading trained model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TrajectoryGRU(
        input_size=4,
        hidden_size=128,
        num_layers=3,
        output_size=4,
    ).to(device)
    model.load_state_dict(torch.load('predictive_module/model/kgru_model.pth'))
    model.eval()
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # 1. Trajectory predictions
    print("\n1. Visualizing trajectory predictions...")
    visualize_trajectory_predictions(
        model, 
        test_trajectories,
        n_samples=6,
        sequence_length=10,
        prediction_horizon=10,  # Reduced from 20 to 10
        device=device
    )
    
    # 2. Error growth over time
    print("\n2. Analyzing error growth over prediction horizon...")
    visualize_error_over_time(
        model,
        test_trajectories,
        n_trajectories=50,
        sequence_length=10,
        prediction_horizon=10,  # Reduced from 20 to 10
        device=device
    )
    
    # 3. Speed comparison
    print("\n3. Comparing low-speed vs high-speed predictions...")
    visualize_speed_comparison(
        model,
        test_trajectories,
        sequence_length=10,
        prediction_horizon=10,  # Reduced from 15 to 10
        device=device
    )
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - predictive_module/plot/trajectory_predictions.png (6 example trajectories)")
    print("  - predictive_module/plot/error_over_time.png (error growth analysis)")
    print("  - predictive_module/plot/speed_comparison.png (low vs high speed accuracy)")