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
    prediction_horizon=20,
    device='cuda'
):
    """
    Visualize model predictions vs ground truth
    
    Args:
        model: Trained K-GRU model
        trajectories: List of test trajectories
        n_samples: Number of examples to show
        sequence_length: Input sequence length
        prediction_horizon: How many steps to predict ahead
    """
    
    model.eval()
    
    # Select diverse trajectories (different speeds)
    selected_indices = np.linspace(0, len(trajectories)-1, n_samples, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, traj_idx in enumerate(selected_indices):
        trajectory = trajectories[traj_idx]
        
        # Skip if trajectory too short
        if len(trajectory) < sequence_length + prediction_horizon:
            continue
        
        # Random starting point
        start_idx = np.random.randint(0, len(trajectory) - sequence_length - prediction_horizon)
        
        # Input sequence
        input_seq = trajectory[start_idx:start_idx+sequence_length]
        
        # Ground truth future
        ground_truth = trajectory[start_idx+sequence_length:start_idx+sequence_length+prediction_horizon]
        
        # Predict future
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
            predictions = model.predict_sequence(input_tensor, prediction_horizon)
            predictions = predictions.cpu().numpy()[0]
        
        # Calculate speed
        avg_speed = np.mean(np.linalg.norm(input_seq[:, 2:4], axis=1))
        
        # Plot
        ax = axes[idx]
        
        # Plot history (input)
        ax.plot(input_seq[:, 0], input_seq[:, 1], 
               'o-', color='gray', linewidth=2, markersize=4, 
               label='History', alpha=0.7)
        
        # Plot ground truth future
        ax.plot(ground_truth[:, 0], ground_truth[:, 1],
               's-', color='green', linewidth=2, markersize=4,
               label='Ground Truth', alpha=0.7)
        
        # Plot predictions
        ax.plot(predictions[:, 0], predictions[:, 1],
               '^-', color='red', linewidth=2, markersize=4,
               label='Predicted', alpha=0.7)
        
        # Mark start and end
        ax.plot(input_seq[-1, 0], input_seq[-1, 1], 
               'o', color='blue', markersize=10, label='Start')
        ax.plot(ground_truth[-1, 0], ground_truth[-1, 1],
               '*', color='gold', markersize=15, label='Goal')
        
        # Calculate error
        position_error = np.linalg.norm(predictions[:, :2] - ground_truth[:, :2], axis=1)
        mean_error = position_error.mean()
        
        ax.set_xlabel('X Position (m)', fontsize=10)
        ax.set_ylabel('Y Position (m)', fontsize=10)
        ax.set_title(f'Trajectory {idx+1}\nSpeed: {avg_speed:.2f} m/s | Error: {mean_error:.3f}m', 
                    fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('predictive_module/plot/trajectory_predictions.png', dpi=150)
    print("Trajectory visualization saved to predictive_module/plot/trajectory_predictions.png")
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
    prediction_horizon=15,
    device='cuda'
):
    """
    Compare prediction accuracy for low-speed vs high-speed obstacles
    """
    
    model.eval()
    
    low_speed_errors = []
    high_speed_errors = []
    
    for trajectory in trajectories[:200]:  # Sample 200 trajectories
        if len(trajectory) < sequence_length + prediction_horizon:
            continue
        
        # Calculate average speed
        speeds = np.linalg.norm(trajectory[:, 2:4], axis=1)
        avg_speed = speeds.mean()
        
        # Random start
        start_idx = np.random.randint(0, len(trajectory) - sequence_length - prediction_horizon)
        
        input_seq = trajectory[start_idx:start_idx+sequence_length]
        ground_truth = trajectory[start_idx+sequence_length:start_idx+sequence_length+prediction_horizon]
        
        # Predict
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
            predictions = model.predict_sequence(input_tensor, prediction_horizon)
            predictions = predictions.cpu().numpy()[0]
        
        # Error
        position_error = np.linalg.norm(predictions[:, :2] - ground_truth[:, :2], axis=1).mean()
        
        # Classify by speed (threshold at 2.0 m/s for pedestrian vs vehicle)
        if avg_speed < 2.0:
            low_speed_errors.append(position_error)
        else:
            high_speed_errors.append(position_error)
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    positions = [1, 2]
    data = [low_speed_errors, high_speed_errors]
    labels = [f'Low Speed\n(<2.0 m/s)\nn={len(low_speed_errors)}',
              f'High Speed\n(≥2.0 m/s)\nn={len(high_speed_errors)}']
    
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                    labels=labels,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    showfliers=True)
    
    ax.set_ylabel('Position Error (m)', fontsize=12)
    ax.set_title('Prediction Accuracy: Low-Speed vs High-Speed Obstacles', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    low_mean = np.mean(low_speed_errors)
    high_mean = np.mean(high_speed_errors)
    
    ax.text(1, low_mean, f'Mean: {low_mean:.4f}m', 
           ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(2, high_mean, f'Mean: {high_mean:.4f}m',
           ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('predictive_module/plot/speed_comparison.png', dpi=150)
    print("Speed comparison saved to predictive_module/plot/speed_comparison.png")
    plt.show()


if __name__ == "__main__":
    # Load test data
    print("Loading test data...")
    with open('predictive_module/data/kgru_training_data_realistic.pkl', 'rb') as f:
        data = pickle.load(f)
    
    trajectories = data['trajectories']
    
    # Use test split (last 15%)
    n_train = int(0.7 * len(trajectories))
    n_val = int(0.15 * len(trajectories))
    test_trajectories = trajectories[n_train+n_val:]
    
    print(f"Using {len(test_trajectories)} test trajectories")
    
    # Load trained model
    print("Loading trained model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TrajectoryGRU(
        input_size=4,
        hidden_size=50,
        num_layers=2,
        output_size=4
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