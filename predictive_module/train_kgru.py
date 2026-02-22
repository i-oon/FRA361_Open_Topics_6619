# train_kgru.py
"""
K-GRU training with validation, early stopping, and monitoring
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictive_module.k_gru_predictor import TrajectoryGRU
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm


class TrajectoryDataset(Dataset):
    """Dataset with data augmentation"""
    
    def __init__(self, trajectories, sequence_length=10, augment=True):
        self.sequence_length = sequence_length
        self.augment = augment
        
        self.sequences = []
        self.targets = []
        
        for traj in trajectories:
            # Skip short trajectories
            if len(traj) < sequence_length + 1:
                continue
            
            # Create overlapping sequences
            for i in range(len(traj) - sequence_length):
                seq = traj[i:i+sequence_length].copy()
                target = traj[i+sequence_length].copy()
                
                self.sequences.append(seq)
                self.targets.append(target)
        
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
        
        print(f"Dataset created: {len(self.sequences)} samples")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        target = self.targets[idx].copy()
        
        # Data augmentation (random flips)
        if self.augment and np.random.random() > 0.5:
            # Flip x-axis
            seq[:, 0] *= -1  # x position
            seq[:, 2] *= -1  # vx
            target[0] *= -1
            target[2] *= -1
        
        if self.augment and np.random.random() > 0.5:
            # Flip y-axis
            seq[:, 1] *= -1  # y position
            seq[:, 3] *= -1  # vy
            target[1] *= -1
            target[3] *= -1
        
        return (
            torch.FloatTensor(seq),
            torch.FloatTensor(target)
        )


def print_gpu_utilization():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def train_model(
    train_loader,
    val_loader,
    epochs=100,
    lr=0.001,
    device='cuda',
    patience=15,
    save_path='kgru_model.pth'
):
    """Train K-GRU with early stopping and learning rate scheduling"""
    
    model = TrajectoryGRU(
        input_size=4,
        hidden_size=128,
        num_layers=3,
        output_size=4
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("="*60)
    print("TRAINING K-GRU MODEL")
    print("="*60)
    print(f"Device: {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print_gpu_utilization()
    print("="*60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_pos_error = 0.0
        train_vel_error = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for sequences, targets in pbar:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            predictions, _ = model(sequences)
            
            # Total loss
            loss = criterion(predictions, targets)
            
            # Track position and velocity errors separately
            pos_error = criterion(predictions[:, :2], targets[:, :2])
            vel_error = criterion(predictions[:, 2:], targets[:, 2:])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_pos_error += pos_error.item()
            train_vel_error += vel_error.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        train_loss /= len(train_loader)
        train_pos_error /= len(train_loader)
        train_vel_error /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_pos_error = 0.0
        val_vel_error = 0.0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                predictions, _ = model(sequences)
                
                loss = criterion(predictions, targets)
                pos_error = criterion(predictions[:, :2], targets[:, :2])
                vel_error = criterion(predictions[:, 2:], targets[:, 2:])
                
                val_loss += loss.item()
                val_pos_error += pos_error.item()
                val_vel_error += vel_error.item()
        
        val_loss /= len(val_loader)
        val_pos_error /= len(val_loader)
        val_vel_error /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.6f} (Pos: {train_pos_error:.6f}, Vel: {train_vel_error:.6f})")
        print(f"  Val Loss:   {val_loss:.6f} (Pos: {val_pos_error:.6f}, Vel: {val_vel_error:.6f})")
        
        # GPU monitoring every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("  ", end="")
            print_gpu_utilization()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Best model saved! (Val loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print("-"*60)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {save_path}")
    
    return model, train_losses, val_losses


def evaluate_predictions(model, test_loader, device='cuda', save_plots=True):
    """Evaluate model with ADE and FDE metrics"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for sequences, targets in tqdm(test_loader, desc="Testing"):
            sequences = sequences.to(device)
            predictions, _ = model(sequences)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    position_errors = np.linalg.norm(predictions[:, :2] - targets[:, :2], axis=1)
    ade = position_errors.mean()
    fde = position_errors.mean()
    
    velocity_errors = np.linalg.norm(predictions[:, 2:] - targets[:, 2:], axis=1)
    vel_error = velocity_errors.mean()
    
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"ADE (Position): {ade:.4f} m")
    print(f"FDE (Position): {fde:.4f} m")
    print(f"Velocity Error: {vel_error:.4f} m/s")
    print("="*60)
    
    if save_plots:
        # Plot error distributions
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].hist(position_errors, bins=50, edgecolor='black')
        axes[0].axvline(ade, color='red', linestyle='--', label=f'Mean: {ade:.4f}')
        axes[0].set_xlabel('Position Error (m)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Position Prediction Errors')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(velocity_errors, bins=50, edgecolor='black')
        axes[1].axvline(vel_error, color='red', linestyle='--', label=f'Mean: {vel_error:.4f}')
        axes[1].set_xlabel('Velocity Error (m/s)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Velocity Prediction Errors')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('kgru_evaluation.png', dpi=150)
        plt.savefig('predictive_module/plot/kgru_evaluation.png', dpi=150)
    
    return ade, fde, vel_error


def analyze_kmeans_clustering(train_trajectories):
    """
    Analyze K-means clustering on training data
    Validates that K-means discovers natural speed boundaries
    """
    print("\n" + "="*60)
    print("K-MEANS CLUSTERING VALIDATION")
    print("="*60)
    
    # Extract speeds from training trajectories
    all_speeds = []
    for traj in train_trajectories:
        speeds = np.linalg.norm(traj[:, 2:4], axis=1)
        all_speeds.append(speeds.mean())
    
    all_speeds = np.array(all_speeds).reshape(-1, 1)
    
    # K-means clustering (discovers natural grouping)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(all_speeds)
    
    # Analyze discovered clusters
    centers = kmeans.cluster_centers_.flatten()
    low_center = centers.min()
    high_center = centers.max()
    discovered_boundary = (low_center + high_center) / 2
    
    # Determine which cluster is low vs high
    low_cluster = 0 if centers[0] < centers[1] else 1
    high_cluster = 1 - low_cluster
    
    n_low = np.sum(labels == low_cluster)
    n_high = np.sum(labels == high_cluster)
    
    print(f"\n✅ K-means Discovered Speed Groups:")
    print(f"  Low-speed cluster:")
    print(f"    Center: {low_center:.2f} m/s")
    print(f"    Count: {n_low} ({n_low/len(labels)*100:.1f}%)")
    print(f"  High-speed cluster:")
    print(f"    Center: {high_center:.2f} m/s")
    print(f"    Count: {n_high} ({n_high/len(labels)*100:.1f}%)")
    print(f"  Discovered boundary: {discovered_boundary:.2f} m/s")
    
    # Compare to manual threshold
    manual_threshold = 2.0
    print(f"\n📊 Validation Against Manual Threshold:")
    print(f"  K-means discovered: {discovered_boundary:.2f} m/s")
    print(f"  Manual threshold: {manual_threshold:.2f} m/s")
    print(f"  Difference: {abs(discovered_boundary - manual_threshold):.3f} m/s")
    
    if abs(discovered_boundary - manual_threshold) < 0.3:
        print(f"  ✅ K-means validates bimodal assumption!")
        print(f"     Natural boundary closely matches 2.0 m/s threshold")
    else:
        print(f"  ⚠️ K-means suggests different boundary: {discovered_boundary:.2f} m/s")
        print(f"     Consider using discovered boundary for evaluation")
    
    # Check for severe imbalance
    balance_ratio = min(n_low, n_high) / max(n_low, n_high)
    print(f"\n📊 Cluster Balance:")
    print(f"  Ratio: {balance_ratio:.2%} (minority/majority)")
    
    if balance_ratio < 0.1:
        print(f"  ⚠️ SEVERE IMBALANCE! ({n_low} vs {n_high})")
        print(f"     K-means clustering may not be beneficial")
        print(f"     Consider regenerating data with balanced speeds")
    elif balance_ratio < 0.3:
        print(f"  ⚠️ Imbalanced clusters")
        print(f"     K-means benefit may be limited")
    else:
        print(f"  ✅ Reasonably balanced clusters")
        print(f"     K-means clustering is meaningful")
    
    print("="*60)
    
    return discovered_boundary, low_center, high_center


if __name__ == "__main__":
    # Load collected data
    print("Loading training data...")
    with open('predictive_module/data/kgru_training_data_realistic.pkl', 'rb') as f:
        data = pickle.load(f)
    
    trajectories = data['trajectories']
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Split data: 70% train, 15% val, 15% test
    n_train = int(0.7 * len(trajectories))
    n_val = int(0.15 * len(trajectories))
    
    train_data = trajectories[:n_train]
    val_data = trajectories[n_train:n_train+n_val]
    test_data = trajectories[n_train+n_val:]
    
    print(f"Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Create datasets
    train_dataset = TrajectoryDataset(train_data, sequence_length=10, augment=True)
    val_dataset = TrajectoryDataset(val_data, sequence_length=10, augment=False)
    test_dataset = TrajectoryDataset(test_data, sequence_length=10, augment=False)
    
    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=256, 
        shuffle=True, 
        num_workers=8,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True
    )
    
    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, train_losses, val_losses = train_model(
        train_loader,
        val_loader,
        epochs=100,
        lr=0.001,
        device=device,
        patience=15,
        save_path='predictive_module/model/kgru_model.pth'
    )
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('predictive_module/model/kgru_model.pth'))
    
    # Evaluate on test set
    ade, fde, vel_error = evaluate_predictions(model, test_loader, device=device)
    
    # ========== K-MEANS CLUSTERING ANALYSIS ==========
    # Validate that K-means discovers natural speed boundaries
    discovered_boundary, low_center, high_center = analyze_kmeans_clustering(train_data)
    # =================================================
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.title('K-GRU Training Progress', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('predictive_module/plot/kgru_training.png', dpi=150)
    print("\nTraining curves saved to kgru_training.png")
