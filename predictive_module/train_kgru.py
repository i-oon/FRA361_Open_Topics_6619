"""
Train K-GRU v2 at 25 Hz with class labels
Complete training script with early stopping and validation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from k_gru_predictor import TrajectoryGRU


class TrajectoryDataset(Dataset):
    """
    Dataset for 8D input (motion + class) at 25 Hz
    """
    
    def __init__(self, trajectories, sequence_length=25, augment=False):
        """
        Args:
            trajectories: List of numpy arrays (T, 8)
            sequence_length: Observation window (25 frames @ 25Hz = 1 second)
            augment: Whether to add noise augmentation
        """
        self.trajectories = trajectories
        self.sequence_length = sequence_length
        self.augment = augment
        self.samples = self._create_samples()
    
    def _create_samples(self):
        """Create training samples: input sequence + target next step"""
        samples = []
        
        for traj in self.trajectories:
            # Need sequence_length + 1 for input + target
            if len(traj) < self.sequence_length + 1:
                continue
            
            # Sliding window
            for i in range(len(traj) - self.sequence_length):
                input_seq = traj[i:i + self.sequence_length]      # (seq_len, 8)
                target = traj[i + self.sequence_length, :4]       # (4,) motion only
                samples.append((input_seq, target))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_seq, target = self.samples[idx]
        input_seq = input_seq.copy()   # always copy — we will mutate below
        target    = target.copy()

        if self.augment:
            # 1. Position noise (realistic GPS/tracking error)
            pos_noise = np.random.normal(0, 0.05, input_seq[:, :2].shape)  # ± 5cm
            input_seq[:, :2] += pos_noise

            # 2. Velocity noise (acceleration/measurement error)
            vel_noise = np.random.normal(0, 0.1, input_seq[:, 2:4].shape)   # ± 0.1 m/s
            input_seq[:, 2:4] += vel_noise

            # 3. Random small rotation (heading error)
            if np.random.rand() < 0.3:  # 30% chance
                angle = np.random.uniform(-3, 3) * np.pi / 180  # ± 3 degrees
                cos_a, sin_a = np.cos(angle), np.sin(angle)

                # Rotate positions relative to first frame
                origin = input_seq[0, :2]
                rel_pos = input_seq[:, :2] - origin

                rotated = np.zeros_like(rel_pos)
                rotated[:, 0] = rel_pos[:, 0] * cos_a - rel_pos[:, 1] * sin_a
                rotated[:, 1] = rel_pos[:, 0] * sin_a + rel_pos[:, 1] * cos_a

                input_seq[:, :2] = rotated + origin

                # Rotate velocities
                vel = input_seq[:, 2:4]
                rotated_vel = np.zeros_like(vel)
                rotated_vel[:, 0] = vel[:, 0] * cos_a - vel[:, 1] * sin_a
                rotated_vel[:, 1] = vel[:, 0] * sin_a + vel[:, 1] * cos_a
                input_seq[:, 2:4] = rotated_vel

            # 4. Random scaling (speed variation)
            if np.random.rand() < 0.2:  # 20% chance
                scale = np.random.uniform(0.95, 1.05)  # ± 5% speed
                input_seq[:, 2:4] *= scale

            # 5. Random dropout frames (occlusion simulation)
            if np.random.rand() < 0.1:  # 10% chance
                n_dropout = np.random.randint(1, 4)  # Drop 1-3 frames
                dropout_idx = np.random.choice(len(input_seq), n_dropout, replace=False)
                # Interpolate dropped frames
                for drop_i in dropout_idx:
                    if drop_i > 0 and drop_i < len(input_seq) - 1:
                        input_seq[drop_i, :4] = (input_seq[drop_i-1, :4] + input_seq[drop_i+1, :4]) / 2

        # Position normalization: shift so the last observed frame is at origin.
        # Makes the model position-invariant → predictions generalize across
        # different coordinate systems (crucial for cross-domain evaluation).
        origin_xy = input_seq[-1, :2].copy()
        input_seq[:, 0] -= origin_xy[0]
        input_seq[:, 1] -= origin_xy[1]
        target[0] -= origin_xy[0]
        target[1] -= origin_xy[1]

        return (
            torch.FloatTensor(input_seq),
            torch.FloatTensor(target)
        )


def train_kgru(
    data_path='predictive_module/data/ind_class.pkl',
    save_path='predictive_module/model/kgru_ind.pth',
    sequence_length=25,      # 1 second @ 25 Hz
    batch_size=256,          # Smaller batch (8D input = more memory)
    epochs=100,
    learning_rate=0.001,
    patience=15,             # Early stopping patience
    device=None
):
    """
    Train K-GRU v2 at 25 Hz with class labels
    
    Args:
        data_path: Path to preprocessed .pkl file
        save_path: Where to save trained model
        sequence_length: Observation window (frames)
        batch_size: Training batch size
        epochs: Maximum training epochs
        learning_rate: Adam learning rate
        patience: Early stopping patience
        device: 'cuda' or 'cpu' (auto-detect if None)
    """
    
    print("="*70)
    print("TRAINING K-GRU V2 @ 25 Hz WITH CLASS LABELS")
    print("="*70)
    
    # Device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load data
    print(f"\n📂 Loading data...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    trajectories = data['trajectories']
    dt = data['dt']
    frequency = data['frequency']
    
    print(f"   Trajectories: {len(trajectories)}")
    print(f"   Frequency: {frequency} Hz (dt={dt}s)")
    print(f"   Input format: {data['input_format']}")
    print(f"   Feature dimension: {data['feature_dim']}D")
    
    # Verify dt matches expected
    expected_dt = 1.0 / 25.0
    if abs(dt - expected_dt) > 0.001:
        print(f"   ⚠️ WARNING: Expected dt={expected_dt:.4f}, got dt={dt:.4f}")
    
    # Split data
    n_train = int(0.7 * len(trajectories))
    n_val = int(0.15 * len(trajectories))
    
    train_data = trajectories[:n_train]
    val_data = trajectories[n_train:n_train+n_val]
    test_data = trajectories[n_train+n_val:]
    
    print(f"\n📊 Data split:")
    print(f"   Train: {len(train_data)} trajectories")
    print(f"   Val:   {len(val_data)} trajectories")
    print(f"   Test:  {len(test_data)} trajectories")
    
    # Create datasets
    print(f"\n🔨 Creating datasets (sequence_length={sequence_length})...")
    train_dataset = TrajectoryDataset(train_data, sequence_length, augment=True)
    val_dataset = TrajectoryDataset(val_data, sequence_length, augment=False)
    test_dataset = TrajectoryDataset(test_data, sequence_length, augment=False)
    
    print(f"   Train samples: {len(train_dataset):,}")
    print(f"   Val samples:   {len(val_dataset):,}")
    print(f"   Test samples:  {len(test_dataset):,}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    # Model
    print(f"\n🤖 Creating model...")
    model = TrajectoryGRU(
        input_size=8,
        hidden_size=256,
        num_layers=4,
        output_size=4,
        dropout=0.3
    ).to(device)
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Training loop
    print(f"\n🚀 Training...")
    print(f"   Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print(f"   Early stopping patience: {patience}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs[:, -1, :], targets)  # Predict last step
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        train_loss /= train_batches
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs, _ = model(inputs)
                loss = criterion(outputs[:, -1, :], targets)
                
                val_loss += loss.item()
                val_batches += 1
        
        val_loss /= val_batches
        val_losses.append(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'input_size': 8,
                    'hidden_size': 256,
                    'num_layers': 4,
                    'output_size': 4,
                    'sequence_length': sequence_length,
                    'frequency': frequency,
                    'dt': dt,
                }
            }, save_path)
            print(f"   ✅ Saved best model (val_loss={val_loss:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                break
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training Curves - K-GRU v2 @ 25 Hz')
    plt.grid(True, alpha=0.3)
    
    plot_path = save_path.replace('.pth', '_training.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Training curves saved: {plot_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"✅ Best validation loss: {best_val_loss:.6f}")
    print(f"✅ Model saved: {save_path}")
    print(f"   Input: {sequence_length} frames @ 25 Hz = {sequence_length/25:.2f}s observation")
    print(f"   Format: 8D [motion(4) + class(4)]")
    print("="*70)


if __name__ == "__main__":
    # Train model
    train_kgru(
        data_path='predictive_module/data/ind_with_class.pkl',
        save_path='predictive_module/model/kgru_ind.pth',
        sequence_length=25,   # 1 second @ 25 Hz
        batch_size=256,       # Adjust based on GPU memory
        epochs=100,
        learning_rate=0.001,
        patience=15
    )