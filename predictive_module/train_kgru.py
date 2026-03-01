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
        
        # Optional augmentation (only for training)
        if self.augment:
            # Add small Gaussian noise to positions and velocities
            noise = np.random.normal(0, 0.01, input_seq[:, :4].shape)
            input_seq = input_seq.copy()
            input_seq[:, :4] += noise
        
        return (
            torch.FloatTensor(input_seq),
            torch.FloatTensor(target)
        )


def train_kgru(
    data_path='predictive_module/data/ind_class.pkl',
    save_path='predictive_module/model/kgru_ind.pth',
    sequence_length=25,      # 1 second @ 25 Hz
    batch_size=128,          # Smaller batch (8D input = more memory)
    epochs=150,
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
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Model
    print(f"\n🤖 Creating model...")
    model = TrajectoryGRU(
        input_size=8,
        hidden_size=128,
        num_layers=3,
        output_size=4,
        dropout=0.2
    ).to(device)
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
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
                    'hidden_size': 128,
                    'num_layers': 3,
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
        batch_size=128,       # Adjust based on GPU memory
        epochs=150,
        learning_rate=0.001,
        patience=15
    )