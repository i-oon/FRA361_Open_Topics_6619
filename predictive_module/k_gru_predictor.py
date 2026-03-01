"""
K-GRU v2: Enhanced with class labels
Input: [x, y, vx, vy, is_car, is_ped, is_truck, is_bicycle] = 8D
Output: [x, y, vx, vy] = 4D (class is constant)

CRITICAL FIX: Hidden state reset to avoid temporal inversion
"""

import torch
import torch.nn as nn


class TrajectoryGRU(nn.Module):
    """
    Enhanced GRU with class label inputs
    
    Architecture:
    - Input: (batch, seq_len, 8) - motion + class
    - GRU: 3 layers, 128 hidden units
    - Output: (batch, seq_len, 4) - motion only
    """
    
    def __init__(self, input_size=8, hidden_size=128, num_layers=3, output_size=4, dropout=0.2):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection (predict motion only, not class)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        """
        Forward pass
        
        Args:
            x: (batch, seq_len, 8) - input sequence with class
            hidden: Optional GRU hidden state
        
        Returns:
            output: (batch, seq_len, 4) - predicted motion
            hidden: Updated hidden state
        """
        # GRU forward
        gru_out, hidden = self.gru(x, hidden)
        
        # Project to motion
        output = self.fc(gru_out)
        
        return output, hidden
    
    def predict_sequence(self, input_seq, horizon=100):
        """
        Autoregressive multi-step prediction
        
        CRITICAL FIX: Reset hidden=None each step to avoid temporal inversion
        
        Args:
            input_seq: (batch, obs_len, 8) - observed sequence with class
            horizon: Number of future steps to predict
        
        Returns:
            predictions: (batch, horizon, 4) - predicted motion only
        """
        self.eval()
        device = input_seq.device
        batch_size = input_seq.shape[0]
        
        predictions = []
        
        # Extract class (constant throughout prediction)
        # Shape: (batch, 4)
        agent_class = input_seq[:, -1, 4:8]
        
        # Start with observed sequence
        current_seq = input_seq.clone()
        
        with torch.no_grad():
            for step in range(horizon):
                # ✅ CRITICAL FIX: Reset hidden state each step!
                # This prevents temporal inversion bug
                pred_motion, _ = self.forward(current_seq, hidden=None)
                
                # Take last prediction
                next_motion = pred_motion[:, -1:, :]  # (batch, 1, 4)
                
                # Append class to create full 8D input
                next_full = torch.cat([
                    next_motion,
                    agent_class.unsqueeze(1)  # (batch, 1, 4)
                ], dim=2)  # (batch, 1, 8)
                
                # Store motion prediction
                predictions.append(next_motion)
                
                # Update sequence (sliding window)
                current_seq = torch.cat([
                    current_seq[:, 1:, :],  # Remove oldest
                    next_full                # Add newest
                ], dim=1)
        
        # Stack predictions
        predictions = torch.cat(predictions, dim=1)  # (batch, horizon, 4)
        
        return predictions


if __name__ == "__main__":
    # Test model
    print("Testing TrajectoryGRU...")
    
    model = TrajectoryGRU(input_size=8, hidden_size=128, num_layers=3, output_size=4)
    
    # Test input: batch=2, seq_len=25 (1s @ 25Hz), features=8
    test_input = torch.randn(2, 25, 8)
    
    # Forward pass
    output, hidden = model(test_input)
    print(f"Forward pass: {test_input.shape} -> {output.shape}")
    
    # Autoregressive prediction
    predictions = model.predict_sequence(test_input, horizon=100)
    print(f"Prediction: {test_input.shape} -> {predictions.shape}")
    
    print("✅ Model test passed!")