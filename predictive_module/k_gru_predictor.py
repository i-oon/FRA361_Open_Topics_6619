"""
K-GRU Kalman Prediction Module
Based on: Liu et al. (2025) - Adaptive Motion Planning Leveraging 
Speed-Differentiated Prediction for Mobile Robots in Dynamic Environments
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from filterpy.kalman import KalmanFilter
from collections import deque

class KGRUPredictor:
    """
    K-GRU Kalman predictor for dynamic obstacle trajectories
    
    Combines:
    1. K-means clustering (speed-based grouping)
    2. Kalman filtering (state estimation)
    3. GRU network (trajectory prediction)
    """
    
    def __init__(
        self,
        history_length=10,
        dt=0.1,  # Timestep (from MuJoCo)
        low_speed_threshold=0.2,
        high_speed_threshold=1.1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.history_length = history_length
        self.dt = dt
        self.low_speed_threshold = low_speed_threshold
        self.high_speed_threshold = high_speed_threshold
        self.device = device
        
        # GRU network
        self.gru_model = TrajectoryGRU(
            input_size=4,
            hidden_size=128,
            num_layers=3,
            output_size=4,
            dropout=0.2
        ).to(device)
        
        # K-means clusterer
        self.kmeans = KMeans(n_clusters=2, random_state=42)
        
        # History buffers for each obstacle
        self.obstacle_histories = {}
        
        # Kalman filters for each obstacle
        self.kalman_filters = {}
        
    def _init_kalman_filter(self):
        """Initialize Kalman filter with Brownian motion model"""
        kf = KalmanFilter(dim_x=4, dim_z=4)
        
        # State transition matrix (Brownian motion)
        kf.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Observation matrix (observe all states)
        kf.H = np.eye(4)
        
        # Process noise covariance
        kf.Q = np.eye(4) * 0.01
        
        # Measurement noise covariance
        kf.R = np.eye(4) * 0.1
        
        # Initial state covariance
        kf.P *= 1.0
        
        return kf
    
    def update(self, obstacle_states):
        """
        Update predictor with new obstacle observations
        
        Args:
            obstacle_states: List of dicts with keys ['id', 'pos', 'vel', 'speed_group']
                            pos: (x, y), vel: (vx, vy)
        """
        for obs in obstacle_states:
            obs_id = obs['id']
            state = np.array([
                obs['pos'][0],
                obs['pos'][1],
                obs['vel'][0],
                obs['vel'][1]
            ])
            
            # Initialize history buffer if new obstacle
            if obs_id not in self.obstacle_histories:
                self.obstacle_histories[obs_id] = deque(maxlen=self.history_length)
                self.kalman_filters[obs_id] = self._init_kalman_filter()
                self.kalman_filters[obs_id].x = state
            
            # Kalman filter update
            kf = self.kalman_filters[obs_id]
            kf.predict()
            kf.update(state)
            
            # Store filtered state in history
            self.obstacle_histories[obs_id].append(kf.x.copy())
    
    def classify_speeds(self, obstacle_states):
        """
        K-means clustering to discover natural speed groups
        
        TRUE K-means: Discovers boundary from data, not manual threshold!
        
        Returns:
            low_speed_ids: List of obstacle IDs in discovered low-speed group
            high_speed_ids: List of obstacle IDs in discovered high-speed group
            boundary: Discovered speed boundary
        """
        if len(obstacle_states) < 2:
            # Not enough obstacles for clustering
            # Use single obstacle's speed vs median of history
            speeds = [np.linalg.norm(obs['vel']) for obs in obstacle_states]
            ids = [obs['id'] for obs in obstacle_states]
            
            # Default boundary if we have history
            if hasattr(self, 'discovered_boundary'):
                boundary = self.discovered_boundary
            else:
                boundary = 2.0  # Fallback only
            
            low_speed_ids = [id for id, s in zip(ids, speeds) if s < boundary]
            high_speed_ids = [id for id, s in zip(ids, speeds) if s >= boundary]
            return low_speed_ids, high_speed_ids, boundary
        
        # Extract velocities for clustering
        velocities = np.array([obs['vel'] for obs in obstacle_states])
        speeds = np.linalg.norm(velocities, axis=1).reshape(-1, 1)
        
        # K-means clustering - DISCOVERS the grouping!
        labels = self.kmeans.fit_predict(speeds)
        
        # Determine which cluster is low-speed vs high-speed
        cluster_means = [speeds[labels == i].mean() for i in range(2)]
        low_cluster = 0 if cluster_means[0] < cluster_means[1] else 1
        high_cluster = 1 - low_cluster
        
        # Calculate discovered boundary
        centers = self.kmeans.cluster_centers_.flatten()
        self.discovered_boundary = (centers[0] + centers[1]) / 2
        
        # Separate obstacle IDs based on K-means labels
        low_speed_ids = [obs['id'] for obs, label in zip(obstacle_states, labels) 
                        if label == low_cluster]
        high_speed_ids = [obs['id'] for obs, label in zip(obstacle_states, labels) 
                        if label == high_cluster]
        
        return low_speed_ids, high_speed_ids, self.discovered_boundary
    
    def predict(self, obstacle_states, prediction_horizon=5):
        """
        Predict future trajectories for all obstacles using K-means clustering
        
        Args:
            obstacle_states: Current obstacle observations
            prediction_horizon: Number of timesteps to predict ahead
            
        Returns:
            predictions: Dict mapping obstacle_id -> predicted trajectory
            boundary: K-means discovered speed boundary (for analysis)
        """
        # Update with current observations
        self.update(obstacle_states)
        
        # Classify into speed groups using K-means
        low_speed_ids, high_speed_ids, boundary = self.classify_speeds(obstacle_states)
        
        predictions = {}
        
        # Process low-speed obstacles first (priority)
        for obs_id in low_speed_ids:
            if obs_id in self.obstacle_histories and len(self.obstacle_histories[obs_id]) >= 3:
                trajectory = self._predict_gru(obs_id, prediction_horizon)
                predictions[obs_id] = trajectory
        
        # Process high-speed obstacles (delayed)
        for obs_id in high_speed_ids:
            if obs_id in self.obstacle_histories and len(self.obstacle_histories[obs_id]) >= 3:
                trajectory = self._predict_gru(obs_id, prediction_horizon)
                predictions[obs_id] = trajectory
        
        return predictions, boundary
    
    def _predict_gru(self, obs_id, horizon):
        """
        Use GRU to predict trajectory for single obstacle
        
        Args:
            obs_id: Obstacle ID
            horizon: Prediction horizon
            
        Returns:
            trajectory: Array of shape (horizon, 4)
        """
        history = np.array(list(self.obstacle_histories[obs_id]))
        
        # Convert to tensor
        history_tensor = torch.FloatTensor(history).unsqueeze(0).to(self.device)
        
        # Predict using GRU
        with torch.no_grad():
            trajectory = self.gru_model.predict_sequence(history_tensor, horizon)
        
        return trajectory.cpu().numpy()[0]  # Shape: (horizon, 4)
    
    def save_model(self, path):
        """Save trained GRU model"""
        torch.save(self.gru_model.state_dict(), path)
    
    def load_model(self, path):
        """Load trained GRU model"""
        self.gru_model.load_state_dict(torch.load(path, map_location=self.device))
        self.gru_model.eval()


class TrajectoryGRU(nn.Module):
    """
    GRU network for trajectory prediction
    
    Architecture from Liu et al. (2025):
    - Input: (batch, seq_len, 4) where 4 = [x, y, vx, vy]
    - GRU: 2 layers, 50 hidden units
    - Output: (batch, 4) predicted next state
    """
    
    def __init__(self, input_size=4, hidden_size=50, num_layers=2, output_size=4):
        super(TrajectoryGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Dense output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden=None):
        """
        Forward pass
        
        Args:
            x: Input sequence (batch, seq_len, 4)
            hidden: Initial hidden state (optional)
            
        Returns:
            output: Predicted next state (batch, 4)
            hidden: Final hidden state
        """
        # GRU forward
        gru_out, hidden = self.gru(x, hidden)
        
        # Take last timestep output
        last_output = gru_out[:, -1, :]
        
        # Dense layer
        output = self.fc(last_output)
        
        return output, hidden
    
    def predict_sequence(self, x, horizon):
        """
        Predict trajectory for multiple timesteps
        
        Args:
            x: Initial sequence (batch, seq_len, 4)
            horizon: Number of steps to predict
            
        Returns:
            predictions: Predicted trajectory (batch, horizon, 4)
        """
        self.eval()
        predictions = []
        
        current_input = x
        hidden = None
        
        for _ in range(horizon):
            # Predict next step
            pred, hidden = self.forward(current_input, hidden)
            predictions.append(pred)
            
            # Update input: append prediction, remove oldest
            current_input = torch.cat([
                current_input[:, 1:, :],
                pred.unsqueeze(1)
            ], dim=1)
        
        return torch.stack(predictions, dim=1)