# FRA361 Open Topics: Anticipatory Navigation with K-GRU Prediction

## Project Overview

This project implements **anticipatory dynamic obstacle navigation** for mobile robots using K-GRU (K-means + GRU) trajectory prediction and TD3 reinforcement learning. The system enables robots to predict future obstacle positions and proactively avoid collisions rather than reactively responding to current states.

**Key Innovation:** Combining prediction-based planning with reinforcement learning for safer, smoother navigation in dynamic environments with both pedestrian and vehicle obstacles.

---

## Table of Contents

1. [Methodology](#methodology)
2. [Implementation Progress](#implementation-progress)
3. [K-GRU Prediction Module](#k-gru-prediction-module)
4. [Results](#results)
5. [File Structure](#file-structure)
6. [Usage Guide](#usage-guide)
7. [Next Steps](#next-steps)

---

## Methodology

### K-GRU Prediction Pipeline

Based on Liu et al. (2025) "Adaptive Motion Planning Leveraging Speed-Differentiated Prediction for Mobile Robots in Dynamic Environments" (Applied Sciences, 15(13), 7551).

**Three-Stage Architecture:**

1. **K-means Clustering (Speed Classification)**
   - Separates obstacles into low-speed (<2.0 m/s) and high-speed (≥2.0 m/s) groups
   - K=2 clusters based on velocity magnitude
   - Prioritizes common low-speed obstacles, buffers high-speed

2. **Kalman Filter (State Estimation)**
   - Brownian motion model for obstacle dynamics
   - State vector: x = [x, y, vx, vy]^T (4D)
   - Smooths noisy sensor data
   - Provides clean estimates for GRU input

3. **GRU Network (Trajectory Prediction)**
   - Input: Sequence of 10 timesteps [x, y, vx, vy]
   - Architecture: 2 layers, 50 hidden units per layer
   - Output: Predicts future positions 5-20 timesteps ahead
   - Learns motion patterns and long-term dependencies

### Reinforcement Learning Integration

**TD3 (Twin Delayed Deep Deterministic Policy Gradient):**
- **Baseline:** Reactive navigation using only current obstacle positions
- **Anticipatory:** Uses K-GRU predictions to avoid future collision zones
- **Reward shaping:** Penalty based on predicted proximity, not just current distance

---

## Implementation Progress

### ✅ Completed

1. **K-GRU Predictor Module** (`k_gru_predictor.py`)
   - Full three-stage pipeline implementation
   - Model persistence (save/load)
   - Integration-ready API

2. **Data Collection Pipeline**
   - **Version 1:** Simple dynamics (3,075 trajectories)
   - **Version 2:** Realistic pedestrian/vehicle dynamics (3,320 trajectories)
   - Diverse configurations (3-10 obstacles, varying speed ratios)

3. **Training Infrastructure**
   - PyTorch training loop with validation
   - Early stopping (patience=15 epochs)
   - Learning rate scheduling
   - Data augmentation (random flips)
   - GPU optimization (batch_size=256, num_workers=8)

4. **Evaluation & Visualization**
   - Trajectory prediction plots (6 examples)
   - Error growth analysis over prediction horizon
   - Speed comparison (low vs high speed obstacles)
   - Comprehensive metrics (ADE, FDE, velocity error)

### 🔄 In Progress

- TD3 baseline agent (reactive navigation)
- TD3 + K-GRU agent (anticipatory navigation)
- Comparative experiments

---

## K-GRU Prediction Module

### Architecture Details

**GRU Network:**
```python
TrajectoryGRU(
    input_size=4,      # [x, y, vx, vy]
    hidden_size=50,    # Hidden units per layer
    num_layers=2,      # Stacked GRU layers
    output_size=4      # [x_next, y_next, vx_next, vy_next]
)
```

**Kalman Filter:**
- State transition matrix: 4×4 with Δt terms
- Process noise Q: 0.01 * I(4)
- Measurement noise R: 0.1 * I(4)
- Full state observability (H = I(4))

**K-means Clustering:**
- K=2 clusters (low-speed, high-speed)
- Threshold: 2.0 m/s (pedestrian vs vehicle boundary)
- Features: velocity magnitude

### Data Collection

#### Version 1: Simple Dynamics
- **Trajectories:** 3,075
- **Characteristics:** Uniform motion, no noise, simple patterns
- **Results:** Excellent accuracy (ADE: 0.0018m), but unrealistic

#### Version 2: Realistic Dynamics
- **Trajectories:** 3,320
- **Episode length:** 2,000 steps (1,001 avg)
- **Speed distribution:**
  - Low-speed (<2.0 m/s): 47.5% (pedestrians)
  - High-speed (≥2.0 m/s): 52.5% (vehicles)
  - Mean: 2.10 m/s, Std: 1.05 m/s

**Pedestrian Characteristics (0.8-1.5 m/s):**
- Random stops: 5% chance per step (5-20 steps duration)
- Direction changes: 8% chance per step (±60° max)
- Speed variations: ±15% (walking rhythm)

**Vehicle Characteristics (2.0-4.0 m/s):**
- Smooth trajectories: momentum-based (0.9 factor)
- Small turn radius: ±30° max, 2% chance per step
- Consistent speed: ±5% variation

**Sensor Noise:**
- Position: ±3cm std (simulates LiDAR uncertainty)
- Velocity: ±5cm/s std (simulates estimation error)

### Training Configuration

```python
Optimizer: Adam (lr=0.001, weight_decay=1e-5)
Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
Batch size: 256
Epochs: 100 (early stopping at ~30-40 typically)
Data split: 70% train, 15% val, 15% test
Augmentation: Random x/y axis flips
Gradient clipping: max_norm=1.0
Device: CUDA (RTX 3050 Laptop GPU)
Training time: ~10-15 hours
```

---

## Results

### Version 1: Simple Dynamics

| Metric | Value | Assessment |
|--------|-------|------------|
| **ADE (Position)** | 0.0018m (1.8mm) | ⭐ Excellent |
| **Velocity Error** | 0.0048 m/s | ⭐ Excellent |
| **Validation Loss** | 0.000968 | Very low |

**Speed Comparison:**
- Low-speed: 0.408m
- High-speed: 0.428m
- **Difference: 5%** (minimal - K-means benefit not visible)

**Conclusion:** Excellent accuracy but unrealistic due to perfect sensing and simple dynamics.

---

### Version 2: Realistic Dynamics (Current)

| Metric | Value | Real-World Implication |
|--------|-------|------------------------|
| **ADE (Position)** | 0.0455m (4.5cm) | ✅ Within sensor noise |
| **Velocity Error** | 0.3860 m/s | ✅ Acceptable for 2-4 m/s speeds |
| **Validation Loss** | 0.085534 | Good generalization |

**Error Growth Over Time:**
- **Step 1:** ~0.05m (excellent)
- **Step 5:** ~0.15m (good for collision avoidance)
- **Step 10:** ~0.30m (acceptable)
- **Step 20:** ~0.45m (still usable)

**Speed Comparison:**
- **Low-speed (<2.0 m/s, pedestrians):** 0.2198m (n=120)
- **High-speed (≥2.0 m/s, vehicles):** 0.2033m (n=80)
- **Difference: 8%** (low-speed slightly worse)

**Key Finding:** Pedestrians are actually harder to predict than vehicles due to random stops and sudden direction changes, despite lower speeds. This validates the realistic dynamics implementation.

**Comparison to Liu et al. (2025) Paper:**
- Paper ADE: 0.024m (real-world, with LiDAR noise)
- Our ADE: 0.045m (simulation, with synthetic noise)
- **Within expected range** considering different noise models

---

### Trajectory Prediction Examples

From 20-step horizon predictions:

| Trajectory | Speed (m/s) | Type | Error (m) | Quality |
|-----------|-------------|------|-----------|---------|
| 1 | 2.70 | Vehicle | 0.045 | ⭐ Excellent |
| 2 | 2.77 | Vehicle | 0.414 | ✅ Good |
| 3 | 1.01 | Pedestrian | 0.066 | ⭐ Excellent |
| 4 | 3.17 | Vehicle | 0.256 | ✅ Good |
| 5 | 1.07 | Pedestrian | 0.975 | ❌ Challenging case |
| 6 | 1.25 | Pedestrian | 0.381 | ✅ Good |

**Overall:** 83% of predictions have <0.5m error over 20 steps (~2 seconds ahead).

---

### Why Errors Increased from Version 1 to Version 2

**Expected and Desirable!**

| Factor | Version 1 | Version 2 | Impact |
|--------|-----------|-----------|--------|
| Sensor noise | None | ±3cm pos, ±5cm/s vel | +~3cm baseline |
| Motion complexity | Simple | Random stops, turns | +~2cm |
| Speed diversity | Narrow | 0.8-4.0 m/s range | +~1cm |
| Prediction difficulty | Easy patterns | Stochastic behaviors | +~1cm |

**Total increase:** ~0.002m → 0.045m (20x), but this represents **realistic performance**.

For navigation:
- Robot safety radius: ~0.3m
- Prediction error at 5 steps: ~0.15m
- **Safety buffer: 0.15m** (50% of robot radius, acceptable!)

---

## File Structure

```
FRA361_Open_Topics_6619/
├── env/
│   ├── dynamic_nav_env.py              # MuJoCo environment (5 dynamic obstacles)
│   └── test_environment.py
├── omni_carver_description/
│   ├── description/
│   │   ├── omni_carver.xml             # Robot model (kv=150, 3 omni wheels)
│   │   └── omni_carver.urdf
│   └── mesh/                           # STL files
├── predictive_module/
│   ├── data/
│   │   └── kgru_training_data_realistic.pkl  # 3,320 trajectories, ~80MB
│   ├── model/
│   │   └── kgru_model.pth              # Trained weights, ~500KB
│   ├── plot/
│   │   ├── kgru_training.png           # Loss curves
│   │   ├── kgru_evaluation.png         # Error distributions
│   │   ├── trajectory_predictions.png  # 6 example predictions
│   │   ├── error_over_time.png         # Error growth analysis
│   │   └── speed_comparison.png        # Low vs high speed comparison
│   ├── k_gru_predictor.py              # Main predictor class
│   ├── data_collection_realistic.py    # Data generation with dynamics
│   ├── train_kgru.py                   # Training script
│   └── visualize_predictions.py        # Evaluation & plotting
├── README.md                            # This file
└── (TD3 implementation - coming next)
```

---

## Usage Guide

### 1. Data Collection

```bash
# Collect realistic training data (2-3 hours)
python3 predictive_module/data_collection_realistic.py

# Output: predictive_module/data/kgru_training_data_realistic.pkl
# Contains: 3,320 trajectories, diverse speeds, sensor noise
```

### 2. Training

```bash
# Train K-GRU model (10-15 hours on RTX 3050)
python3 predictive_module/train_kgru.py

# Output: 
#   - predictive_module/model/kgru_model.pth
#   - predictive_module/plot/kgru_training.png
#   - predictive_module/plot/kgru_evaluation.png
```

**Monitor GPU during training:**
```bash
watch -n 1 nvidia-smi
```

### 3. Visualization

```bash
# Generate prediction visualizations (5 minutes)
python3 predictive_module/visualize_predictions.py

# Output:
#   - predictive_module/plot/trajectory_predictions.png
#   - predictive_module/plot/error_over_time.png
#   - predictive_module/plot/speed_comparison.png
```

### 4. Using the Predictor

```python
from predictive_module.k_gru_predictor import KGRUPredictor

# Initialize predictor
predictor = KGRUPredictor(
    n_obstacles=5,
    sequence_length=10,
    device='cuda'
)

# Load trained model
predictor.load_model('predictive_module/model/kgru_model.pth')

# Predict future trajectories
obstacle_states = [
    [x1, y1, vx1, vy1],  # Obstacle 1
    [x2, y2, vx2, vy2],  # Obstacle 2
    # ...
]

predictions = predictor.predict(obstacle_states, horizon=5)
# Returns: {obstacle_id: trajectory_array}
# trajectory_array.shape = (horizon, 4)  # [x, y, vx, vy] for each step
```

---

## Key Design Decisions

### Why Prioritize Low-Speed Obstacles?
- Most obstacles in real environments are low-speed (pedestrians, slow robots)
- Easier to predict accurately due to lower velocities
- High-speed obstacles processed separately to avoid noise contamination

### Why Kalman Filter Before GRU?
- Raw sensor data is noisy (especially in real LiDAR)
- Kalman provides physics-based smoothing
- GRU receives cleaner, more consistent inputs
- Combines model-based (Kalman) + learning-based (GRU) approaches

### Why 10-Timestep History?
- Enough to capture motion patterns
- Not too long (computational efficiency)
- Matches paper configuration
- Balances short-term reactivity vs long-term planning

### Why Data Augmentation?
- Doubles effective dataset size via random flips
- Prevents overfitting to specific directions
- Robot should handle obstacles from any direction
- Preserves physics (flipping is valid transformation)

### Why Realistic Dynamics Matter?
- Simple dynamics → unrealistic perfect predictions
- Real world has:
  - Pedestrians: unpredictable, sudden stops/turns
  - Vehicles: momentum, smooth trajectories
  - Sensor noise: ±3-5cm typical for LiDAR
- Model trained on realistic data generalizes better to hardware

---

## Dependencies

```bash
# Core
pip install torch numpy matplotlib scipy

# Environment
pip install mujoco gymnasium

# K-GRU specific
pip install filterpy scikit-learn tqdm

# Visualization
pip install matplotlib
```

**System Requirements:**
- Python 3.10+
- CUDA-capable GPU (recommended, RTX 3050 or better)
- 8GB+ RAM
- 2GB+ disk space for data and models

---

## Next Steps

### Phase 1: TD3 Baseline (Reactive Navigation)
1. Implement TD3 agent architecture
2. Define observation space (robot state + current obstacle positions)
3. Define action space (vx, vy, omega)
4. Design reactive reward function (current distance-based penalties)
5. Train baseline agent (no predictions)

### Phase 2: TD3 + K-GRU (Anticipatory Navigation)
1. Integrate K-GRU predictor with environment
2. Extend observation space (predicted future positions)
3. Design anticipatory reward function (predicted collision zones)
4. Train anticipatory agent (with predictions)

### Phase 3: Comparative Experiments
1. Success rate (% episodes reaching goal)
2. Collision rate (% episodes with collisions)
3. Average time to goal
4. Path smoothness (angular velocity variance)
5. Freeze time (% time robot stopped/stuck)
6. Statistical significance testing (t-test)

### Phase 4: Analysis & Reporting
1. Performance comparison tables
2. Trajectory visualizations (baseline vs anticipatory)
3. Ablation studies (prediction horizon effects)
4. Discussion of limitations and future work

---

## Related Work

**Main Reference:**
Liu, Y., et al. (2025). Adaptive Motion Planning Leveraging Speed-Differentiated Prediction for Mobile Robots in Dynamic Environments. *Applied Sciences*, 15(13), 7551.

**Key Contributions:**
- K-means clustering for speed-based obstacle separation
- GRU for learning-based trajectory prediction
- Demonstrated superiority over LSTM, Kalman-only, particle filters

**Our Extensions:**
- TD3 reinforcement learning integration
- Realistic pedestrian/vehicle dynamics modeling
- Simulation-to-simulation validation framework
- Comprehensive evaluation metrics

---

## Authors

**Student:** Beam (i-oon)  
**Course:** FRA361 Open Topics in Field and Service Robotics  
**Institution:** Vidyasirimedhi Institute of Science and Technology (VISTEC)  
**Semester:** 2024-2025

---

## License

Educational project for academic purposes.

---

## Acknowledgments

- Liu et al. (2025) for K-GRU methodology
- Anthropic Claude for development assistance
- VISTEC FRA361 course staff
- MuJoCo physics engine team

---

**Last Updated:** February 17, 2026  
**Status:** K-GRU prediction module complete ✅ | TD3 integration pending 🔄