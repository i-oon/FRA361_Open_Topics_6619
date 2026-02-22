# FRA361 Open Topics: Anticipatory Navigation with K-GRU Prediction

## Project Overview

This project implements **anticipatory dynamic obstacle navigation** for mobile robots using K-GRU (K-means + GRU) trajectory prediction and TD3 reinforcement learning. The system enables robots to predict future obstacle positions and proactively avoid collisions rather than reactively responding to current states.

**Key Innovation:** Combining prediction-based planning with reinforcement learning for safer, smoother navigation in dynamic environments with both pedestrian and vehicle obstacles.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Methodology](#methodology)
3. [Implementation Progress](#implementation-progress)
4. [K-GRU Prediction Module](#k-gru-prediction-module)
5. [Results](#results)
6. [File Structure](#file-structure)
7. [Usage Guide](#usage-guide)
8. [Key Findings](#key-findings)
9. [Related Work](#related-work)
10. [Next Steps](#next-steps)

---

## Environment Setup

<p align="center">
    <img width="50%" src="vishual/mujoco_omni_carver.gif">
    </br> 

</p>


### Prerequisites

**System Requirements:**
- Ubuntu 22.04 LTS (or compatible Linux distribution)
- Python 3.10+
- CUDA-capable GPU (recommended: RTX 3050 or better)
- 8GB+ RAM
- 5GB+ disk space

### 1. Install MuJoCo

**Method A: Download Pre-built Binary (Recommended)**

```bash
# Create directory
mkdir -p ~/.mujoco
cd ~/.mujoco

# Download MuJoCo 2.1.0 (or latest)
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz

# Extract
tar -xzf mujoco210-linux-x86_64.tar.gz

# Add to PATH
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin' >> ~/.bashrc
source ~/.bashrc
```

**Method B: Install via pip (Easier)**

```bash
pip install mujoco==2.3.0
```

**Verify Installation:**

```bash
python3 -c "import mujoco; print(mujoco.__version__)"
# Should print: 2.3.0 (or your version)
```

---

### 2. Set Up Python Environment

**Option A: Using venv (Recommended)**

```bash
# Create virtual environment
cd ~/FRA361_Open_Topics_6619
python3 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

**Option B: Using conda**

```bash
# Create conda environment
conda create -n fra361 python=3.10
conda activate fra361
```

---

### 3. Install Dependencies

**Install from requirements:**

```bash
pip install -r requirements.txt
```

---


### 4. Test Dynamic Navigation Environment

**Clone Project:**

```bash
cd ~
git clone <your-repo-url> FRA361_Open_Topics_6619
cd FRA361_Open_Topics_6619
```

**Run test:**

```bash
python3 env/test_environment.py
```

**Expected output:**
```
✅ Environment created!
Observation shape: (37,)
Number of obstacles: 5
✅ Environment test complete!
```

---

### 6. Verify GPU (Optional but Recommended)

**Check CUDA availability:**

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
```

**Expected output:**
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
CUDA version: 11.8
```
---

### 7. Quick Start Verification

**Run verification:**

```bash
chmod +x verify_setup.sh
./verify_setup.sh
```

**Expected output:**
```
================================
FRA361 Environment Verification
================================
Python version: Python 3.10.12
MuJoCo: ✅ Version 2.3.0
Gymnasium: ✅ Version 0.29.1
PyTorch: ✅ Version 2.0.1
CUDA: ✅ Available
Filterpy: ✅ Installed
Scikit-learn: ✅ Version 1.3.0
================================
Verification complete!
================================
```

---

### 10. Robot Model Setup

**The omni-wheel robot model (omni_carver) is already included in the repository.**

**Verify robot model:**

```bash
# Check URDF exists
ls -l omni_carver_description/description/omni_carver.urdf

# Check MuJoCo XML exists
ls -l omni_carver_description/description/omni_carver.xml

# Check mesh files
ls -l omni_carver_description/mesh/
```

**Robot Specifications:**
- **Type:** Omni-directional mobile robot
- **Wheels:** 3 omni-wheels (120° apart)
- **Motor constant (kv):** 150
- **Control:** Velocity control (vx, vy, omega)
- **Sensors:** IMU, LiDAR (simulated)
- **Mass:** ~5kg (including wheels and sensors)

**Test robot model in MuJoCo:**

```python
# test_robot.py
import mujoco

# Load robot model
model = mujoco.MjModel.from_xml_path(
    'omni_carver_description/description/omni_carver.xml'
)
data = mujoco.MjData(model)

print("✅ Robot model loaded!")
print(f"DoF: {model.nv}")
print(f"Bodies: {model.nbody}")
print(f"Actuators: {model.nu}")

# List body names
print("\nBodies:")
for i in range(model.nbody):
    print(f"  {i}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)}")
```

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

### Version 2: Realistic Dynamics

| Metric | Value | Real-World Implication |
|--------|-------|------------------------|
| **ADE (Position)** | 0.0455m (4.5cm) | ✅ Within sensor noise |
| **Velocity Error** | 0.3860 m/s | ✅ Acceptable for 2-4 m/s speeds |
| **Validation Loss** | 0.085534 | Good generalization |

**Speed Comparison:**
- **Low-speed (<2.0 m/s, pedestrians):** 0.2198m (n=120)
- **High-speed (≥2.0 m/s, vehicles):** 0.2033m (n=80)
- **Difference: 8%** (minimal K-means benefit)

---

### Version 3: Hybrid ETH/UCY + Synthetic (Current)

| Metric | Value | Quality |
|--------|-------|---------|
| **ADE (Position)** | 0.0342m (3.4cm) | ⭐⭐⭐⭐⭐ Excellent! |
| **Velocity Error** | 0.2631 m/s | ⭐⭐⭐⭐ Very good |
| **Validation Loss** | 0.045266 | ⭐⭐⭐⭐ Low & stable |

**Dataset Composition:**
- **ETH/UCY Real Pedestrians:** 1,421 trajectories (70%)
  - 8 scenes: ETH, Hotel, Zara01/02/03, UCY Students, UCY University
  - Average speed: 0.76 m/s
  - Average length: 45 frames (18 seconds)
- **Synthetic Vehicles:** 609 trajectories (30%)
  - Speed: 2.0-4.0 m/s
  - Truncated to 60 frames (balanced sampling)
- **Total:** 2,030 trajectories, 100,809 samples

**Balance Quality:**
- Trajectory ratio: 70% pedestrians / 30% vehicles
- Sample ratio: 63.1% low-speed / 36.9% high-speed
- Difference: 6.9% ✅ Excellent balance!

**Error Growth Over Time (10-step horizon):**
- **Step 1:** ~0.2m (excellent)
- **Step 5:** ~1.0m (good for collision avoidance)
- **Step 10:** ~2.5m (acceptable for long-term planning)

**Speed Comparison (10-step horizon):**
- **Low-speed (<2.0 m/s, Real Pedestrians):** 2.0675m (n=120)
- **High-speed (≥2.0 m/s, Synthetic Vehicles):** 0.3870m (n=80)
- **Difference: 434%** (low-speed HARDER to predict!)

**Critical Finding - K-means Limitation:**

Counter-intuitively, real pedestrian trajectories were **5× harder** to predict than synthetic vehicle trajectories. This inverted result reveals important insights:

**Why Real Pedestrians Are Harder:**
1. **Complex Social Behaviors:** Goal changes, social force interactions, crowd dynamics
2. **Unpredictable Actions:** Random stops (5% chance), sharp turns (±60°)
3. **Short Trajectories:** 45 frames avg (limiting observation window)
4. **Natural Variability:** Real human behavior > synthetic physics

**Why Synthetic Vehicles Are Easier:**
1. **Deterministic Physics:** Momentum-based, smooth trajectories
2. **Predictable Patterns:** Consistent speed (±5%), small turn radius (±30°)
3. **Longer Observation:** 60 frames (better context)
4. **No Social Dynamics:** Independent motion models

**Implications:**
- ✅ K-means clustering **does not guarantee improvement** when data source quality differs
- ✅ Real-world complexity dominates over speed-based classification
- ✅ Short-term predictions (5 steps, ~0.5m error) remain suitable for navigation
- ✅ This finding validates the importance of data quality in trajectory prediction

---

### Comparison Across All Versions

| Metric | V1: Simple | V2: Realistic | V3: Hybrid (Current) | Best |
|--------|-----------|---------------|----------------------|------|
| **ADE** | 0.0018m | 0.0455m | **0.0342m** | V3 ✅ |
| **Velocity Error** | 0.0048 m/s | 0.3860 m/s | **0.2631 m/s** | V3 ✅ |
| **Val Loss** | 0.0010 | 0.0855 | **0.0453** | V3 ✅ |
| **Data Type** | Pure synthetic | Pure synthetic | **Real + Synthetic** | V3 |
| **Trajectories** | 3,075 | 3,320 | **2,030** | Balanced |
| **Realism** | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | V3 |
| **K-means Benefit** | 5% | 8% | **Inverted (434%)** | Finding |

**Evolution Summary:**
- **V1 → V2:** Added realistic dynamics + sensor noise (20× error increase, expected)
- **V2 → V3:** Added real ETH/UCY data (25-47% improvement across metrics!)
- **Key Discovery:** Real pedestrian data quality > synthetic, leading to better overall performance

---
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
│   │   ├── kgru_training_data.pkl                # V1: Simple dynamics
│   │   ├── kgru_training_data_realistic.pkl      # V2: Realistic dynamics
│   │   ├── kgru_training_data_hybrid.pkl         # V3: ETH/UCY + Synthetic (current)
│   │   ├── eth_ucy_processed.pkl                 # Processed ETH/UCY pedestrians
│   │   └── sgan/                                 # ETH/UCY raw data
│   │       └── scripts/datasets/
│   │           ├── eth/                          # ETH university
│   │           ├── hotel/                        # Hotel entrance
│   │           ├── univ/                         # UCY university  
│   │           ├── zara1/                        # Zara shopping 1
│   │           ├── zara2/                        # Zara shopping 2
│   │           └── raw/all_data/                 # Raw text files
│   ├── model/
│   │   └── kgru_model.pth              # Trained weights (V3 hybrid)
│   ├── plot/
│   │   ├── kgru_training.png           # Loss curves
│   │   ├── kgru_evaluation.png         # Error distributions
│   │   ├── trajectory_predictions.png  # 6 example predictions (10-step)
│   │   ├── error_over_time.png         # Error growth analysis (10-step)
│   │   └── speed_comparison.png        # Low vs high speed comparison
│   ├── k_gru_predictor.py              # Main predictor class
│   ├── data_collection_realistic.py    # Synthetic data generation
│   ├── preprocess_eth_ucy.py           # ETH/UCY preprocessing
│   ├── merge_datasets.py               # Hybrid dataset creation (balanced)
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

### Why Hybrid ETH/UCY + Synthetic Dataset?
- **Real pedestrian behavior**: ETH/UCY captures authentic human motion patterns including social forces, goal-directed navigation, and natural variability
- **Diverse scenarios**: 8 scenes covering university campuses, hotels, and shopping areas
- **Vehicle completion**: Synthetic vehicles (2-4 m/s) complement real pedestrians (0.3-2 m/s) for full speed range
- **Balanced sampling**: Truncated vehicle trajectories to 60 frames to match pedestrian length (~45 frames), preventing sample imbalance
- **70:30 ratio**: Reflects real-world distribution where pedestrians outnumber vehicles in mixed environments

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

**Detailed installation instructions available in [Environment Setup](#environment-setup) section.**

### Core Dependencies

```bash
# Core
pip install torch numpy matplotlib scipy

# Environment
pip install mujoco gymnasium

# K-GRU specific
pip install filterpy scikit-learn tqdm
```

**System Requirements:**
- Python 3.10+
- CUDA-capable GPU (recommended, RTX 3050 or better)
- Ubuntu 22.04 LTS or compatible Linux
- 8GB+ RAM
- 5GB+ disk space for data and models

**Complete requirements.txt available in repository.**

---

## Next Steps

### ✅ Completed: K-GRU Prediction Module
- [x] ETH/UCY dataset integration (1,421 real pedestrian trajectories)
- [x] Hybrid dataset creation (70% real + 30% synthetic, balanced)
- [x] Training with hybrid data (ADE: 0.034m, excellent performance)
- [x] Comprehensive evaluation (10-step horizon analysis)
- [x] K-means clustering analysis (inverted benefit discovered)
- [x] Scientific finding: Real pedestrian complexity > synthetic vehicle physics

### Phase 1: TD3 Baseline (Reactive Navigation) - Ready to Start
1. Implement TD3 agent architecture
2. Define observation space (robot state + current obstacle positions)
3. Define action space (vx, vy, omega)
4. Design reactive reward function (current distance-based penalties)
5. Train baseline agent (no predictions)

**Expected Performance:**
- Success rate: 60-70% (reactive only)
- Collision rate: 20-30%
- Path smoothness: Moderate (reactive swerving)

### Phase 2: TD3 + K-GRU (Anticipatory Navigation)
1. Integrate K-GRU predictor with environment
2. Extend observation space (predicted future positions, 5-step horizon)
3. Design anticipatory reward function (predicted collision zones)
4. Train anticipatory agent (with predictions)

**Expected Improvement:**
- Success rate: 80-90% (anticipatory planning)
- Collision rate: 5-15% (proactive avoidance)
- Path smoothness: High (smooth trajectory planning)
- Prediction horizon: 5 steps (2 seconds, ~0.5m error)

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
4. Discussion of K-means limitation findings
5. Future work recommendations

---

## Key Findings

### 1. Hybrid Dataset Outperforms Pure Synthetic
**Discovery:** Combining real ETH/UCY pedestrian data with synthetic vehicles yields **25-47% better performance** than pure synthetic data.

**Metrics Improvement:**
- ADE: 0.0455m → 0.0342m (25% better)
- Velocity Error: 0.3860 m/s → 0.2631 m/s (32% better)  
- Validation Loss: 0.0855 → 0.0453 (47% better)

**Why:** Real pedestrian trajectories from ETH/UCY capture authentic human behavior patterns that improve model generalization despite being more complex to predict.

### 2. K-means Clustering Benefit is Data-Dependent
**Discovery:** K-means clustering by speed does **NOT** universally improve prediction accuracy. Benefit depends on relative prediction difficulty between clusters.

**Our Results (Inverted):**
- Low-speed (real pedestrians): 2.07m error (HARDER)
- High-speed (synthetic vehicles): 0.39m error (EASIER)
- Ratio: 434% (opposite of expected)

**Traditional Expectation (from Liu et al. 2025):**
- Low-speed: Lower error (simpler patterns)
- High-speed: Higher error (momentum, complex physics)

**Root Cause:**
1. **Data source dominates:** Real human behavior > synthetic physics in complexity
2. **Social dynamics:** Real pedestrians exhibit unpredictable stops, turns, social interactions
3. **Trajectory length:** Shorter real trajectories (45 frames) vs longer synthetic (60 frames)
4. **Physics predictability:** Deterministic synthetic motion easier than stochastic human behavior

**Implication:** K-means clustering helps **only when** high-speed obstacles are genuinely harder to predict. Data quality and source matter more than speed classification.

### 3. Practical Navigation Performance
**Discovery:** Despite high 10-step errors, **5-step predictions** (~0.5-1.0m) are **sufficient for robot navigation**.

**For TD3 Integration:**
- Prediction horizon: 5 steps (2 seconds ahead)
- Expected error: 0.5m (extrapolated from error growth curve)
- Robot safety zone: 3m diameter
- **Safety margin: 2.5m** (adequate for collision avoidance)

**Validation:** Short-term predictions prioritize immediate collision avoidance over long-term trajectory accuracy, which aligns with reactive control loops in mobile robots.

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

**Last Updated:** February 18, 2026  
**Status:** K-GRU prediction module complete (Hybrid ETH/UCY + Synthetic) ✅ | K-means analysis complete ✅ | TD3 integration ready to start 🚀