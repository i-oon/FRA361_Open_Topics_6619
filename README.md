# FRA361 Open Topics: Anticipatory Navigation with K-GRU Prediction

## Project Overview

This project implements **K-GRU (K-means + GRU) trajectory prediction** for dynamic obstacle navigation in mobile robotics. The system predicts future obstacle positions to enable anticipatory collision avoidance in mixed-speed environments containing both pedestrians and vehicles.

**Key Innovation:** Speed-differentiated trajectory prediction using K-means clustering, Kalman filtering, and GRU networks — validated through systematic data quality improvement and real-world data testing.

---

## Table of Contents

1. [Current Status](#current-status)
2. [Environment Setup](#environment-setup)
3. [Methodology](#methodology)
4. [Results & Validation](#results--validation)
5. [Key Findings](#key-findings)
6. [Implementation Journey](#implementation-journey)
7. [File Structure](#file-structure)
8. [Usage Guide](#usage-guide)
9. [Next Steps](#next-steps)
10. [Dependencies](#dependencies)
11. [Related Work](#related-work)

---

## Current Status

### K-GRU Prediction Module: Complete & Validated

**Primary Model: Trained and evaluated on ETH/UCY real pedestrian data (1,421 trajectories, 8 real-world scenes).**

**Model Performance (ETH/UCY Real Data, Corrected Inference):**
```
Dataset:          ETH/UCY real pedestrian trajectories
Training ADE:     0.034m  (teacher forcing)
Autoregressive:   ~0.18m – 1.34m ADE per sample (10-step horizon)
Typical range:    0.35m – 0.83m ADE on straight-walking segments
```

> **Note:** An earlier inference bug (GRU hidden state carried across sliding-window steps,
> causing temporal inversion) produced inflated ADE of ~3.17m. This has been fixed in
> `predict_sequence()` — hidden state is now correctly reset each step.

**K-means Clustering (Natural Discovery on Real Data):**
```
Discovered boundary: 0.97 m/s  (data-driven from ETH/UCY, not manually set)
Low-speed cluster:   52%  — slow walkers (~0.76 m/s)
High-speed cluster:  48%  — fast walkers (~1.35 m/s)
Balance:             Near-perfect 52% / 48%
Elbow method:        Confirms K=2 is optimal
```

**Model Stability:**
```
Architecture:    3-layer GRU, 128 hidden units
Training:        Converges reliably with early stopping
Generalization:  No overfitting, stable validation
Reproducibility: Consistent results across experiments
```

**Known Limitation:**
```
The model predicts straight-line continuations and does not anticipate turns.
This is a fundamental property of deterministic MSE-trained models — the model
learns the mean of all possible futures, which collapses multimodal turn
distributions into a straight-line prediction.
Acceptable for short-horizon collision avoidance; not suitable for long-horizon
planning in high-curvature environments.
```

---

## Environment Setup

<p align="center">
    <img width="50%" src="visual/mujoco_omni_carver.gif">
    </br>
</p>

### Prerequisites

- Ubuntu 22.04 LTS (or compatible Linux)
- Python 3.10+
- CUDA-capable GPU (recommended: RTX 3050 or better)
- 8 GB+ RAM, 5 GB+ disk space

### Quick Installation
```bash
git clone <your-repo-url> FRA361_Open_Topics_6619
cd FRA361_Open_Topics_6619

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python3 env/test_environment.py
```

---

## Methodology

### K-GRU Prediction Pipeline

Based on Liu et al. (2025) *"Adaptive Motion Planning Leveraging Speed-Differentiated Prediction for Mobile Robots in Dynamic Environments"*.

**Three-Stage Architecture:**

1. **K-means Clustering** — Discovers natural speed boundaries from data (not manual thresholds)
2. **Kalman Filter** — Brownian motion model for state estimation and noise filtering
3. **GRU Network** — Learns motion patterns and predicts future trajectories

**Network Architecture:**
- Input: 10-timestep sequence `[x, y, vx, vy]`
- GRU: 3 layers, 128 hidden units, dropout 0.2
- Output: Future positions 10 timesteps ahead (1 second at 10 Hz)

**Key Insight:**
Unlike manual speed thresholds (e.g., a hard-coded 2.0 m/s cutoff), K-means discovers the natural separation boundary from data. Our implementation found 2.19 m/s, validating the bimodal pedestrian–vehicle assumption while allowing data-driven adaptation to different environments.

---

## Results & Validation

### Primary Results: ETH/UCY Real Pedestrian Data

The final model is trained and evaluated on ETH/UCY real-world pedestrian trajectories (1,421 total, 8 scenes: ETH, Hotel, Zara01/02/03, Students01/03, University).

**Training:**
```
Dataset:          ETH/UCY real pedestrian trajectories
Training ADE:     0.034m  (teacher forcing, single-step)
Validation Loss:  Stable, early stopping
```

**Autoregressive Evaluation (10-step horizon, corrected inference):**
```
Straight-walking segments:  ~0.18m – 0.63m ADE  ★★★★  Good
Turning/complex segments:   ~0.81m – 1.34m ADE  ★★    Limited (straight-line bias)

Note: Model reliably predicts straight-line continuations.
      Turn prediction is not supported (MSE averaging effect).
```

**K-means Natural Discovery on ETH/UCY:**
```
Discovered boundary: 0.97 m/s  (data-driven — no manual threshold used)
Low-speed cluster:   52%  (737 trajectories, slow walkers ~0.76 m/s)
High-speed cluster:  48%  (684 trajectories, fast walkers ~1.35 m/s)
Balance:             Near-perfect — elbow method confirms K=2 is optimal
Benefit:             3.7% ADE difference (both clusters are pedestrians,
                     behavioral complexity dominates over speed)
```

**Error Growth Over Time (ETH/UCY, corrected):**
```
Step 1:   ~0.05m  (excellent short-term)
Step 5:   ~0.35m  (good for navigation)
Step 10:  ~0.65m  (acceptable for planning on straight paths)

Accumulation is approximately linear for straight-walking cases.
Turns cause error to grow faster after the direction change.
```

---

### Inference Bug Fixed

A bug in `predict_sequence()` was discovered and fixed: the GRU hidden state was incorrectly carried across sliding-window steps, causing temporal inversion (the model processed `t_9` context then received `t_1` next, going backwards in time). This inflated the previously reported ETH/UCY ADE to ~3.17m. After the fix, hidden state is reset to `None` each step so the full window is processed fresh.

---

### Comparison: Real vs. Synthetic Data

| Scenario | Data | Boundary (K-means) | Balance | ADE (straight) | Notes |
|---|---|---|---|---|---|
| ETH/UCY real pedestrians | Real | **0.97 m/s** (natural) | 52% / 48% | ~0.35–0.65m | Primary model |
| Synthetic mixed traffic | Synthetic | 2.39 m/s | 76% / 24% | 0.27m / 0.96m | Validation only |
| Hybrid real+synthetic | Mixed | 2.0 m/s (manual) | 60% / 40% | Inverted pattern | Historical |

**Key takeaway:** K-means discovers fundamentally different boundaries depending on the data: 0.97 m/s separates slow vs. fast walkers in pedestrian-only data; ~2.2–2.4 m/s separates pedestrians from vehicles in mixed traffic. Both are valid — the boundary adapts to the actual data distribution.

---

## Key Findings

### 1. K-means Discovers Natural Boundaries — No Manual Threshold Needed

K-means successfully discovers speed clusters from **real** data without any fixed threshold. The 52%/48% balance in ETH/UCY and the data-driven 0.97 m/s boundary confirm that Liu et al.'s approach is scientifically sound. The elbow method confirms K=2 is optimal. The discovered boundary adapts to the environment: 0.97 m/s for pedestrian-only data, ~2.2 m/s for mixed-traffic data.

### 2. Speed Differentiation Effectiveness is Context-Dependent

K-means works best when speed indicates **fundamentally different motion types** (e.g., pedestrians vs. vehicles). For homogeneous motion (pedestrians only), behavioral complexity dominates and clustering provides minimal prediction benefit, even though the clusters themselves are valid.

| Environment | Motion Types | Discovered Boundary | K-means Benefit |
|---|---|---|---|
| Urban intersections, mixed traffic | Social + physics-based | ~2.2 m/s | Large (71.6%) ✅ |
| Pedestrian-only zones, indoor spaces | Social behavior only | ~0.97 m/s | Limited (3.7%) ⚠️ |

### 3. Model Architecture is Stable and Deployment-Ready

The 3-layer GRU with 128 hidden units converges reliably across all dataset versions. Trained on ETH/UCY real pedestrian data, the model achieves ~0.35–0.65m ADE on straight-walking segments (corrected inference). Performance differences across datasets reflect motion complexity, not model limitations.

### 4. Deterministic MSE Models Cannot Predict Turns

The model reliably predicts straight-line continuations but collapses turn distributions to a straight average. This is not a data or architecture bug — it is inherent to deterministic MSE training: when a pedestrian could turn left or right with equal probability, the MSE-optimal prediction is to go straight. Turning prediction requires probabilistic models (CVAE, diffusion) or goal-conditioned approaches.

### 5. Data Quality Drives Performance

| Dataset Version | Type | ADE | Note |
|---|---|---|---|
| V1: Simple synthetic | Constant velocity | 0.0018m | Unrealistic |
| V2: Stochastic motion | Random changes | 1.00m | Unpredictable by design |
| V3: Imbalanced (98%/2%) | Goal-directed | 0.55m | K-means meaningless |
| V4: Balanced (76%/24%) | Goal-directed | 0.27m | Validates method on synthetic |
| **ETH/UCY real (final)** | **Human behavior** | **~0.35–0.65m** | **Primary model, corrected inference** |

Same model architecture across all versions — **data quality drives performance, not model capacity**.

---

## Implementation Journey

### Data Version Evolution

**V1 — Simple Synthetic (Baseline)**
- Constant velocity, no noise, perfect sensing
- Training ADE: 0.0018m — unrealistically good
- Lesson: Perfect data ≠ useful model

**V2 — Stochastic Motion**
- Added random direction changes (0.5%/step), random stops (5%), sensor noise
- Real ADE: 1.0m — unpredictable by design
- Lesson: Stochastic ≠ realistic; neural networks learn patterns, not randomness

**V3 — Predictable Physics (Improved)**
- Removed random changes, longer episodes, larger model
- Real ADE: 0.55m — better, but speed ranges (0.1–0.3 vs 0.5–1.0 m/s) both fall in pedestrian zone
- K-means split: 98% / 2% — clustering is meaningless
- Lesson: Predictability matters, but so does data balance

**V4 — Balanced Speeds (Current)**
- Fixed speed ranges: low = (0.5, 1.5) m/s, high = (2.5, 4.5) m/s
- Low-speed ADE: 0.27m, High-speed ADE: 0.96m
- K-means split: 76% / 24% — meaningful clustering
- Lesson: A 2-line code change fixed data distribution and enabled full validation

**ETH/UCY Hybrid (Attempted)**
- Real pedestrians (1,421) + synthetic vehicles (609)
- Training ADE: 0.034m (best), but inverted pattern (low-speed harder)
- Lesson: Real behavioral complexity dominates synthetic physics

### Critical Lessons

| Lesson | Evidence |
|---|---|
| Motion predictability is essential | Random: 1.0m error; predictable physics: 0.27m |
| Data balance enables validation | 98%/2% → K-means useless; 76%/24% → K-means works |
| Model capacity helps but has limits | 128/3-layer GRU better than 50/2, but cannot fix poor data |
| Speed vs. behavioral complexity | Synthetic: high-speed harder; Real humans: both equally hard |

---

## File Structure
```
FRA361_Open_Topics_6619/
├── env/
│   ├── dynamic_nav_env.py              # MuJoCo environment (fixed speed ranges)
│   └── test_environment.py
├── omni_carver_description/
│   ├── description/
│   │   ├── omni_carver.xml             # Robot model
│   │   └── omni_carver.urdf
│   └── mesh/                           # STL files
├── predictive_module/
│   ├── data/
│   │   ├── kgru_training_data_realistic.pkl     # Current (balanced)
│   │   ├── kgru_training_data_hybrid.pkl        # ETH/UCY + Synthetic
│   │   └── eth_ucy_processed.pkl                # Real pedestrian data
│   ├── model/
│   │   └── kgru_model_eth_ucy.pth      # Trained weights (ETH/UCY real data)
│   ├── plot/
│   │   ├── trajectory_predictions_improved.png
│   │   ├── error_over_time.png
│   │   ├── speed_comparison.png
│   │   ├── kmeans_analysis.png
│   │   ├── kgru_training.png
│   │   └── kgru_evaluation.png
│   ├── k_gru_predictor.py              # Main predictor class
│   ├── data_collection_realistic.py    # Synthetic data generation
│   ├── train_kgru.py                   # Training with K-means analysis
│   ├── visualize_predictions.py        # Evaluation and visualization
│   └── analyze_kmeans.py              # K-means clustering analysis
├── README.md
└── requirements.txt
```

---

## Usage Guide

### 1. Data Collection
```bash
# Collect balanced training data (~3 hours)
python3 predictive_module/data_collection_realistic.py
# Output: ~3,320 trajectories (low-speed: ~2,500, high-speed: ~800)
```

### 2. K-means Analysis
```bash
# Analyze clustering BEFORE training to verify data balance
python3 predictive_module/analyze_kmeans.py
# Expected: boundary ~2.2 m/s, balance ratio 30-40%
```

### 3. Training
```bash
# Train K-GRU model (~15-20 hours on RTX 3050)
python3 predictive_module/train_kgru.py
# Outputs: model weights, training curves, evaluation metrics
```

### 4. Visualization
```bash
# Generate trajectory prediction visualizations
python3 predictive_module/visualize_predictions.py
# Outputs: 6 examples (low/high speed), error growth, K-means comparison
```

### 5. Using the Predictor
```python
from predictive_module.k_gru_predictor import KGRUPredictor

predictor = KGRUPredictor(history_length=10, dt=0.1, device='cuda')
predictor.load_model('predictive_module/model/kgru_model_eth_ucy.pth')

obstacle_states = [
    {'id': 0, 'pos': [1.0, 2.0], 'vel': [0.5, 0.3]},
    {'id': 1, 'pos': [3.0, 1.0], 'vel': [3.5, 0.8]},
]

predictions, boundary = predictor.predict(obstacle_states, prediction_horizon=10)
print(f"K-means discovered boundary: {boundary:.2f} m/s")
# predictions: {id: array(10, 4)}  — [x, y, vx, vy] for each future step
```

---

## Next Steps

### Completed

- [x] Model trained on ETH/UCY real pedestrian data (1,421 trajectories)
- [x] Inference bug fixed (hidden state reset in autoregressive rollout)
- [x] K-means discovers natural boundary from real data: 0.97 m/s (no manual threshold)
- [x] Natural cluster balance confirmed: 52% / 48% (ETH/UCY)
- [x] Speed differentiation validated for mixed traffic (71.6% benefit, synthetic)
- [x] Context-dependency identified: limited benefit for pedestrian-only environments
- [x] Liu et al. (2025) methodology confirmed on real data
- [x] Model architecture stable across all datasets

### Next Phase: Navigation Integration

The prediction module is ready for:
- TD3 reinforcement learning integration
- Reactive vs. anticipatory navigation comparison
- Mixed-traffic environment deployment

**For thesis completion:**
1. Integrate K-GRU with TD3 navigation
2. Compare reactive vs. anticipatory approaches
3. Demonstrate prediction benefit in navigation tasks

**Optional future work:**
1. Test on inD/INTERACTION real intersection datasets
2. Explore social force models for pedestrian-only scenarios
3. Investigate K=3 clustering (slow/medium/fast)

---

## Dependencies
```
# Core
torch  numpy  matplotlib  scipy  pandas

# Environment
mujoco  gymnasium

# K-GRU specific
filterpy  scikit-learn  tqdm

# Analysis
seaborn
```

See `requirements.txt` for pinned versions.

---

## Related Work

**Main Reference:**
Liu, Y., et al. (2025). Adaptive Motion Planning Leveraging Speed-Differentiated Prediction for Mobile Robots in Dynamic Environments. *Applied Sciences*, 15(13), 7551.

**Dataset Sources:**
- ETH/UCY: Pedestrian trajectories (Pellegrini et al., 2009)
- inD Dataset: Real intersection data (Bock et al., 2020) — Recommended for future work
- INTERACTION: Multi-agent scenarios (Zhan et al., 2019) — Alternative

---

## Authors

**Student:** Disthorn Suttawet
**Course:** FRA361 Open Topics
**Institution:** Institute of Field Robotics
**Semester:** 2024–2025

---

## Acknowledgments

- Liu et al. (2025) for the K-GRU methodology
- VISTEC FRA361 course staff
- MuJoCo physics engine team
- Anthropic Claude for development assistance and debugging support

---

**Last Updated:** February 23, 2026
**Status:** K-GRU module complete & validated ✅ | Trained on ETH/UCY real data ✅ | Inference bug fixed ✅ | Ready for TD3 integration 🚀

---

## Quick Reference

```
Model Architecture:
  GRU:    3 layers, 128 hidden units, dropout 0.2
  Input:  10 timesteps × 4 features [x, y, vx, vy]
  Output: 10-step predictions (1 second at 10 Hz)
  Weights: kgru_model_eth_ucy.pth

Training Data (Primary):
  Dataset:       ETH/UCY real pedestrian trajectories
  Trajectories:  1,421 (8 real-world scenes)
  Training ADE:  0.034m  (teacher forcing)

Inference Performance (ETH/UCY, corrected):
  Straight paths:  ~0.18m – 0.63m ADE  (good)
  Turning paths:   ~0.81m – 1.34m ADE  (limited, straight-line bias)

K-means (Natural Discovery on ETH/UCY):
  Boundary:  0.97 m/s  (data-driven, not fixed)
  Balance:   52% / 48%
  Benefit:   3.7%  (both clusters are pedestrians)

K-means (Natural Discovery on Synthetic Mixed Traffic):
  Boundary:  2.39 m/s  (data-driven)
  Balance:   76% / 24%
  Benefit:   71.6%  (diverse motion types)

Validated:
  ✅ Natural K-means boundary from real data (0.97 m/s)
  ✅ Inference bug fixed (hidden state reset per step)
  ✅ Speed differentiation effective for mixed traffic
  ✅ Model stable across all datasets
  ✅ Liu et al. methodology confirmed
```
