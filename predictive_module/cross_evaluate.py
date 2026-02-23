"""
Cross-evaluate: 2 datasets × 2 models = 4 combinations
"""
import numpy as np
import torch
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predictive_module.k_gru_predictor import TrajectoryGRU

# Temporal resolution each model was trained at
MODEL_DTS = {
    'Synthetic': 0.4,   # 2.5 Hz (downsampled from 10 Hz to match ETH/UCY)
    'ETH/UCY':   0.4,   # 2.5 Hz
}


def resample_trajectory(traj, source_dt, target_dt):
    """
    Resample a trajectory from source_dt to target_dt using linear interpolation.
    Fixes dt mismatch: ETH/UCY is 2.5 Hz (dt=0.4s), synthetic is 10 Hz (dt=0.1s).
    Velocities are recomputed from interpolated positions at the new dt.
    """
    if abs(source_dt - target_dt) < 1e-6:
        return traj

    source_times = np.arange(len(traj)) * source_dt
    target_times = np.arange(0, source_times[-1], target_dt)

    resampled = np.zeros((len(target_times), traj.shape[1]))

    # Interpolate x and y positions
    resampled[:, 0] = np.interp(target_times, source_times, traj[:, 0])
    resampled[:, 1] = np.interp(target_times, source_times, traj[:, 1])

    # Recompute vx, vy from interpolated positions at new dt
    for i in range(len(resampled) - 1):
        resampled[i, 2] = (resampled[i+1, 0] - resampled[i, 0]) / target_dt
        resampled[i, 3] = (resampled[i+1, 1] - resampled[i, 1]) / target_dt
    resampled[-1, 2:4] = resampled[-2, 2:4]  # copy last velocity

    return resampled


def normalize_sequence(seq):
    """
    Normalize positions relative to the last observed frame.
    Fixes position scale mismatch: MuJoCo arena (~0-10m) vs ETH/UCY world (~0-50m+).
    The model sees displacement-from-last-observation instead of absolute coordinates.
    Returns (normalized_seq, origin) — add origin back to convert predictions to world coords.
    """
    origin = seq[-1, :2].copy()  # last observed position
    normalized = seq.copy()
    normalized[:, 0] -= origin[0]
    normalized[:, 1] -= origin[1]
    return normalized, origin


def evaluate(model_path, data_path, model_name, data_name, source_dt, model_dt, n_samples=100):
    """
    Evaluate model on dataset.
    
    Normalization is ONLY applied for cross-domain evaluation!
    Domain-matched evaluation uses data as-is (models expect their training distribution).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = TrajectoryGRU(4, 128, 3, 4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    trajectories = data['trajectories']
    n_train = int(0.7 * len(trajectories))
    n_val = int(0.15 * len(trajectories))
    test_traj = trajectories[n_train+n_val:]

    # Determine if cross-domain
    is_cross_domain = (model_name != data_name)
    
    print(f"   {'Cross-domain' if is_cross_domain else 'Domain-matched'}: ", end="")
    if is_cross_domain:
        print(f"Resample {source_dt}s→{model_dt}s + Normalize positions")
    else:
        print(f"Resample {source_dt}s→{model_dt}s (no normalization)")

    errors = []
    n_processed = 0

    for traj in test_traj:
        if n_processed >= n_samples:
            break

        # Resample to model's training dt (applied to all cases)
        traj = resample_trajectory(traj, source_dt=source_dt, target_dt=model_dt)

        if len(traj) < 20:
            continue

        max_start = len(traj) - 20
        if max_start <= 0:
            continue

        start_idx = np.random.randint(0, max_start)
        input_seq = traj[start_idx:start_idx+10]
        ground_truth = traj[start_idx+10:start_idx+20]

        # Normalize positions ONLY for cross-domain ← KEY FIX
        if is_cross_domain:
            input_seq, origin = normalize_sequence(input_seq)
            ground_truth = ground_truth.copy()
            ground_truth[:, 0] -= origin[0]
            ground_truth[:, 1] -= origin[1]

        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
            predictions = model.predict_sequence(input_tensor, 10)
            predictions = predictions.cpu().numpy()[0]

        error = np.linalg.norm(predictions[:, :2] - ground_truth[:, :2], axis=1).mean()
        errors.append(error)
        n_processed += 1

    return np.mean(errors), np.std(errors), len(errors)


if __name__ == "__main__":

    models = [
        ('predictive_module/model/kgru_synthetic.pth', 'Synthetic'),
        ('predictive_module/model/kgru_eth_ucy.pth', 'ETH/UCY'),
    ]

    # source_dt: temporal resolution of each dataset
    #   Synthetic:  10 Hz → dt = 0.1s
    #   ETH/UCY:   2.5 Hz → dt = 0.4s
    datasets = [
        ('predictive_module/data/synthetic_mixed_traffic.pkl',  'Synthetic', 0.1),
        ('predictive_module/data/eth_ucy_real_pedestrians.pkl', 'ETH/UCY',   0.4),
    ]

    print("\n" + "="*70)
    print("CROSS-EVALUATION: 2×2 Matrix (with dt-resampling + position normalization)")
    print("="*70)
    print(f"  Fix 1: Data resampled to each model's training dt (Synthetic=0.1s, ETH/UCY=0.4s)")
    print(f"  Fix 2: Positions normalized relative to last observed frame")
    print()

    results = {}

    for model_path, model_name in models:
        print(f"\n📦 Model: {model_name}")
        print("-"*70)

        for data_path, data_name, source_dt in datasets:
            key = f"{model_name} → {data_name}"
            print(f"   Testing on {data_name} (source_dt={source_dt}s)...", end=" ", flush=True)

            try:
                ade, std, n = evaluate(
                    model_path, data_path,
                    model_name=model_name,    # ← ADD THIS
                    data_name=data_name,      # ← ADD THIS
                    source_dt=source_dt,
                    model_dt=MODEL_DTS[model_name],
                    n_samples=100
                )
                results[key] = (ade, std, n)
                print(f"✅ {ade:.4f}m ± {std:.4f}m (n={n})")
            except Exception as e:
                print(f"❌ Error: {e}")
                results[key] = (None, None, 0)
    
    # Summary Table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Training → Testing':<30} {'ADE (m)':<20} {'Samples'}")
    print("-"*70)
    
    for key in ['Synthetic → Synthetic', 'Synthetic → ETH/UCY', 
                'ETH/UCY → ETH/UCY', 'ETH/UCY → Synthetic']:
        if key in results:
            ade, std, n = results[key]
            if ade is not None:
                print(f"{key:<30} {ade:.4f} ± {std:.4f}       {n}")
            else:
                print(f"{key:<30} FAILED")
    
    print("="*70)
    
    # Analysis
    print("\n📊 KEY INSIGHTS:")
    print("-"*70)
    
    syn_syn = results.get('Synthetic → Synthetic', (None, None, None))[0]
    syn_eth = results.get('Synthetic → ETH/UCY', (None, None, None))[0]
    eth_eth = results.get('ETH/UCY → ETH/UCY', (None, None, None))[0]
    eth_syn = results.get('ETH/UCY → Synthetic', (None, None, None))[0]
    
    if syn_syn and eth_eth:
        print(f"\n1. Domain-Matched Performance:")
        print(f"   Synthetic → Synthetic: {syn_syn:.4f}m")
        print(f"   ETH/UCY → ETH/UCY:     {eth_eth:.4f}m")
        
        if syn_syn < eth_eth:
            ratio = eth_eth / syn_syn
            print(f"   → Synthetic is {ratio:.1f}× easier (cleaner data)")
        
    if syn_eth and eth_eth:
        print(f"\n2. Synthetic → Real Transfer Gap:")
        gap = syn_eth - eth_eth
        pct = (gap / eth_eth) * 100
        print(f"   ETH/UCY → ETH/UCY: {eth_eth:.4f}m (baseline)")
        print(f"   Synthetic → ETH/UCY: {syn_eth:.4f}m")
        print(f"   → Transfer gap: +{gap:.4f}m ({pct:.1f}% worse)")
    
    if eth_syn and syn_syn:
        print(f"\n3. Real → Synthetic Transfer:")
        gap = eth_syn - syn_syn
        pct = (gap / syn_syn) * 100
        print(f"   Synthetic → Synthetic: {syn_syn:.4f}m (baseline)")
        print(f"   ETH/UCY → Synthetic: {eth_syn:.4f}m")
        print(f"   → Transfer gap: +{gap:.4f}m ({pct:.1f}% worse)")
    
    print("="*70)