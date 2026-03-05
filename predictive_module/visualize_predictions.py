# visualize_predictions.py
"""
Visualize K-GRU predictions vs ground truth trajectories.
Supports 4 evaluation modes: 2 domain-matched + 2 cross-domain.
K-means is computed ONCE per mode in __main__ and passed to all functions.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictive_module.k_gru_predictor import TrajectoryGRU
from predictive_module.utils import (
    kmeans_speed_clusters,
    downsample_trajectories,
    upsample_trajectories,
    normalize_sequence,
)


def _apply_normalization(input_seq, ground_truth):
    """Shift so last observed frame is at origin. Used for cross-domain only."""
    norm_input, origin = normalize_sequence(input_seq)
    norm_gt = ground_truth.copy()
    norm_gt[:, 0] -= origin[0]
    norm_gt[:, 1] -= origin[1]
    return norm_input, norm_gt


def visualize_trajectory_predictions(
    model,
    valid_trajs,        # pre-filtered: len >= seq_len + horizon + 1
    labels,             # K-means labels aligned with valid_trajs (0=low, 1=high)
    boundary,           # K-means boundary (m/s) — for title label only
    n_samples=6,
    sequence_length=10,
    prediction_horizon=10,
    device='cuda',
    save_dir='predictive_module/plot',
    cross_domain=False,      # controls plot label only
    normalize_input=False,   # controls position normalization at inference
    mode_label='',
):
    """Plot 3 low-speed + 3 high-speed examples with ground truth vs predicted."""
    model.eval()

    low_idx  = np.where(labels == 0)[0]
    high_idx = np.where(labels == 1)[0]

    # Pick as balanced as possible; if one cluster is small, fill from the other
    half    = n_samples // 2
    n_low   = min(half, len(low_idx))
    n_high  = min(half, len(high_idx))
    n_low   = min(n_low  + max(0, half - n_high), len(low_idx))
    n_high  = min(n_high + max(0, half - n_low),  len(high_idx))
    chosen_low  = low_idx[np.linspace(0, len(low_idx)-1,  n_low,  dtype=int)] if n_low  > 0 else np.array([], dtype=int)
    chosen_high = high_idx[np.linspace(0, len(high_idx)-1, n_high, dtype=int)] if n_high > 0 else np.array([], dtype=int)
    selected_indices = np.concatenate([chosen_low, chosen_high]).astype(int)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, traj_idx in enumerate(selected_indices):
        trajectory = valid_trajs[traj_idx]

        max_start = len(trajectory) - sequence_length - prediction_horizon
        if max_start <= 0:
            continue
        start_idx = np.random.randint(0, max_start)

        input_seq    = trajectory[start_idx:start_idx+sequence_length].copy()
        ground_truth = trajectory[start_idx+sequence_length:start_idx+sequence_length+prediction_horizon].copy()

        avg_speed = np.mean(np.linalg.norm(input_seq[:, 2:4], axis=1))

        if normalize_input:
            input_seq, ground_truth = _apply_normalization(input_seq, ground_truth)

        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
            predictions = model.predict_sequence(input_tensor, prediction_horizon)
            predictions = predictions.cpu().numpy()[0]

        ax = axes[idx]

        # Observation arrows
        for i in range(len(input_seq)-1):
            ax.arrow(input_seq[i, 0], input_seq[i, 1],
                     input_seq[i+1, 0] - input_seq[i, 0],
                     input_seq[i+1, 1] - input_seq[i, 1],
                     head_width=0.02, head_length=0.03,
                     fc='gray', ec='gray', alpha=0.6, linewidth=2)

        ax.plot(ground_truth[:, 0], ground_truth[:, 1],
                'o-', color='#2ecc71', linewidth=3, markersize=6,
                label='Ground Truth', zorder=3)
        ax.plot(predictions[:, 0], predictions[:, 1],
                's--', color='#e74c3c', linewidth=3, markersize=6,
                label='Predicted', alpha=0.8, zorder=2)

        for i in range(len(predictions)):
            ax.plot([predictions[i, 0], ground_truth[i, 0]],
                    [predictions[i, 1], ground_truth[i, 1]],
                    ':', color='orange', linewidth=1, alpha=0.5, zorder=1)

        ax.plot(input_seq[0, 0], input_seq[0, 1],
                'D', color='blue', markersize=12, label='Start', zorder=4)
        ax.plot(ground_truth[-1, 0], ground_truth[-1, 1],
                '*', color='gold', markersize=18,
                markeredgecolor='black', markeredgewidth=1.5,
                label='True End', zorder=5)
        ax.plot(predictions[-1, 0], predictions[-1, 1],
                'X', color='red', markersize=12,
                markeredgecolor='darkred', markeredgewidth=1.5,
                label='Pred End', zorder=5)

        position_errors = np.linalg.norm(predictions[:, :2] - ground_truth[:, :2], axis=1)
        ade = position_errors.mean()
        fde = position_errors[-1]

        category = "HIGH-SPEED" if labels[traj_idx] == 1 else "LOW-SPEED"
        color = '#3498db' if labels[traj_idx] == 1 else '#9b59b6'

        ax.set_xlabel('X Position (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y Position (m)', fontsize=11, fontweight='bold')
        ax.set_title(
            f'{category}  |  {avg_speed:.2f} m/s\nADE: {ade:.3f}m  FDE: {fde:.3f}m',
            fontsize=12, fontweight='bold', color=color
        )
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
        ax.set_facecolor('#f8f9fa')

    cross_tag = '(cross-domain, normalized)' if cross_domain else '(domain-matched)'
    fig.suptitle(f'{mode_label}  {cross_tag}  |  K-means boundary: {boundary:.2f} m/s',
                 fontsize=13, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])   # leave 5% at top for suptitle
    save_path = os.path.join(save_dir, 'trajectory_predictions_improved.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"   ✅ Trajectory plot saved → {save_path}")
    plt.show()


def visualize_error_over_time(
    model,
    valid_trajs,
    n_trajectories=50,
    sequence_length=10,
    prediction_horizon=10,
    device='cuda',
    save_dir='predictive_module/plot',
    cross_domain=False,
    normalize_input=False,
    mode_label='',
):
    """Plot how position and velocity error grows step-by-step."""
    model.eval()

    all_position_errors = []
    all_velocity_errors = []

    for trajectory in valid_trajs[:n_trajectories]:
        max_start = len(trajectory) - sequence_length - prediction_horizon
        if max_start <= 0:
            continue
        start_idx = np.random.randint(0, max_start)

        input_seq    = trajectory[start_idx:start_idx+sequence_length].copy()
        ground_truth = trajectory[start_idx+sequence_length:start_idx+sequence_length+prediction_horizon].copy()

        if normalize_input:
            input_seq, ground_truth = _apply_normalization(input_seq, ground_truth)

        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
            predictions = model.predict_sequence(input_tensor, prediction_horizon)
            predictions = predictions.cpu().numpy()[0]

        all_position_errors.append(np.linalg.norm(predictions[:, :2] - ground_truth[:, :2], axis=1))
        all_velocity_errors.append(np.linalg.norm(predictions[:, 2:4] - ground_truth[:, 2:4], axis=1))

    all_position_errors = np.array(all_position_errors)
    all_velocity_errors = np.array(all_velocity_errors)

    mean_pos = all_position_errors.mean(axis=0)
    std_pos  = all_position_errors.std(axis=0)
    mean_vel = all_velocity_errors.mean(axis=0)
    std_vel  = all_velocity_errors.std(axis=0)

    timesteps = np.arange(1, prediction_horizon+1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(timesteps, mean_pos, 'b-', linewidth=2, label='Mean Error')
    axes[0].fill_between(timesteps, mean_pos - std_pos, mean_pos + std_pos,
                         alpha=0.3, label='±1 Std Dev')
    axes[0].set_xlabel('Prediction Step', fontsize=12)
    axes[0].set_ylabel('Position Error (m)', fontsize=12)
    axes[0].set_title('Position Error Growth Over Time', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(timesteps, mean_vel, 'r-', linewidth=2, label='Mean Error')
    axes[1].fill_between(timesteps, mean_vel - std_vel, mean_vel + std_vel,
                         alpha=0.3, label='±1 Std Dev')
    axes[1].set_xlabel('Prediction Step', fontsize=12)
    axes[1].set_ylabel('Velocity Error (m/s)', fontsize=12)
    axes[1].set_title('Velocity Error Growth Over Time', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    cross_tag = '(cross-domain)' if cross_domain else '(domain-matched)'
    fig.suptitle(f'{mode_label}  {cross_tag}', fontsize=13, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'error_over_time.png')
    plt.savefig(save_path, dpi=150)
    print(f"   ✅ Error-over-time plot saved → {save_path}")
    plt.show()


def visualize_speed_comparison(
    model,
    valid_trajs,        # pre-filtered and aligned with labels
    labels,             # K-means labels (0=low, 1=high)
    boundary,           # K-means boundary (m/s)
    low_center,
    high_center,
    sequence_length=10,
    prediction_horizon=10,
    device='cuda',
    save_dir='predictive_module/plot',
    cross_domain=False,
    normalize_input=False,
    mode_label='',
):
    """Boxplot comparing prediction error between K-means speed clusters."""
    model.eval()

    low_speed_errors  = []
    high_speed_errors = []

    for trajectory, label in zip(valid_trajs, labels):
        max_start = len(trajectory) - sequence_length - prediction_horizon
        if max_start <= 0:
            continue
        start_idx = np.random.randint(0, max_start)

        input_seq    = trajectory[start_idx:start_idx+sequence_length].copy()
        ground_truth = trajectory[start_idx+sequence_length:start_idx+sequence_length+prediction_horizon].copy()

        if normalize_input:
            input_seq, ground_truth = _apply_normalization(input_seq, ground_truth)

        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
            predictions = model.predict_sequence(input_tensor, prediction_horizon)
            predictions = predictions.cpu().numpy()[0]

        error = np.linalg.norm(predictions[:, :2] - ground_truth[:, :2], axis=1).mean()
        if label == 0:
            low_speed_errors.append(error)
        else:
            high_speed_errors.append(error)

    low_mean  = np.mean(low_speed_errors)
    high_mean = np.mean(high_speed_errors)

    if low_mean > high_mean:
        benefit_pct = (low_mean - high_mean) / low_mean * 100
        winner = "High-speed"
        print(f"   ✅ High-speed predictions {benefit_pct:.1f}% better than low-speed")
    else:
        benefit_pct = (high_mean - low_mean) / high_mean * 100
        winner = "Low-speed"
        print(f"   ⚠️  Low-speed predictions {benefit_pct:.1f}% better than high-speed")

    fig, ax = plt.subplots(figsize=(10, 6))
    labels_text = [
        f'Low-Speed\n(μ={low_center:.2f} m/s)\nn={len(low_speed_errors)}',
        f'High-Speed\n(μ={high_center:.2f} m/s)\nn={len(high_speed_errors)}',
    ]
    ax.boxplot([low_speed_errors, high_speed_errors],
               positions=[1, 2], widths=0.6, patch_artist=True,
               labels=labels_text,
               boxprops=dict(facecolor='lightblue', alpha=0.7),
               medianprops=dict(color='red', linewidth=2),
               showfliers=True)

    ax.text(1, low_mean,  f'Mean: {low_mean:.4f}m',  ha='center', va='bottom',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(2, high_mean, f'Mean: {high_mean:.4f}m', ha='center', va='bottom',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(0.5, 0.95,
            f'{winner} cluster: {benefit_pct:.1f}% lower error\nBoundary: {boundary:.2f} m/s',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    cross_tag = '(cross-domain)' if cross_domain else '(domain-matched)'
    ax.set_ylabel('Position Error (m)', fontsize=12, fontweight='bold')
    ax.set_title(f'K-means Speed Clusters — {mode_label}  {cross_tag}\n'
                 f'Boundary: {boundary:.2f} m/s',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'speed_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Speed comparison saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Select which modes to run (comment out any to skip) ──────────────────
    EVAL_MODES = [
        # {
        #     'key':         'eth_eth',
        #     'label':       'ETH/UCY → ETH/UCY',
        #     'model_path':  'predictive_module/model/kgru_eth_ucy.pth',
        #     'data_path':   'predictive_module/data/eth_ucy_real_pedestrians.pkl',
        #     'downsample':  False,
        #     'cross_domain': False,
        #     'plot_dir':    'predictive_module/plot/eth',
        #     'input_size':  4,
        #     'seq_len':     10,   # 10 × 0.4s = 4s observation
        #     'horizon':     10,   # 10 × 0.4s = 4s prediction
        #     'dropout':     0.2,
        # },
        # {
        #     'key':         'syn_syn',
        #     'label':       'Synthetic → Synthetic',
        #     'model_path':  'predictive_module/model/kgru_synthetic.pth',
        #     'data_path':   'predictive_module/data/synthetic_mixed_traffic.pkl',
        #     'downsample':  True,
        #     'cross_domain': False,
        #     'plot_dir':    'predictive_module/plot/synthetic',
        #     'input_size':  4,
        #     'seq_len':     10,
        #     'horizon':     10,
        #     'dropout':     0.2,
        # },
        # {
        #     'key':         'syn_eth',
        #     'label':       'Synthetic → ETH/UCY',
        #     'model_path':  'predictive_module/model/kgru_synthetic.pth',
        #     'data_path':   'predictive_module/data/eth_ucy_real_pedestrians.pkl',
        #     'downsample':  False,
        #     'cross_domain': True,
        #     'plot_dir':    'predictive_module/plot/syn_to_eth',
        #     'input_size':  4,
        #     'seq_len':     10,
        #     'horizon':     10,
        #     'dropout':     0.2,
        # },
        # {
        #     'key':         'eth_syn',
        #     'label':       'ETH/UCY → Synthetic',
        #     'model_path':  'predictive_module/model/kgru_eth_ucy.pth',
        #     'data_path':   'predictive_module/data/synthetic_mixed_traffic.pkl',
        #     'downsample':  True,
        #     'cross_domain': True,
        #     'plot_dir':    'predictive_module/plot/eth_to_syn',
        #     'input_size':  4,
        #     'seq_len':     10,
        #     'horizon':     10,
        #     'dropout':     0.2,
        # },
        {
            'key':           'ind',
            'label':         'inD → inD',
            'model_path':    'predictive_module/model/kgru_ind.pth',
            'data_path':     'predictive_module/data/ind_with_class.pkl',
            'downsample':    False,   # already at 25 Hz
            'cross_domain':  False,
            'normalize_input': True,  # model trained with position normalization
            'plot_dir':      'predictive_module/plot/ind',
            'input_size':    8,       # [x, y, vx, vy, is_car, is_ped, is_truck, is_bicycle]
            'seq_len':       25,      # 25 × 0.04s = 1s observation
            'horizon':       25,      # 25 × 0.04s = 1s prediction
            'dropout':       0.5,
            'pad_class':     None,
            'max_speed':     20.0,    # filter tracking-error outliers (99th%=15.88 m/s)
        },
        # ── Cross-domain: ETH/Syn model → inD data (truncate 8D→4D) ─────────
        {
            'key':                  'eth_ind',
            'label':                'ETH/UCY → inD',
            'model_path':           'predictive_module/model/kgru_eth_ucy.pth',
            'data_path':            'predictive_module/data/ind_with_class.pkl',
            'downsample':           True,
            'downsample_source_dt': 0.04,   # inD 25Hz (dt=0.04s) → 2.5Hz (dt=0.4s)
            'downsample_target_dt': 0.4,
            'cross_domain':         True,
            'plot_dir':             'predictive_module/plot/eth_to_ind',
            'input_size':           4,       # ETH model: 4D; class cols truncated at runtime
            'seq_len':              10,
            'horizon':              10,
            'dropout':              0.2,
            'pad_class':            None,
            'max_speed':            20.0,
        },
        {
            'key':                  'syn_ind',
            'label':                'Synthetic → inD',
            'model_path':           'predictive_module/model/kgru_synthetic.pth',
            'data_path':            'predictive_module/data/ind_with_class.pkl',
            'downsample':           True,
            'downsample_source_dt': 0.04,   # inD 25Hz (dt=0.04s) → 2.5Hz (dt=0.4s)
            'downsample_target_dt': 0.4,
            'cross_domain':         True,
            'plot_dir':             'predictive_module/plot/syn_to_ind',
            'input_size':           4,
            'seq_len':              10,
            'horizon':              10,
            'dropout':              0.2,
            'pad_class':            None,
            'max_speed':            20.0,
        },
        # ── Cross-domain: inD model → ETH/Syn data (upsample + pad class) ──────
        # ETH/UCY is all pedestrians → pad [is_car=0, is_ped=1, is_truck=0, is_bicycle=0]
        # Upsample ETH 2.5Hz → 25Hz so temporal scale matches inD training
        {
            'key':               'ind_eth',
            'label':             'inD → ETH/UCY',
            'model_path':        'predictive_module/model/kgru_ind.pth',
            'data_path':         'predictive_module/data/eth_ucy_real_pedestrians.pkl',
            'downsample':        False,
            'upsample_source_dt': 0.4,   # ETH 2.5Hz → 25Hz (×10)
            'cross_domain':      True,
            'normalize_input':   True,
            'plot_dir':          'predictive_module/plot/ind_to_eth',
            'input_size':        8,
            'seq_len':           25,      # 25 × 0.04s = 1s (matches inD training)
            'horizon':           25,
            'dropout':           0.5,
            'pad_class':         [0, 1, 0, 0],   # all ETH agents are pedestrians
        },
        # Synthetic 10Hz → 25Hz (×2.5); class unknown → pad zeros
        {
            'key':               'ind_syn',
            'label':             'inD → Synthetic',
            'model_path':        'predictive_module/model/kgru_ind.pth',
            'data_path':         'predictive_module/data/synthetic_mixed_traffic.pkl',
            'downsample':        False,
            'upsample_source_dt': 0.1,   # Syn 10Hz → 25Hz (×2.5)
            'cross_domain':      True,
            'normalize_input':   True,
            'plot_dir':          'predictive_module/plot/ind_to_syn',
            'input_size':        8,
            'seq_len':           25,
            'horizon':           25,
            'dropout':           0.5,
            'pad_class':         [0, 0, 0, 0],   # class unknown for synthetic agents
        },
    ]
    # ─────────────────────────────────────────────────────────────────────────

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Cache loaded datasets so we don't reload the same file twice
    _data_cache  = {}
    _model_cache = {}

    for mode in EVAL_MODES:
        print("\n" + "="*65)
        print(f"  {mode['label']}  {'[cross-domain]' if mode['cross_domain'] else '[domain-matched]'}")
        print("="*65)

        os.makedirs(mode['plot_dir'], exist_ok=True)

        # ── Load data ────────────────────────────────────────────────────────
        pad_class        = mode.get('pad_class')
        max_speed        = mode.get('max_speed')
        upsample_src_dt  = mode.get('upsample_source_dt')
        ds_src_key       = mode.get('downsample_source_dt', 0.1)
        ds_tgt_key       = mode.get('downsample_target_dt', 0.4)
        cache_key = (mode['data_path'], mode['downsample'], ds_src_key, ds_tgt_key,
                     upsample_src_dt,
                     tuple(pad_class) if pad_class is not None else None,
                     max_speed)
        if cache_key not in _data_cache:
            with open(mode['data_path'], 'rb') as f:
                raw = pickle.load(f)
            trajs = raw['trajectories']
            if mode['downsample']:
                ds_src = mode.get('downsample_source_dt', 0.1)
                ds_tgt = mode.get('downsample_target_dt', 0.4)
                before = len(trajs)
                trajs = downsample_trajectories(trajs, source_dt=ds_src, target_dt=ds_tgt)
                factor = int(round(ds_tgt / ds_src))
                print(f"   Downsample: {ds_src}s → {ds_tgt}s  (stride={factor}x, "
                      f"{before} → {len(trajs)} trajectories)")
            if upsample_src_dt is not None:
                before = len(trajs)
                trajs = upsample_trajectories(trajs, source_dt=upsample_src_dt,
                                              target_dt=0.04)
                print(f"   Upsample: {upsample_src_dt}s → 0.04s  "
                      f"({before} → {len(trajs)} trajectories)")
            if max_speed is not None:
                before = len(trajs)
                trajs = [t for t in trajs
                         if np.mean(np.linalg.norm(t[:, 2:4], axis=1)) <= max_speed]
                print(f"   Speed filter: {before} → {len(trajs)} trajectories "
                      f"(removed {before - len(trajs)} outliers > {max_speed} m/s)")
            if pad_class is not None:
                pad = np.array(pad_class, dtype=np.float32)
                trajs = [np.hstack([t, np.tile(pad, (len(t), 1))]) for t in trajs]
            _data_cache[cache_key] = trajs
        trajectories = _data_cache[cache_key]

        n_train = int(0.7 * len(trajectories))
        n_val   = int(0.15 * len(trajectories))
        test_trajectories = trajectories[n_train + n_val:]

        # ── Load model ───────────────────────────────────────────────────────
        if mode['model_path'] not in _model_cache:
            checkpoint = torch.load(mode['model_path'], map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                cfg = checkpoint.get('config', {})
                hidden_size = cfg.get('hidden_size', 128)
                num_layers  = cfg.get('num_layers',  3)
                state_dict  = checkpoint['model_state_dict']
            else:
                hidden_size = 128
                num_layers  = 3
                state_dict  = checkpoint
            m = TrajectoryGRU(input_size=mode['input_size'], hidden_size=hidden_size,
                              num_layers=num_layers, output_size=4,
                              dropout=mode['dropout']).to(device)
            m.load_state_dict(state_dict)
            m.eval()
            _model_cache[mode['model_path']] = m
        model = _model_cache[mode['model_path']]

        # ── K-means on ALL trajectories → boundary matches global analysis ──────
        # (same distribution as analyze_kmeans_clustering in train_kgru.py)
        boundary, low_center, high_center, all_labels = kmeans_speed_clusters(trajectories)

        # Slice to test split, then filter by minimum length
        test_labels = all_labels[n_train + n_val:]
        valid_mask  = np.array([len(t) >= mode['seq_len'] + mode['horizon'] + 1 for t in test_trajectories])
        valid_trajs = [t for t, m in zip(test_trajectories, valid_mask) if m]
        labels      = test_labels[valid_mask]

        # Truncate feature dim if model expects fewer cols than data provides
        # (e.g. ETH/Syn model on inD 8D data → keep only [x,y,vx,vy])
        feat_dim = mode['input_size']
        if valid_trajs and valid_trajs[0].shape[1] > feat_dim:
            valid_trajs = [t[:, :feat_dim] for t in valid_trajs]

        if len(valid_trajs) < 6:
            print(f"   ⚠️  Only {len(valid_trajs)} valid trajectories — skipping mode.")
            continue

        print(f"   K-means: {low_center:.2f} m/s (low) | {boundary:.2f} m/s (boundary)"
              f" | {high_center:.2f} m/s (high)")
        print(f"   Samples: {len(valid_trajs)} valid  "
              f"(low={int(np.sum(labels==0))}, high={int(np.sum(labels==1))})")

        # normalize_input: True for inD model (trained w/ normalization) and
        # all cross-domain modes; falls back to cross_domain for ETH/Syn modes
        normalize_input = mode.get('normalize_input', mode['cross_domain'])

        # ── 1. Trajectory predictions ─────────────────────────────────────────
        print("\n   1. Trajectory predictions...")
        visualize_trajectory_predictions(
            model, valid_trajs, labels, boundary,
            n_samples=6, sequence_length=mode['seq_len'], prediction_horizon=mode['horizon'],
            device=device, save_dir=mode['plot_dir'],
            cross_domain=mode['cross_domain'], normalize_input=normalize_input,
            mode_label=mode['label'],
        )

        # ── 2. Error over time ────────────────────────────────────────────────
        print("   2. Error over time...")
        visualize_error_over_time(
            model, valid_trajs,
            n_trajectories=50, sequence_length=mode['seq_len'], prediction_horizon=mode['horizon'],
            device=device, save_dir=mode['plot_dir'],
            cross_domain=mode['cross_domain'], normalize_input=normalize_input,
            mode_label=mode['label'],
        )

        # ── 3. Speed comparison ───────────────────────────────────────────────
        print("   3. Speed comparison (K-means clusters)...")
        visualize_speed_comparison(
            model, valid_trajs, labels, boundary, low_center, high_center,
            sequence_length=mode['seq_len'], prediction_horizon=mode['horizon'],
            device=device, save_dir=mode['plot_dir'],
            cross_domain=mode['cross_domain'], normalize_input=normalize_input,
            mode_label=mode['label'],
        )

    print("\n" + "="*65)
    print("VISUALIZATION COMPLETE")
    print("="*65)
    print("Output directories:")
    for mode in EVAL_MODES:
        print(f"  {mode['plot_dir']}/  ({mode['label']})")
