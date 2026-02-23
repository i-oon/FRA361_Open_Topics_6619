"""
Shared utilities for K-GRU trajectory prediction.
"""
import numpy as np
from sklearn.cluster import KMeans


def kmeans_speed_clusters(trajectories, random_state=42):
    """
    Discover low/high speed clusters from trajectory data using K-means.

    Returns:
        boundary   : midpoint between the two cluster centers (m/s)
        low_center : mean speed of the slow cluster (m/s)
        high_center: mean speed of the fast cluster (m/s)
        labels     : per-trajectory int array — 0 = low-speed, 1 = high-speed
    """
    avg_speeds = np.array([
        np.linalg.norm(t[:, 2:4], axis=1).mean() for t in trajectories
    ])

    kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    raw_labels = kmeans.fit_predict(avg_speeds.reshape(-1, 1))

    centers = kmeans.cluster_centers_.flatten()
    low_cluster = int(centers[0] > centers[1])   # index of the lower center
    high_cluster = 1 - low_cluster

    # Normalise labels so 0 = low-speed, 1 = high-speed
    labels = np.where(raw_labels == low_cluster, 0, 1)

    return (centers[low_cluster] + centers[high_cluster]) / 2, \
           centers[low_cluster], centers[high_cluster], labels


def normalize_sequence(seq):
    """
    Shift positions so the last observed frame is at the origin.
    Used for cross-domain evaluation to remove absolute coordinate bias.

    Returns:
        normalized : copy of seq with x/y positions shifted
        origin     : (x, y) of the last frame before shifting
    """
    origin = seq[-1, :2].copy()
    normalized = seq.copy()
    normalized[:, 0] -= origin[0]
    normalized[:, 1] -= origin[1]
    return normalized, origin


def downsample_trajectories(trajectories, source_dt=0.1, target_dt=0.4, min_length=11):
    """
    Downsample trajectories from source_dt to target_dt by striding, then
    recompute velocities at the new temporal resolution.

    Args:
        trajectories : list of (N, 4) arrays [x, y, vx, vy]
        source_dt    : time step of the input data (seconds)
        target_dt    : desired time step (seconds)
        min_length   : minimum number of frames to keep a trajectory

    Returns:
        List of downsampled trajectories that meet the length requirement.
    """
    factor = int(round(target_dt / source_dt))
    result = []
    for traj in trajectories:
        ds = traj[::factor].copy()
        for i in range(len(ds) - 1):
            ds[i, 2] = (ds[i+1, 0] - ds[i, 0]) / target_dt
            ds[i, 3] = (ds[i+1, 1] - ds[i, 1]) / target_dt
        ds[-1, 2:4] = ds[-2, 2:4]
        if len(ds) >= min_length:
            result.append(ds)
    return result
