"""
generate_maritime_radar_dataset.py
----------------------------------
Synthetic maritime radar dataset generator.

This script creates a dataset containing millions of radar detections with the
following required columns:
    - TrackID  : str (unique identifier per track)
    - Range    : float (metres)
    - Azimuth  : float (degrees)
    - Elevation: float (degrees)
    - Doppler  : float (m/s)
    - RCS      : float (dBsm)
    - SNR      : float (dB)
    - Timestamp: str (ISO-8601 UTC)
    - Label    : int (1 = vessel/target, 0 = sea clutter)

Key features
------------
* Supports three sea-state models (calm, moderate, rough) that influence sea-clutter
  statistics via a Weibull amplitude model and Doppler spread.
* Generates realistic vessel tracks with user-controllable motion & signature
  parameters.
* Creates multiple sequential detections per TrackID (track length randomly
  sampled within a range) with a configurable revisit interval.
* Produces a dataset large enough for deep-learning pipelines (scale to 1 GiB+
  by adjusting CLI args).
* Automatically splits the full dataset into train/val/test partitions.

Usage (example)
---------------
python generate_maritime_radar_dataset.py \
    --num_target_tracks 50000 \
    --num_clutter_tracks 25000 \
    --min_track_len 50 \
    --max_track_len 200 \
    --sea_state rough \
    --output_dir data/ \
    --split_train 0.8 --split_val 0.1 --split_test 0.1 \
    --file_format parquet

The above command will yield ≈12.5 million detections (∼2.5–3 GiB in Parquet)
under a rough-sea scenario.
"""

import argparse
import os
import uuid
from datetime import datetime, timedelta
from typing import Tuple, List

import numpy as np
import pandas as pd

ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"

# ---------------------------- Utility functions ---------------------------- #

def _iso_now() -> str:
    """Return current time in ISO-8601 UTC format (ms resolution)."""
    return datetime.utcnow().strftime(ISO_FMT)[:-3] + "Z"


def _sample_weibull_rcs(size: int, sea_state: str) -> np.ndarray:
    """Sample RCS values (dBsm) using a Weibull shape controlled by sea state."""
    sea_state = sea_state.lower()
    # Shape k and scale λ parameters chosen heuristically
    params = {
        "calm": (0.6, 0.2),
        "moderate": (0.5, 0.4),
        "rough": (0.4, 0.8),
    }
    k, lam = params.get(sea_state, params["moderate"])
    # Generate amplitude then convert to dBsm (10*log10)
    amp = np.random.weibull(k, size) * lam
    rcs = 10.0 * np.log10(amp + 1e-12)  # avoid log(0)
    return rcs


def _sample_snr(rcs: np.ndarray) -> np.ndarray:
    """Derive an SNR proxy from RCS with added gaussian noise."""
    snr = rcs + np.random.normal(20.0, 3.0, size=rcs.shape)  # 1:1 slope approx
    return snr


def _simulate_clutter_track(track_id: str, num_points: int, sea_state: str,
                            start_time: datetime, dt: float) -> pd.DataFrame:
    """Generate a single sea-clutter track (quasi-stationary micro-dynamics)."""
    # For clutter we assume range variation around some patch; Doppler near 0
    base_range = np.random.uniform(100.0, 3000.0)  # metres
    base_az = np.random.uniform(0.0, 360.0)
    base_el = np.random.normal(0.0, 0.2)  # small elevation angles (sea surface)

    # Small jitter per detection
    rng = base_range + np.random.normal(0.0, 5.0, size=num_points)
    az = (base_az + np.random.normal(0.0, 0.5, size=num_points)) % 360.0
    el = base_el + np.random.normal(0.0, 0.05, size=num_points)

    # Doppler for clutter ~ near 0 with sea-state-dependent spread
    doppler_sigma = {"calm": 0.2, "moderate": 0.5, "rough": 1.0}.get(sea_state, 0.5)
    doppler = np.random.normal(0.0, doppler_sigma, size=num_points)

    rcs = _sample_weibull_rcs(num_points, sea_state)
    snr = _sample_snr(rcs)

    # Timestamps
    times = [start_time + timedelta(seconds=i * dt) for i in range(num_points)]
    timestamps = [t.strftime(ISO_FMT)[:-3] + "Z" for t in times]

    df = pd.DataFrame({
        "TrackID": track_id,
        "Range": rng,
        "Azimuth": az,
        "Elevation": el,
        "Doppler": doppler,
        "RCS": rcs,
        "SNR": snr,
        "Timestamp": timestamps,
        "Label": 0,  # clutter
    })
    return df


def _simulate_target_track(track_id: str, num_points: int,
                           start_time: datetime, dt: float) -> pd.DataFrame:
    """Generate a moving vessel track with random kinematics."""
    # Initial position in polar co-ordinates
    range0 = np.random.uniform(500.0, 20000.0)
    az0 = np.random.uniform(0.0, 360.0)
    el0 = np.random.normal(0.0, 1.0)  # vessels slightly above horizon due to mast

    # Velocity components (converted to polar increments)
    speed = np.random.uniform(1.0, 15.0)  # m/s (small boats) to 30 kn ≈ 15 m/s
    heading = np.random.uniform(0.0, 360.0)
    # Convert to Cartesian increments per dt then to dRange/dAz for small motions
    vx = speed * np.cos(np.deg2rad(heading))
    vy = speed * np.sin(np.deg2rad(heading))

    # Pre-allocate arrays
    rng = np.empty(num_points)
    az = np.empty(num_points)
    el = np.empty(num_points)
    doppler = np.full(num_points, speed) * np.cos(np.deg2rad(heading - az0))

    x0 = range0 * np.cos(np.deg2rad(az0))
    y0 = range0 * np.sin(np.deg2rad(az0))

    for i in range(num_points):
        x = x0 + vx * dt * i
        y = y0 + vy * dt * i
        rng[i] = np.hypot(x, y)
        az[i] = (np.rad2deg(np.arctan2(y, x)) + 360.0) % 360.0
        el[i] = el0 + np.random.normal(0.0, 0.05)

    # RCS ~ log-normal around mean 10 dBsm with vessel-specific variance
    rcs_mu = np.random.normal(10.0, 2.0)
    rcs_sigma = 1.5
    rcs = np.random.lognormal(mean=rcs_mu / 10.0 * np.log(10), sigma=rcs_sigma / 10.0, size=num_points)
    rcs = 10.0 * np.log10(rcs + 1e-12)

    snr = _sample_snr(rcs) + 3.0  # typically higher than clutter

    times = [start_time + timedelta(seconds=i * dt) for i in range(num_points)]
    timestamps = [t.strftime(ISO_FMT)[:-3] + "Z" for t in times]

    df = pd.DataFrame({
        "TrackID": track_id,
        "Range": rng,
        "Azimuth": az,
        "Elevation": el,
        "Doppler": doppler,
        "RCS": rcs,
        "SNR": snr,
        "Timestamp": timestamps,
        "Label": 1,  # target
    })
    return df

# ------------------------------ Main routine ------------------------------ #

def generate_dataset(num_target_tracks: int,
                     num_clutter_tracks: int,
                     min_track_len: int,
                     max_track_len: int,
                     sea_state: str,
                     dt: float,
                     splits: Tuple[float, float, float],
                     output_dir: str,
                     file_format: str = "parquet",
                     seed: int = 42) -> None:
    """Generate full dataset and write to disk in the chosen format."""
    rng_global = np.random.default_rng(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    dfs: List[pd.DataFrame] = []
    now = datetime.utcnow()

    # Generate clutter tracks
    for _ in range(num_clutter_tracks):
        n_pts = rng_global.integers(min_track_len, max_track_len + 1)
        track_id = f"clutter_{uuid.uuid4().hex[:10]}"
        df = _simulate_clutter_track(track_id, n_pts, sea_state, now, dt)
        dfs.append(df)

    # Generate target tracks
    for _ in range(num_target_tracks):
        n_pts = rng_global.integers(min_track_len, max_track_len + 1)
        track_id = f"target_{uuid.uuid4().hex[:10]}"
        df = _simulate_target_track(track_id, n_pts, now, dt)
        dfs.append(df)

    print("Concatenating...")
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Total detections: {len(full_df):,}")

    # Shuffle detections to remove ordering bias
    full_df = full_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Split indices
    train_ratio, val_ratio, test_ratio = splits
    n = len(full_df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    split_tags = np.empty(n, dtype=object)
    split_tags[:train_end] = "train"
    split_tags[train_end:val_end] = "val"
    split_tags[val_end:] = "test"
    full_df["Split"] = split_tags

    # Persist
    print(f"Saving to {output_dir} in {file_format} format...")
    if file_format == "parquet":
        import pyarrow as pa  # deferred import
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(full_df)
        pq.write_table(table, os.path.join(output_dir, "maritime_radar.parquet"), compression="zstd")
    else:
        # fall back to csv (may be slower/larger)
        full_df.to_csv(os.path.join(output_dir, "maritime_radar.csv"), index=False)

    print("Dataset generation complete.")

# ------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic maritime radar dataset generator")
    p.add_argument("--num_target_tracks", type=int, default=10000, help="Number of vessel tracks to simulate")
    p.add_argument("--num_clutter_tracks", type=int, default=5000, help="Number of sea-clutter tracks to simulate")
    p.add_argument("--min_track_len", type=int, default=20, help="Minimum detections per track")
    p.add_argument("--max_track_len", type=int, default=100, help="Maximum detections per track")
    p.add_argument("--sea_state", choices=["calm", "moderate", "rough"], default="moderate", help="Sea state model")
    p.add_argument("--dt", type=float, default=0.5, help="Time between detections (seconds)")
    p.add_argument("--split_train", type=float, default=0.8, help="Train split ratio")
    p.add_argument("--split_val", type=float, default=0.1, help="Validation split ratio")
    p.add_argument("--split_test", type=float, default=0.1, help="Test split ratio")
    p.add_argument("--output_dir", type=str, default="dataset_out", help="Directory to write dataset")
    p.add_argument("--file_format", choices=["parquet", "csv"], default="parquet", help="Output file format")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    total = args.split_train + args.split_val + args.split_test
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    generate_dataset(
        num_target_tracks=args.num_target_tracks,
        num_clutter_tracks=args.num_clutter_tracks,
        min_track_len=args.min_track_len,
        max_track_len=args.max_track_len,
        sea_state=args.sea_state,
        dt=args.dt,
        splits=(args.split_train, args.split_val, args.split_test),
        output_dir=args.output_dir,
        file_format=args.file_format,
        seed=args.seed,
    )