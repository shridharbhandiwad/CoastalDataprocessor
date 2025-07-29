# Maritime Radar Dataset Generator

This repository contains a self-contained Python script for generating large-scale, labeled maritime radar datasets suitable for machine-learning research.

## Features

* **Realistic clutter** based on Weibull sea-clutter amplitude statistics for three sea-states (calm, moderate, rough).
* **Moving and static vessel tracks** with sequential detections (TrackID) and kinematic consistency.
* **Customisable size** – produce gigabyte-scale datasets by tweaking CLI parameters.
* **Supervised labels** identifying each detection as `target` (vessel) or `clutter`.
* **Automatic train/val/test split** baked into the output file.
* Output to **Parquet** (fast, compressed) or **CSV**.

## Quick start

```bash
# 1. Install deps (ideally in a virtual environment)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate a ~1 GB dataset under rough sea conditions
python generate_maritime_radar_dataset.py \
  --num_target_tracks 40000 \
  --num_clutter_tracks 20000 \
  --min_track_len 50 --max_track_len 150 \
  --sea_state rough \
  --output_dir data/ \
  --file_format parquet
```

The script prints progress and writes `data/maritime_radar.parquet` containing millions of detections. The table includes a `Split` column (`train`, `val`, `test`) for downstream workflows.

## File schema

| Column     | Type   | Description                             |
|------------|--------|-----------------------------------------|
| TrackID    | string | Unique track identifier                 |
| Range      | float  | Slant-range in metres                   |
| Azimuth    | float  | Bearing angle in degrees [0, 360)       |
| Elevation  | float  | Elevation angle in degrees              |
| Doppler    | float  | Radial velocity in m/s                  |
| RCS        | float  | Radar cross section in dBsm             |
| SNR        | float  | Signal-to-Noise Ratio in dB             |
| Timestamp  | string | ISO-8601 UTC timestamp (ms resolution)  |
| Label      | int    | 1 = vessel target, 0 = sea clutter       |
| Split      | string | train / val / test                      |

## Customising generation

Run `python generate_maritime_radar_dataset.py --help` to see all available options, including:

* `--dt` revisit interval (s) between detections within a track.
* `--seed` RNG seed for reproducibility.
* `--file_format csv` if Parquet is unsuitable.

## Pre-processing suggestions

* Convert Azimuth and Elevation to Cartesian direction cosines for neural networks.
* Standardise/normalise continuous features and encode categorical ones.
* Use windowing to build temporal sequences per TrackID.

Feel free to modify the generator to match sensor specifics (antenna pattern, PRF, noise figure, etc.). Pull requests welcome!