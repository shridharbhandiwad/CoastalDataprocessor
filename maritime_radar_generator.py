#!/usr/bin/env python3
"""
Maritime Radar Dataset Generator

Generates realistic synthetic maritime radar data with labeled tracks for:
- Sea clutter (using Weibull and K-distribution models)
- Vessel targets (with realistic movement patterns)
- Multiple sea states (calm, moderate, rough)

Author: AI Assistant
License: MIT
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import h5py
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RadarDetection:
    """Single radar detection point"""
    track_id: str
    range_m: float
    azimuth_deg: float
    elevation_deg: float
    doppler_ms: float
    rcs_dbsm: float
    snr_db: float
    timestamp: str
    label: str  # 'target' or 'clutter'
    sea_state: int  # 1-6 scale

@dataclass
class RadarConfig:
    """Radar system configuration"""
    max_range_m: float = 50000.0  # 50km max range
    range_resolution_m: float = 30.0  # 30m range resolution
    azimuth_beamwidth_deg: float = 1.0  # 1 degree azimuth beamwidth
    elevation_beamwidth_deg: float = 2.0  # 2 degree elevation beamwidth
    frequency_ghz: float = 9.4  # X-band radar
    prf_hz: float = 3000.0  # Pulse repetition frequency
    tx_power_w: float = 25000.0  # 25kW transmit power
    antenna_gain_db: float = 33.0  # Antenna gain
    noise_figure_db: float = 3.0  # Receiver noise figure

class SeaClutterModel:
    """Sea clutter simulation using Weibull and K-distribution models"""
    
    def __init__(self, radar_config: RadarConfig):
        self.radar_config = radar_config
        
    def weibull_clutter(self, size: int, shape: float = 1.2, scale: float = 1.0) -> np.ndarray:
        """Generate Weibull-distributed clutter amplitudes"""
        return np.random.weibull(shape, size) * scale
        
    def k_distribution_clutter(self, size: int, shape: float = 1.5, scale: float = 1.0) -> np.ndarray:
        """Generate K-distributed clutter amplitudes"""
        # K-distribution approximated as Gamma distribution
        return np.random.gamma(shape, scale, size)
        
    def calculate_sea_clutter_rcs(self, range_m: float, azimuth_deg: float, 
                                 sea_state: int, wind_speed_ms: float = 10.0) -> float:
        """Calculate sea clutter RCS using empirical models"""
        # Base clutter coefficient (depends on frequency, polarization, grazing angle)
        grazing_angle_deg = np.arctan(10.0 / range_m) * 180.0 / np.pi  # Assume 10m antenna height
        
        # Sea state factor (1=calm, 6=very rough)
        sea_state_factor = 1.0 + (sea_state - 1) * 0.5
        
        # Wind speed factor
        wind_factor = (wind_speed_ms / 10.0) ** 0.5
        
        # Range dependence (typically range^3 to range^4)
        range_factor = (range_m / 1000.0) ** 3.5
        
        # Base sigma_0 for X-band, VV polarization
        sigma_0_db = -25.0 + 10.0 * np.log10(sea_state_factor * wind_factor)
        
        # Convert to RCS (sigma_0 * resolution cell area)
        resolution_cell_area = (self.radar_config.range_resolution_m * 
                               range_m * np.radians(self.radar_config.azimuth_beamwidth_deg))
        
        rcs_dbsm = sigma_0_db + 10.0 * np.log10(resolution_cell_area)
        
        return rcs_dbsm
        
    def generate_clutter_detections(self, num_detections: int, sea_state: int, 
                                   time_span_hours: float = 1.0) -> List[RadarDetection]:
        """Generate sea clutter detections"""
        detections = []
        start_time = datetime.utcnow()
        
        for i in range(num_detections):
            # Random spatial distribution
            range_m = np.random.uniform(1000, self.radar_config.max_range_m)
            azimuth_deg = np.random.uniform(0, 360)
            elevation_deg = np.random.normal(0, 0.5)  # Small elevation spread
            
            # Clutter Doppler (near zero with small variations from sea motion)
            if sea_state <= 2:
                doppler_ms = np.random.normal(0, 0.2)  # Calm seas
            elif sea_state <= 4:
                doppler_ms = np.random.normal(0, 0.5)  # Moderate seas
            else:
                doppler_ms = np.random.normal(0, 1.0)   # Rough seas
            
            # Calculate RCS
            rcs_dbsm = self.calculate_sea_clutter_rcs(range_m, azimuth_deg, sea_state)
            
            # Add clutter amplitude variations
            if sea_state <= 2:
                rcs_variation = self.weibull_clutter(1, shape=1.8)[0]
            else:
                rcs_variation = self.k_distribution_clutter(1, shape=1.2)[0]
            
            rcs_dbsm += 10.0 * np.log10(rcs_variation) - 3.0  # Normalize and add variation
            
            # Calculate SNR
            snr_db = self.calculate_snr(range_m, rcs_dbsm)
            
            # Random timestamp within time span
            time_offset = np.random.uniform(0, time_span_hours * 3600)
            timestamp = (start_time + timedelta(seconds=time_offset)).isoformat() + 'Z'
            
            detection = RadarDetection(
                track_id=f"clutter_{i:08d}",
                range_m=range_m,
                azimuth_deg=azimuth_deg,
                elevation_deg=elevation_deg,
                doppler_ms=doppler_ms,
                rcs_dbsm=rcs_dbsm,
                snr_db=snr_db,
                timestamp=timestamp,
                label='clutter',
                sea_state=sea_state
            )
            detections.append(detection)
            
        return detections
    
    def calculate_snr(self, range_m: float, rcs_dbsm: float) -> float:
        """Calculate SNR using radar equation"""
        # Radar equation: SNR = (Pt * G^2 * λ^2 * σ) / ((4π)^3 * R^4 * kT * B * F * L)
        
        pt_db = 10.0 * np.log10(self.radar_config.tx_power_w)
        wavelength_m = 3e8 / (self.radar_config.frequency_ghz * 1e9)
        wavelength_db = 20.0 * np.log10(wavelength_m)
        range_db = 40.0 * np.log10(range_m)
        
        # System losses and constants
        k_boltzmann = 1.38e-23
        temp_k = 290.0
        bandwidth_hz = 1e6  # 1 MHz bandwidth
        system_losses_db = 3.0
        
        noise_power_db = 10.0 * np.log10(k_boltzmann * temp_k * bandwidth_hz)
        
        snr_db = (pt_db + 2 * self.radar_config.antenna_gain_db + wavelength_db + 
                 rcs_dbsm - 103.4 - range_db - noise_power_db - 
                 self.radar_config.noise_figure_db - system_losses_db)
        
        return snr_db

class VesselTargetModel:
    """Vessel target simulation with realistic movement patterns"""
    
    def __init__(self, radar_config: RadarConfig):
        self.radar_config = radar_config
        self.vessel_types = {
            'cargo': {'length': 200, 'beam': 32, 'speed_range': (5, 15), 'rcs_base': 1000},
            'tanker': {'length': 300, 'beam': 45, 'speed_range': (8, 18), 'rcs_base': 2000},
            'container': {'length': 350, 'beam': 40, 'speed_range': (12, 22), 'rcs_base': 1500},
            'bulk_carrier': {'length': 250, 'beam': 38, 'speed_range': (6, 14), 'rcs_base': 1200},
            'fishing': {'length': 50, 'beam': 12, 'speed_range': (3, 10), 'rcs_base': 100},
            'yacht': {'length': 30, 'beam': 8, 'speed_range': (8, 25), 'rcs_base': 50},
            'patrol': {'length': 80, 'beam': 15, 'speed_range': (15, 35), 'rcs_base': 200}
        }
        
    def calculate_vessel_rcs(self, vessel_type: str, aspect_angle_deg: float) -> float:
        """Calculate vessel RCS based on type and aspect angle"""
        base_rcs = self.vessel_types[vessel_type]['rcs_base']
        
        # Aspect angle dependence (simplified)
        aspect_rad = np.radians(aspect_angle_deg)
        aspect_factor = 0.5 + 0.5 * np.cos(aspect_rad) ** 2  # Beam aspect stronger
        
        # Add some random variation
        variation_db = np.random.normal(0, 2.0)
        
        rcs_dbsm = 10.0 * np.log10(base_rcs * aspect_factor) + variation_db
        
        return rcs_dbsm
    
    def generate_vessel_track(self, vessel_type: str, track_id: str, 
                            track_duration_hours: float = 2.0,
                            detection_interval_s: float = 10.0) -> List[RadarDetection]:
        """Generate a complete vessel track with realistic movement"""
        detections = []
        vessel_params = self.vessel_types[vessel_type]
        
        # Initial conditions
        start_time = datetime.utcnow()
        initial_range = np.random.uniform(5000, 40000)
        initial_azimuth = np.random.uniform(0, 360)
        speed_ms = np.random.uniform(*vessel_params['speed_range']) * 0.514444  # knots to m/s
        course_deg = np.random.uniform(0, 360)
        
        # Current position in Cartesian coordinates
        x = initial_range * np.cos(np.radians(initial_azimuth))
        y = initial_range * np.sin(np.radians(initial_azimuth))
        
        num_detections = int(track_duration_hours * 3600 / detection_interval_s)
        
        for i in range(num_detections):
            # Update position
            if i > 0:
                # Add some course variation
                course_change = np.random.normal(0, 2.0)  # Small course changes
                course_deg += course_change
                course_deg = course_deg % 360
                
                # Update position
                dx = speed_ms * detection_interval_s * np.cos(np.radians(course_deg))
                dy = speed_ms * detection_interval_s * np.sin(np.radians(course_deg))
                x += dx
                y += dy
            
            # Convert back to polar coordinates
            range_m = np.sqrt(x**2 + y**2)
            azimuth_deg = np.degrees(np.arctan2(y, x)) % 360
            
            # Skip if out of radar range
            if range_m > self.radar_config.max_range_m:
                continue
            
            # Elevation (small variations due to sea state)
            elevation_deg = np.random.normal(0, 0.3)
            
            # Doppler calculation
            radial_velocity = speed_ms * np.cos(np.radians(course_deg - azimuth_deg))
            doppler_ms = radial_velocity
            
            # Calculate RCS
            aspect_angle = abs(course_deg - azimuth_deg) % 180
            rcs_dbsm = self.calculate_vessel_rcs(vessel_type, aspect_angle)
            
            # Calculate SNR
            snr_db = self.calculate_snr(range_m, rcs_dbsm)
            
            # Timestamp
            timestamp = (start_time + timedelta(seconds=i * detection_interval_s)).isoformat() + 'Z'
            
            detection = RadarDetection(
                track_id=track_id,
                range_m=range_m,
                azimuth_deg=azimuth_deg,
                elevation_deg=elevation_deg,
                doppler_ms=doppler_ms,
                rcs_dbsm=rcs_dbsm,
                snr_db=snr_db,
                timestamp=timestamp,
                label='target',
                sea_state=np.random.randint(1, 7)  # Random sea state for diversity
            )
            detections.append(detection)
            
        return detections
    
    def calculate_snr(self, range_m: float, rcs_dbsm: float) -> float:
        """Calculate SNR using radar equation (same as clutter model)"""
        pt_db = 10.0 * np.log10(self.radar_config.tx_power_w)
        wavelength_m = 3e8 / (self.radar_config.frequency_ghz * 1e9)
        wavelength_db = 20.0 * np.log10(wavelength_m)
        range_db = 40.0 * np.log10(range_m)
        
        k_boltzmann = 1.38e-23
        temp_k = 290.0
        bandwidth_hz = 1e6
        system_losses_db = 3.0
        
        noise_power_db = 10.0 * np.log10(k_boltzmann * temp_k * bandwidth_hz)
        
        snr_db = (pt_db + 2 * self.radar_config.antenna_gain_db + wavelength_db + 
                 rcs_dbsm - 103.4 - range_db - noise_power_db - 
                 self.radar_config.noise_figure_db - system_losses_db)
        
        return snr_db

class MaritimeRadarDatasetGenerator:
    """Main dataset generator class"""
    
    def __init__(self, output_dir: str = "maritime_radar_dataset"):
        self.output_dir = output_dir
        self.radar_config = RadarConfig()
        self.clutter_model = SeaClutterModel(self.radar_config)
        self.vessel_model = VesselTargetModel(self.radar_config)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_dataset(self, target_size_gb: float = 1.0,
                        clutter_ratio: float = 0.7,
                        num_vessel_tracks: int = 1000) -> Dict[str, str]:
        """Generate complete maritime radar dataset"""
        
        logger.info(f"Generating maritime radar dataset ({target_size_gb:.1f} GB)")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Estimate number of detections needed
        bytes_per_detection = 200  # Approximate
        target_detections = int(target_size_gb * 1e9 / bytes_per_detection)
        
        clutter_detections_target = int(target_detections * clutter_ratio)
        vessel_detections_target = target_detections - clutter_detections_target
        
        logger.info(f"Target detections: {target_detections:,}")
        logger.info(f"Clutter detections: {clutter_detections_target:,}")
        logger.info(f"Vessel detections: {vessel_detections_target:,}")
        
        all_detections = []
        
        # Generate sea clutter for different sea states
        for sea_state in range(1, 7):
            state_detections = clutter_detections_target // 6
            logger.info(f"Generating {state_detections:,} clutter detections for sea state {sea_state}")
            
            clutter_detections = self.clutter_model.generate_clutter_detections(
                state_detections, sea_state, time_span_hours=24.0
            )
            all_detections.extend(clutter_detections)
        
        # Generate vessel tracks
        vessel_types = list(self.vessel_model.vessel_types.keys())
        detections_per_track = vessel_detections_target // num_vessel_tracks
        
        logger.info(f"Generating {num_vessel_tracks:,} vessel tracks")
        
        for i in range(num_vessel_tracks):
            vessel_type = np.random.choice(vessel_types)
            track_id = f"vessel_{vessel_type}_{i:06d}"
            
            track_detections = self.vessel_model.generate_vessel_track(
                vessel_type, track_id, track_duration_hours=4.0
            )
            all_detections.extend(track_detections)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1:,} vessel tracks")
        
        logger.info(f"Total detections generated: {len(all_detections):,}")
        
        # Save dataset
        return self.save_dataset(all_detections)
    
    def save_dataset(self, detections: List[RadarDetection]) -> Dict[str, str]:
        """Save dataset in multiple formats with train/validation/test splits"""
        
        logger.info("Saving dataset...")
        
        # Convert to DataFrame
        data = []
        for det in detections:
            data.append({
                'track_id': det.track_id,
                'range_m': det.range_m,
                'azimuth_deg': det.azimuth_deg,
                'elevation_deg': det.elevation_deg,
                'doppler_ms': det.doppler_ms,
                'rcs_dbsm': det.rcs_dbsm,
                'snr_db': det.snr_db,
                'timestamp': det.timestamp,
                'label': det.label,
                'sea_state': det.sea_state
            })
        
        df = pd.DataFrame(data)
        
        # Shuffle data
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Create train/validation/test splits
        n_total = len(df)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        df_train = df[:n_train]
        df_val = df[n_train:n_train + n_val]
        df_test = df[n_train + n_val:]
        
        # Save in multiple formats
        files_created = {}
        
        # CSV format
        train_csv = os.path.join(self.output_dir, 'train.csv')
        val_csv = os.path.join(self.output_dir, 'validation.csv')
        test_csv = os.path.join(self.output_dir, 'test.csv')
        
        df_train.to_csv(train_csv, index=False)
        df_val.to_csv(val_csv, index=False)
        df_test.to_csv(test_csv, index=False)
        
        files_created['train_csv'] = train_csv
        files_created['val_csv'] = val_csv
        files_created['test_csv'] = test_csv
        
        # HDF5 format (more efficient for large datasets)
        hdf5_file = os.path.join(self.output_dir, 'maritime_radar_dataset.h5')
        with h5py.File(hdf5_file, 'w') as f:
            # Create groups for splits
            train_group = f.create_group('train')
            val_group = f.create_group('validation')
            test_group = f.create_group('test')
            
            # Save data arrays
            for col in df.columns:
                if col == 'track_id' or col == 'timestamp' or col == 'label':
                    # String data
                    train_group.create_dataset(col, data=df_train[col].astype('S'))
                    val_group.create_dataset(col, data=df_val[col].astype('S'))
                    test_group.create_dataset(col, data=df_test[col].astype('S'))
                else:
                    # Numeric data
                    train_group.create_dataset(col, data=df_train[col].values)
                    val_group.create_dataset(col, data=df_val[col].values)
                    test_group.create_dataset(col, data=df_test[col].values)
        
        files_created['hdf5'] = hdf5_file
        
        # Save metadata
        metadata = {
            'dataset_info': {
                'total_detections': len(df),
                'train_detections': len(df_train),
                'validation_detections': len(df_val),
                'test_detections': len(df_test),
                'clutter_detections': len(df[df['label'] == 'clutter']),
                'target_detections': len(df[df['label'] == 'target']),
                'unique_tracks': df['track_id'].nunique(),
                'sea_states': sorted(df['sea_state'].unique().tolist()),
                'time_span': {
                    'start': df['timestamp'].min(),
                    'end': df['timestamp'].max()
                }
            },
            'radar_config': {
                'max_range_m': self.radar_config.max_range_m,
                'range_resolution_m': self.radar_config.range_resolution_m,
                'azimuth_beamwidth_deg': self.radar_config.azimuth_beamwidth_deg,
                'elevation_beamwidth_deg': self.radar_config.elevation_beamwidth_deg,
                'frequency_ghz': self.radar_config.frequency_ghz,
                'prf_hz': self.radar_config.prf_hz,
                'tx_power_w': self.radar_config.tx_power_w,
                'antenna_gain_db': self.radar_config.antenna_gain_db,
                'noise_figure_db': self.radar_config.noise_figure_db
            },
            'field_descriptions': {
                'track_id': 'Unique identifier for each track',
                'range_m': 'Range to target in meters',
                'azimuth_deg': 'Azimuth angle in degrees (0-360)',
                'elevation_deg': 'Elevation angle in degrees',
                'doppler_ms': 'Doppler velocity in m/s',
                'rcs_dbsm': 'Radar cross section in dBsm',
                'snr_db': 'Signal-to-noise ratio in dB',
                'timestamp': 'ISO-8601 UTC timestamp',
                'label': 'Classification: target or clutter',
                'sea_state': 'Sea state (1=calm to 6=very rough)'
            }
        }
        
        metadata_file = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        files_created['metadata'] = metadata_file
        
        # Calculate file sizes
        total_size_mb = sum(os.path.getsize(f) for f in files_created.values()) / (1024 * 1024)
        
        logger.info(f"Dataset saved successfully!")
        logger.info(f"Total size: {total_size_mb:.1f} MB")
        logger.info(f"Files created: {list(files_created.keys())}")
        
        return files_created

def main():
    parser = argparse.ArgumentParser(description='Generate maritime radar dataset')
    parser.add_argument('--size-gb', type=float, default=1.0,
                       help='Target dataset size in GB (default: 1.0)')
    parser.add_argument('--output-dir', type=str, default='maritime_radar_dataset',
                       help='Output directory (default: maritime_radar_dataset)')
    parser.add_argument('--clutter-ratio', type=float, default=0.7,
                       help='Ratio of clutter to total detections (default: 0.7)')
    parser.add_argument('--num-tracks', type=int, default=1000,
                       help='Number of vessel tracks to generate (default: 1000)')
    
    args = parser.parse_args()
    
    # Create generator
    generator = MaritimeRadarDatasetGenerator(args.output_dir)
    
    # Generate dataset
    files_created = generator.generate_dataset(
        target_size_gb=args.size_gb,
        clutter_ratio=args.clutter_ratio,
        num_vessel_tracks=args.num_tracks
    )
    
    print("\nDataset generation complete!")
    print(f"Files created:")
    for file_type, filepath in files_created.items():
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  {file_type}: {filepath} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()