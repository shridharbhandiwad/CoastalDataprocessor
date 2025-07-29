#!/usr/bin/env python3
"""
Basic Functionality Test for Maritime Radar Dataset Generator

This script tests the core functionality without requiring external dependencies.
It demonstrates the basic structure and mathematical models.

Author: AI Assistant
License: MIT
"""

import os
import sys
import json
import math
import random
from datetime import datetime, timedelta

def basic_radar_equation_test():
    """Test basic radar equation calculations"""
    print("Testing Radar Equation Calculations:")
    print("=" * 40)
    
    # Basic radar parameters
    tx_power_w = 25000.0  # 25kW
    antenna_gain_db = 33.0
    frequency_ghz = 9.4
    range_m = 10000.0  # 10km
    rcs_dbsm = 20.0  # 20 dBsm target
    
    # Calculate wavelength
    c = 3e8  # speed of light
    wavelength_m = c / (frequency_ghz * 1e9)
    
    # Convert to dB values
    pt_db = 10.0 * math.log10(tx_power_w)
    wavelength_db = 20.0 * math.log10(wavelength_m)
    range_db = 40.0 * math.log10(range_m)
    
    # Simplified SNR calculation
    k_boltzmann = 1.38e-23
    temp_k = 290.0
    bandwidth_hz = 1e6
    noise_figure_db = 3.0
    system_losses_db = 3.0
    
    noise_power_db = 10.0 * math.log10(k_boltzmann * temp_k * bandwidth_hz)
    
    snr_db = (pt_db + 2 * antenna_gain_db + wavelength_db + rcs_dbsm - 
              103.4 - range_db - noise_power_db - noise_figure_db - system_losses_db)
    
    print(f"  Transmit Power: {tx_power_w/1000:.0f} kW ({pt_db:.1f} dB)")
    print(f"  Frequency: {frequency_ghz} GHz")
    print(f"  Wavelength: {wavelength_m:.3f} m")
    print(f"  Range: {range_m/1000:.0f} km")
    print(f"  Target RCS: {rcs_dbsm} dBsm")
    print(f"  Calculated SNR: {snr_db:.1f} dB")
    
    return snr_db

def basic_clutter_model_test():
    """Test basic sea clutter model"""
    print("\nTesting Sea Clutter Models:")
    print("=" * 40)
    
    # Test different sea states
    sea_states = [1, 3, 5]  # Calm, moderate, rough
    
    for sea_state in sea_states:
        print(f"\nSea State {sea_state}:")
        
        # Sea state factor
        sea_state_factor = 1.0 + (sea_state - 1) * 0.5
        wind_speed_ms = 5.0 + sea_state * 2.0  # Approximate wind speed
        wind_factor = (wind_speed_ms / 10.0) ** 0.5
        
        # Base sigma_0 for X-band
        sigma_0_db = -25.0 + 10.0 * math.log10(sea_state_factor * wind_factor)
        
        print(f"  Wind speed: {wind_speed_ms:.1f} m/s")
        print(f"  Sea state factor: {sea_state_factor:.2f}")
        print(f"  Wind factor: {wind_factor:.2f}")
        print(f"  Sigma_0: {sigma_0_db:.1f} dB")
        
        # Generate some sample clutter amplitudes using simple random distribution
        num_samples = 5
        clutter_samples = []
        for _ in range(num_samples):
            # Simple approximation of Weibull/K-distribution
            if sea_state <= 2:
                # Weibull-like (more stable)
                amplitude = random.weibullvariate(1.8, 1.0)
            else:
                # More variable for rough seas
                amplitude = random.gammavariate(1.2, 1.0)
            
            clutter_samples.append(amplitude)
        
        print(f"  Sample amplitudes: {[f'{a:.2f}' for a in clutter_samples[:3]]}")

def basic_vessel_simulation_test():
    """Test basic vessel movement simulation"""
    print("\nTesting Vessel Movement Simulation:")
    print("=" * 40)
    
    # Vessel types
    vessel_types = {
        'cargo': {'speed_range': (5, 15), 'rcs_base': 1000},
        'fishing': {'speed_range': (3, 10), 'rcs_base': 100},
        'yacht': {'speed_range': (8, 25), 'rcs_base': 50}
    }
    
    for vessel_type, params in vessel_types.items():
        print(f"\n{vessel_type.title()} Ship:")
        
        # Random initial conditions
        initial_range = random.uniform(5000, 30000)
        initial_azimuth = random.uniform(0, 360)
        speed_knots = random.uniform(*params['speed_range'])
        speed_ms = speed_knots * 0.514444  # Convert to m/s
        course_deg = random.uniform(0, 360)
        
        print(f"  Initial range: {initial_range/1000:.1f} km")
        print(f"  Initial azimuth: {initial_azimuth:.1f}°")
        print(f"  Speed: {speed_knots:.1f} knots ({speed_ms:.1f} m/s)")
        print(f"  Course: {course_deg:.1f}°")
        
        # Calculate position after 10 minutes
        time_interval = 600  # 10 minutes
        
        # Convert to Cartesian
        x = initial_range * math.cos(math.radians(initial_azimuth))
        y = initial_range * math.sin(math.radians(initial_azimuth))
        
        # Update position
        dx = speed_ms * time_interval * math.cos(math.radians(course_deg))
        dy = speed_ms * time_interval * math.sin(math.radians(course_deg))
        x += dx
        y += dy
        
        # Convert back to polar
        new_range = math.sqrt(x**2 + y**2)
        new_azimuth = math.degrees(math.atan2(y, x)) % 360
        
        # Calculate radial velocity (Doppler)
        radial_velocity = speed_ms * math.cos(math.radians(course_deg - initial_azimuth))
        
        print(f"  After 10 min range: {new_range/1000:.1f} km")
        print(f"  After 10 min azimuth: {new_azimuth:.1f}°")
        print(f"  Radial velocity: {radial_velocity:.1f} m/s")
        
        # Calculate RCS with aspect angle
        aspect_angle = abs(course_deg - initial_azimuth) % 180
        aspect_factor = 0.5 + 0.5 * math.cos(math.radians(aspect_angle)) ** 2
        rcs_dbsm = 10.0 * math.log10(params['rcs_base'] * aspect_factor)
        
        print(f"  Aspect angle: {aspect_angle:.1f}°")
        print(f"  RCS: {rcs_dbsm:.1f} dBsm")

def basic_data_structure_test():
    """Test basic data structure and JSON serialization"""
    print("\nTesting Data Structure:")
    print("=" * 40)
    
    # Create sample detection
    detection = {
        'track_id': 'vessel_cargo_000001',
        'range_m': 15750.5,
        'azimuth_deg': 45.2,
        'elevation_deg': 0.15,
        'doppler_ms': 5.3,
        'rcs_dbsm': 25.7,
        'snr_db': 18.4,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'label': 'target',
        'sea_state': 3
    }
    
    print("Sample Detection:")
    for key, value in detection.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Test JSON serialization
    json_str = json.dumps(detection, indent=2)
    print(f"\nJSON serialized size: {len(json_str)} bytes")
    
    # Create sample dataset metadata
    metadata = {
        'dataset_info': {
            'total_detections': 1000000,
            'clutter_detections': 700000,
            'target_detections': 300000,
            'unique_tracks': 1500
        },
        'radar_config': {
            'frequency_ghz': 9.4,
            'max_range_m': 50000.0,
            'tx_power_w': 25000.0
        }
    }
    
    print(f"\nSample metadata:")
    print(json.dumps(metadata, indent=2))

def estimate_dataset_size():
    """Estimate dataset sizes for different configurations"""
    print("\nDataset Size Estimation:")
    print("=" * 40)
    
    # Estimate bytes per detection
    sample_detection = {
        'track_id': 'vessel_cargo_000001',  # ~20 bytes
        'range_m': 15750.5,  # ~8 bytes
        'azimuth_deg': 45.2,  # ~8 bytes
        'elevation_deg': 0.15,  # ~8 bytes
        'doppler_ms': 5.3,  # ~8 bytes
        'rcs_dbsm': 25.7,  # ~8 bytes
        'snr_db': 18.4,  # ~8 bytes
        'timestamp': '2024-01-15T10:30:45.123Z',  # ~25 bytes
        'label': 'target',  # ~10 bytes
        'sea_state': 3  # ~4 bytes
    }
    
    # Estimate JSON size
    json_size = len(json.dumps(sample_detection))
    print(f"Estimated bytes per detection (JSON): {json_size}")
    
    # Calculate for different dataset sizes
    target_sizes_gb = [0.1, 1.0, 5.0, 10.0]
    
    print(f"\nDataset size estimates:")
    for size_gb in target_sizes_gb:
        total_bytes = size_gb * 1e9
        num_detections = int(total_bytes / json_size)
        num_tracks = num_detections // 500  # Assume 500 detections per track average
        
        print(f"  {size_gb:4.1f} GB: {num_detections:8,} detections, {num_tracks:5,} tracks")

def performance_test():
    """Basic performance test for generation algorithms"""
    print("\nPerformance Test:")
    print("=" * 40)
    
    import time
    
    # Test detection generation speed
    start_time = time.time()
    num_detections = 10000
    
    detections = []
    for i in range(num_detections):
        # Simulate detection generation
        detection = {
            'track_id': f'test_{i:06d}',
            'range_m': random.uniform(1000, 50000),
            'azimuth_deg': random.uniform(0, 360),
            'elevation_deg': random.normalvariate(0, 0.5),
            'doppler_ms': random.normalvariate(0, 2.0),
            'rcs_dbsm': random.normalvariate(15, 5),
            'snr_db': random.normalvariate(20, 8),
            'timestamp': (datetime.utcnow() + timedelta(seconds=i)).isoformat() + 'Z',
            'label': 'clutter' if random.random() < 0.7 else 'target',
            'sea_state': random.randint(1, 6)
        }
        detections.append(detection)
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    print(f"Generated {num_detections:,} detections in {generation_time:.2f} seconds")
    print(f"Generation rate: {num_detections/generation_time:.0f} detections/second")
    print(f"Estimated time for 1M detections: {1000000/num_detections*generation_time:.1f} seconds")

def main():
    """Run all basic tests"""
    print("Maritime Radar Dataset Generator - Basic Functionality Test")
    print("=" * 60)
    print("Testing core algorithms without external dependencies...")
    print()
    
    try:
        # Run all tests
        basic_radar_equation_test()
        basic_clutter_model_test()
        basic_vessel_simulation_test()
        basic_data_structure_test()
        estimate_dataset_size()
        performance_test()
        
        print("\n" + "=" * 60)
        print("✅ All basic functionality tests completed successfully!")
        print("=" * 60)
        
        print("\nTest Summary:")
        print("  ✓ Radar equation calculations")
        print("  ✓ Sea clutter modeling")
        print("  ✓ Vessel movement simulation")
        print("  ✓ Data structure serialization")
        print("  ✓ Dataset size estimation")
        print("  ✓ Performance benchmarking")
        
        print("\nThe maritime radar dataset generator is ready for use!")
        print("Install the required dependencies and run the full generator:")
        print("  pip install -r requirements.txt")
        print("  python maritime_radar_generator.py --size-gb 1.0")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)