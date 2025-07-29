#!/usr/bin/env python3
"""
Example Usage of Maritime Radar Dataset Generator

This script demonstrates how to generate and analyze maritime radar datasets
with various configurations and use cases.

Author: AI Assistant
License: MIT
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from maritime_radar_generator import MaritimeRadarDatasetGenerator
from data_analysis_tools import MaritimeRadarAnalyzer

def example_1_quick_test_dataset():
    """Generate a small test dataset for quick validation"""
    print("=" * 60)
    print("Example 1: Quick Test Dataset Generation")
    print("=" * 60)
    
    # Create generator
    generator = MaritimeRadarDatasetGenerator("test_dataset")
    
    print("Generating small test dataset (100 MB)...")
    start_time = time.time()
    
    # Generate small dataset
    files_created = generator.generate_dataset(
        target_size_gb=0.1,  # 100 MB
        clutter_ratio=0.7,
        num_vessel_tracks=50
    )
    
    end_time = time.time()
    print(f"Generation completed in {end_time - start_time:.2f} seconds")
    
    # Print file information
    print("\nFiles created:")
    for file_type, filepath in files_created.items():
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  {file_type}: {filepath} ({size_mb:.1f} MB)")
    
    return files_created

def example_2_custom_configuration():
    """Generate dataset with custom radar and vessel configurations"""
    print("\n" + "=" * 60)
    print("Example 2: Custom Configuration Dataset")
    print("=" * 60)
    
    # Create generator with custom output directory
    generator = MaritimeRadarDatasetGenerator("custom_radar_dataset")
    
    # Modify radar configuration for C-band radar
    generator.radar_config.frequency_ghz = 5.6  # C-band instead of X-band
    generator.radar_config.max_range_m = 75000  # 75 km range
    generator.radar_config.tx_power_w = 50000   # 50 kW power
    generator.radar_config.antenna_gain_db = 35.0  # Higher gain antenna
    
    # Add custom vessel type
    generator.vessel_model.vessel_types['research_vessel'] = {
        'length': 120,
        'beam': 20,
        'speed_range': (5, 18),
        'rcs_base': 600
    }
    
    print("Custom radar configuration:")
    print(f"  Frequency: {generator.radar_config.frequency_ghz} GHz")
    print(f"  Max range: {generator.radar_config.max_range_m / 1000} km")
    print(f"  TX power: {generator.radar_config.tx_power_w / 1000} kW")
    
    print(f"\nVessel types available: {list(generator.vessel_model.vessel_types.keys())}")
    
    # Generate dataset with high clutter ratio
    print("\nGenerating custom dataset (200 MB)...")
    files_created = generator.generate_dataset(
        target_size_gb=0.2,
        clutter_ratio=0.8,  # High clutter environment
        num_vessel_tracks=75
    )
    
    print("Custom dataset generation complete!")
    return files_created

def example_3_large_production_dataset():
    """Generate a large production-ready dataset"""
    print("\n" + "=" * 60)
    print("Example 3: Large Production Dataset")
    print("=" * 60)
    
    # For demonstration, we'll use a smaller size
    # In practice, you might want 5-10 GB or more
    target_size = 0.5  # 500 MB for demo
    
    generator = MaritimeRadarDatasetGenerator("production_dataset")
    
    print(f"Generating production dataset ({target_size} GB)...")
    print("This may take several minutes...")
    
    start_time = time.time()
    
    files_created = generator.generate_dataset(
        target_size_gb=target_size,
        clutter_ratio=0.65,
        num_vessel_tracks=500
    )
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    print(f"\nProduction dataset generation completed!")
    print(f"Total time: {generation_time:.2f} seconds")
    print(f"Generation rate: {target_size * 1024 / generation_time:.1f} MB/min")
    
    return files_created

def example_4_comprehensive_analysis():
    """Perform comprehensive analysis of generated dataset"""
    print("\n" + "=" * 60)
    print("Example 4: Comprehensive Dataset Analysis")
    print("=" * 60)
    
    # Use the test dataset from example 1
    dataset_dir = "test_dataset"
    
    if not os.path.exists(dataset_dir):
        print("No dataset found. Running Example 1 first...")
        example_1_quick_test_dataset()
    
    # Create analyzer
    analyzer = MaritimeRadarAnalyzer(dataset_dir)
    
    # Generate summary report
    print("Generating summary report...")
    summary = analyzer.generate_summary_report()
    
    print("\nDataset Summary:")
    print(f"  Total detections: {summary['basic_stats']['total_detections']:,}")
    print(f"  Unique tracks: {summary['basic_stats']['unique_tracks']:,}")
    print(f"  Time span: {summary['basic_stats']['time_span_hours']:.1f} hours")
    print(f"  Clutter/Target ratio: {summary['basic_stats']['label_distribution']}")
    
    print(f"\nRange statistics (km):")
    range_stats = summary['spatial_stats']['range_stats']
    print(f"  Min: {range_stats['min'] / 1000:.1f}")
    print(f"  Max: {range_stats['max'] / 1000:.1f}")
    print(f"  Mean: {range_stats['mean'] / 1000:.1f}")
    
    print(f"\nSNR statistics (dB):")
    snr_stats = summary['signal_stats']['snr_stats']
    print(f"  Min: {snr_stats['min']:.1f}")
    print(f"  Max: {snr_stats['max']:.1f}")
    print(f"  Mean: {snr_stats['mean']:.1f}")
    
    # Create output directory for analysis results
    analysis_dir = "analysis_results"
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Generate visualizations (in a real scenario, these would display)
    print(f"\nGenerating analysis plots in '{analysis_dir}'...")
    
    try:
        # Note: In a headless environment, plots won't display but will be saved
        analyzer.plot_dataset_overview(os.path.join(analysis_dir, "overview.png"))
        print("  ✓ Dataset overview plot created")
        
        analyzer.plot_polar_coverage(os.path.join(analysis_dir, "polar_coverage.png"))
        print("  ✓ Polar coverage plot created")
        
        analyzer.plot_track_analysis(save_path=os.path.join(analysis_dir, "track_analysis.png"))
        print("  ✓ Track analysis plot created")
        
        analyzer.create_interactive_dashboard(os.path.join(analysis_dir, "dashboard.html"))
        print("  ✓ Interactive dashboard created")
        
    except Exception as e:
        print(f"  Note: Plotting skipped in headless environment ({e})")
    
    # Run classification benchmark
    print("\nRunning classification benchmark...")
    benchmark_results = analyzer.benchmark_classification()
    
    print(f"Classification Results:")
    print(f"  Accuracy: {benchmark_results['test_accuracy']:.3f}")
    print(f"  Feature importance:")
    for feature, importance in sorted(benchmark_results['feature_importance'].items(), 
                                    key=lambda x: x[1], reverse=True):
        print(f"    {feature}: {importance:.3f}")
    
    return summary, benchmark_results

def example_5_data_loading_and_processing():
    """Demonstrate how to load and process the generated dataset"""
    print("\n" + "=" * 60)
    print("Example 5: Data Loading and Processing")
    print("=" * 60)
    
    dataset_dir = "test_dataset"
    
    if not os.path.exists(dataset_dir):
        print("No dataset found. Running Example 1 first...")
        example_1_quick_test_dataset()
    
    # Load data in different formats
    analyzer = MaritimeRadarAnalyzer(dataset_dir)
    
    print("Loading data from CSV format...")
    df_csv = analyzer.load_data('train', format='csv')
    print(f"CSV data shape: {df_csv.shape}")
    
    try:
        print("Loading data from HDF5 format...")
        df_hdf5 = analyzer.load_data('train', format='hdf5')
        print(f"HDF5 data shape: {df_hdf5.shape}")
    except Exception as e:
        print(f"HDF5 loading failed: {e}")
    
    # Basic data exploration
    print("\nData exploration:")
    print(f"Columns: {list(df_csv.columns)}")
    print(f"Data types:")
    for col, dtype in df_csv.dtypes.items():
        print(f"  {col}: {dtype}")
    
    # Sample data analysis
    print(f"\nLabel distribution:")
    label_counts = df_csv['label'].value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count:,} ({count/len(df_csv)*100:.1f}%)")
    
    # Track analysis
    print(f"\nTrack statistics:")
    track_lengths = df_csv.groupby('track_id').size()
    print(f"  Number of tracks: {len(track_lengths)}")
    print(f"  Average detections per track: {track_lengths.mean():.1f}")
    print(f"  Min/Max detections per track: {track_lengths.min()}/{track_lengths.max()}")
    
    # Show sample vessel tracks
    vessel_tracks = df_csv[df_csv['label'] == 'target']['track_id'].unique()[:3]
    print(f"\nSample vessel tracks:")
    for track_id in vessel_tracks:
        track_data = df_csv[df_csv['track_id'] == track_id]
        print(f"  {track_id}: {len(track_data)} detections, "
              f"avg speed: {track_data['doppler_ms'].abs().mean():.1f} m/s")
    
    return df_csv

def example_6_machine_learning_preparation():
    """Demonstrate preparing data for machine learning"""
    print("\n" + "=" * 60)
    print("Example 6: Machine Learning Data Preparation")
    print("=" * 60)
    
    dataset_dir = "test_dataset"
    
    if not os.path.exists(dataset_dir):
        print("No dataset found. Running Example 1 first...")
        example_1_quick_test_dataset()
    
    # Load data
    analyzer = MaritimeRadarAnalyzer(dataset_dir)
    df_train = analyzer.load_data('train', format='csv')
    df_val = analyzer.load_data('validation', format='csv')
    df_test = analyzer.load_data('test', format='csv')
    
    print(f"Dataset splits:")
    print(f"  Training: {len(df_train):,} samples")
    print(f"  Validation: {len(df_val):,} samples")
    print(f"  Test: {len(df_test):,} samples")
    
    # Feature engineering examples
    print(f"\nFeature engineering:")
    
    # Basic features
    feature_columns = ['range_m', 'azimuth_deg', 'elevation_deg', 'doppler_ms', 
                      'rcs_dbsm', 'snr_db', 'sea_state']
    
    # Derived features
    df_train['range_km'] = df_train['range_m'] / 1000
    df_train['doppler_abs'] = df_train['doppler_ms'].abs()
    df_train['rcs_linear'] = 10 ** (df_train['rcs_dbsm'] / 10)
    df_train['snr_linear'] = 10 ** (df_train['snr_db'] / 10)
    
    print(f"  Original features: {len(feature_columns)}")
    print(f"  Derived features: 4")
    print(f"  Total features: {len(feature_columns) + 4}")
    
    # Feature statistics
    print(f"\nFeature statistics:")
    for col in feature_columns[:4]:  # Show first 4 features
        mean_val = df_train[col].mean()
        std_val = df_train[col].std()
        print(f"  {col}: μ={mean_val:.2f}, σ={std_val:.2f}")
    
    # Label encoding
    print(f"\nLabel encoding:")
    label_map = {'clutter': 0, 'target': 1}
    df_train['label_encoded'] = df_train['label'].map(label_map)
    print(f"  {label_map}")
    
    # Feature correlation analysis
    import numpy as np
    
    feature_data = df_train[feature_columns].select_dtypes(include=[np.number])
    correlation_matrix = feature_data.corr()
    
    print(f"\nHighest feature correlations:")
    # Find pairs with highest absolute correlation (excluding diagonal)
    corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            corr_pairs.append((correlation_matrix.columns[i], 
                             correlation_matrix.columns[j], abs(corr_val)))
    
    corr_pairs.sort(key=lambda x: x[2], reverse=True)
    for feat1, feat2, corr in corr_pairs[:3]:
        print(f"  {feat1} - {feat2}: {corr:.3f}")
    
    return df_train, df_val, df_test

def main():
    """Run all examples"""
    print("Maritime Radar Dataset Generator - Example Usage")
    print("=" * 60)
    
    try:
        # Example 1: Quick test dataset
        example_1_quick_test_dataset()
        
        # Example 2: Custom configuration
        example_2_custom_configuration()
        
        # Example 3: Large dataset (commented out for demo)
        # example_3_large_production_dataset()
        
        # Example 4: Analysis
        example_4_comprehensive_analysis()
        
        # Example 5: Data loading
        example_5_data_loading_and_processing()
        
        # Example 6: ML preparation
        example_6_machine_learning_preparation()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
        print("\nGenerated datasets:")
        for dataset_dir in ["test_dataset", "custom_radar_dataset"]:
            if os.path.exists(dataset_dir):
                total_size = sum(os.path.getsize(os.path.join(dataset_dir, f)) 
                               for f in os.listdir(dataset_dir) 
                               if os.path.isfile(os.path.join(dataset_dir, f)))
                print(f"  {dataset_dir}: {total_size / (1024*1024):.1f} MB")
        
        print("\nNext steps:")
        print("  1. Explore the generated datasets")
        print("  2. Run the analysis tools")
        print("  3. Use the data for your machine learning projects")
        print("  4. Customize the generator for your specific needs")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()