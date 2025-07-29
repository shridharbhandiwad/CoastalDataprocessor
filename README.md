# Maritime Radar Dataset Generator

A comprehensive synthetic maritime radar dataset generator that creates realistic labeled tracks for sea clutter and vessel targets using physical models and empirical radar equations.

## Features

- **Realistic Physical Modeling**: Uses Weibull and K-distribution models for sea clutter simulation
- **Vessel Target Simulation**: Multiple vessel types with realistic movement patterns
- **Multiple Sea States**: Supports calm to very rough sea conditions (1-6 scale)
- **Complete Track Generation**: Sequential detections per TrackID with realistic temporal evolution
- **Supervised Learning Ready**: Pre-labeled data for target vs. clutter classification
- **Large-Scale Generation**: Capable of producing GB-scale datasets with millions of detections
- **Multiple Formats**: Output in CSV and HDF5 formats with train/validation/test splits
- **Comprehensive Analysis Tools**: Built-in visualization and analysis utilities

## Dataset Fields

Each radar detection includes the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `track_id` | string | Unique identifier for each track |
| `range_m` | float | Range to target in meters |
| `azimuth_deg` | float | Azimuth angle in degrees (0-360) |
| `elevation_deg` | float | Elevation angle in degrees |
| `doppler_ms` | float | Doppler velocity in m/s |
| `rcs_dbsm` | float | Radar cross section in dBsm |
| `snr_db` | float | Signal-to-noise ratio in dB |
| `timestamp` | string | ISO-8601 UTC timestamp |
| `label` | string | Classification: 'target' or 'clutter' |
| `sea_state` | int | Sea state (1=calm to 6=very rough) |

## Installation

```bash
# Clone or download the repository
git clone <repository-url>
cd maritime-radar-dataset

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Dataset Generation

```bash
# Generate a 1GB dataset with default settings
python maritime_radar_generator.py --size-gb 1.0

# Generate larger dataset with custom parameters
python maritime_radar_generator.py \
    --size-gb 5.0 \
    --num-tracks 5000 \
    --clutter-ratio 0.75 \
    --output-dir my_radar_dataset
```

### Analyze Generated Dataset

```bash
# Run comprehensive analysis
python data_analysis_tools.py \
    --dataset-dir maritime_radar_dataset \
    --output-dir analysis_results
```

## Physical Models

### Sea Clutter Simulation

The generator uses empirically-based sea clutter models:

- **Weibull Distribution**: For calm to moderate sea states
- **K-Distribution**: For rough sea conditions
- **Empirical σ₀ Model**: Based on frequency, polarization, and grazing angle
- **Sea State Scaling**: Accounts for wind speed and wave height effects

### Vessel Target Models

Realistic vessel simulation includes:

- **7 Vessel Types**: Cargo, tanker, container, bulk carrier, fishing, yacht, patrol
- **Aspect Angle Dependence**: RCS varies with viewing angle
- **Realistic Movement**: Course variations and speed profiles
- **Doppler Calculation**: Based on radial velocity components

### Radar System Model

Uses realistic X-band radar parameters:

- **Frequency**: 9.4 GHz (X-band)
- **Max Range**: 50 km
- **Range Resolution**: 30 m
- **Transmit Power**: 25 kW
- **Antenna Gain**: 33 dB
- **SNR Calculation**: Full radar equation implementation

## Dataset Generation Examples

### Small Test Dataset

```python
from maritime_radar_generator import MaritimeRadarDatasetGenerator

# Create generator
generator = MaritimeRadarDatasetGenerator("test_dataset")

# Generate small dataset for testing
files = generator.generate_dataset(
    target_size_gb=0.1,  # 100 MB
    clutter_ratio=0.7,
    num_vessel_tracks=100
)
```

### Large Production Dataset

```python
# Generate large dataset for training
generator = MaritimeRadarDatasetGenerator("production_dataset")

files = generator.generate_dataset(
    target_size_gb=10.0,  # 10 GB
    clutter_ratio=0.65,
    num_vessel_tracks=10000
)
```

## Analysis and Visualization

The toolkit includes comprehensive analysis tools:

### Statistical Analysis

```python
from data_analysis_tools import MaritimeRadarAnalyzer

analyzer = MaritimeRadarAnalyzer("maritime_radar_dataset")

# Generate summary statistics
summary = analyzer.generate_summary_report()
print(f"Total detections: {summary['basic_stats']['total_detections']:,}")
```

### Visualization Options

- **Dataset Overview**: Multi-panel statistical plots
- **Polar Coverage**: Spatial distribution in radar coordinates
- **Track Analysis**: Individual vessel track evolution
- **Interactive Dashboard**: Plotly-based interactive exploration
- **Classification Benchmark**: Basic ML performance metrics

### Example Plots

```python
# Create comprehensive overview
analyzer.plot_dataset_overview("dataset_overview.png")

# Analyze spatial coverage
analyzer.plot_polar_coverage("spatial_coverage.png")

# Track specific vessels
analyzer.plot_track_analysis(["vessel_cargo_000001", "vessel_tanker_000005"])

# Interactive dashboard
analyzer.create_interactive_dashboard("dashboard.html")
```

## Dataset Splits

The generator automatically creates train/validation/test splits:

- **Training**: 70% of data
- **Validation**: 15% of data  
- **Test**: 15% of data

Data is shuffled before splitting to ensure representative distributions across all splits.

## File Formats

### CSV Format
- `train.csv`, `validation.csv`, `test.csv`
- Human-readable, widely compatible
- Good for initial exploration and small datasets

### HDF5 Format
- `maritime_radar_dataset.h5`
- Efficient storage and fast access
- Recommended for large datasets
- Organized in groups: `/train`, `/validation`, `/test`

### Metadata
- `metadata.json`: Complete dataset information
- Radar configuration parameters
- Statistical summaries
- Field descriptions

## Performance Considerations

### Memory Usage
- Large datasets are generated iteratively to minimize memory usage
- HDF5 format provides memory-efficient access
- Consider using data generators for training very large models

### Generation Time
- ~1-2 minutes per 100MB on modern hardware
- Parallelization possible for very large datasets
- Progress logging included for long-running generations

### Storage Requirements
- CSV format: ~200 bytes per detection
- HDF5 format: ~150 bytes per detection (compressed)
- 1GB dataset ≈ 5M detections ≈ 1000 tracks

## Customization

### Custom Vessel Types

```python
# Add custom vessel type
generator.vessel_model.vessel_types['custom_ship'] = {
    'length': 150,
    'beam': 25, 
    'speed_range': (10, 20),
    'rcs_base': 800
}
```

### Custom Radar Parameters

```python
# Modify radar configuration
generator.radar_config.frequency_ghz = 5.6  # C-band
generator.radar_config.max_range_m = 75000  # 75 km range
generator.radar_config.tx_power_w = 50000   # 50 kW
```

### Custom Sea State Models

```python
# Customize clutter statistics
def custom_clutter_model(sea_state):
    if sea_state <= 3:
        return generator.clutter_model.weibull_clutter(size, shape=2.0)
    else:
        return generator.clutter_model.k_distribution_clutter(size, shape=1.0)
```

## Applications

This dataset is suitable for:

- **Machine Learning**: Target detection and classification
- **Algorithm Development**: Radar signal processing research
- **Benchmarking**: Performance evaluation of detection algorithms
- **Education**: Teaching radar principles and data science
- **System Testing**: Validation of maritime surveillance systems

## Data Quality

The generated data includes realistic characteristics:

- **Temporal Coherence**: Sequential detections follow physical motion
- **Spatial Coherence**: Consistent with radar geometry
- **Statistical Accuracy**: Matches empirical clutter distributions
- **Physical Realism**: Based on established radar and maritime models
- **Measurement Noise**: Appropriate noise levels for radar systems

## Contributing

To contribute improvements:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

Potential areas for enhancement:
- Additional vessel types or behaviors
- More sophisticated sea clutter models
- Weather effects (rain, fog, etc.)
- Multi-static radar configurations
- Electronic warfare scenarios

## License

MIT License - see LICENSE file for details.

## References

1. Ward, K.D., Tough, R.J.A., & Watts, S. (2013). *Sea Clutter: Scattering, the K-Distribution and Radar Performance*
2. Skolnik, M.I. (2008). *Radar Handbook, Third Edition*
3. Nathanson, F.E., et al. (1999). *Radar Design Principles*
4. IEC 62388:2013 - *Maritime navigation and radiocommunication equipment and systems*

## Support

For questions or issues:
- Create an issue in the repository
- Check the documentation
- Review example notebooks for usage patterns

---

**Note**: This is synthetic data generated using physical models. While realistic, it should be validated against real radar data for production applications.