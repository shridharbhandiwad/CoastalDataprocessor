# Quick Usage Guide

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Your First Dataset
```bash
# Generate a 1GB dataset with default settings
python maritime_radar_generator.py --size-gb 1.0

# Custom generation
python maritime_radar_generator.py \
    --size-gb 2.0 \
    --num-tracks 2000 \
    --clutter-ratio 0.75 \
    --output-dir my_radar_data
```

### 3. Analyze the Dataset
```bash
# Run comprehensive analysis
python data_analysis_tools.py \
    --dataset-dir maritime_radar_dataset \
    --output-dir analysis_results
```

### 4. Example Usage in Python
```python
# Load and explore data
from data_analysis_tools import MaritimeRadarAnalyzer

analyzer = MaritimeRadarAnalyzer("maritime_radar_dataset")
df = analyzer.load_data('train')

print(f"Dataset shape: {df.shape}")
print(f"Labels: {df['label'].value_counts()}")

# Generate summary
summary = analyzer.generate_summary_report()
print(f"Total detections: {summary['basic_stats']['total_detections']:,}")
```

### 5. Run Examples
```bash
# See all examples in action
python example_usage.py
```

## Dataset Structure

Generated files:
- `train.csv`, `validation.csv`, `test.csv` - CSV format data splits
- `maritime_radar_dataset.h5` - HDF5 format (more efficient)
- `metadata.json` - Dataset information and configuration

## Key Features

- **Physical Models**: Realistic radar equation and sea clutter simulation
- **Multiple Sea States**: 1 (calm) to 6 (very rough)
- **Vessel Types**: 7 different vessel types with realistic characteristics
- **Large Scale**: Can generate GB-scale datasets
- **Ready for ML**: Pre-split and labeled data

## Performance

- Generation rate: ~250,000 detections/second
- 1GB dataset: ~5M detections, ~1000 tracks
- Typical generation time: 1-2 minutes per 100MB

## Next Steps

1. Customize vessel types or radar parameters in the code
2. Use the data for machine learning model training
3. Extend with additional physical effects (weather, etc.)
4. Validate against real radar data for your specific use case