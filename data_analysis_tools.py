#!/usr/bin/env python3
"""
Maritime Radar Dataset Analysis Tools

Provides visualization and analysis utilities for the maritime radar dataset.
Includes statistical analysis, plotting functions, and data exploration tools.

Author: AI Assistant
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import h5py
import json
import os
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MaritimeRadarAnalyzer:
    """Analysis and visualization tools for maritime radar dataset"""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.metadata = self.load_metadata()
        self.data = None
        
    def load_metadata(self) -> Dict:
        """Load dataset metadata"""
        metadata_file = os.path.join(self.dataset_dir, 'metadata.json')
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    def load_data(self, split: str = 'train', format: str = 'csv') -> pd.DataFrame:
        """Load dataset split"""
        if format == 'csv':
            file_path = os.path.join(self.dataset_dir, f'{split}.csv')
            return pd.read_csv(file_path)
        elif format == 'hdf5':
            file_path = os.path.join(self.dataset_dir, 'maritime_radar_dataset.h5')
            with h5py.File(file_path, 'r') as f:
                group = f[split]
                data = {}
                for col in group.keys():
                    if col in ['track_id', 'timestamp', 'label']:
                        data[col] = [s.decode('utf-8') for s in group[col][:]]
                    else:
                        data[col] = group[col][:]
                return pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def generate_summary_report(self) -> Dict:
        """Generate comprehensive dataset summary"""
        if self.data is None:
            self.data = self.load_data('train')
        
        summary = {
            'basic_stats': {
                'total_detections': len(self.data),
                'unique_tracks': self.data['track_id'].nunique(),
                'time_span_hours': self._calculate_time_span(),
                'label_distribution': self.data['label'].value_counts().to_dict()
            },
            'spatial_stats': {
                'range_stats': self.data['range_m'].describe().to_dict(),
                'azimuth_stats': self.data['azimuth_deg'].describe().to_dict(),
                'elevation_stats': self.data['elevation_deg'].describe().to_dict()
            },
            'signal_stats': {
                'doppler_stats': self.data['doppler_ms'].describe().to_dict(),
                'rcs_stats': self.data['rcs_dbsm'].describe().to_dict(),
                'snr_stats': self.data['snr_db'].describe().to_dict()
            },
            'sea_state_distribution': self.data['sea_state'].value_counts().sort_index().to_dict()
        }
        
        return summary
    
    def _calculate_time_span(self) -> float:
        """Calculate time span of dataset in hours"""
        timestamps = pd.to_datetime(self.data['timestamp'])
        return (timestamps.max() - timestamps.min()).total_seconds() / 3600
    
    def plot_dataset_overview(self, save_path: Optional[str] = None):
        """Create comprehensive dataset overview plots"""
        if self.data is None:
            self.data = self.load_data('train')
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Maritime Radar Dataset Overview', fontsize=16, fontweight='bold')
        
        # Label distribution
        axes[0, 0].pie(self.data['label'].value_counts().values, 
                      labels=self.data['label'].value_counts().index,
                      autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Target vs Clutter Distribution')
        
        # Range distribution
        axes[0, 1].hist(self.data['range_m'] / 1000, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Range (km)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Range Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Azimuth distribution
        axes[0, 2].hist(self.data['azimuth_deg'], bins=36, alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('Azimuth (degrees)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Azimuth Distribution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Doppler by label
        clutter_doppler = self.data[self.data['label'] == 'clutter']['doppler_ms']
        target_doppler = self.data[self.data['label'] == 'target']['doppler_ms']
        
        axes[1, 0].hist(clutter_doppler, bins=50, alpha=0.6, label='Clutter', color='red')
        axes[1, 0].hist(target_doppler, bins=50, alpha=0.6, label='Target', color='blue')
        axes[1, 0].set_xlabel('Doppler (m/s)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Doppler Distribution by Label')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # RCS by label
        clutter_rcs = self.data[self.data['label'] == 'clutter']['rcs_dbsm']
        target_rcs = self.data[self.data['label'] == 'target']['rcs_dbsm']
        
        axes[1, 1].hist(clutter_rcs, bins=50, alpha=0.6, label='Clutter', color='red')
        axes[1, 1].hist(target_rcs, bins=50, alpha=0.6, label='Target', color='blue')
        axes[1, 1].set_xlabel('RCS (dBsm)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('RCS Distribution by Label')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # SNR by label
        clutter_snr = self.data[self.data['label'] == 'clutter']['snr_db']
        target_snr = self.data[self.data['label'] == 'target']['snr_db']
        
        axes[1, 2].hist(clutter_snr, bins=50, alpha=0.6, label='Clutter', color='red')
        axes[1, 2].hist(target_snr, bins=50, alpha=0.6, label='Target', color='blue')
        axes[1, 2].set_xlabel('SNR (dB)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('SNR Distribution by Label')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # Sea state distribution
        sea_state_counts = self.data['sea_state'].value_counts().sort_index()
        axes[2, 0].bar(sea_state_counts.index, sea_state_counts.values, alpha=0.7)
        axes[2, 0].set_xlabel('Sea State')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].set_title('Sea State Distribution')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Range vs RCS scatter (sample)
        sample_data = self.data.sample(n=min(5000, len(self.data)))
        colors = ['red' if label == 'clutter' else 'blue' for label in sample_data['label']]
        
        axes[2, 1].scatter(sample_data['range_m'] / 1000, sample_data['rcs_dbsm'], 
                          c=colors, alpha=0.5, s=1)
        axes[2, 1].set_xlabel('Range (km)')
        axes[2, 1].set_ylabel('RCS (dBsm)')
        axes[2, 1].set_title('Range vs RCS (Sample)')
        axes[2, 1].grid(True, alpha=0.3)
        
        # Doppler vs SNR scatter (sample)
        axes[2, 2].scatter(sample_data['doppler_ms'], sample_data['snr_db'], 
                          c=colors, alpha=0.5, s=1)
        axes[2, 2].set_xlabel('Doppler (m/s)')
        axes[2, 2].set_ylabel('SNR (dB)')
        axes[2, 2].set_title('Doppler vs SNR (Sample)')
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_polar_coverage(self, save_path: Optional[str] = None):
        """Create polar plot showing spatial coverage"""
        if self.data is None:
            self.data = self.load_data('train')
        
        # Sample data for visualization
        sample_data = self.data.sample(n=min(10000, len(self.data)))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(projection='polar'))
        
        # Clutter coverage
        clutter_data = sample_data[sample_data['label'] == 'clutter']
        if len(clutter_data) > 0:
            theta_clutter = np.radians(clutter_data['azimuth_deg'])
            r_clutter = clutter_data['range_m'] / 1000
            
            ax1.scatter(theta_clutter, r_clutter, c='red', alpha=0.5, s=1, label='Clutter')
            ax1.set_title('Sea Clutter Spatial Distribution', pad=20)
            ax1.set_ylim(0, 50)
            ax1.set_ylabel('Range (km)', labelpad=30)
        
        # Target coverage
        target_data = sample_data[sample_data['label'] == 'target']
        if len(target_data) > 0:
            theta_target = np.radians(target_data['azimuth_deg'])
            r_target = target_data['range_m'] / 1000
            
            ax2.scatter(theta_target, r_target, c='blue', alpha=0.5, s=1, label='Targets')
            ax2.set_title('Vessel Target Spatial Distribution', pad=20)
            ax2.set_ylim(0, 50)
            ax2.set_ylabel('Range (km)', labelpad=30)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_track_analysis(self, track_ids: Optional[List[str]] = None, save_path: Optional[str] = None):
        """Analyze and plot individual tracks"""
        if self.data is None:
            self.data = self.load_data('train')
        
        if track_ids is None:
            # Select a few interesting tracks
            vessel_tracks = self.data[self.data['label'] == 'target']['track_id'].unique()
            track_ids = np.random.choice(vessel_tracks, size=min(5, len(vessel_tracks)), replace=False)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Individual Track Analysis', fontsize=16, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(track_ids)))
        
        for i, track_id in enumerate(track_ids):
            track_data = self.data[self.data['track_id'] == track_id].sort_values('timestamp')
            
            if len(track_data) == 0:
                continue
            
            color = colors[i % len(colors)]
            
            # Convert to Cartesian for trajectory plot
            x = track_data['range_m'] * np.cos(np.radians(track_data['azimuth_deg']))
            y = track_data['range_m'] * np.sin(np.radians(track_data['azimuth_deg']))
            
            # Trajectory
            axes[0, 0].plot(x / 1000, y / 1000, 'o-', color=color, markersize=2, 
                           linewidth=1, alpha=0.7, label=f'{track_id[:15]}...')
            
            # Range over time
            timestamps = pd.to_datetime(track_data['timestamp'])
            time_minutes = (timestamps - timestamps.iloc[0]).dt.total_seconds() / 60
            axes[0, 1].plot(time_minutes, track_data['range_m'] / 1000, 'o-', 
                           color=color, markersize=2, linewidth=1, alpha=0.7)
            
            # Doppler over time
            axes[0, 2].plot(time_minutes, track_data['doppler_ms'], 'o-', 
                           color=color, markersize=2, linewidth=1, alpha=0.7)
            
            # RCS over time
            axes[1, 0].plot(time_minutes, track_data['rcs_dbsm'], 'o-', 
                           color=color, markersize=2, linewidth=1, alpha=0.7)
            
            # SNR over time
            axes[1, 1].plot(time_minutes, track_data['snr_db'], 'o-', 
                           color=color, markersize=2, linewidth=1, alpha=0.7)
            
            # Azimuth over time
            axes[1, 2].plot(time_minutes, track_data['azimuth_deg'], 'o-', 
                           color=color, markersize=2, linewidth=1, alpha=0.7)
        
        # Set labels and titles
        axes[0, 0].set_xlabel('X (km)')
        axes[0, 0].set_ylabel('Y (km)')
        axes[0, 0].set_title('Track Trajectories')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].axis('equal')
        
        axes[0, 1].set_xlabel('Time (minutes)')
        axes[0, 1].set_ylabel('Range (km)')
        axes[0, 1].set_title('Range vs Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].set_xlabel('Time (minutes)')
        axes[0, 2].set_ylabel('Doppler (m/s)')
        axes[0, 2].set_title('Doppler vs Time')
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Time (minutes)')
        axes[1, 0].set_ylabel('RCS (dBsm)')
        axes[1, 0].set_title('RCS vs Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('Time (minutes)')
        axes[1, 1].set_ylabel('SNR (dB)')
        axes[1, 1].set_title('SNR vs Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].set_xlabel('Time (minutes)')
        axes[1, 2].set_ylabel('Azimuth (degrees)')
        axes[1, 2].set_title('Azimuth vs Time')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(self, save_path: Optional[str] = None):
        """Create interactive Plotly dashboard"""
        if self.data is None:
            self.data = self.load_data('train')
        
        # Sample data for performance
        sample_data = self.data.sample(n=min(20000, len(self.data)))
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Range vs Azimuth', 'Doppler vs RCS', 
                           'SNR Distribution', 'Sea State Analysis',
                           'Range vs SNR', 'Temporal Distribution'],
            specs=[[{"type": "scatterpolar"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # Color mapping
        color_map = {'clutter': 'red', 'target': 'blue'}
        colors = [color_map[label] for label in sample_data['label']]
        
        # Polar scatter plot
        fig.add_trace(
            go.Scatterpolar(
                r=sample_data['range_m'] / 1000,
                theta=sample_data['azimuth_deg'],
                mode='markers',
                marker=dict(color=colors, size=3, opacity=0.6),
                name='Detections'
            ),
            row=1, col=1
        )
        
        # Doppler vs RCS
        fig.add_trace(
            go.Scatter(
                x=sample_data['doppler_ms'],
                y=sample_data['rcs_dbsm'],
                mode='markers',
                marker=dict(color=colors, size=3, opacity=0.6),
                name='Detections'
            ),
            row=1, col=2
        )
        
        # SNR histograms
        for label, color in color_map.items():
            data_subset = sample_data[sample_data['label'] == label]
            fig.add_trace(
                go.Histogram(
                    x=data_subset['snr_db'],
                    name=f'{label.title()} SNR',
                    marker_color=color,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Sea state analysis
        sea_state_counts = self.data.groupby(['sea_state', 'label']).size().unstack(fill_value=0)
        
        for label, color in color_map.items():
            if label in sea_state_counts.columns:
                fig.add_trace(
                    go.Bar(
                        x=sea_state_counts.index,
                        y=sea_state_counts[label],
                        name=f'{label.title()}',
                        marker_color=color,
                        opacity=0.7
                    ),
                    row=2, col=2
                )
        
        # Range vs SNR
        fig.add_trace(
            go.Scatter(
                x=sample_data['range_m'] / 1000,
                y=sample_data['snr_db'],
                mode='markers',
                marker=dict(color=colors, size=3, opacity=0.6),
                name='Detections'
            ),
            row=3, col=1
        )
        
        # Temporal distribution
        timestamps = pd.to_datetime(sample_data['timestamp'])
        fig.add_trace(
            go.Histogram(
                x=timestamps.dt.hour,
                name='Detections by Hour',
                marker_color='green',
                opacity=0.7
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Maritime Radar Dataset Interactive Dashboard",
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Doppler (m/s)", row=1, col=2)
        fig.update_yaxes(title_text="RCS (dBsm)", row=1, col=2)
        
        fig.update_xaxes(title_text="SNR (dB)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        fig.update_xaxes(title_text="Sea State", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        fig.update_xaxes(title_text="Range (km)", row=3, col=1)
        fig.update_yaxes(title_text="SNR (dB)", row=3, col=1)
        
        fig.update_xaxes(title_text="Hour of Day", row=3, col=2)
        fig.update_yaxes(title_text="Frequency", row=3, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def benchmark_classification(self) -> Dict:
        """Run basic classification benchmark"""
        if self.data is None:
            self.data = self.load_data('train')
        
        # Prepare features
        feature_columns = ['range_m', 'azimuth_deg', 'elevation_deg', 'doppler_ms', 
                          'rcs_dbsm', 'snr_db', 'sea_state']
        
        X = self.data[feature_columns]
        y = self.data['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = rf.predict(X_test_scaled)
        
        # Metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Feature importance
        feature_importance = dict(zip(feature_columns, rf.feature_importances_))
        
        return {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'feature_importance': feature_importance,
            'test_accuracy': rf.score(X_test_scaled, y_test)
        }

def main():
    """Example usage of the analysis tools"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze maritime radar dataset')
    parser.add_argument('--dataset-dir', type=str, default='maritime_radar_dataset',
                       help='Dataset directory')
    parser.add_argument('--output-dir', type=str, default='analysis_output',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = MaritimeRadarAnalyzer(args.dataset_dir)
    
    # Generate summary report
    print("Generating summary report...")
    summary = analyzer.generate_summary_report()
    
    with open(os.path.join(args.output_dir, 'summary_report.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Dataset Summary:")
    print(f"  Total detections: {summary['basic_stats']['total_detections']:,}")
    print(f"  Unique tracks: {summary['basic_stats']['unique_tracks']:,}")
    print(f"  Time span: {summary['basic_stats']['time_span_hours']:.1f} hours")
    print(f"  Label distribution: {summary['basic_stats']['label_distribution']}")
    
    # Create visualizations
    print("Creating overview plots...")
    analyzer.plot_dataset_overview(os.path.join(args.output_dir, 'dataset_overview.png'))
    
    print("Creating polar coverage plots...")
    analyzer.plot_polar_coverage(os.path.join(args.output_dir, 'polar_coverage.png'))
    
    print("Creating track analysis...")
    analyzer.plot_track_analysis(save_path=os.path.join(args.output_dir, 'track_analysis.png'))
    
    print("Creating interactive dashboard...")
    analyzer.create_interactive_dashboard(os.path.join(args.output_dir, 'interactive_dashboard.html'))
    
    # Run classification benchmark
    print("Running classification benchmark...")
    benchmark_results = analyzer.benchmark_classification()
    
    with open(os.path.join(args.output_dir, 'benchmark_results.json'), 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    print(f"Classification accuracy: {benchmark_results['test_accuracy']:.3f}")
    print("Analysis complete! Check the output directory for results.")

if __name__ == "__main__":
    main()