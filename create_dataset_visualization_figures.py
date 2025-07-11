#!/usr/bin/env python3
"""
Create Nature-Style Dataset Visualization Figures

This script generates publication-quality figures showing representative time series
from both training domains and evaluation datasets to illustrate the diversity
and complexity captured by the foundation model approach.

Figures:
1. Training Domain Diversity (9 subplots)
2. Evaluation Dataset Examples (all available UCR datasets)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from scipy.signal import detrend
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

# Set Nature-style plotting parameters
plt.style.use('default')
sns.set_palette("Set2")

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.linewidth': 0.5,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.transparent': False,
    'savefig.facecolor': 'white'
})

class DatasetVisualizer:
    """Create publication-quality visualizations of training and evaluation datasets"""
    
    def __init__(self):
        self.training_data_dir = Path("../training_datasets")
        self.ucr_data_dir = Path("../ucr_test_datasets")
        self.output_dir = Path("figures")
        self.output_dir.mkdir(exist_ok=True)
        
        # Define colors for different domains (Nature-style palette)
        self.training_colors = {
            'air_quality': '#1f77b4',      # Blue
            'economics': '#ff7f0e',        # Orange  
            'electricity': '#2ca02c',      # Green
            'm4_hourly': '#d62728',        # Red
            'smart_meter': '#9467bd',      # Purple
            'solar': '#8c564b',            # Brown
            'traffic': '#e377c2',          # Pink
            'weather': '#7f7f7f',          # Gray
            'web_traffic': '#bcbd22'       # Olive
        }
        
        # Extended color palette for UCR datasets
        self.ucr_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78'
        ]
    
    def preprocess_for_visualization(self, ts, target_length=512):
        """Apply same preprocessing as training pipeline for consistency"""
        # Handle different input types
        if hasattr(ts, 'values'):
            ts = ts.values
        if not isinstance(ts, np.ndarray):
            ts = np.array(ts)
        
        # Flatten if needed
        if ts.ndim > 1:
            ts = ts.flatten()
        
        # Remove NaNs
        ts = ts[~np.isnan(ts)]
        
        if len(ts) == 0:
            return np.zeros(target_length)
        
        # Detrend
        if len(ts) > 1:
            ts = detrend(ts)
        
        # Outlier clipping
        std = np.std(ts)
        mean = np.mean(ts)
        if std > 1e-6:
            ts = np.clip(ts, mean - 3 * std, mean + 3 * std)
            # Z-score normalization for visualization
            ts = zscore(ts)
        
        # Resample to target length for consistent visualization
        if len(ts) != target_length:
            original_indices = np.linspace(0, 1, len(ts))
            new_indices = np.linspace(0, 1, target_length)
            ts = np.interp(new_indices, original_indices, ts)
        
        return ts
    
    def load_training_domain_examples(self):
        """Load representative examples from each training domain"""
        domain_examples = {}
        
        # Define which datasets represent which domains best
        domain_mapping = {
            'air_quality': 'Air Quality Monitoring',
            'economics': 'Economic Indicators', 
            'electricity': 'Electricity Consumption',
            'm4_hourly': 'M4 Hourly Forecasting',
            'smart_meter': 'Smart Meter Usage',
            'solar': 'Solar Power Generation',
            'traffic': 'Traffic Flow',
            'weather': 'Weather Measurements',
            'web_traffic': 'Web Traffic Logs'
        }
        
        for domain_key, domain_name in domain_mapping.items():
            pq_file = self.training_data_dir / f"{domain_key}.parquet"
            
            if pq_file.exists():
                print(f"Loading {domain_name} from {pq_file}")
                df = pd.read_parquet(pq_file)
                
                # Select a representative time series (not the first, to avoid edge cases)
                idx = min(len(df) // 4, 50)  # Take from first quarter but not beginning
                ts_values = df['target'].iloc[idx]
                
                # Preprocess for visualization
                ts_processed = self.preprocess_for_visualization(ts_values)
                
                domain_examples[domain_name] = {
                    'data': ts_processed,
                    'color': self.training_colors[domain_key],
                    'n_series': len(df)
                }
                print(f"✓ Loaded {domain_name}: {len(df)} series")
        
        return domain_examples
    
    def load_ucr_examples(self):
        """Load representative examples from all available UCR evaluation datasets"""
        ucr_examples = {}
        
        # Get all available UCR datasets, excluding synthetic TwoPatterns
        available_datasets = []
        for item in self.ucr_data_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.') and item.name != 'TwoPatterns':
                available_datasets.append(item.name)
        
        available_datasets.sort()  # Sort for consistent ordering
        print(f"Found {len(available_datasets)} UCR datasets: {available_datasets}")
        
        for i, dataset_name in enumerate(available_datasets):
            dataset_dir = self.ucr_data_dir / dataset_name
            train_path = dataset_dir / f"{dataset_name}_train.pkl"
            
            if train_path.exists():
                print(f"Loading {dataset_name} from {train_path}")
                import pickle
                try:
                    with open(train_path, 'rb') as f:
                        train_data = pickle.load(f)
                    
                    # Extract a representative time series
                    X = train_data['X']
                    
                    # Handle different data formats
                    if hasattr(X, 'iloc'):
                        if len(X.columns) == 1:
                            ts_values = X.iloc[0, 0]
                        else:
                            # Multi-column, concatenate
                            row_series = []
                            for col in range(min(len(X.columns), 3)):  # Limit to avoid too long series
                                series = X.iloc[0, col]
                                if hasattr(series, 'values'):
                                    series = series.values
                                row_series.append(series)
                            ts_values = np.concatenate(row_series)
                    else:
                        ts_values = X[0]
                    
                    # Preprocess for visualization
                    ts_processed = self.preprocess_for_visualization(ts_values)
                    
                    ucr_examples[dataset_name] = {
                        'data': ts_processed,
                        'color': self.ucr_colors[i % len(self.ucr_colors)],
                        'n_samples': len(X)
                    }
                    print(f"✓ Loaded {dataset_name}: {len(X)} samples")
                    
                except Exception as e:
                    print(f"⚠️  Could not load {dataset_name}: {e}")
        
        return ucr_examples
    
    def create_training_domains_figure(self):
        """Create Figure showing all training domain diversity"""
        domain_examples = self.load_training_domain_examples()
        
        # Create figure with 3x3 subplots
        fig, axes = plt.subplots(3, 3, figsize=(12, 9))
        axes = axes.flatten()
        
        time_axis = np.linspace(0, 1, 512)  # Normalized time axis
        
        for i, (domain_name, data) in enumerate(domain_examples.items()):
            ax = axes[i]
            
            # Plot time series with shaded confidence region
            ts_data = data['data']
            
            # Add subtle noise bands to show variability
            noise_band = 0.1
            upper_band = ts_data + noise_band
            lower_band = ts_data - noise_band
            
            # Fill between for visual appeal
            ax.fill_between(time_axis, lower_band, upper_band, 
                           color=data['color'], alpha=0.2)
            
            # Main time series line
            ax.plot(time_axis, ts_data, color=data['color'], 
                   linewidth=1.0, alpha=0.8)
            
            # Styling
            ax.set_title(f"{domain_name}", 
                        fontsize=8, fontweight='bold', pad=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(-3, 3)  # Standardized y-axis
            
            # Clean axes
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=7)
            
            # Only show axis labels on edge plots
            if i >= 6:  # Bottom row
                ax.set_xlabel('Normalized Time', fontsize=7)
            if i % 3 == 0:  # Left column
                ax.set_ylabel('Normalized\nAmplitude', fontsize=7)
        
        plt.tight_layout()
        
        # Save figure in both formats
        output_path_pdf = self.output_dir / "training_domain_diversity.pdf"
        output_path_png = self.output_dir / "training_domain_diversity.png"
        
        plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.savefig(output_path_png, format='png', bbox_inches='tight', 
                   facecolor='white', edgecolor='none', dpi=300)
        
        print(f"✅ Training domains figure saved:")
        print(f"    PDF: {output_path_pdf}")
        print(f"    PNG: {output_path_png}")
        plt.close()
    
    def create_evaluation_datasets_figure(self):
        """Create Figure showing all available UCR evaluation dataset examples"""
        ucr_examples = self.load_ucr_examples()
        
        # Determine grid size based on number of datasets
        n_datasets = len(ucr_examples)
        if n_datasets <= 12:
            rows, cols = 3, 4
        elif n_datasets <= 15:
            rows, cols = 3, 5
        elif n_datasets <= 20:
            rows, cols = 4, 5
        else:
            rows, cols = 5, 5
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        axes = axes.flatten()
        
        time_axis = np.linspace(0, 1, 512)  # Normalized time axis
        
        for i, (dataset_name, data) in enumerate(ucr_examples.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Plot time series
            ts_data = data['data']
            
            # Add subtle background shading
            ax.axhspan(-3, 3, alpha=0.05, color=data['color'])
            
            # Main time series line with slight thickness variation for visual interest
            ax.plot(time_axis, ts_data, color=data['color'], 
                   linewidth=1.2, alpha=0.9)
            
            # Add subtle markers at peaks for visual interest
            peaks = []
            for j in range(1, len(ts_data)-1):
                if ts_data[j] > ts_data[j-1] and ts_data[j] > ts_data[j+1] and ts_data[j] > 1.5:
                    peaks.append(j)
            
            if peaks:
                peak_times = [time_axis[p] for p in peaks[:3]]  # Limit to 3 peaks
                peak_values = [ts_data[p] for p in peaks[:3]]
                ax.scatter(peak_times, peak_values, color=data['color'], 
                          s=15, alpha=0.7, zorder=5)
            
            # Styling
            ax.set_title(f"{dataset_name}", 
                        fontsize=8, fontweight='bold', pad=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(-3, 3)  # Standardized y-axis
            
            # Clean axes
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.2, linewidth=0.5)
            
            # Only show axis labels on edge plots
            if i >= (rows-1)*cols:  # Bottom row
                ax.set_xlabel('Normalized Time', fontsize=7)
            if i % cols == 0:  # Left column
                ax.set_ylabel('Normalized\nAmplitude', fontsize=7)
        
        # Hide unused subplots
        for i in range(len(ucr_examples), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure in both formats
        output_path_pdf = self.output_dir / "evaluation_benchmark_examples.pdf"
        output_path_png = self.output_dir / "evaluation_benchmark_examples.png"
        
        plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.savefig(output_path_png, format='png', bbox_inches='tight', 
                   facecolor='white', edgecolor='none', dpi=300)
        
        print(f"✅ Evaluation datasets figure saved:")
        print(f"    PDF: {output_path_pdf}")
        print(f"    PNG: {output_path_png}")
        plt.close()

def main():
    """Generate dataset visualization figures"""
    print("🎨 Creating Nature-style dataset visualization figures...")
    
    visualizer = DatasetVisualizer()
    
    # Create the two main figures
    print("\n📊 Creating training domains figure...")
    visualizer.create_training_domains_figure()
    
    print("\n📈 Creating evaluation datasets figure...")
    visualizer.create_evaluation_datasets_figure()
    
    print(f"\n✅ All figures saved to: {visualizer.output_dir}")
    print("\nFigures created:")
    print("  - training_domain_diversity.pdf/.png (9 training domains)")
    print("  - evaluation_benchmark_examples.pdf/.png (13 UCR datasets)")

if __name__ == "__main__":
    main() 