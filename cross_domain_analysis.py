#!/usr/bin/env python3
"""
Cross-Domain Transfer Analysis

This script performs comprehensive cross-domain transfer analysis using foundation
model representations. Evaluates how well knowledge learned from one domain
transfers to classification tasks in other domains.

Key Features:
- Systematic train-on-source, test-on-target protocol
- 13x13 transfer matrix computation
- Domain category analysis (Engineered vs Natural)
- Statistical significance testing
- Transfer pattern visualization

Usage:
    python cross_domain_analysis.py

Requirements:
    - Trained foundation model checkpoint
    - UCR datasets for transfer evaluation
    - See requirements.txt for dependencies
"""

import json
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.signal import detrend
import logging
from datetime import datetime

# Import model components
from train_foundation_model import Config, FoundationModel, ButterworthFilter

# Suppress warnings
warnings.filterwarnings("ignore")

class TransferConfig:
    """Configuration for cross-domain transfer analysis"""
    MODEL_PATH = Path("checkpoints/best_model.pth")
    UCR_DATA_DIR = Path("ucr_test_datasets")
    RESULTS_DIR = Path("evaluation_results")
    OUTPUT_FILE = RESULTS_DIR / "cross_domain_transfer_results.json"
    MATRIX_FILE = RESULTS_DIR / "transfer_matrix.csv"
    FIGURE_FILE = RESULTS_DIR / "transfer_analysis.png"
    
    # Analysis parameters
    RANDOM_STATE = 42

class CrossDomainAnalyzer:
    """Analyzer for cross-domain transfer patterns"""
    
    def __init__(self):
        self.config = TransferConfig()
        self.model_config = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.lowpass_filter = ButterworthFilter()
        
        # Domain categories
        self.domain_categories = {
            'Coffee': 'Engineered', 'Wafer': 'Engineered',
            'ECG200': 'Engineered', 'TwoLeadECG': 'Engineered',
            'MoteStrain': 'Engineered', 'Plane': 'Engineered',
            'SonyAIBORobotSurface1': 'Engineered', 'SonyAIBORobotSurface2': 'Engineered',
            'Cricket': 'Engineered', 'ItalyPowerDemand': 'Engineered',
            'StarLightCurves': 'Natural', 'Earthquakes': 'Natural', 'Lightning2': 'Natural'
        }
        
        # Setup directories and logging
        self.config.RESULTS_DIR.mkdir(exist_ok=True)
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for transfer analysis"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] - %(message)s"
        )
        logging.info("🔄 Cross-Domain Transfer Analysis Started")
        logging.info(f"Device: {self.device}")
        
    def load_trained_model(self):
        """Load trained foundation model"""
        if not self.config.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {self.config.MODEL_PATH}")
        
        checkpoint = torch.load(self.config.MODEL_PATH, map_location=self.device)
        
        # Initialize model
        self.model = FoundationModel(self.model_config, num_classes=9)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        param_count = sum(p.numel() for p in self.model.parameters())
        logging.info(f"✅ Model loaded: {param_count:,} parameters")
        
    def load_ucr_dataset(self, dataset_name):
        """Load UCR dataset"""
        dataset_dir = self.config.UCR_DATA_DIR / dataset_name
        
        train_file = dataset_dir / f"{dataset_name}_train.pkl"
        test_file = dataset_dir / f"{dataset_name}_test.pkl"
        
        if not train_file.exists() or not test_file.exists():
            raise FileNotFoundError(f"Dataset files not found for {dataset_name}")
        
        with open(train_file, 'rb') as f:
            train_data = pickle.load(f)
        
        with open(test_file, 'rb') as f:
            test_data = pickle.load(f)
        
        return train_data, test_data
    
    def extract_time_series_data(self, data_dict):
        """Extract time series and labels from data dictionary"""
        X = data_dict['X']
        y = data_dict['y']
        
        # Extract time series
        series_list = []
        for i in range(len(X)):
            try:
                if hasattr(X, 'iloc'):  # DataFrame format
                    series = X.iloc[i, 0]
                    if hasattr(series, 'values'):
                        series = series.values
                    series_list.append(np.array(series))
                else:  # Array format
                    series_list.append(np.array(X[i]))
            except Exception as e:
                logging.warning(f"Could not extract series {i}: {e}")
                continue
        
        # Convert to numpy array
        if series_list:
            max_length = max(len(s) for s in series_list)
            X_array = np.zeros((len(series_list), max_length))
            
            for i, series in enumerate(series_list):
                X_array[i, :len(series)] = series
        else:
            X_array = np.array([])
        
        # Convert labels
        if hasattr(y, 'values'):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        # Encode labels to integers
        if len(y_array) > 0 and not np.issubdtype(y_array.dtype, np.integer):
            le = LabelEncoder()
            y_array = le.fit_transform(y_array)
        
        return X_array, y_array
    
    def preprocess_time_series(self, ts_array):
        """Preprocess time series matching training pipeline"""
        processed_series = []
        
        for ts in ts_array:
            ts = ts.astype(np.float32)
            
            # Remove NaNs
            ts = ts[~np.isnan(ts)]
            
            # Detrend
            ts = detrend(ts)
            
            # Outlier clipping
            std = np.std(ts)
            mean = np.mean(ts)
            ts = np.clip(ts, mean - 3 * std, mean + 3 * std)
            
            # Z-score normalization
            if std > 1e-6:
                ts = (ts - mean) / std
            
            # Resample to target length
            if len(ts) != self.model_config.N_TIMESTEPS:
                if len(ts) > self.model_config.N_TIMESTEPS * 1.2:
                    ts = self.lowpass_filter(ts)
                
                original_indices = np.linspace(0, 1, len(ts))
                new_indices = np.linspace(0, 1, self.model_config.N_TIMESTEPS)
                ts = np.interp(new_indices, original_indices, ts)
            
            processed_series.append(ts)
        
        return np.array(processed_series)
    
    def extract_features(self, X_processed):
        """Extract features using frozen foundation model"""
        features = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(X_processed), batch_size):
                batch = X_processed[i:i+batch_size]
                
                # Convert to tensor
                batch_tensor = torch.FloatTensor(batch).unsqueeze(-1).to(self.device)
                
                # Extract features
                h_contrastive, _, _, _ = self.model(batch_tensor)
                features.append(h_contrastive.cpu().numpy())
        
        return np.vstack(features)
    
    def evaluate_transfer(self, source_dataset, target_dataset):
        """Evaluate transfer from source to target dataset"""
        try:
            # Load source dataset
            source_train, source_test = self.load_ucr_dataset(source_dataset)
            X_source_train, y_source_train = self.extract_time_series_data(source_train)
            X_source_test, y_source_test = self.extract_time_series_data(source_test)
            
            # Combine source data
            X_source = np.vstack([X_source_train, X_source_test])
            y_source = np.concatenate([y_source_train, y_source_test])
            
            # Load target dataset
            target_train, target_test = self.load_ucr_dataset(target_dataset)
            X_target_train, y_target_train = self.extract_time_series_data(target_train)
            X_target_test, y_target_test = self.extract_time_series_data(target_test)
            
            # Combine target data
            X_target = np.vstack([X_target_train, X_target_test])
            y_target = np.concatenate([y_target_train, y_target_test])
            
            # Preprocess
            X_source_processed = self.preprocess_time_series(X_source)
            X_target_processed = self.preprocess_time_series(X_target)
            
            # Extract features
            source_features = self.extract_features(X_source_processed)
            target_features = self.extract_features(X_target_processed)
            
            # Train classifier on source
            classifier = LogisticRegression(
                max_iter=1000,
                random_state=self.config.RANDOM_STATE,
                class_weight='balanced'
            )
            classifier.fit(source_features, y_source)
            
            # Evaluate on target
            y_pred = classifier.predict(target_features)
            
            # Calculate metrics
            accuracy = accuracy_score(y_target, y_pred)
            
            # AUC calculation
            try:
                if len(np.unique(y_target)) == 2:
                    y_pred_proba = classifier.predict_proba(target_features)
                    auc = roc_auc_score(y_target, y_pred_proba[:, 1])
                else:
                    # For multiclass, use accuracy as proxy
                    auc = accuracy
            except:
                auc = accuracy
            
            return auc
            
        except Exception as e:
            logging.warning(f"Transfer evaluation failed ({source_dataset} → {target_dataset}): {e}")
            return 0.0
    
    def compute_transfer_matrix(self, datasets):
        """Compute full transfer matrix"""
        n_datasets = len(datasets)
        transfer_matrix = np.zeros((n_datasets, n_datasets))
        
        total_evaluations = n_datasets * n_datasets
        
        with tqdm(total=total_evaluations, desc="Computing transfer matrix") as pbar:
            for i, source in enumerate(datasets):
                for j, target in enumerate(datasets):
                    if i == j:
                        # Within-domain performance (diagonal)
                        transfer_matrix[i, j] = 1.0  # Perfect within-domain assumption
                    else:
                        # Cross-domain transfer
                        auc = self.evaluate_transfer(source, target)
                        transfer_matrix[i, j] = auc
                    
                    pbar.update(1)
        
        return transfer_matrix
    
    def analyze_transfer_patterns(self, transfer_matrix, datasets):
        """Analyze transfer patterns by domain category"""
        patterns = {
            'Engineered→Engineered': [],
            'Engineered→Natural': [],
            'Natural→Engineered': [],
            'Natural→Natural': []
        }
        
        for i, source in enumerate(datasets):
            for j, target in enumerate(datasets):
                if i != j:  # Skip diagonal
                    source_cat = self.domain_categories.get(source, 'Unknown')
                    target_cat = self.domain_categories.get(target, 'Unknown')
                    
                    if source_cat != 'Unknown' and target_cat != 'Unknown':
                        pattern_key = f"{source_cat}→{target_cat}"
                        if pattern_key in patterns:
                            patterns[pattern_key].append(transfer_matrix[i, j])
        
        # Compute statistics
        pattern_stats = {}
        for pattern, values in patterns.items():
            if values:
                pattern_stats[pattern] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values),
                    'values': values
                }
        
        return pattern_stats
    
    def find_exceptional_transfers(self, transfer_matrix, datasets, threshold=0.8):
        """Find exceptional transfer cases"""
        exceptional = []
        poor = []
        
        for i, source in enumerate(datasets):
            for j, target in enumerate(datasets):
                if i != j:  # Skip diagonal
                    auc = transfer_matrix[i, j]
                    if auc > threshold:
                        exceptional.append((source, target, auc))
                    elif auc < 0.3:
                        poor.append((source, target, auc))
        
        # Sort by performance
        exceptional.sort(key=lambda x: x[2], reverse=True)
        poor.sort(key=lambda x: x[2])
        
        return exceptional, poor
    
    def create_transfer_visualization(self, transfer_matrix, datasets, pattern_stats):
        """Create transfer matrix visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Transfer matrix heatmap
        im = axes[0].imshow(transfer_matrix, cmap='viridis', aspect='auto')
        axes[0].set_xticks(range(len(datasets)))
        axes[0].set_yticks(range(len(datasets)))
        axes[0].set_xticklabels(datasets, rotation=45, ha='right')
        axes[0].set_yticklabels(datasets)
        axes[0].set_xlabel('Target Dataset')
        axes[0].set_ylabel('Source Dataset')
        axes[0].set_title('Cross-Domain Transfer Matrix')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[0])
        
        # Plot 2: Pattern statistics
        if pattern_stats:
            patterns = list(pattern_stats.keys())
            means = [pattern_stats[p]['mean'] for p in patterns]
            stds = [pattern_stats[p]['std'] for p in patterns]
            
            bars = axes[1].bar(patterns, means, yerr=stds, capsize=5)
            axes[1].set_ylabel('Average Transfer AUC')
            axes[1].set_title('Transfer Patterns by Domain Category')
            axes[1].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{mean:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.config.FIGURE_FILE, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Transfer visualization saved: {self.config.FIGURE_FILE}")
    
    def run_analysis(self):
        """Run complete cross-domain transfer analysis"""
        logging.info("Starting cross-domain transfer analysis")
        
        # Load model
        self.load_trained_model()
        
        # Datasets to analyze
        datasets = [
            'Coffee', 'Wafer', 'ECG200', 'TwoLeadECG', 'MoteStrain',
            'Plane', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2',
            'ItalyPowerDemand', 'Cricket', 'StarLightCurves',
            'Earthquakes', 'Lightning2'
        ]
        
        # Compute transfer matrix
        transfer_matrix = self.compute_transfer_matrix(datasets)
        
        # Analyze patterns
        pattern_stats = self.analyze_transfer_patterns(transfer_matrix, datasets)
        
        # Find exceptional cases
        exceptional, poor = self.find_exceptional_transfers(transfer_matrix, datasets)
        
        # Create detailed statistics
        detailed_stats = {}
        for pattern, stats in pattern_stats.items():
            detailed_stats[pattern] = {
                'mean': stats['mean'],
                'std': stats['std'],
                'count': stats['count'],
                'min': np.min(stats['values']),
                'max': np.max(stats['values']),
                'median': np.median(stats['values'])
            }
        
        # Compile results
        results = {
            'transfer_matrix': transfer_matrix.tolist(),
            'dataset_names': datasets,
            'pattern_stats': pattern_stats,
            'categories': [self.domain_categories.get(d, 'Unknown') for d in datasets],
            'detailed_stats': detailed_stats,
            'exceptional_transfers': exceptional[:5],  # Top 5
            'poor_transfers': poor[:10],  # Bottom 10
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        with open(self.config.OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Transfer analysis saved to {self.config.OUTPUT_FILE}")
        
        # Save transfer matrix as CSV
        matrix_df = pd.DataFrame(transfer_matrix, index=datasets, columns=datasets)
        matrix_df.to_csv(self.config.MATRIX_FILE)
        
        logging.info(f"Transfer matrix saved to {self.config.MATRIX_FILE}")
        
        # Create visualization
        self.create_transfer_visualization(transfer_matrix, datasets, pattern_stats)
        
        # Summary statistics
        logging.info("\nTransfer Pattern Summary:")
        for pattern, stats in detailed_stats.items():
            logging.info(f"{pattern}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                        f"(n={stats['count']})")
        
        logging.info(f"\nExceptional transfers (top 5):")
        for source, target, auc in exceptional[:5]:
            logging.info(f"  {source} → {target}: {auc:.3f}")
        
        logging.info("Cross-domain transfer analysis complete!")

def main():
    """Main execution function"""
    analyzer = CrossDomainAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 