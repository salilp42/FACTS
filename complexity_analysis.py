#!/usr/bin/env python3
"""
Complexity Analysis for Time Series Datasets

This script computes various complexity metrics for time series data to understand
the relationship between signal complexity and model performance. Implements
permutation entropy, spectral entropy, and other complexity measures.

Key Features:
- Permutation entropy using Bandt-Pompe method
- Spectral entropy from power spectral density
- Statistical complexity measures
- Correlation analysis with model performance
- Robust handling of different time series formats

Usage:
    python complexity_analysis.py

Requirements:
    - UCR datasets in ucr_test_datasets/ directory
    - Foundation model evaluation results
    - See requirements.txt for dependencies
"""

import numpy as np
import pandas as pd
import json
import pickle
import warnings
from pathlib import Path
from scipy import stats
from scipy.signal import welch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

class ComplexityConfig:
    """Configuration for complexity analysis"""
    UCR_DATA_DIR = Path("ucr_test_datasets")
    RESULTS_DIR = Path("evaluation_results")
    OUTPUT_FILE = RESULTS_DIR / "complexity_analysis.json"
    CORRELATIONS_FILE = RESULTS_DIR / "complexity_correlations.json"
    FIGURE_FILE = RESULTS_DIR / "complexity_correlation.png"
    
    # Analysis parameters
    PERMUTATION_ORDER = 3
    PERMUTATION_DELAY = 1
    MAX_SAMPLES_PER_DATASET = 100
    RANDOM_STATE = 42

class ComplexityAnalyzer:
    """Analyzer for time series complexity metrics"""
    
    def __init__(self):
        self.config = ComplexityConfig()
        self.config.RESULTS_DIR.mkdir(exist_ok=True)
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for complexity analysis"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] - %(message)s"
        )
        logging.info("🔬 Complexity Analysis Started")
    
    def permutation_entropy(self, time_series, order=None, delay=None, normalize=True):
        """
        Calculate permutation entropy using Bandt-Pompe method
        
        Args:
            time_series: 1D array of time series values
            order: Embedding dimension (default: 3)
            delay: Time delay (default: 1)
            normalize: Whether to normalize by maximum entropy
            
        Returns:
            Permutation entropy value
        """
        if order is None:
            order = self.config.PERMUTATION_ORDER
        if delay is None:
            delay = self.config.PERMUTATION_DELAY
            
        time_series = np.array(time_series).flatten()
        
        if len(time_series) < order:
            return np.nan
        
        try:
            # Create ordinal patterns
            patterns = []
            for i in range(len(time_series) - order + 1):
                segment = time_series[i:i + order]
                pattern = tuple(np.argsort(segment))
                patterns.append(pattern)
            
            # Count pattern frequencies
            unique_patterns, counts = np.unique(patterns, return_counts=True, axis=0)
            
            # Calculate entropy
            probabilities = counts / len(patterns)
            pe = -np.sum(probabilities * np.log2(probabilities))
            
            if normalize:
                max_entropy = np.log2(np.math.factorial(order))
                pe = pe / max_entropy if max_entropy > 0 else 0
            
            return pe
            
        except Exception as e:
            logging.warning(f"Permutation entropy calculation failed: {e}")
            return np.nan
    
    def spectral_entropy(self, time_series):
        """
        Calculate spectral entropy from power spectral density
        
        Args:
            time_series: 1D array of time series values
            
        Returns:
            Spectral entropy value
        """
        try:
            time_series = np.array(time_series).flatten()
            
            # Remove DC component
            time_series = time_series - np.mean(time_series)
            
            # Compute power spectral density using Welch's method
            freqs, psd = welch(time_series, nperseg=min(256, len(time_series)//4))
            
            # Normalize to probability distribution
            psd_norm = psd / np.sum(psd)
            psd_norm = psd_norm[psd_norm > 1e-12]  # Avoid log(0)
            
            # Calculate entropy
            se = -np.sum(psd_norm * np.log2(psd_norm))
            
            return se
            
        except Exception as e:
            logging.warning(f"Spectral entropy calculation failed: {e}")
            return np.nan
    
    def statistical_complexity(self, time_series):
        """
        Calculate statistical complexity measures
        
        Args:
            time_series: 1D array of time series values
            
        Returns:
            Dictionary of complexity metrics
        """
        try:
            time_series = np.array(time_series).flatten()
            
            # Basic statistics
            mean_val = np.mean(time_series)
            std_val = np.std(time_series)
            
            # Coefficient of variation
            cv = std_val / (np.abs(mean_val) + 1e-12)
            
            # Normalized range
            range_norm = (np.max(time_series) - np.min(time_series)) / (std_val + 1e-12)
            
            # Lag-1 autocorrelation
            if len(time_series) > 1:
                autocorr = np.corrcoef(time_series[:-1], time_series[1:])[0, 1]
                if np.isnan(autocorr):
                    autocorr = 0.0
            else:
                autocorr = 0.0
            
            return {
                'std_deviation': std_val,
                'coefficient_variation': cv,
                'range_normalized': range_norm,
                'autocorr_lag1': autocorr
            }
            
        except Exception as e:
            logging.warning(f"Statistical complexity calculation failed: {e}")
            return {
                'std_deviation': np.nan,
                'coefficient_variation': np.nan,
                'range_normalized': np.nan,
                'autocorr_lag1': np.nan
            }
    
    def load_ucr_dataset(self, dataset_name):
        """Load UCR dataset from pickle files"""
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
    
    def extract_time_series_safely(self, data_dict):
        """Safely extract time series from various data formats"""
        X = data_dict['X']
        
        series_list = []
        for i in range(len(X)):
            try:
                if hasattr(X, 'iloc'):  # DataFrame format
                    series = X.iloc[i, 0]
                    if hasattr(series, 'values'):
                        series = series.values
                    series_array = np.array(series).flatten()
                else:  # Array format
                    series_array = np.array(X[i]).flatten()
                
                if len(series_array) > 3:  # Minimum length for complexity
                    series_list.append(series_array)
                    
            except Exception as e:
                logging.warning(f"Could not extract series {i}: {e}")
                continue
        
        return series_list
    
    def load_foundation_model_results(self):
        """Load foundation model evaluation results"""
        results_file = self.config.RESULTS_DIR / "foundation_model_evaluation_results.json"
        
        if not results_file.exists():
            raise FileNotFoundError(f"Foundation model results not found: {results_file}")
        
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Convert to dataset-keyed dictionary
        results = {}
        for dataset_result in data['dataset_results']:
            dataset_name = dataset_result['dataset']
            results[dataset_name] = {
                'auc_roc': dataset_result['auc_roc'],
                'accuracy': dataset_result['accuracy'],
                'f1_score': dataset_result['f1_score']
            }
        
        return results
    
    def analyze_dataset_complexity(self, dataset_name):
        """Analyze complexity metrics for a single dataset"""
        logging.info(f"Analyzing complexity for {dataset_name}")
        
        try:
            # Load dataset
            train_data, test_data = self.load_ucr_dataset(dataset_name)
            
            # Extract time series
            train_series = self.extract_time_series_safely(train_data)
            test_series = self.extract_time_series_safely(test_data)
            
            all_series = train_series + test_series
            
            if not all_series:
                logging.warning(f"No valid series found for {dataset_name}")
                return None
            
            # Sample for efficiency
            max_samples = self.config.MAX_SAMPLES_PER_DATASET
            if len(all_series) > max_samples:
                np.random.seed(self.config.RANDOM_STATE)
                indices = np.random.choice(len(all_series), max_samples, replace=False)
                sample_series = [all_series[i] for i in indices]
            else:
                sample_series = all_series
            
            # Compute complexity metrics
            pe_values = []
            se_values = []
            stat_metrics = []
            
            for series in sample_series:
                pe = self.permutation_entropy(series)
                se = self.spectral_entropy(series)
                stats_dict = self.statistical_complexity(series)
                
                if not np.isnan(pe):
                    pe_values.append(pe)
                if not np.isnan(se):
                    se_values.append(se)
                stat_metrics.append(stats_dict)
            
            # Aggregate results
            dataset_metrics = {
                'dataset': dataset_name,
                'n_series_analyzed': len(sample_series),
                'n_series_total': len(all_series),
                'series_length': len(sample_series[0]) if sample_series else 0,
                
                # Permutation entropy
                'permutation_entropy_mean': np.mean(pe_values) if pe_values else np.nan,
                'permutation_entropy_std': np.std(pe_values) if pe_values else np.nan,
                'permutation_entropy_median': np.median(pe_values) if pe_values else np.nan,
                
                # Spectral entropy
                'spectral_entropy_mean': np.mean(se_values) if se_values else np.nan,
                'spectral_entropy_std': np.std(se_values) if se_values else np.nan,
                'spectral_entropy_median': np.median(se_values) if se_values else np.nan,
            }
            
            # Add statistical complexity metrics
            if stat_metrics:
                for key in stat_metrics[0].keys():
                    values = [m[key] for m in stat_metrics if not np.isnan(m[key])]
                    if values:
                        dataset_metrics[f"{key}_mean"] = np.mean(values)
                        dataset_metrics[f"{key}_std"] = np.std(values)
                        dataset_metrics[f"{key}_median"] = np.median(values)
                    else:
                        dataset_metrics[f"{key}_mean"] = np.nan
                        dataset_metrics[f"{key}_std"] = np.nan
                        dataset_metrics[f"{key}_median"] = np.nan
            
            logging.info(f"Complexity analysis complete for {dataset_name}")
            return dataset_metrics
            
        except Exception as e:
            logging.error(f"Error analyzing {dataset_name}: {e}")
            return None
    
    def compute_correlations(self, complexity_results, performance_results):
        """Compute correlations between complexity and performance"""
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(complexity_results)
        
        # Add performance metrics
        for idx, row in df.iterrows():
            dataset_name = row['dataset']
            if dataset_name in performance_results:
                df.loc[idx, 'foundation_auc'] = performance_results[dataset_name]['auc_roc']
                df.loc[idx, 'foundation_accuracy'] = performance_results[dataset_name]['accuracy']
                df.loc[idx, 'foundation_f1'] = performance_results[dataset_name]['f1_score']
        
        # Complexity metrics to analyze
        complexity_metrics = [
            'permutation_entropy_mean', 'spectral_entropy_mean',
            'std_deviation_mean', 'coefficient_variation_mean'
        ]
        
        correlations = {}
        for metric in complexity_metrics:
            if metric in df.columns and 'foundation_auc' in df.columns:
                valid_data = df[[metric, 'foundation_auc']].dropna()
                if len(valid_data) > 3:
                    corr, p_value = stats.pearsonr(valid_data[metric], valid_data['foundation_auc'])
                    correlations[metric] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'n_samples': len(valid_data)
                    }
        
        return correlations, df
    
    def create_correlation_figure(self, df, correlations):
        """Create correlation visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Define domain categories
        domain_categories = {
            'Coffee': 'Manufacturing', 'Wafer': 'Manufacturing',
            'ECG200': 'Medical', 'TwoLeadECG': 'Medical',
            'MoteStrain': 'Engineering', 'Plane': 'Engineering',
            'SonyAIBORobotSurface1': 'Engineering', 'SonyAIBORobotSurface2': 'Engineering',
            'Cricket': 'Engineering', 'StarLightCurves': 'Natural',
            'Earthquakes': 'Natural', 'Lightning2': 'Natural',
            'ItalyPowerDemand': 'Transportation'
        }
        
        df['domain_category'] = df['dataset'].map(domain_categories)
        
        # Domain colors
        domain_colors = {
            'Manufacturing': '#1f77b4', 'Medical': '#ff7f0e', 'Engineering': '#2ca02c',
            'Natural': '#d62728', 'Transportation': '#9467bd'
        }
        
        # Plot 1: Spectral Entropy vs Performance
        if 'spectral_entropy_mean' in df.columns and 'foundation_auc' in df.columns:
            for domain, color in domain_colors.items():
                domain_data = df[df['domain_category'] == domain]
                if len(domain_data) > 0:
                    axes[0].scatter(domain_data['spectral_entropy_mean'], 
                                   domain_data['foundation_auc'],
                                   c=color, label=domain, alpha=0.7, s=100, 
                                   edgecolors='black', linewidth=1)
            
            if 'spectral_entropy_mean' in correlations:
                corr_data = correlations['spectral_entropy_mean']
                axes[0].text(0.05, 0.95, f'r = {corr_data["correlation"]:.3f}\np = {corr_data["p_value"]:.3f}', 
                            transform=axes[0].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            axes[0].set_xlabel('Spectral Entropy', fontsize=12)
            axes[0].set_ylabel('Foundation Model AUC', fontsize=12)
            axes[0].set_title('Complexity vs Performance', fontsize=14)
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
        
        # Plot 2: Permutation Entropy by Domain
        if 'permutation_entropy_mean' in df.columns:
            pe_by_domain = []
            domain_labels = []
            
            for domain in domain_colors.keys():
                domain_data = df[df['domain_category'] == domain]
                if len(domain_data) > 0:
                    pe_values = domain_data['permutation_entropy_mean'].dropna().values
                    if len(pe_values) > 0:
                        pe_by_domain.extend(pe_values)
                        domain_labels.extend([domain] * len(pe_values))
            
            if pe_by_domain:
                pe_df = pd.DataFrame({'complexity': pe_by_domain, 'domain': domain_labels})
                sns.boxplot(data=pe_df, x='domain', y='complexity', ax=axes[1])
                axes[1].set_title('Permutation Entropy by Domain', fontsize=14)
                axes[1].set_xlabel('Domain Category', fontsize=12)
                axes[1].set_ylabel('Permutation Entropy', fontsize=12)
                axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.config.FIGURE_FILE, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Correlation figure saved: {self.config.FIGURE_FILE}")
    
    def run_analysis(self):
        """Run complete complexity analysis"""
        logging.info("Starting complexity analysis")
        
        # Load foundation model results
        try:
            performance_results = self.load_foundation_model_results()
            logging.info(f"Loaded performance results for {len(performance_results)} datasets")
        except Exception as e:
            logging.error(f"Could not load performance results: {e}")
            return
        
        # UCR datasets to analyze
        datasets = [
            'Coffee', 'Cricket', 'Earthquakes', 'ECG200', 'ItalyPowerDemand',
            'Lightning2', 'MoteStrain', 'Plane', 'SonyAIBORobotSurface1',
            'SonyAIBORobotSurface2', 'StarLightCurves', 'TwoLeadECG', 'Wafer'
        ]
        
        complexity_results = []
        
        for dataset_name in tqdm(datasets, desc="Analyzing datasets"):
            result = self.analyze_dataset_complexity(dataset_name)
            if result is not None:
                # Add performance metrics
                if dataset_name in performance_results:
                    result['foundation_auc'] = performance_results[dataset_name]['auc_roc']
                    result['foundation_accuracy'] = performance_results[dataset_name]['accuracy']
                    result['foundation_f1'] = performance_results[dataset_name]['f1_score']
                
                complexity_results.append(result)
        
        # Save complexity results
        with open(self.config.OUTPUT_FILE, 'w') as f:
            json.dump(complexity_results, f, indent=2)
        
        logging.info(f"Complexity analysis saved to {self.config.OUTPUT_FILE}")
        
        # Compute correlations
        if len(complexity_results) > 5:
            correlations, df = self.compute_correlations(complexity_results, performance_results)
            
            # Save correlations
            with open(self.config.CORRELATIONS_FILE, 'w') as f:
                json.dump(correlations, f, indent=2)
            
            # Print summary
            logging.info("\nComplexity-Performance Correlations:")
            for metric, corr_data in correlations.items():
                significance = "**" if corr_data['p_value'] < 0.05 else ""
                logging.info(f"{metric}: r={corr_data['correlation']:.3f}, "
                           f"p={corr_data['p_value']:.3f} (n={corr_data['n_samples']}) {significance}")
            
            # Create visualization
            self.create_correlation_figure(df, correlations)
            
            logging.info("Complexity analysis complete!")
        else:
            logging.warning("Insufficient data for correlation analysis")

def main():
    """Main execution function"""
    analyzer = ComplexityAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 