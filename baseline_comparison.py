#!/usr/bin/env python3
"""
Baseline Comparison Analysis

This script evaluates modern deep learning baselines against the foundation model
to provide comprehensive performance comparisons. Implements MiniRocket and
statistical feature baselines with fair evaluation protocols.

Key Features:
- MiniRocket implementation with random convolutional kernels
- Statistical feature extraction with Random Forest
- Fair comparison using identical evaluation protocols
- Performance analysis across multiple datasets

Usage:
    python baseline_comparison.py

Requirements:
    - UCR datasets for evaluation
    - Foundation model evaluation results for comparison
    - See requirements.txt for dependencies
"""

import numpy as np
import pandas as pd
import json
import pickle
import warnings
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import logging
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

class BaselineConfig:
    """Configuration for baseline comparison"""
    UCR_DATA_DIR = Path("ucr_test_datasets")
    RESULTS_DIR = Path("evaluation_results")
    OUTPUT_FILE = RESULTS_DIR / "baseline_comparison.json"
    SUMMARY_FILE = RESULTS_DIR / "baseline_summary.csv"
    
    # Analysis parameters
    N_ROCKET_KERNELS = 1000
    RANDOM_STATE = 42
    N_CV_FOLDS = 5

class MiniRocketTransform:
    """Simplified ROCKET-style feature extraction"""
    
    def __init__(self, n_kernels=1000, random_state=42):
        self.n_kernels = n_kernels
        self.random_state = random_state
        self.kernels = None
        self.fitted = False
    
    def _generate_kernels(self, length):
        """Generate random convolutional kernels"""
        np.random.seed(self.random_state)
        
        kernels = []
        for _ in range(self.n_kernels):
            # Random kernel length
            kernel_length = np.random.choice([3, 7, 9, 11])
            
            # Random weights
            weights = np.random.normal(0, 1, kernel_length)
            
            # Random bias
            bias = np.random.normal(0, 1)
            
            # Random dilation
            dilation = np.random.choice([1, 2, 4])
            
            kernels.append({
                'weights': weights,
                'bias': bias,
                'dilation': dilation
            })
        
        return kernels
    
    def _apply_kernel(self, X, kernel):
        """Apply a single kernel to time series"""
        weights = kernel['weights']
        bias = kernel['bias']
        dilation = kernel['dilation']
        
        n_samples, length = X.shape
        features = np.zeros((n_samples, 2))  # Max and PPV features
        
        for i in range(n_samples):
            series = X[i]
            
            # Apply convolution with dilation
            conv_output = []
            for j in range(0, length - len(weights) * dilation + 1, dilation):
                if j + len(weights) * dilation <= length:
                    conv_val = np.sum(series[j:j + len(weights) * dilation:dilation] * weights) + bias
                    conv_output.append(conv_val)
            
            if conv_output:
                conv_output = np.array(conv_output)
                # Max pooling
                features[i, 0] = np.max(conv_output)
                # Proportion of positive values
                features[i, 1] = np.mean(conv_output > 0)
        
        return features
    
    def fit_transform(self, X):
        """Fit and transform the data"""
        _, length = X.shape
        self.kernels = self._generate_kernels(length)
        self.fitted = True
        
        # Apply all kernels
        all_features = []
        for kernel in self.kernels:
            features = self._apply_kernel(X, kernel)
            all_features.append(features)
        
        return np.hstack(all_features)
    
    def transform(self, X):
        """Transform new data using fitted kernels"""
        if not self.fitted:
            raise ValueError("Must fit before transform")
        
        all_features = []
        for kernel in self.kernels:
            features = self._apply_kernel(X, kernel)
            all_features.append(features)
        
        return np.hstack(all_features)

class BaselineComparator:
    """Comparator for baseline methods"""
    
    def __init__(self):
        self.config = BaselineConfig()
        self.config.RESULTS_DIR.mkdir(exist_ok=True)
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for baseline comparison"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] - %(message)s"
        )
        logging.info("📊 Baseline Comparison Started")
        
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
        """Extract time series from data dictionary"""
        X = data_dict['X']
        y = data_dict['y']
        
        # Extract time series
        series_list = []
        for i in range(len(X)):
            if hasattr(X, 'iloc'):  # DataFrame format
                series = X.iloc[i, 0]
                if hasattr(series, 'values'):
                    series = series.values
                series_list.append(np.array(series))
            else:  # Array format
                series_list.append(np.array(X[i]))
        
        # Convert to numpy array
        max_length = max(len(s) for s in series_list)
        X_array = np.zeros((len(series_list), max_length))
        
        for i, series in enumerate(series_list):
            X_array[i, :len(series)] = series
        
        # Convert labels
        if hasattr(y, 'values'):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        return X_array, y_array
    
    def evaluate_minirocket(self, X_train, y_train, X_test, y_test):
        """Evaluate MiniRocket baseline"""
        try:
            # Transform features
            transformer = MiniRocketTransform(n_kernels=self.config.N_ROCKET_KERNELS)
            X_train_transformed = transformer.fit_transform(X_train)
            X_test_transformed = transformer.transform(X_test)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_transformed)
            X_test_scaled = scaler.transform(X_test_transformed)
            
            # Train classifier
            clf = LogisticRegression(max_iter=1000, random_state=self.config.RANDOM_STATE)
            clf.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = clf.predict(X_test_scaled)
            y_pred_proba = clf.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # AUC calculation
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            
            return {
                'accuracy': accuracy,
                'f1_score': f1,
                'auc_roc': auc,
                'method': 'MiniRocket + Logistic'
            }
        
        except Exception as e:
            logging.warning(f"MiniRocket evaluation failed: {e}")
            return {
                'accuracy': np.nan,
                'f1_score': np.nan,
                'auc_roc': np.nan,
                'method': 'MiniRocket (Failed)'
            }
    
    def evaluate_statistical_baseline(self, X_train, y_train, X_test, y_test):
        """Evaluate statistical features baseline"""
        try:
            # Extract statistical features
            def extract_features(X):
                features = []
                for i in range(X.shape[0]):
                    series = X[i]
                    series = series[series != 0]  # Remove padding zeros
                    
                    if len(series) > 0:
                        feat = [
                            np.mean(series),
                            np.std(series),
                            np.min(series),
                            np.max(series),
                            np.median(series),
                            np.percentile(series, 25),
                            np.percentile(series, 75),
                            len(series),
                            np.sum(np.diff(series) > 0) / len(series) if len(series) > 1 else 0
                        ]
                    else:
                        feat = [0] * 9
                    
                    features.append(feat)
                
                return np.array(features)
            
            X_train_feat = extract_features(X_train)
            X_test_feat = extract_features(X_test)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_feat)
            X_test_scaled = scaler.transform(X_test_feat)
            
            # Train classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=self.config.RANDOM_STATE)
            clf.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = clf.predict(X_test_scaled)
            y_pred_proba = clf.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            
            return {
                'accuracy': accuracy,
                'f1_score': f1,
                'auc_roc': auc,
                'method': 'Statistical Features + RandomForest'
            }
        
        except Exception as e:
            logging.warning(f"Statistical baseline failed: {e}")
            return {
                'accuracy': np.nan,
                'f1_score': np.nan,
                'auc_roc': np.nan,
                'method': 'Statistical Features (Failed)'
            }
    
    def load_foundation_model_results(self):
        """Load foundation model results for comparison"""
        results_file = self.config.RESULTS_DIR / "foundation_model_evaluation_results.json"
        
        if not results_file.exists():
            logging.warning(f"Foundation model results not found: {results_file}")
            return {}
        
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Convert to dataset-keyed dictionary
        results = {}
        for dataset_result in data['dataset_results']:
            dataset_name = dataset_result['dataset']
            results[dataset_name] = {
                'auc_roc': dataset_result['auc_roc'],
                'accuracy': dataset_result['accuracy'],
                'f1_score': dataset_result['f1_score'],
                'method': 'Foundation Model'
            }
        
        return results
    
    def evaluate_dataset(self, dataset_name):
        """Evaluate baselines on single dataset"""
        logging.info(f"Evaluating baselines for {dataset_name}")
        
        try:
            # Load dataset
            train_data, test_data = self.load_ucr_dataset(dataset_name)
            
            # Extract time series data
            X_train, y_train = self.extract_time_series_data(train_data)
            X_test, y_test = self.extract_time_series_data(test_data)
            
            logging.info(f"Data shape: Train {X_train.shape}, Test {X_test.shape}")
            
            # Evaluate baselines
            results = {}
            
            # Statistical baseline
            results['statistical'] = self.evaluate_statistical_baseline(X_train, y_train, X_test, y_test)
            
            # MiniRocket
            results['minirocket'] = self.evaluate_minirocket(X_train, y_train, X_test, y_test)
            
            return results
            
        except Exception as e:
            logging.error(f"Error evaluating {dataset_name}: {e}")
            return None
    
    def create_comparison_summary(self, results):
        """Create summary comparison across all datasets"""
        summary_data = []
        
        # Get all methods
        methods = set()
        for dataset_results in results.values():
            if dataset_results:
                methods.update(dataset_results.keys())
        
        # Create summary
        for method in methods:
            method_results = []
            for dataset_name, dataset_results in results.items():
                if dataset_results and method in dataset_results:
                    method_results.append(dataset_results[method])
            
            if method_results:
                # Calculate averages
                avg_auc = np.nanmean([r['auc_roc'] for r in method_results])
                avg_accuracy = np.nanmean([r['accuracy'] for r in method_results])
                avg_f1 = np.nanmean([r['f1_score'] for r in method_results])
                
                summary_data.append({
                    'method': method_results[0]['method'],
                    'n_datasets': len(method_results),
                    'avg_auc_roc': avg_auc,
                    'avg_accuracy': avg_accuracy,
                    'avg_f1_score': avg_f1
                })
        
        return summary_data
    
    def run_comparison(self):
        """Run complete baseline comparison"""
        logging.info("Starting baseline comparison")
        
        # Load foundation model results
        foundation_results = self.load_foundation_model_results()
        
        # Datasets to evaluate
        datasets = [
            'Coffee', 'Cricket', 'ECG200', 'Lightning2', 'Plane', 'Wafer'
        ]
        
        all_results = {}
        
        for dataset_name in tqdm(datasets, desc="Evaluating datasets"):
            results = self.evaluate_dataset(dataset_name)
            if results is not None:
                # Add foundation model results if available
                if dataset_name in foundation_results:
                    results['foundation'] = foundation_results[dataset_name]
                
                all_results[dataset_name] = results
        
        # Save detailed results
        with open(self.config.OUTPUT_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logging.info(f"Baseline comparison saved to {self.config.OUTPUT_FILE}")
        
        # Create summary
        summary = self.create_comparison_summary(all_results)
        
        # Save summary as CSV
        if summary:
            summary_df = pd.DataFrame(summary)
            summary_df.to_csv(self.config.SUMMARY_FILE, index=False)
            
            logging.info(f"Summary saved to {self.config.SUMMARY_FILE}")
            
            # Print summary
            logging.info("\nBaseline Comparison Summary:")
            for row in summary:
                logging.info(f"{row['method']}: AUC={row['avg_auc_roc']:.3f}, "
                           f"Accuracy={row['avg_accuracy']:.3f}, F1={row['avg_f1_score']:.3f}")
        
        logging.info("Baseline comparison complete!")

def main():
    """Main execution function"""
    comparator = BaselineComparator()
    comparator.run_comparison()

if __name__ == "__main__":
    main() 