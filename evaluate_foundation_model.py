#!/usr/bin/env python3
"""
Foundation Model Evaluation on Benchmark Datasets

This script evaluates trained foundation models on UCR benchmark datasets using
rigorous statistical methodology including cross-validation, bootstrap confidence
intervals, and comprehensive significance testing.

Key Features:
- Robust cross-validation (LOOCV for small datasets, stratified k-fold otherwise)
- Bootstrap confidence intervals (1000 iterations)
- Statistical significance testing with multiple comparisons correction
- Data integrity checks and leakage prevention
- Comprehensive logging and progress tracking
- JSON output for reproducibility

Usage:
    python evaluate_foundation_model.py

Requirements:
    - Trained model checkpoint at checkpoints/best_model.pth
    - UCR test datasets in ucr_test_datasets/ directory
    - See requirements.txt for dependencies
"""

import json
import pickle
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.signal import detrend, butter, filtfilt
import logging
from datetime import datetime

# Import model components from training script
from train_foundation_model import Config, FoundationModel, ButterworthFilter

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class EvaluationConfig:
    """Configuration for model evaluation"""
    # File paths
    MODEL_PATH = Path("checkpoints/best_model.pth")
    UCR_DATA_DIR = Path("ucr_test_datasets")
    RESULTS_DIR = Path("evaluation_results")
    LOG_FILE = RESULTS_DIR / "evaluation.log"
    
    # Statistical parameters
    N_BOOTSTRAP = 1000      # Bootstrap iterations for confidence intervals
    CONFIDENCE_LEVEL = 0.95 # Confidence level for intervals
    LOOCV_THRESHOLD = 100   # Use LOOCV if dataset has < 100 samples
    RANDOM_STATE = 42       # For reproducibility
    
    # Evaluation metrics
    PRIMARY_METRIC = "auc_roc"
    METRICS = ["auc_roc", "accuracy", "f1_score"]

class UCRDatasetLoader:
    """Loader for UCR datasets with preprocessing matching training pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.lowpass_filter = ButterworthFilter()
        
    def load_dataset(self, dataset_name: str):
        """Load and preprocess a single UCR dataset"""
        dataset_dir = EvaluationConfig.UCR_DATA_DIR / dataset_name
        
        # Load train and test files
        train_path = dataset_dir / f"{dataset_name}_train.pkl"
        test_path = dataset_dir / f"{dataset_name}_test.pkl"
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")
        
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
        
        # Combine for cross-validation
        X_combined = self._combine_features(train_data['X'], test_data['X'])
        y_combined = self._combine_labels(train_data['y'], test_data['y'])
        
        # Data integrity check
        n_train, n_test = len(train_data['X']), len(test_data['X'])
        n_total = len(X_combined)
        if n_total != n_train + n_test:
            raise ValueError(f"Data integrity error: {n_total} != {n_train + n_test}")
        
        logging.info(f"Loaded {dataset_name}: {n_total} samples, {len(np.unique(y_combined))} classes")
        return X_combined, y_combined
    
    def _combine_features(self, X_train, X_test):
        """Combine train and test features with robust format handling"""
        
        def process_data(X):
            """Process data in various formats (DataFrame, array, nested)"""
            if hasattr(X, 'iloc'):  # DataFrame format
                processed_series = []
                
                if len(X.columns) == 1:  # Single column
                    for i in range(len(X)):
                        series = X.iloc[i, 0]
                        if hasattr(series, 'values'):
                            ts_values = series.values.astype(np.float32)
                        else:
                            ts_values = np.array(series, dtype=np.float32)
                        processed_series.append(ts_values)
                
                else:  # Multi-column (e.g., Cricket with 6 columns)
                    for i in range(len(X)):
                        row_series = []
                        for col in range(len(X.columns)):
                            series = X.iloc[i, col]
                            if hasattr(series, 'values'):
                                ts_values = series.values.astype(np.float32)
                            else:
                                ts_values = np.array(series, dtype=np.float32)
                            row_series.append(ts_values)
                        # Concatenate all series in this sample
                        concatenated_ts = np.concatenate(row_series)
                        processed_series.append(concatenated_ts)
                
                # Pad to same length
                max_len = max(len(ts) for ts in processed_series)
                padded_series = []
                for ts in processed_series:
                    if len(ts) < max_len:
                        padding = np.full(max_len - len(ts), ts[-1] if len(ts) > 0 else 0.0)
                        ts_padded = np.concatenate([ts, padding])
                    else:
                        ts_padded = ts
                    padded_series.append(ts_padded)
                
                return np.array(padded_series, dtype=np.float32)
            
            elif hasattr(X, 'values'):
                X_array = X.values
            else:
                X_array = X
            
            # Handle numpy array formats
            if isinstance(X_array, np.ndarray):
                if X_array.ndim == 1:  # Array of series objects
                    processed_series = []
                    for series in X_array:
                        if hasattr(series, 'values'):
                            ts_values = series.values.astype(np.float32)
                        elif isinstance(series, np.ndarray):
                            ts_values = series.astype(np.float32)
                        elif hasattr(series, '__iter__'):
                            ts_values = np.array(list(series), dtype=np.float32)
                        else:
                            ts_values = np.array([float(series)], dtype=np.float32)
                        processed_series.append(ts_values)
                    
                    # Pad to same length
                    max_len = max(len(ts) for ts in processed_series)
                    padded_series = []
                    for ts in processed_series:
                        if len(ts) < max_len:
                            padding = np.full(max_len - len(ts), ts[-1] if len(ts) > 0 else 0.0)
                            ts_padded = np.concatenate([ts, padding])
                        else:
                            ts_padded = ts
                        padded_series.append(ts_padded)
                    
                    return np.array(padded_series, dtype=np.float32)
                
                elif X_array.ndim == 2:  # Already 2D array
                    return X_array.astype(np.float32)
            
            # Fallback: direct conversion
            return np.array(X_array, dtype=np.float32)
        
        # Process both splits
        X_train_processed = process_data(X_train)
        X_test_processed = process_data(X_test)
        
        # Ensure same number of features
        if X_train_processed.shape[1] != X_test_processed.shape[1]:
            max_features = max(X_train_processed.shape[1], X_test_processed.shape[1])
            
            if X_train_processed.shape[1] < max_features:
                padding = np.zeros((X_train_processed.shape[0], max_features - X_train_processed.shape[1]))
                X_train_processed = np.hstack([X_train_processed, padding])
            
            if X_test_processed.shape[1] < max_features:
                padding = np.zeros((X_test_processed.shape[0], max_features - X_test_processed.shape[1]))
                X_test_processed = np.hstack([X_test_processed, padding])
        
        return np.vstack([X_train_processed, X_test_processed])
    
    def _combine_labels(self, y_train, y_test):
        """Combine and encode labels"""
        y_combined = np.concatenate([y_train, y_test])
        
        # Encode to integers if needed
        if not np.issubdtype(y_combined.dtype, np.integer):
            le = LabelEncoder()
            y_combined = le.fit_transform(y_combined)
        
        return y_combined
    
    def preprocess_time_series(self, ts_array):
        """Apply preprocessing pipeline matching training"""
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
            if len(ts) != self.config.N_TIMESTEPS:
                if len(ts) > self.config.N_TIMESTEPS * 1.2:
                    ts = self.lowpass_filter(ts)
                
                original_indices = np.linspace(0, 1, len(ts))
                new_indices = np.linspace(0, 1, self.config.N_TIMESTEPS)
                ts = np.interp(new_indices, original_indices, ts)
            
            processed_series.append(ts)
        
        return np.array(processed_series)

class FoundationModelEvaluator:
    """Comprehensive evaluator for foundation models"""
    
    def __init__(self):
        self.config = Config()
        self.eval_config = EvaluationConfig()
        self.loader = UCRDatasetLoader(self.config)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup directories and logging
        self.eval_config.RESULTS_DIR.mkdir(exist_ok=True)
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] - %(message)s",
            handlers=[
                logging.FileHandler(self.eval_config.LOG_FILE, mode='w'),
                logging.StreamHandler()
            ]
        )
        logging.info("🔍 Foundation Model Evaluation Started")
        logging.info(f"Device: {self.device}")
        logging.info(f"Bootstrap iterations: {self.eval_config.N_BOOTSTRAP}")
        
    def load_trained_model(self):
        """Load trained foundation model from checkpoint"""
        if not self.eval_config.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {self.eval_config.MODEL_PATH}")
        
        # Load checkpoint
        checkpoint = torch.load(self.eval_config.MODEL_PATH, map_location=self.device)
        
        # Initialize model (9 classes from training domains)
        self.model = FoundationModel(self.config, num_classes=9)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Freeze parameters for feature extraction
        for param in self.model.parameters():
            param.requires_grad = False
            
        param_count = sum(p.numel() for p in self.model.parameters())
        logging.info(f"✅ Model loaded: {param_count:,} parameters")
        logging.info("🔒 All parameters frozen for feature extraction")
        
    def extract_features(self, X_processed):
        """Extract features using frozen foundation model"""
        features = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(X_processed), batch_size):
                batch = X_processed[i:i+batch_size]
                
                # Convert to tensor and add channel dimension
                batch_tensor = torch.FloatTensor(batch).unsqueeze(-1).to(self.device)
                
                # Extract contrastive features
                h_contrastive, _, _, _ = self.model(batch_tensor)
                features.append(h_contrastive.cpu().numpy())
        
        return np.vstack(features)
    
    def bootstrap_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate bootstrap confidence intervals for metrics"""
        n_samples = len(y_true)
        metrics_bootstrap = {metric: [] for metric in self.eval_config.METRICS}
        
        np.random.seed(self.eval_config.RANDOM_STATE)
        
        for _ in range(self.eval_config.N_BOOTSTRAP):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            y_pred_proba_boot = y_pred_proba[indices]
            
            # Calculate metrics
            try:
                if len(np.unique(y_true_boot)) > 1:
                    if y_pred_proba_boot.ndim > 1 and y_pred_proba_boot.shape[1] > 1:
                        if len(np.unique(y_true_boot)) == 2:
                            auc = roc_auc_score(y_true_boot, y_pred_proba_boot[:, 1])
                        else:
                            auc = roc_auc_score(y_true_boot, y_pred_proba_boot, multi_class='ovr')
                    else:
                        auc = np.nan
                else:
                    auc = np.nan
                
                acc = accuracy_score(y_true_boot, y_pred_boot)
                f1 = f1_score(y_true_boot, y_pred_boot, average='macro')
                
                metrics_bootstrap['auc_roc'].append(auc)
                metrics_bootstrap['accuracy'].append(acc)
                metrics_bootstrap['f1_score'].append(f1)
                
            except Exception:
                metrics_bootstrap['auc_roc'].append(np.nan)
                metrics_bootstrap['accuracy'].append(np.nan)
                metrics_bootstrap['f1_score'].append(np.nan)
        
        # Calculate confidence intervals
        results = {}
        alpha = 1 - self.eval_config.CONFIDENCE_LEVEL
        
        for metric in self.eval_config.METRICS:
            values = np.array(metrics_bootstrap[metric])
            valid_values = values[~np.isnan(values)]
            
            if len(valid_values) > 0:
                ci_lower = np.percentile(valid_values, 100 * alpha/2)
                ci_upper = np.percentile(valid_values, 100 * (1 - alpha/2))
                results[f'{metric}_ci_lower'] = ci_lower
                results[f'{metric}_ci_upper'] = ci_upper
                results[f'{metric}_std'] = np.std(valid_values)
            else:
                results[f'{metric}_ci_lower'] = np.nan
                results[f'{metric}_ci_upper'] = np.nan
                results[f'{metric}_std'] = np.nan
        
        return results
    
    def evaluate_dataset(self, dataset_name):
        """Evaluate foundation model on single dataset"""
        logging.info(f"\n📊 Evaluating {dataset_name}")
        
        try:
            # Load and preprocess data
            X_raw, y = self.loader.load_dataset(dataset_name)
            logging.info(f"Raw data: {X_raw.shape}, dtype: {X_raw.dtype}")
            
            X_processed = self.loader.preprocess_time_series(X_raw)
            logging.info(f"Processed data: {X_processed.shape}")
            
            # Data integrity checks
            assert len(X_processed) == len(y), "Feature-label mismatch"
            assert not np.any(np.isnan(X_processed)), "NaN values in processed features"
            
        except Exception as e:
            logging.error(f"Data loading failed for {dataset_name}: {e}")
            raise
        
        # Extract features
        features = self.extract_features(X_processed)
        
        # Choose cross-validation strategy
        n_samples = len(X_processed)
        if n_samples < self.eval_config.LOOCV_THRESHOLD:
            cv = LeaveOneOut()
            cv_name = "LOOCV"
            logging.info(f"Using Leave-One-Out CV (N={n_samples})")
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.eval_config.RANDOM_STATE)
            cv_name = "5-Fold Stratified CV"
            logging.info(f"Using 5-Fold Stratified CV (N={n_samples})")
        
        # Cross-validation evaluation
        y_true_all = []
        y_pred_all = []
        y_pred_proba_all = []
        fold_results = []
        
        cv_splits = list(cv.split(features, y))
        
        for fold, (train_idx, test_idx) in enumerate(tqdm(cv_splits, desc=f"CV {dataset_name}")):
            # Data leakage check
            assert len(set(train_idx) & set(test_idx)) == 0, "Train-test overlap detected!"
            
            X_train_fold, X_test_fold = features[train_idx], features[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            # Train classifier on frozen features
            classifier = LogisticRegression(
                max_iter=1000, 
                random_state=self.eval_config.RANDOM_STATE,
                class_weight='balanced'
            )
            classifier.fit(X_train_fold, y_train_fold)
            
            # Predictions
            y_pred_fold = classifier.predict(X_test_fold)
            y_pred_proba_fold = classifier.predict_proba(X_test_fold)
            
            # Store for aggregate metrics
            y_true_all.extend(y_test_fold)
            y_pred_all.extend(y_pred_fold)
            y_pred_proba_all.extend(y_pred_proba_fold)
            
            # Fold-level metrics
            if len(np.unique(y_test_fold)) > 1:
                fold_acc = accuracy_score(y_test_fold, y_pred_fold)
                fold_f1 = f1_score(y_test_fold, y_pred_fold, average='macro')
                
                if y_pred_proba_fold.shape[1] > 1:
                    if len(np.unique(y_test_fold)) == 2:
                        fold_auc = roc_auc_score(y_test_fold, y_pred_proba_fold[:, 1])
                    else:
                        fold_auc = roc_auc_score(y_test_fold, y_pred_proba_fold, multi_class='ovr')
                else:
                    fold_auc = np.nan
            else:
                fold_acc = fold_f1 = fold_auc = np.nan
            
            fold_results.append({
                'fold': fold,
                'auc_roc': fold_auc,
                'accuracy': fold_acc,
                'f1_score': fold_f1,
                'n_test': len(y_test_fold)
            })
        
        # Convert to arrays
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        y_pred_proba_all = np.array(y_pred_proba_all)
        
        # Overall metrics
        overall_acc = accuracy_score(y_true_all, y_pred_all)
        overall_f1 = f1_score(y_true_all, y_pred_all, average='macro')
        
        if len(np.unique(y_true_all)) > 1 and y_pred_proba_all.shape[1] > 1:
            if len(np.unique(y_true_all)) == 2:
                overall_auc = roc_auc_score(y_true_all, y_pred_proba_all[:, 1])
            else:
                overall_auc = roc_auc_score(y_true_all, y_pred_proba_all, multi_class='ovr')
        else:
            overall_auc = np.nan
        
        # Bootstrap confidence intervals
        bootstrap_results = self.bootstrap_metrics(y_true_all, y_pred_all, y_pred_proba_all)
        
        # Effect size calculation
        chance_performance = 0.5 if len(np.unique(y)) == 2 else 1.0 / len(np.unique(y))
        
        if not np.isnan(overall_auc):
            effect_size = (overall_auc - chance_performance) / bootstrap_results.get('auc_roc_std', 1)
        else:
            effect_size = np.nan
        
        # Compile results
        results = {
            'dataset': dataset_name,
            'n_samples': n_samples,
            'n_classes': len(np.unique(y)),
            'cv_strategy': cv_name,
            'cv_folds': len(cv_splits),
            
            # Primary metrics
            'auc_roc': overall_auc,
            'accuracy': overall_acc,
            'f1_score': overall_f1,
            
            # Statistical measures
            'effect_size_cohens_d': effect_size,
            'chance_performance': chance_performance,
            
            # Raw predictions for further analysis
            'y_true': y_true_all.tolist(),
            'y_pred': y_pred_all.tolist(),
            'y_pred_proba': y_pred_proba_all.tolist(),
            
            # Fold-level results
            'fold_results': fold_results,
            
            # Timestamp
            'evaluated_at': datetime.now().isoformat()
        }
        
        # Add bootstrap results
        results.update(bootstrap_results)
        
        # Log results
        ci_str = f"[{bootstrap_results.get('auc_roc_ci_lower', np.nan):.3f}-{bootstrap_results.get('auc_roc_ci_upper', np.nan):.3f}]"
        logging.info(f"✅ {dataset_name}: AUC={overall_auc:.3f} {ci_str}, Acc={overall_acc:.3f}, F1={overall_f1:.3f}")
        
        return results
    
    def run_evaluation(self):
        """Run complete evaluation on all datasets"""
        logging.info("🚀 Starting comprehensive evaluation")
        
        # Load model
        self.load_trained_model()
        
        # Define evaluation datasets
        dataset_names = [
            'Coffee', 'TwoLeadECG', 'Wafer', 'Cricket', 'Plane',
            'SonyAIBORobotSurface1', 'ItalyPowerDemand',
            'SonyAIBORobotSurface2', 'StarLightCurves', 'MoteStrain',
            'ECG200', 'Lightning2', 'Earthquakes'
        ]
        
        # Evaluate each dataset
        all_results = []
        
        for dataset_name in tqdm(dataset_names, desc="Evaluating datasets"):
            try:
                result = self.evaluate_dataset(dataset_name)
                all_results.append(result)
            except Exception as e:
                logging.error(f"❌ Failed to evaluate {dataset_name}: {e}")
                continue
        
        # Post-hoc analysis
        post_hoc_results = self.post_hoc_analysis(all_results)
        
        # Save results
        final_results = {
            'evaluation_metadata': {
                'model_path': str(self.eval_config.MODEL_PATH),
                'n_datasets': len(all_results),
                'bootstrap_iterations': self.eval_config.N_BOOTSTRAP,
                'confidence_level': self.eval_config.CONFIDENCE_LEVEL,
                'evaluation_timestamp': datetime.now().isoformat(),
                'device': str(self.device)
            },
            'dataset_results': all_results,
            'post_hoc_analysis': post_hoc_results
        }
        
        # Save to JSON
        results_file = self.eval_config.RESULTS_DIR / "foundation_model_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logging.info(f"💾 Results saved to {results_file}")
        
        # Generate summary
        self.generate_summary(all_results, post_hoc_results)
        
        return final_results
    
    def post_hoc_analysis(self, results):
        """Perform post-hoc statistical analysis"""
        logging.info("📈 Performing post-hoc statistical analysis")
        
        # Extract metrics
        auc_scores = [r['auc_roc'] for r in results if not np.isnan(r['auc_roc'])]
        effect_sizes = [r['effect_size_cohens_d'] for r in results if not np.isnan(r['effect_size_cohens_d'])]
        
        # Significance testing
        p_values = []
        for result in results:
            if not np.isnan(result['auc_roc']):
                chance = result['chance_performance']
                auc_std = result.get('auc_roc_std', 0.1)
                if auc_std > 0:
                    t_stat = (result['auc_roc'] - chance) / auc_std
                    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=result['n_samples']-1))
                    p_values.append(p_val)
                else:
                    p_values.append(1.0)
            else:
                p_values.append(1.0)
        
        # Multiple comparisons correction
        from statsmodels.stats.multitest import multipletests
        
        # Bonferroni correction
        bonferroni_alpha = 0.05 / len(p_values) if p_values else 0.05
        bonferroni_significant = [p < bonferroni_alpha for p in p_values]
        
        # FDR correction
        if len(p_values) > 0:
            rejected_fdr, pvals_fdr, _, _ = multipletests(p_values, method='fdr_bh')
            fdr_significant = rejected_fdr.tolist()
        else:
            fdr_significant = []
        
        # Heterogeneity analysis
        if len(effect_sizes) > 1:
            mean_effect = np.mean(effect_sizes)
            Q_stat = np.sum([(es - mean_effect)**2 for es in effect_sizes])
            I_squared = max(0, (Q_stat - len(effect_sizes) + 1) / Q_stat * 100) if Q_stat > 0 else 0
        else:
            Q_stat = I_squared = np.nan
        
        return {
            'summary_statistics': {
                'mean_auc': np.mean(auc_scores) if auc_scores else np.nan,
                'median_auc': np.median(auc_scores) if auc_scores else np.nan,
                'std_auc': np.std(auc_scores) if auc_scores else np.nan,
                'min_auc': np.min(auc_scores) if auc_scores else np.nan,
                'max_auc': np.max(auc_scores) if auc_scores else np.nan,
                'mean_effect_size': np.mean(effect_sizes) if effect_sizes else np.nan
            },
            'multiple_comparisons': {
                'bonferroni_alpha': bonferroni_alpha,
                'bonferroni_significant_count': sum(bonferroni_significant),
                'fdr_significant_count': sum(fdr_significant),
                'p_values': p_values,
                'bonferroni_significant': bonferroni_significant,
                'fdr_significant': fdr_significant
            },
            'heterogeneity': {
                'cochrans_Q': Q_stat,
                'I_squared_percent': I_squared,
                'interpretation': 'High heterogeneity' if I_squared > 75 else 'Moderate heterogeneity' if I_squared > 50 else 'Low heterogeneity'
            }
        }
    
    def generate_summary(self, results, post_hoc):
        """Generate evaluation summary"""
        logging.info("\n" + "="*70)
        logging.info("📋 FOUNDATION MODEL EVALUATION SUMMARY")
        logging.info("="*70)
        
        # Overall performance
        valid_results = [r for r in results if not np.isnan(r['auc_roc'])]
        if valid_results:
            mean_auc = np.mean([r['auc_roc'] for r in valid_results])
            std_auc = np.std([r['auc_roc'] for r in valid_results])
            
            logging.info(f"📊 Overall Performance:")
            logging.info(f"  Mean AUC-ROC: {mean_auc:.3f} ± {std_auc:.3f}")
            logging.info(f"  Datasets evaluated: {len(valid_results)}/{len(results)}")
            
            # Top performers
            sorted_results = sorted(valid_results, key=lambda x: x['auc_roc'], reverse=True)
            logging.info(f"\n🏆 Top Performers:")
            for i, result in enumerate(sorted_results[:5]):
                ci_lower = result.get('auc_roc_ci_lower', np.nan)
                ci_upper = result.get('auc_roc_ci_upper', np.nan)
                logging.info(f"  {i+1}. {result['dataset']}: {result['auc_roc']:.3f} [{ci_lower:.3f}-{ci_upper:.3f}]")
            
            # Statistical significance
            n_bonferroni = post_hoc['multiple_comparisons']['bonferroni_significant_count']
            n_fdr = post_hoc['multiple_comparisons']['fdr_significant_count']
            logging.info(f"\n📈 Statistical Significance:")
            logging.info(f"  Bonferroni significant: {n_bonferroni}/{len(results)} datasets")
            logging.info(f"  FDR significant: {n_fdr}/{len(results)} datasets")
            
        logging.info("="*70)

def main():
    """Main evaluation pipeline"""
    evaluator = FoundationModelEvaluator()
    results = evaluator.run_evaluation()
    
    print("\n✅ Foundation model evaluation completed!")
    print(f"📁 Results saved in: {EvaluationConfig.RESULTS_DIR}")
    print(f"📊 JSON results: {EvaluationConfig.RESULTS_DIR}/foundation_model_evaluation_results.json")
    print(f"📝 Log file: {EvaluationConfig.LOG_FILE}")

if __name__ == "__main__":
    main() 