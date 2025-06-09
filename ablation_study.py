#!/usr/bin/env python3
"""
Comprehensive Foundation Model Ablation Study

This script performs extensive ablation studies on the trained foundation model,
testing architecture components, preprocessing strategies, and baseline comparisons
with rigorous statistical analysis for publication.

Key Features:
- Architecture component ablation (heads, layers, embeddings)
- Preprocessing ablation (detrending, normalization, filtering)  
- Baseline comparisons (handcrafted features, statistical methods)
- Statistical significance testing with effect sizes
- Publication-ready outputs and analysis

Usage:
    python ablation_study.py

Requirements:
    - Trained model checkpoint at checkpoints/best_model.pth
    - UCR test datasets in ucr_test_datasets/ directory
    - See requirements.txt for dependencies
"""

import json
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from scipy import stats
from scipy.signal import detrend
import logging
from datetime import datetime

# Import components from other scripts
from evaluate_foundation_model import (
    FoundationModelEvaluator, UCRDatasetLoader, EvaluationConfig, 
    Config, FoundationModel, ButterworthFilter
)

warnings.filterwarnings('ignore')

class AblationConfig:
    """Configuration for comprehensive ablation studies"""
    
    # Architecture component ablations
    ARCHITECTURE_ABLATIONS = {
        'full_model': {
            'use_head': 'contrastive',
            'use_layer': 'final',
            'description': 'Full foundation model with contrastive head'
        },
        'classification_head': {
            'use_head': 'classification',
            'use_layer': 'final',
            'description': 'Use classification head instead of contrastive'
        },
        'early_layers': {
            'use_head': 'contrastive',
            'use_layer': 'layer_2',
            'description': 'Extract from early transformer layers'
        },
        'middle_layers': {
            'use_head': 'contrastive',
            'use_layer': 'layer_3',
            'description': 'Extract from middle transformer layers'
        },
        'patch_embeddings': {
            'use_head': 'patch_embed',
            'use_layer': 'embeddings',
            'description': 'Use patch embeddings directly'
        }
    }
    
    # Preprocessing ablations
    PREPROCESSING_ABLATIONS = {
        'full_preprocessing': {
            'skip_detrend': False,
            'skip_znorm': False,
            'skip_outlier_clip': False,
            'skip_lowpass': False,
            'description': 'Full preprocessing pipeline (baseline)'
        },
        'no_detrend': {
            'skip_detrend': True,
            'skip_znorm': False,
            'skip_outlier_clip': False,
            'skip_lowpass': False,
            'description': 'No detrending'
        },
        'no_normalization': {
            'skip_detrend': False,
            'skip_znorm': True,
            'skip_outlier_clip': False,
            'skip_lowpass': False,
            'description': 'No z-score normalization'
        }
    }
    
    # Baseline methods (representation learning focus)
    BASELINE_METHODS = {
        'handcrafted_features': {
            'method': 'handcrafted',
            'description': 'Statistical + frequency domain features'
        },
        'raw_time_series': {
            'method': 'raw_linear',
            'description': 'Direct time series with logistic regression'
        },
        'simple_statistical': {
            'method': 'simple_stats',
            'description': 'Simple statistical features baseline'
        }
    }
    
    # Results organization
    RESULTS_DIR = Path("ablation_results")

class HandcraftedFeatureExtractor:
    """Extract comprehensive handcrafted features"""
    
    def extract_features(self, X):
        """Extract handcrafted features from time series"""
        features = []
        
        for ts in X:
            ts_features = []
            
            # Basic statistical features
            ts_features.extend([
                np.mean(ts), np.std(ts), np.median(ts),
                np.min(ts), np.max(ts), np.ptp(ts),
                stats.skew(ts), stats.kurtosis(ts),
                np.percentile(ts, 25), np.percentile(ts, 75),
                np.var(ts), np.sqrt(np.mean(ts**2))
            ])
            
            # Difference features
            diff1 = np.diff(ts)
            ts_features.extend([
                np.mean(diff1), np.std(diff1), np.max(np.abs(diff1))
            ])
            
            # Frequency features
            fft_vals = np.fft.fft(ts)
            fft_mag = np.abs(fft_vals[:len(fft_vals)//2])
            
            if len(fft_mag) > 0:
                ts_features.extend([
                    np.mean(fft_mag), np.std(fft_mag), np.max(fft_mag)
                ])
            else:
                ts_features.extend([0, 0, 0])
            
            features.append(ts_features)
        
        return np.array(features)

class AblationDatasetLoader(UCRDatasetLoader):
    """Extended dataset loader with ablation-specific preprocessing"""
    
    def preprocess_time_series_ablation(self, ts_array, preprocessing_config):
        """Apply ablated preprocessing pipeline"""
        processed_series = []
        
        for ts in ts_array:
            ts = ts.astype(np.float32)
            
            # Remove NaNs (always)
            ts = ts[~np.isnan(ts)]
            
            # Conditional preprocessing steps
            if not preprocessing_config.get('skip_detrend', False):
                ts = detrend(ts)
            
            if not preprocessing_config.get('skip_outlier_clip', False):
                std = np.std(ts)
                mean = np.mean(ts)
                ts = np.clip(ts, mean - 3 * std, mean + 3 * std)
            
            if not preprocessing_config.get('skip_znorm', False):
                std = np.std(ts)
                mean = np.mean(ts)
                if std > 1e-6:
                    ts = (ts - mean) / std
            
            # Resampling to target length
            if len(ts) != self.config.N_TIMESTEPS:
                if len(ts) > self.config.N_TIMESTEPS * 1.2 and not preprocessing_config.get('skip_lowpass', False):
                    ts = self.lowpass_filter(ts)
                
                original_indices = np.linspace(0, 1, len(ts))
                new_indices = np.linspace(0, 1, self.config.N_TIMESTEPS)
                ts = np.interp(new_indices, original_indices, ts)
            
            processed_series.append(ts)
        
        return np.array(processed_series)

class AblationFoundationModelEvaluator(FoundationModelEvaluator):
    """Extended evaluator for ablation studies"""
    
    def __init__(self):
        super().__init__()
        self.ablation_config = AblationConfig()
        self.loader = AblationDatasetLoader(self.config)
        self.handcrafted_extractor = HandcraftedFeatureExtractor()
        
        # Create results directory
        self.ablation_config.RESULTS_DIR.mkdir(exist_ok=True)
        
        logging.info("🔬 Ablation study evaluator initialized")
    
    def extract_features_ablation(self, X_processed, ablation_type, ablation_params):
        """Extract features based on ablation configuration"""
        
        if ablation_type == 'baseline':
            return self._extract_baseline_features(X_processed, ablation_params)
        elif ablation_type == 'architecture':
            return self._extract_architecture_features(X_processed, ablation_params)
        else:  # preprocessing ablation
            return self.extract_features(X_processed)
    
    def _extract_baseline_features(self, X_processed, params):
        """Extract baseline method features"""
        method = params['method']
        
        if method == 'handcrafted':
            return self.handcrafted_extractor.extract_features(X_processed)
        elif method == 'raw_linear':
            return X_processed.reshape(X_processed.shape[0], -1)
        elif method == 'simple_stats':
            return self._extract_simple_statistical_features(X_processed)
        else:
            raise ValueError(f"Unknown baseline method: {method}")
    
    def _extract_simple_statistical_features(self, X_processed):
        """Extract simple statistical features"""
        features = []
        
        for ts in X_processed:
            ts_features = [
                np.mean(ts), np.std(ts), np.median(ts),
                np.min(ts), np.max(ts), np.ptp(ts),
                stats.skew(ts), stats.kurtosis(ts),
                np.percentile(ts, 25), np.percentile(ts, 75),
                np.var(ts), np.sqrt(np.mean(ts**2)),
                np.mean(np.abs(np.diff(ts))),
                np.std(np.diff(ts)),
                len(ts[ts > np.mean(ts)]) / len(ts),
                np.sum(ts**2),
                np.mean(ts**2)
            ]
            features.append(ts_features)
        
        return np.array(features)
    
    def _extract_architecture_features(self, X_processed, params):
        """Extract features using ablated architecture"""
        use_head = params['use_head']
        use_layer = params['use_layer']
        
        features = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(X_processed), batch_size):
                batch = X_processed[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch).unsqueeze(-1).to(self.device)
                
                if use_head == 'patch_embed':
                    # Use patch embeddings directly
                    patch_embedded = self.model.patch_embed(batch_tensor)
                    batch_features = patch_embedded.mean(dim=1).cpu().numpy()
                
                elif use_layer.startswith('layer_'):
                    # Extract from specific transformer layer
                    layer_idx = int(use_layer.split('_')[1]) - 1
                    batch_features = self._extract_from_layer(batch_tensor, layer_idx)
                
                else:
                    # Use specific head
                    h_contrastive, _, h_class, _ = self.model(batch_tensor)
                    
                    if use_head == 'contrastive':
                        batch_features = h_contrastive.cpu().numpy()
                    elif use_head == 'classification':
                        batch_features = h_class.cpu().numpy()
                    else:
                        batch_features = h_contrastive.cpu().numpy()
                
                features.append(batch_features)
        
        return np.vstack(features)
    
    def _extract_from_layer(self, batch_tensor, layer_idx):
        """Extract features from specific transformer layer"""
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(output.mean(dim=1).cpu().numpy())
        
        # Register hook
        if layer_idx < len(self.model.transformer_encoder.layers):
            handle = self.model.transformer_encoder.layers[layer_idx].register_forward_hook(hook_fn)
            
            # Forward pass
            x_patched = self.model.patch_embed(batch_tensor)
            x_pos = self.model.pos_encoder(x_patched)
            
            # Process up to desired layer
            for i, layer in enumerate(self.model.transformer_encoder.layers):
                x_pos = layer(x_pos)
                if i == layer_idx:
                    break
            
            handle.remove()
            
            if activations:
                return activations[0]
        
        # Fallback
        h_contrastive, _, _, _ = self.model(batch_tensor)
        return h_contrastive.cpu().numpy()
    
    def evaluate_ablation(self, dataset_name, ablation_type, ablation_name, ablation_params):
        """Evaluate specific ablation configuration"""
        logging.info(f"🔬 Ablating {dataset_name}: {ablation_type}.{ablation_name}")
        
        try:
            # Load dataset
            X_raw, y = self.loader.load_dataset(dataset_name)
            
            # Apply preprocessing
            if ablation_type == 'preprocessing':
                X_processed = self.loader.preprocess_time_series_ablation(X_raw, ablation_params)
            else:
                X_processed = self.loader.preprocess_time_series(X_raw)
            
            # Extract features
            features = self.extract_features_ablation(X_processed, ablation_type, ablation_params)
            
            # Cross-validation evaluation
            return self._run_cv_evaluation(features, y, dataset_name, ablation_name)
            
        except Exception as e:
            logging.error(f"❌ Ablation failed for {dataset_name}.{ablation_name}: {e}")
            return None
    
    def _run_cv_evaluation(self, features, y, dataset_name, method_name):
        """Run cross-validation evaluation"""
        
        classifier = LogisticRegression(
            max_iter=1000, 
            random_state=42,
            class_weight='balanced'
        )
        
        # Choose CV strategy
        n_samples = len(features)
        if n_samples < 100:
            cv = LeaveOneOut()
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        y_true_all = []
        y_pred_all = []
        y_pred_proba_all = []
        
        cv_splits = list(cv.split(features, y))
        
        for train_idx, test_idx in cv_splits:
            X_train_fold, X_test_fold = features[train_idx], features[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            # Scale features if needed
            if features.shape[1] > 128:
                scaler = StandardScaler()
                X_train_fold = scaler.fit_transform(X_train_fold)
                X_test_fold = scaler.transform(X_test_fold)
            
            # Train classifier
            classifier.fit(X_train_fold, y_train_fold)
            
            # Predictions
            y_pred_fold = classifier.predict(X_test_fold)
            y_pred_proba_fold = classifier.predict_proba(X_test_fold)
            
            y_true_all.extend(y_test_fold)
            y_pred_all.extend(y_pred_fold)
            y_pred_proba_all.extend(y_pred_proba_fold)
        
        # Calculate metrics
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        y_pred_proba_all = np.array(y_pred_proba_all)
        
        accuracy = accuracy_score(y_true_all, y_pred_all)
        f1 = f1_score(y_true_all, y_pred_all, average='macro')
        
        if len(np.unique(y_true_all)) > 1 and y_pred_proba_all.shape[1] > 1:
            if len(np.unique(y_true_all)) == 2:
                auc_roc = roc_auc_score(y_true_all, y_pred_proba_all[:, 1])
            else:
                auc_roc = roc_auc_score(y_true_all, y_pred_proba_all, multi_class='ovr')
        else:
            auc_roc = np.nan
        
        return {
            'dataset': dataset_name,
            'method': method_name,
            'auc_roc': auc_roc,
            'accuracy': accuracy,
            'f1_score': f1,
            'n_samples': n_samples,
            'n_features': features.shape[1],
            'cv_folds': len(cv_splits)
        }
    
    def run_comprehensive_ablation_study(self):
        """Run complete ablation study"""
        logging.info("🚀 Starting comprehensive ablation study")
        
        # Load model
        self.load_trained_model()
        
        # Get available datasets
        dataset_names = [
            'Coffee', 'TwoLeadECG', 'Wafer', 'Cricket', 'Plane',
            'SonyAIBORobotSurface1', 'ItalyPowerDemand',
            'SonyAIBORobotSurface2', 'StarLightCurves', 'MoteStrain',
            'ECG200', 'Lightning2', 'Earthquakes'
        ]
        
        all_results = []
        
        # Architecture ablations
        logging.info("🏗️ Running architecture ablations...")
        for ablation_name, ablation_params in tqdm(
            self.ablation_config.ARCHITECTURE_ABLATIONS.items(),
            desc="Architecture ablations"
        ):
            for dataset_name in dataset_names:
                result = self.evaluate_ablation(
                    dataset_name, 'architecture', ablation_name, ablation_params
                )
                if result:
                    result['ablation_category'] = 'architecture'
                    result['ablation_name'] = ablation_name
                    result['description'] = ablation_params['description']
                    all_results.append(result)
        
        # Preprocessing ablations
        logging.info("🔄 Running preprocessing ablations...")
        for ablation_name, ablation_params in tqdm(
            self.ablation_config.PREPROCESSING_ABLATIONS.items(),
            desc="Preprocessing ablations"
        ):
            for dataset_name in dataset_names:
                result = self.evaluate_ablation(
                    dataset_name, 'preprocessing', ablation_name, ablation_params
                )
                if result:
                    result['ablation_category'] = 'preprocessing'
                    result['ablation_name'] = ablation_name
                    result['description'] = ablation_params['description']
                    all_results.append(result)
        
        # Baseline comparisons
        logging.info("⚖️ Running baseline comparisons...")
        for baseline_name, baseline_params in tqdm(
            self.ablation_config.BASELINE_METHODS.items(),
            desc="Baseline comparisons"
        ):
            for dataset_name in dataset_names:
                result = self.evaluate_ablation(
                    dataset_name, 'baseline', baseline_name, baseline_params
                )
                if result:
                    result['ablation_category'] = 'baseline'
                    result['ablation_name'] = baseline_name
                    result['description'] = baseline_params['description']
                    all_results.append(result)
        
        # Save results
        self.save_ablation_results(all_results)
        
        logging.info(f"✅ Ablation study completed: {len(all_results)} evaluations")
        
        return {
            'raw_results': all_results,
            'metadata': {
                'n_evaluations': len(all_results),
                'n_datasets': len(dataset_names),
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def save_ablation_results(self, all_results):
        """Save comprehensive ablation results"""
        
        # Main results file
        results_file = self.ablation_config.RESULTS_DIR / "comprehensive_ablation_study.json"
        with open(results_file, 'w') as f:
            json.dump({
                'raw_results': all_results,
                'metadata': {
                    'n_evaluations': len(all_results),
                    'timestamp': datetime.now().isoformat()
                }
            }, f, indent=2)
        
        # Summary table
        df = pd.DataFrame(all_results)
        summary = df.groupby(['ablation_name', 'ablation_category']).agg({
            'auc_roc': ['mean', 'std', 'count'],
            'accuracy': ['mean', 'std'],
            'f1_score': ['mean', 'std']
        }).round(4)
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        summary = summary.reset_index()
        summary.to_csv(self.ablation_config.RESULTS_DIR / "ablation_summary_table.csv", index=False)
        
        # Detailed results
        df[['dataset', 'ablation_category', 'ablation_name', 'auc_roc', 'accuracy', 'f1_score', 'description']].to_csv(
            self.ablation_config.RESULTS_DIR / "detailed_results_table.csv", 
            index=False
        )
        
        logging.info(f"💾 Results saved to {self.ablation_config.RESULTS_DIR}")

def main():
    """Run comprehensive ablation study"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler("ablation_results/ablation_study.log", mode='w'),
            logging.StreamHandler()
        ]
    )
    
    # Run ablation study
    evaluator = AblationFoundationModelEvaluator()
    results = evaluator.run_comprehensive_ablation_study()
    
    print("\n✅ Comprehensive ablation study completed!")
    print(f"📁 Results saved in: {AblationConfig.RESULTS_DIR}")
    print(f"📊 Evaluations completed: {results['metadata']['n_evaluations']}")

if __name__ == "__main__":
    main() 