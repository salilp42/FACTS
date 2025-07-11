# FACTS: Foundation model Analysis for Cross-domain Time Series

A comprehensive framework for training and evaluating foundation models on time series data across multiple scientific domains.

## Overview

This repository contains the complete pipeline for:
- Training transformer-based foundation models on multi-domain time series data
- Evaluating model performance on benchmark datasets with statistical rigor
- Conducting comprehensive ablation studies and statistical analysis
- Analyzing complexity metrics and their relationship to performance
- Demonstrating generative capabilities through masked prediction
- Comparing against modern deep learning baselines
- Performing cross-domain transfer analysis

## Key Features

- **Multi-domain Training**: Trains on time series from 9 diverse scientific domains
- **Robust Evaluation**: Cross-validation with bootstrap confidence intervals
- **Statistical Analysis**: Publication-ready statistical testing with multiple comparisons correction
- **Ablation Studies**: Component-wise analysis of model architecture
- **Complexity Analysis**: Permutation entropy, spectral entropy, and statistical complexity measures
- **Generative Capabilities**: Masked prediction reconstruction across domains
- **Baseline Comparisons**: MiniRocket and statistical feature baselines
- **Transfer Analysis**: Cross-domain transfer matrix and pattern analysis
- **Reproducible**: Fixed random seeds and comprehensive logging

## Repository Structure

```
FACTS/
├── train_foundation_model.py         # Main training script
├── evaluate_foundation_model.py      # Evaluation on benchmark datasets
├── statistical_analysis.py           # Comprehensive statistical analysis
├── ablation_study.py                # Component ablation analysis
├── complexity_analysis.py            # NEW: Complexity metrics analysis
├── generative_analysis.py            # NEW: Generative capabilities demonstration
├── baseline_comparison.py            # NEW: Modern DL baseline comparisons
├── cross_domain_analysis.py          # NEW: Cross-domain transfer analysis
├── create_dataset_visualization_figures.py  # Dataset visualization
├── requirements.txt                  # Python dependencies
└── README.md                        # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/salilp42/FACTS.git
cd FACTS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Ensure your training data is organized as:
```
training_datasets/
├── domain1.parquet
├── domain2.parquet
└── ...
```

Each parquet file should contain a 'target' column with time series values.

### 2. Train Foundation Model

```bash
python train_foundation_model.py
```

This will:
- Load all training data from multiple domains
- Train a transformer-based foundation model
- Save checkpoints and training logs
- Achieve convergence in ~1-2 hours on modern hardware

### 3. Evaluate Model

```bash
python evaluate_foundation_model.py
```

This will:
- Load the trained model
- Evaluate on UCR benchmark datasets
- Perform cross-validation with statistical testing
- Generate comprehensive results with confidence intervals

### 4. Statistical Analysis

```bash
python statistical_analysis.py
```

This will:
- Combine evaluation and ablation results
- Perform statistical significance testing
- Apply multiple comparisons correction
- Generate publication-ready statistical summaries

### 5. Ablation Study

```bash
python ablation_study.py
```

This will:
- Test individual model components
- Compare against baseline methods
- Quantify component importance
- Generate ablation analysis results

### 6. Complexity Analysis

```bash
python complexity_analysis.py
```

This will:
- Compute permutation entropy using Bandt-Pompe method
- Calculate spectral entropy from power spectral density
- Analyze statistical complexity measures
- Correlate complexity metrics with model performance
- Generate complexity correlation visualizations

### 7. Generative Analysis

```bash
python generative_analysis.py
```

This will:
- Demonstrate masked prediction reconstruction capabilities
- Show original vs reconstructed time series patterns
- Analyze reconstruction quality across domains
- Validate foundation model designation through generative tasks

### 8. Baseline Comparison

```bash
python baseline_comparison.py
```

This will:
- Implement MiniRocket with random convolutional kernels
- Evaluate statistical feature baselines with Random Forest
- Compare against foundation model performance
- Generate comprehensive baseline comparison results

### 9. Cross-Domain Transfer Analysis

```bash
python cross_domain_analysis.py
```

This will:
- Compute 13×13 cross-domain transfer matrix
- Analyze transfer patterns by domain category
- Identify exceptional and poor transfer cases
- Generate transfer pattern visualizations
- Validate engineered vs natural system dichotomy

## Model Architecture

The foundation model uses a transformer-based architecture with:
- **Patch Embedding**: Converts time series to patch tokens
- **Positional Encoding**: Sinusoidal position embeddings
- **Transformer Encoder**: Multi-head attention layers

### Complexity Metrics
- **Permutation Entropy**: Temporal complexity using ordinal patterns
- **Spectral Entropy**: Frequency domain complexity from PSD
- **Statistical Measures**: Coefficient of variation, autocorrelation
- **Performance Correlation**: Strong negative correlation with complexity

### Generative Capabilities
- **Masked Prediction**: 15% masking ratio for reconstruction
- **Quality Assessment**: MSE and correlation metrics
- **Domain Validation**: Reconstruction quality mirrors classification performance

### Transfer Analysis
- **Systematic Evaluation**: All pairwise domain transfers
- **Pattern Analysis**: Engineered→Engineered vs Natural→Natural
- **Statistical Validation**: Mann-Whitney U tests with correction

## Statistical Methods

- **Cross-validation**: Leave-one-out (N<100) or 5-fold stratified
- **Confidence intervals**: Bootstrap with 1000 iterations
- **Effect sizes**: Cohen's d relative to chance performance
- **Multiple comparisons**: Bonferroni and FDR correction
- **Heterogeneity**: I² statistic for meta-analysis
- **Correlation analysis**: Pearson with permutation tests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Requirements

- Python 3.8+
- PyTorch 1.8+
- scikit-learn
- pandas
- numpy
- scipy
- matplotlib
- seaborn
- tqdm

See `requirements.txt` for complete dependency list.

