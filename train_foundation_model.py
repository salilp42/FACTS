#!/usr/bin/env python3
"""
Foundation Model Training for Cross-Domain Time Series Classification

This script implements a comprehensive training pipeline for transformer-based 
foundation models on multi-domain time series data. The model uses multi-objective
training with contrastive learning, masked prediction, and classification objectives.

Key Features:
- Multi-domain time series training (9 scientific domains)
- Transformer architecture with patch-based tokenization
- Multi-objective training (contrastive + masked + classification)
- Comprehensive preprocessing pipeline
- Robust training with progress tracking and checkpointing
- Efficient implementation optimized for modern hardware

Usage:
    python train_foundation_model.py

Requirements:
    - Training data in parquet format with 'target' column
    - See requirements.txt for dependencies
"""

import logging
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from scipy.signal import detrend, butter, filtfilt
import math
import time
from datetime import datetime, timedelta

class Config:
    """Configuration for foundation model training"""
    # Data paths
    DATA_DIR = Path("./training_datasets")
    CHECKPOINT_DIR = Path("./checkpoints")
    LOG_FILE = CHECKPOINT_DIR / "training.log"
    
    # Model architecture
    N_TIMESTEPS = 512       # Target time series length
    PATCH_SIZE = 16         # Patch size for tokenization
    PATCH_STRIDE = 8        # Patch stride for overlapping windows
    D_MODEL = 192          # Model dimension
    N_HEADS = 6            # Number of attention heads
    N_LAYERS = 4           # Number of transformer layers
    D_FF = 768             # Feed-forward dimension
    DROPOUT = 0.1          # Dropout rate
    
    # Training parameters
    NUM_EPOCHS = 3         # Number of training epochs
    BATCH_SIZE = 256       # Batch size for training
    LEARNING_RATE = 5e-4   # Learning rate
    WEIGHT_DECAY = 1e-5    # Weight decay for regularization
    GRAD_CLIP_NORM = 1.0   # Gradient clipping norm
    
    # Multi-objective loss weights
    W_CONTRASTIVE = 0.4    # Weight for contrastive loss
    W_MASKED = 0.3         # Weight for masked prediction loss
    W_CLASSIFICATION = 0.3 # Weight for classification loss
    
    # Self-supervised learning parameters
    CONTRASTIVE_TEMP = 0.07 # Temperature for contrastive learning
    MASK_RATIO = 0.15       # Ratio of patches to mask

class ButterworthFilter:
    """Butterworth low-pass filter for time series preprocessing"""
    
    def __init__(self, cutoff=0.05, order=4):
        self.cutoff = cutoff
        self.order = order
        self.b, self.a = butter(self.order, self.cutoff / 0.5, btype='low', analog=False)

    def __call__(self, data):
        return filtfilt(self.b, self.a, data)

class MultiDomainTimeSeriesDataset(Dataset):
    """Dataset for multi-domain time series with preprocessing and augmentation"""
    
    def __init__(self, series_data, label_data, config: Config, is_validation=False):
        self.config = config
        self.is_validation = is_validation
        self.series = series_data
        self.labels = label_data
        self.domain_map = {i: i for i in range(len(set(label_data)))}
        self.num_classes = len(self.domain_map)
        self.lowpass_filter = ButterworthFilter()
        
        logging.info(f"Dataset created: {len(self.series)} samples, validation={self.is_validation}")

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        ts = self.series[idx].astype(np.float32)
        label = self.labels[idx]

        # Preprocessing pipeline
        ts = self._preprocess_series(ts)
        ts_tensor = torch.from_numpy(ts).float().unsqueeze(-1)

        if not self.is_validation:
            # Training: return augmented pairs for contrastive learning
            ts_aug_1 = self._augment(ts_tensor)
            ts_aug_2 = self._augment(ts_tensor)
            return ts_aug_1, ts_aug_2, torch.tensor(label).long()
        else:
            # Validation: return original series
            return ts_tensor, torch.tensor(label).long()

    def _preprocess_series(self, ts):
        """Apply comprehensive preprocessing pipeline"""
        # Remove NaN values
        ts = ts[~np.isnan(ts)]
        
        # Detrending
        ts = detrend(ts)
        
        # Outlier clipping (±3σ)
        std = np.std(ts)
        mean = np.mean(ts)
        ts = np.clip(ts, mean - 3 * std, mean + 3 * std)
        
        # Z-score normalization
        if std > 1e-6:
            ts = (ts - mean) / std

        # Resampling to target length
        if len(ts) != self.config.N_TIMESTEPS:
            # Apply anti-aliasing filter for significant downsampling
            if len(ts) > self.config.N_TIMESTEPS * 1.2:
                ts = self.lowpass_filter(ts)
            
            # Linear interpolation to target length
            original_indices = np.linspace(0, 1, len(ts))
            new_indices = np.linspace(0, 1, self.config.N_TIMESTEPS)
            ts = np.interp(new_indices, original_indices, ts)
        
        return ts

    def _augment(self, ts_tensor):
        """Apply data augmentation for contrastive learning"""
        # Gaussian noise injection
        noise = torch.randn_like(ts_tensor) * 0.1
        ts_aug = ts_tensor + noise
        
        # Amplitude scaling
        scale = 1 + (torch.rand(1) - 0.5) * 0.4  # Scale factor: 0.8 to 1.2
        ts_aug *= scale

        return ts_aug

class PatchEmbed(nn.Module):
    """Patch embedding layer for time series tokenization"""
    
    def __init__(self, patch_size, stride, d_model):
        super().__init__()
        self.proj = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T) -> (B, D, N_patches) -> (B, N_patches, D)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = x.permute(0, 2, 1)
        return x

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer input"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class FoundationModel(nn.Module):
    """Transformer-based foundation model for time series classification"""
    
    def __init__(self, config: Config, num_classes: int):
        super().__init__()
        self.patch_embed = PatchEmbed(config.PATCH_SIZE, config.PATCH_STRIDE, config.D_MODEL)
        self.pos_encoder = PositionalEncoding(config.D_MODEL)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.D_MODEL, 
            nhead=config.N_HEADS,
            dim_feedforward=config.D_FF, 
            dropout=config.DROPOUT,
            activation='gelu', 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.N_LAYERS)

        # Multi-objective output heads
        self.contrastive_head = nn.Linear(config.D_MODEL, 128)
        self.masked_pred_head = nn.Linear(config.D_MODEL, config.PATCH_SIZE)
        self.classification_head = nn.Linear(config.D_MODEL, num_classes)

    def forward(self, x, mask=None):
        """Forward pass with optional masking for self-supervised learning"""
        # Patch embedding and positional encoding
        x_patched = self.patch_embed(x)
        x_pos = self.pos_encoder(x_patched)

        # Apply mask if provided
        if mask is not None:
            x_pos = x_pos * mask.unsqueeze(-1)

        # Transformer encoding
        encoded = self.transformer_encoder(x_pos)
        
        # Global average pooling for classification
        pooled_output = encoded.mean(dim=1)
        
        # Multi-objective outputs
        h_contrastive = self.contrastive_head(pooled_output)
        h_masked = self.masked_pred_head(encoded)
        h_class = self.classification_head(pooled_output)
        
        return h_contrastive, h_masked, h_class, x_patched

class Trainer:
    """Comprehensive trainer with multi-objective learning"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.LEARNING_RATE, 
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        self.best_val_loss = float('inf')

        logging.info(f"Trainer initialized on {self.device}")
        logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    def _calculate_contrastive_loss(self, h1, h2):
        """Calculate contrastive loss between two views"""
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        
        batch_size = h1.shape[0]
        labels = torch.arange(batch_size).to(self.device)
        
        # Similarity matrix
        sim_matrix = h1 @ h2.T / self.config.CONTRASTIVE_TEMP
        
        # Symmetric contrastive loss
        loss_i = F.cross_entropy(sim_matrix, labels)
        loss_j = F.cross_entropy(sim_matrix.T, labels)
        
        return (loss_i + loss_j) / 2

    def _calculate_masked_loss(self, y_pred, y_true, patch_mask):
        """Calculate masked prediction loss"""
        B, N_patches, _ = y_pred.shape
        
        # Expand mask to match prediction dimensions
        expanded_mask = patch_mask.unsqueeze(-1).expand_as(y_pred)
        
        # Apply mask to predictions and targets
        y_pred_masked = y_pred[expanded_mask]
        y_true_truncated = y_true[:, :, :y_pred.shape[-1]]
        y_true_masked = y_true_truncated[expanded_mask]
        
        if y_pred_masked.numel() == 0:
            return torch.tensor(0.0, device=self.device)
            
        return F.mse_loss(y_pred_masked, y_true_masked)

    def _train_one_epoch(self, epoch):
        """Train model for one epoch"""
        self.model.train()
        total_loss = 0
        total_batches = len(self.train_loader)
        epoch_start_time = time.time()
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}", 
            leave=True, 
            dynamic_ncols=True
        )
        
        for batch_idx, (ts1, ts2, labels) in enumerate(progress_bar):
            ts1, ts2, labels = ts1.to(self.device), ts2.to(self.device), labels.to(self.device)

            # Create random mask for masked prediction
            with torch.no_grad():
                x_patched = self.model.patch_embed(ts1)
            B, N_patches, _ = x_patched.shape
            patch_mask = torch.rand(B, N_patches) < self.config.MASK_RATIO
            patch_mask_bool = patch_mask.to(self.device)
            input_mask = (~patch_mask_bool).float()

            # Forward pass
            self.optimizer.zero_grad()
            h_contrast_1, h_masked_1, h_class_1, x_patched_1 = self.model(ts1, mask=input_mask)
            h_contrast_2, _, h_class_2, _ = self.model(ts2)

            # Multi-objective loss calculation
            loss_c = self._calculate_contrastive_loss(h_contrast_1, h_contrast_2)
            target_patches = x_patched_1.detach()
            loss_m = self._calculate_masked_loss(h_masked_1, target_patches, patch_mask_bool)
            loss_s = (F.cross_entropy(h_class_1, labels) + F.cross_entropy(h_class_2, labels)) / 2
            
            # Weighted total loss
            loss = (self.config.W_CONTRASTIVE * loss_c + 
                    self.config.W_MASKED * loss_m +
                    self.config.W_CLASSIFICATION * loss_s)

            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP_NORM)
            self.optimizer.step()

            total_loss += loss.item()
            
            # Update progress
            if batch_idx % 10 == 0:
                elapsed_time = time.time() - epoch_start_time
                batches_per_sec = (batch_idx + 1) / elapsed_time if elapsed_time > 0 else 0
                eta_seconds = (total_batches - batch_idx - 1) / batches_per_sec if batches_per_sec > 0 else 0
                eta_str = str(timedelta(seconds=int(eta_seconds)))
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'L_c': f'{loss_c.item():.3f}',
                    'L_m': f'{loss_m.item():.3f}', 
                    'L_s': f'{loss_s.item():.3f}',
                    'ETA': eta_str,
                    'B/s': f'{batches_per_sec:.1f}'
                })
        
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")
        
        self.scheduler.step()
        return total_loss / len(self.train_loader)

    def _validate_one_epoch(self):
        """Validate model performance"""
        self.model.eval()
        total_val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for ts, labels in tqdm(self.val_loader, desc="Validating", leave=False):
                ts, labels = ts.to(self.device), labels.to(self.device)
                
                _, _, h_class, _ = self.model(ts)
                
                loss = F.cross_entropy(h_class, labels)
                total_val_loss += loss.item()
                
                _, predicted = torch.max(h_class.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return total_val_loss / len(self.val_loader), accuracy

    def train(self):
        """Main training loop"""
        training_start_time = time.time()
        
        logging.info("🚀 Starting foundation model training...")
        logging.info(f"Configuration:")
        logging.info(f"  - Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        logging.info(f"  - Epochs: {self.config.NUM_EPOCHS}")
        logging.info(f"  - Batch size: {self.config.BATCH_SIZE}")
        logging.info(f"  - Learning rate: {self.config.LEARNING_RATE}")
        logging.info(f"  - Device: {self.device}")
        
        for epoch in range(self.config.NUM_EPOCHS):
            epoch_start = time.time()
            train_loss = self._train_one_epoch(epoch)
            val_loss, val_acc = self._validate_one_epoch()
            epoch_time = time.time() - epoch_start
            
            # Calculate ETA
            elapsed_total = time.time() - training_start_time
            avg_epoch_time = elapsed_total / (epoch + 1)
            remaining_epochs = self.config.NUM_EPOCHS - (epoch + 1)
            eta_total = remaining_epochs * avg_epoch_time
            eta_str = str(timedelta(seconds=int(eta_total)))
            
            logging.info(
                f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} ({epoch_time:.1f}s) | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"Acc: {val_acc:.2f}% | ETA: {eta_str}"
            )
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, "best_model.pth")
                logging.info(f"💾 Best model saved (val_loss: {val_loss:.4f})")
            
            # Periodic checkpoints
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch, f"checkpoint_epoch_{epoch+1}.pth")
        
        total_time = time.time() - training_start_time
        logging.info(f"✅ Training completed in {str(timedelta(seconds=int(total_time)))}")

    def _save_checkpoint(self, epoch, filename):
        """Save model checkpoint"""
        self.config.CHECKPOINT_DIR.mkdir(exist_ok=True)
        checkpoint_path = self.config.CHECKPOINT_DIR / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, checkpoint_path)

def load_training_data(config: Config):
    """Load and prepare training data from multiple domains"""
    logging.info("Loading multi-domain training data...")
    
    all_series = []
    all_labels = []
    domain_map = {}
    
    # Load data from all parquet files
    for pq_file in sorted(list(config.DATA_DIR.glob("*.parquet"))):
        domain_key = pq_file.stem
        if domain_key not in domain_map:
            domain_map[domain_key] = len(domain_map)
        domain_idx = domain_map[domain_key]
        
        df = pd.read_parquet(pq_file)
        domain_count = 0
        
        # Process each time series in the domain
        for ts_values in tqdm(df['target'], desc=f"Loading {domain_key}", total=len(df)):
            # Quality checks
            if len(ts_values) >= 50 and not (pd.Series(ts_values).isnull().sum() / len(ts_values) > 0.5):
                all_series.append(ts_values)
                all_labels.append(domain_idx)
                domain_count += 1
        
        logging.info(f"  Loaded {domain_count} series from {domain_key}")

    logging.info(f"Total: {len(all_series)} series from {len(domain_map)} domains")
    return all_series, all_labels, domain_map

def main():
    """Main training pipeline"""
    config = Config()
    config.CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_FILE, mode='w'),
            logging.StreamHandler()
        ]
    )

    # Load training data
    all_series, all_labels, domain_map = load_training_data(config)
    
    # Create train/validation split
    indices = list(range(len(all_series)))
    val_size = int(0.1 * len(all_series))
    train_indices, val_indices = random_split(indices, [len(all_series) - val_size, val_size])

    # Prepare datasets
    train_series = [all_series[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    val_series = [all_series[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]

    train_dataset = MultiDomainTimeSeriesDataset(train_series, train_labels, config, is_validation=False)
    val_dataset = MultiDomainTimeSeriesDataset(val_series, val_labels, config, is_validation=True)

    # Create data loaders
    num_workers = min(4, torch.get_num_threads() // 2)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True
    )
    
    # Initialize model and trainer
    model = FoundationModel(config, num_classes=len(domain_map))
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Train model
    trainer.train()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module='torch')
    main() 