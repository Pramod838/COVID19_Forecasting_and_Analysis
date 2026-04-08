"""
PyTorch LSTM model with attention mechanism for time series forecasting.

This model uses:
- Bidirectional LSTM for capturing patterns in both directions
- Multi-head self-attention for focusing on important time steps
- Dropout for regularization
- Layer normalization for stable training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """Dataset for time series sequences."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 sequence_length: int = 14):
        """
        Initialize dataset.
        
        Args:
            X: Features array (n_samples, n_features)
            y: Target array (n_samples,)
            sequence_length: Length of input sequences
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.X) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        # Get sequence
        x_seq = self.X[idx:idx + self.sequence_length]
        # Target is the value at the end of sequence
        y_target = self.y[idx + self.sequence_length - 1]
        return x_seq, y_target


class AttentionLayer(nn.Module):
    """
    Multi-head self-attention layer.
    
    Allows model to focus on different parts of the sequence.
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super(AttentionLayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        
        return x, attn_weights


class LSTMAttentionModel(nn.Module):
    """
    LSTM with attention for time series forecasting.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        super(LSTMAttentionModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate output size after LSTM
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention layer
        self.attention = AttentionLayer(lstm_output_size, num_heads, dropout)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        fc_input_size = lstm_output_size
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * num_directions)
        
        # Attention
        attn_out, attn_weights = self.attention(lstm_out)
        
        # Global average pooling over sequence dimension
        # Transpose for pooling: (batch, hidden_size, seq_len)
        attn_out = attn_out.transpose(1, 2)
        pooled = self.global_pool(attn_out).squeeze(-1)  # (batch, hidden_size)
        
        # Fully connected
        output = self.fc(pooled).squeeze(-1)
        
        return output, attn_weights


class LSTMForecaster:
    """Wrapper for LSTM model with training and prediction logic."""
    
    def __init__(self,
                 sequence_length: int = 14,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 patience: int = 15,
                 device: str = None):
        """
        Initialize LSTM forecaster.
        
        Args:
            sequence_length: Length of input sequences
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size for training
            epochs: Maximum training epochs
            patience: Early stopping patience
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.feature_names = None
        self.scaler = None
    
    def _normalize_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize features to [0, 1] range."""
        if fit:
            self.scaler = {
                'min': X.min(axis=0),
                'max': X.max(axis=0),
                'range': X.max(axis=0) - X.min(axis=0)
            }
            # Avoid division by zero
            self.scaler['range'] = np.where(self.scaler['range'] == 0, 1, self.scaler['range'])
        
        X_norm = (X - self.scaler['min']) / self.scaler['range']
        return X_norm
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
           X_val: np.ndarray = None, y_val: np.ndarray = None,
           feature_names: List[str] = None):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training targets (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            feature_names: Names of features
        """
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        
        # Normalize
        X_train_norm = self._normalize_features(X_train, fit=True)
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train_norm, y_train, self.sequence_length)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Validation dataset
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_norm = self._normalize_features(X_val, fit=False)
            val_dataset = TimeSeriesDataset(X_val_norm, y_val, self.sequence_length)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize model
        input_size = X_train.shape[1]
        self.model = LSTMAttentionModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            bidirectional=True
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Training LSTM model for up to {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs, _ = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs, _ = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_losses.append(loss.item())
                
                avg_val_loss = np.mean(val_losses)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs} - "
                              f"Train Loss: {avg_train_loss:.4f}, "
                              f"Val Loss: {avg_val_loss:.4f}")
                
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    # Restore best model
                    self.model.load_state_dict(self.best_state)
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs} - "
                              f"Train Loss: {avg_train_loss:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Predictions (n_samples - sequence_length + 1,)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        
        # Normalize
        X_norm = self._normalize_features(X, fit=False)
        
        # Create dataset
        dataset = TimeSeriesDataset(X_norm, np.zeros(len(X_norm)), self.sequence_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs, _ = self.model(batch_X)
                predictions.extend(outputs.cpu().numpy())
        
        return np.array(predictions)
    
    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Get attention weights for interpretability.
        
        Args:
            X: Single sequence or batch of sequences
        
        Returns:
            Attention weights
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        
        # Handle single sample
        if len(X.shape) == 2:
            X = X.reshape(1, *X.shape)
        
        X_norm = self._normalize_features(X, fit=False)
        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        
        with torch.no_grad():
            _, attn_weights = self.model(X_tensor)
        
        return attn_weights.cpu().numpy()
    
    def save(self, filepath: str):
        """Save model to file."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, device: str = None):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=device or 'cpu')
        
        # Create instance with saved parameters
        instance = cls(
            sequence_length=checkpoint['sequence_length'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            num_heads=checkpoint['num_heads'],
            dropout=checkpoint['dropout'],
            device=device
        )
        
        # Restore scaler and feature names
        instance.scaler = checkpoint['scaler']
        instance.feature_names = checkpoint['feature_names']
        
        # Initialize model
        input_size = len(instance.feature_names)
        instance.model = LSTMAttentionModel(
            input_size=input_size,
            hidden_size=instance.hidden_size,
            num_layers=instance.num_layers,
            num_heads=instance.num_heads,
            dropout=instance.dropout
        ).to(instance.device)
        
        # Load weights
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.model.eval()
        
        logger.info(f"Model loaded from {filepath}")
        return instance
