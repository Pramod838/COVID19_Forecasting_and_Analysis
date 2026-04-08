"""
Evaluation metrics and validation utilities for time series forecasting.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MAPE (Mean Absolute Percentage Error).
    
    Avoids division by zero by clipping denominator.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    if mask.sum() == 0:
        return np.inf
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def mean_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MDA (Mean Directional Accuracy).
    
    Measures whether the model correctly predicts the direction of change.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate direction of change
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    
    # Calculate accuracy
    return np.mean(true_direction == pred_direction) * 100


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate RMSE."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


class MetricsCalculator:
    """Calculate multiple evaluation metrics."""
    
    METRICS = {
        'mae': mean_absolute_error,
        'rmse': root_mean_squared_error,
        'mape': mean_absolute_percentage_error,
        'mda': mean_directional_accuracy,
        'r2': r2_score
    }
    
    @classmethod
    def calculate_all(cls, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all metrics."""
        results = {}
        for name, func in cls.METRICS.items():
            try:
                results[name] = func(y_true, y_pred)
            except Exception as e:
                logger.warning(f"Failed to calculate {name}: {e}")
                results[name] = np.nan
        return results
    
    @classmethod
    def print_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray, 
                     prefix: str = ""):
        """Calculate and print all metrics."""
        metrics = cls.calculate_all(y_true, y_pred)
        
        print(f"\n{prefix}Metrics:")
        print("-" * 40)
        for name, value in metrics.items():
            if not np.isnan(value) and not np.isinf(value):
                print(f"  {name.upper():8}: {value:.4f}")
            else:
                print(f"  {name.upper():8}: N/A")
        
        return metrics


class TimeSeriesValidator:
    """
    Time series cross-validation.
    
    Unlike regular CV, this respects temporal ordering.
    """
    
    def __init__(self, n_splits: int = 5, gap: int = 0):
        """
        Initialize validator.
        
        Args:
            n_splits: Number of splits
            gap: Gap between train and test (to avoid data leakage)
        """
        self.n_splits = n_splits
        self.gap = gap
    
    def expanding_window_split(self, df: pd.DataFrame, 
                               date_col: str = 'date') -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create expanding window splits.
        
        Each split uses all previous data as train and next block as test.
        """
        df = df.sort_values(date_col).reset_index(drop=True)
        n_samples = len(df)
        
        splits = []
        
        # Calculate split points
        test_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Train: from start to (i+1) * test_size
            # Test: from (i+1) * test_size to (i+2) * test_size
            train_end = (i + 1) * test_size
            test_start = train_end + self.gap
            test_end = min(test_start + test_size, n_samples)
            
            if test_start >= n_samples:
                break
            
            train_df = df.iloc[:train_end]
            test_df = df.iloc[test_start:test_end]
            
            splits.append((train_df, test_df))
        
        return splits
    
    def sliding_window_split(self, df: pd.DataFrame, 
                            window_size: int,
                            date_col: str = 'date') -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create sliding window splits.
        
        Each split uses a fixed-size window as train.
        """
        df = df.sort_values(date_col).reset_index(drop=True)
        n_samples = len(df)
        
        splits = []
        
        # Calculate step size
        step = (n_samples - window_size) // self.n_splits
        
        for i in range(self.n_splits):
            train_start = i * step
            train_end = train_start + window_size
            test_start = train_end + self.gap
            test_end = min(test_start + step, n_samples)
            
            if test_start >= n_samples:
                break
            
            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[test_start:test_end]
            
            splits.append((train_df, test_df))
        
        return splits
    
    def time_based_split(self, df: pd.DataFrame, 
                        train_days: int,
                        test_days: int,
                        date_col: str = 'date') -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create splits based on calendar days.
        """
        df = df.sort_values(date_col).reset_index(drop=True)
        
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        
        splits = []
        
        current_start = min_date
        
        for i in range(self.n_splits):
            train_end = current_start + pd.Timedelta(days=train_days)
            test_start = train_end + pd.Timedelta(days=self.gap)
            test_end = test_start + pd.Timedelta(days=test_days)
            
            if test_end > max_date:
                break
            
            train_df = df[df[date_col] < train_end]
            test_df = df[(df[date_col] >= test_start) & (df[date_col] < test_end)]
            
            if len(train_df) > 0 and len(test_df) > 0:
                splits.append((train_df, test_df))
            
            # Move window forward
            current_start = test_start
        
        return splits


class ForecastEvaluator:
    """Evaluate forecasts with multiple horizons."""
    
    def __init__(self, horizons: List[int] = None):
        """
        Initialize evaluator.
        
        Args:
            horizons: List of forecast horizons (days) to evaluate
        """
        self.horizons = horizons or [1, 3, 7, 14]
    
    def evaluate_horizon(self, y_true: np.ndarray, y_pred: np.ndarray,
                        horizon: int) -> Dict[str, float]:
        """Evaluate forecast at specific horizon."""
        # Adjust for horizon
        if len(y_true) > horizon:
            y_true_horizon = y_true[horizon:]
            y_pred_horizon = y_pred[:-horizon] if len(y_pred) > horizon else y_pred
            
            # Ensure same length
            min_len = min(len(y_true_horizon), len(y_pred_horizon))
            y_true_horizon = y_true_horizon[:min_len]
            y_pred_horizon = y_pred_horizon[:min_len]
            
            return MetricsCalculator.calculate_all(y_true_horizon, y_pred_horizon)
        
        return {k: np.nan for k in MetricsCalculator.METRICS.keys()}
    
    def evaluate_all_horizons(self, y_true: np.ndarray, 
                             y_pred: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Evaluate all forecast horizons."""
        results = {}
        for horizon in self.horizons:
            results[horizon] = self.evaluate_horizon(y_true, y_pred, horizon)
        return results
    
    def print_horizon_results(self, results: Dict[int, Dict[str, float]]):
        """Print results for all horizons."""
        print("\nForecast Evaluation by Horizon:")
        print("=" * 60)
        
        for horizon, metrics in results.items():
            print(f"\nHorizon: {horizon} day(s)")
            print("-" * 40)
            for metric, value in metrics.items():
                if not np.isnan(value) and not np.isinf(value):
                    print(f"  {metric.upper():8}: {value:.4f}")


def train_test_split_by_date(df: pd.DataFrame, 
                             test_size: float = 0.2,
                             val_size: float = 0.1,
                             date_col: str = 'date') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train/val/test by date.
    
    Args:
        df: DataFrame with time series data
        test_size: Proportion for test set
        val_size: Proportion for validation set (from training data)
        date_col: Date column name
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    
    n_samples = len(df)
    test_start = int(n_samples * (1 - test_size))
    
    # First split: train+val vs test
    train_val_df = df.iloc[:test_start]
    test_df = df.iloc[test_start:]
    
    # Second split: train vs val
    n_train_val = len(train_val_df)
    val_start = int(n_train_val * (1 - val_size))
    
    train_df = train_val_df.iloc[:val_start]
    val_df = train_val_df.iloc[val_start:]
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df
