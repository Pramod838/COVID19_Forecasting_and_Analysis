"""
Ensemble model that combines Prophet, LSTM, and XGBoost predictions.

Uses weighted averaging with optional dynamic weighting based on recent performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Ensemble forecaster combining multiple models."""
    
    def __init__(self,
                 prophet_weight: float = 0.3,
                 lstm_weight: float = 0.4,
                 xgboost_weight: float = 0.3,
                 dynamic_weighting: bool = True,
                 lookback_window: int = 14):
        """
        Initialize ensemble.
        
        Args:
            prophet_weight: Initial weight for Prophet
            lstm_weight: Initial weight for LSTM
            xgboost_weight: Initial weight for XGBoost
            dynamic_weighting: Whether to adjust weights based on recent performance
            lookback_window: Window for dynamic weight calculation
        """
        # Normalize weights to sum to 1
        total = prophet_weight + lstm_weight + xgboost_weight
        self.weights = {
            'prophet': prophet_weight / total,
            'lstm': lstm_weight / total,
            'xgboost': xgboost_weight / total
        }
        
        self.dynamic_weighting = dynamic_weighting
        self.lookback_window = lookback_window
        
        self.models = {}
        self.weight_history = []
    
    def add_model(self, name: str, model, predictions: np.ndarray):
        """
        Add a model to the ensemble.
        
        Args:
            name: Model name ('prophet', 'lstm', or 'xgboost')
            model: Model object
            predictions: Model predictions
        """
        self.models[name] = {
            'model': model,
            'predictions': predictions
        }
    
    def _calculate_dynamic_weights(self, y_true: np.ndarray,
                                   predictions_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate dynamic weights based on recent performance.
        
        Uses inverse RMSE weighting.
        """
        errors = {}
        
        for name, pred in predictions_dict.items():
            # Calculate recent RMSE
            recent_true = y_true[-self.lookback_window:]
            recent_pred = pred[-self.lookback_window:]
            
            if len(recent_pred) < len(recent_true):
                # Handle length mismatch
                recent_pred = pred[-len(recent_true):]
            
            # Calculate RMSE
            mse = np.mean((recent_true - recent_pred[:len(recent_true)]) ** 2)
            rmse = np.sqrt(mse)
            
            # Avoid division by zero
            errors[name] = max(rmse, 1e-8)
        
        # Inverse error weighting
        inv_errors = {name: 1.0 / err for name, err in errors.items()}
        total_inv = sum(inv_errors.values())
        
        weights = {name: inv_err / total_inv for name, inv_err in inv_errors.items()}
        
        return weights
    
    def _optimize_weights(self, y_true: np.ndarray,
                         predictions_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Optimize ensemble weights to minimize RMSE.
        
        Uses scipy optimization.
        """
        model_names = list(predictions_dict.keys())
        n_models = len(model_names)
        
        # Initial weights
        x0 = np.array([self.weights.get(name, 1.0 / n_models) for name in model_names])
        
        # Constraint: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Bounds: weights between 0 and 1
        bounds = [(0, 1) for _ in range(n_models)]
        
        def objective(weights):
            # Calculate weighted prediction
            ensemble_pred = np.zeros(len(y_true))
            for i, name in enumerate(model_names):
                pred = predictions_dict[name]
                # Ensure same length
                min_len = min(len(ensemble_pred), len(pred))
                ensemble_pred[:min_len] += weights[i] * pred[:min_len]
            
            # Calculate RMSE
            mse = np.mean((y_true - ensemble_pred) ** 2)
            return np.sqrt(mse)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        optimal_weights = {name: result.x[i] for i, name in enumerate(model_names)}
        
        return optimal_weights
    
    def predict(self, predictions_dict: Dict[str, np.ndarray],
               y_true: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Make ensemble prediction.
        
        Args:
            predictions_dict: Dictionary of model predictions
            y_true: True values (for dynamic weighting)
        
        Returns:
            Tuple of (ensemble predictions, weights used)
        """
        # Get model names
        model_names = list(predictions_dict.keys())
        
        if len(model_names) == 0:
            raise ValueError("No models in ensemble!")
        
        # Determine weights
        if self.dynamic_weighting and y_true is not None:
            if len(y_true) >= self.lookback_window:
                weights = self._calculate_dynamic_weights(y_true, predictions_dict)
            else:
                weights = self.weights
        elif y_true is not None:
            # Optimize weights using all data
            weights = self._optimize_weights(y_true, predictions_dict)
        else:
            weights = self.weights
        
        # Store weights for history
        self.weight_history.append(weights.copy())
        
        # Calculate ensemble prediction
        # Find minimum length among all predictions
        min_length = min(len(pred) for pred in predictions_dict.values())
        
        ensemble_pred = np.zeros(min_length)
        
        for name in model_names:
            pred = predictions_dict[name][:min_length]
            weight = weights.get(name, 0)
            ensemble_pred += weight * pred
        
        return ensemble_pred, weights
    
    def get_prediction_intervals(self, predictions_dict: Dict[str, np.ndarray],
                                 confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals using model disagreement.
        
        Args:
            predictions_dict: Dictionary of model predictions
            confidence: Confidence level (e.g., 0.95 for 95%)
        
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Stack predictions
        preds_array = np.array(list(predictions_dict.values()))
        
        # Calculate mean and std across models
        mean_pred = np.mean(preds_array, axis=0)
        std_pred = np.std(preds_array, axis=0)
        
        # For 95% confidence, use approximately 2 standard deviations
        z_score = 1.96 if confidence == 0.95 else 1.0
        
        lower = mean_pred - z_score * std_pred
        upper = mean_pred + z_score * std_pred
        
        # Ensure non-negative for cases
        lower = np.clip(lower, 0, None)
        
        return lower, upper
    
    def get_weight_history(self) -> pd.DataFrame:
        """Get history of weight changes over time."""
        if not self.weight_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.weight_history)
    
    def evaluate(self, y_true: np.ndarray, predictions_dict: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Evaluate all models and ensemble.
        
        Args:
            y_true: True values
            predictions_dict: Dictionary of predictions
        
        Returns:
            Dictionary of evaluation results
        """
        from ..evaluation.metrics import MetricsCalculator
        
        results = {}
        
        # Evaluate individual models
        for name, pred in predictions_dict.items():
            # Ensure same length
            min_len = min(len(y_true), len(pred))
            metrics = MetricsCalculator.calculate_all(y_true[:min_len], pred[:min_len])
            results[name] = metrics
        
        # Evaluate ensemble
        ensemble_pred, _ = self.predict(predictions_dict, y_true)
        min_len = min(len(y_true), len(ensemble_pred))
        ensemble_metrics = MetricsCalculator.calculate_all(y_true[:min_len], ensemble_pred[:min_len])
        results['ensemble'] = ensemble_metrics
        
        return results
    
    def save_weights(self, filepath: str):
        """Save current weights to file."""
        import json
        
        data = {
            'weights': self.weights,
            'dynamic_weighting': self.dynamic_weighting,
            'lookback_window': self.lookback_window,
            'weight_history': self.weight_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Weights saved to {filepath}")
    
    @classmethod
    def load_weights(cls, filepath: str) -> 'EnsembleModel':
        """Load weights from file and create instance."""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        instance = cls(
            prophet_weight=data['weights'].get('prophet', 0.33),
            lstm_weight=data['weights'].get('lstm', 0.33),
            xgboost_weight=data['weights'].get('xgboost', 0.34),
            dynamic_weighting=data.get('dynamic_weighting', True),
            lookback_window=data.get('lookback_window', 14)
        )
        
        instance.weight_history = data.get('weight_history', [])
        
        logger.info(f"Weights loaded from {filepath}")
        return instance


class StackingEnsemble:
    """
    Stacking ensemble using a meta-learner.
    
    Uses predictions from base models as features for a meta-model.
    """
    
    def __init__(self, meta_learner=None):
        """
        Initialize stacking ensemble.
        
        Args:
            meta_learner: Meta-learner model (default: Ridge regression)
        """
        if meta_learner is None:
            from sklearn.linear_model import Ridge
            self.meta_learner = Ridge(alpha=1.0)
        else:
            self.meta_learner = meta_learner
        
        self.base_predictions = {}
    
    def fit(self, predictions_dict: Dict[str, np.ndarray], y_true: np.ndarray):
        """
        Fit meta-learner on base model predictions.
        
        Args:
            predictions_dict: Dictionary of base model predictions
            y_true: True values
        """
        # Stack predictions
        X_meta = np.column_stack(list(predictions_dict.values()))
        
        # Ensure same length
        min_len = min(len(X_meta), len(y_true))
        X_meta = X_meta[:min_len]
        y_true = y_true[:min_len]
        
        # Fit meta-learner
        self.meta_learner.fit(X_meta, y_true)
        
        # Store feature names (model names)
        self.base_predictions = list(predictions_dict.keys())
        
        logger.info(f"Meta-learner fitted with coefficients: {self.meta_learner.coef_}")
        
        return self
    
    def predict(self, predictions_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Make predictions using meta-learner.
        
        Args:
            predictions_dict: Dictionary of base model predictions
        
        Returns:
            Predictions from meta-learner
        """
        # Stack predictions in same order
        X_meta = np.column_stack([
            predictions_dict[name] for name in self.base_predictions
        ])
        
        return self.meta_learner.predict(X_meta)
