"""
XGBoost model for COVID-19 forecasting.

XGBoost is excellent for:
- Tabular time series features
- Feature importance analysis
- Handling non-linear relationships
- Robustness to outliers
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import pickle

logger = logging.getLogger(__name__)


class XGBoostModel:
    """Wrapper for XGBoost time series forecasting."""
    
    def __init__(self,
                 n_estimators: int = 500,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 objective: str = 'reg:squarederror',
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: int = 3,
                 gamma: float = 0,
                 early_stopping_rounds: int = 50,
                 random_state: int = 42):
        """
        Initialize XGBoost model.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Row subsample ratio
            colsample_bytree: Column subsample ratio
            objective: Objective function
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            min_child_weight: Minimum sum of instance weight in child
            gamma: Minimum loss reduction for split
            early_stopping_rounds: Rounds for early stopping
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.objective = objective
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        
        self.model = None
        self.feature_names = None
        self.best_iteration = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
           X_val: Optional[np.ndarray] = None,
           y_val: Optional[np.ndarray] = None,
           feature_names: Optional[List[str]] = None,
           eval_metric: str = 'rmse'):
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            feature_names: Names of features
            eval_metric: Evaluation metric
        """
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("XGBoost not installed. Install with: pip install xgboost")
            raise
        
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            evals.append((dval, 'validation'))
        
        # Parameters
        params = {
            'objective': self.objective,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'seed': self.random_state,
            'eval_metric': eval_metric
        }
        
        logger.info(f"Training XGBoost with {self.n_estimators} trees...")
        
        # Train
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds if len(evals) > 1 else None,
            verbose_eval=50
        )
        
        self.best_iteration = self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.n_estimators
        
        logger.info(f"Training complete. Best iteration: {self.best_iteration}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
        
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("XGBoost not installed")
            raise
        
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        
        # Predict with best iteration if available
        if self.best_iteration:
            predictions = self.model.predict(dtest, iteration_range=(0, self.best_iteration))
        else:
            predictions = self.model.predict(dtest)
        
        return predictions
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            importance_type: 'weight', 'gain', or 'cover'
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        importance = self.model.get_score(importance_type=importance_type)
        
        # Convert to DataFrame
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ])
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Get SHAP values for interpretability.
        
        Args:
            X: Features
        
        Returns:
            SHAP values
        """
        try:
            import shap
            import xgboost as xgb
        except ImportError:
            logger.warning("SHAP not installed. Install with: pip install shap")
            return None
        
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Create explainer
        explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X)
        
        return shap_values
    
    def save(self, filepath: str):
        """Save model to file."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.save_model(filepath)
        
        # Save additional metadata
        metadata = {
            'feature_names': self.feature_names,
            'best_iteration': self.best_iteration,
            'params': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate
            }
        }
        
        metadata_path = filepath.replace('.json', '_metadata.pkl').replace('.model', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load model from file."""
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("XGBoost not installed")
            raise
        
        # Load metadata
        metadata_path = filepath.replace('.json', '_metadata.pkl').replace('.model', '_metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        instance = cls(**metadata['params'])
        instance.feature_names = metadata['feature_names']
        instance.best_iteration = metadata['best_iteration']
        
        # Load model
        instance.model = xgb.Booster()
        instance.model.load_model(filepath)
        
        logger.info(f"Model loaded from {filepath}")
        return instance


class XGBoostTuner:
    """Hyperparameter tuning for XGBoost using Optuna."""
    
    def __init__(self, n_trials: int = 50, timeout: int = 600):
        """
        Initialize tuner.
        
        Args:
            n_trials: Number of Optuna trials
            timeout: Maximum time in seconds
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_params = None
    
    def tune(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            feature_names: List[str] = None) -> Dict:
        """
        Tune hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: Feature names
        
        Returns:
            Best parameters
        """
        try:
            import optuna
            import xgboost as xgb
        except ImportError:
            logger.error("Optuna not installed. Install with: pip install optuna")
            self.best_params = None
            return None
        
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'seed': 42,
                'eval_metric': 'rmse'
            }
            
            # Train model
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=params['n_estimators'],
                evals=[(dval, 'validation')],
                early_stopping_rounds=20,
                verbose_eval=False
            )
            
            # Evaluate
            predictions = model.predict(dval)
            rmse = np.sqrt(np.mean((y_val - predictions) ** 2))
            
            return rmse
        
        # Create study
        try:
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=True)
            
            self.best_params = study.best_params
            logger.info(f"Best RMSE: {study.best_value:.4f}")
            logger.info(f"Best params: {self.best_params}")
            
            return self.best_params
        except Exception as e:
            logger.error(f"Tuning failed: {e}")
            self.best_params = None
            return None
    
    def create_best_model(self, **kwargs) -> XGBoostModel:
        """Create XGBoost model with best parameters."""
        if self.best_params is None:
            raise ValueError("Must run tune() first!")
        
        return XGBoostModel(
            n_estimators=self.best_params.get('n_estimators', 500),
            max_depth=self.best_params.get('max_depth', 6),
            learning_rate=self.best_params.get('learning_rate', 0.1),
            subsample=self.best_params.get('subsample', 0.8),
            colsample_bytree=self.best_params.get('colsample_bytree', 0.8),
            reg_alpha=self.best_params.get('reg_alpha', 0.1),
            reg_lambda=self.best_params.get('reg_lambda', 1.0),
            min_child_weight=self.best_params.get('min_child_weight', 3),
            gamma=self.best_params.get('gamma', 0),
            **kwargs
        )
