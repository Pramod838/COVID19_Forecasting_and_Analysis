
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path

from src.models.prophet_model import ProphetModel
from src.models.lstm_model import LSTMForecaster
from src.models.xgboost_model import XGBoostModel, XGBoostTuner
from src.models.ensemble import EnsembleModel
from src.evaluation.metrics import MetricsCalculator, train_test_split_by_date
from src.features.builders import FeatureBuilder, prepare_features_for_modeling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_prophet(train_df, val_df, test_df, config):
  
    logger.info("="*50)
    logger.info("Training Prophet Model")
    logger.info("="*50)
    
    regressors = config['models']['prophet'].get('regressors', ['workplaces', 'retail_recreation'])
    regressors = [r for r in regressors if r in train_df.columns]
    
    model = ProphetModel(**{k: v for k, v in config['models']['prophet'].items() if k != 'regressors'})
    model.fit(train_df, target_col='new_cases', regressors=regressors)

    test_predictions = model.predict_in_sample(test_df)
    y_pred = test_predictions['yhat'].values
    
    metrics = MetricsCalculator.calculate_all(test_df['new_cases'].values, y_pred)
    logger.info(f"Prophet Test MAE: {metrics['mae']:.0f}")
   
    os.makedirs('models', exist_ok=True)
    model.save('models/prophet_model.pkl')
    
    return y_pred, model


def train_lstm(X_train, y_train, X_val, y_val, X_test, feature_names, config):
    """Train LSTM model."""
    logger.info("\n" + "="*50)
    logger.info("Training LSTM Model")
    logger.info("="*50)
    
    lstm_config = config['models']['lstm']
    model = LSTMForecaster(**lstm_config)
    
    model.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)
    
    y_pred = model.predict(X_test)
    
    y_test_adjusted = y_test[lstm_config['sequence_length']-1:lstm_config['sequence_length']-1+len(y_pred)]
    metrics = MetricsCalculator.calculate_all(y_test_adjusted, y_pred)
    logger.info(f"LSTM Test MAE: {metrics['mae']:.0f}")
    
    model.save('models/lstm_model.pt')
    
    return y_pred, model


def train_xgboost(X_train, y_train, X_val, y_val, X_test, feature_names, config):
    """Train XGBoost model."""
    logger.info("\n" + "="*50)
    logger.info("Training XGBoost Model")
    logger.info("="*50)
    
    if config.get('tune_xgboost', False):
        tuner = XGBoostTuner(n_trials=20, timeout=300)
        best_params = tuner.tune(X_train, y_train, X_val, y_val, feature_names)
        model = tuner.create_best_model()
    else:
        model = XGBoostModel(**config['models']['xgboost'])
    
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    model.fit(X_train_full, y_train_full, feature_names=feature_names)
    
    y_pred = model.predict(X_test)
    
    y_test_actual = test_df['new_cases'].values
    metrics = MetricsCalculator.calculate_all(y_test_actual, y_pred)
    logger.info(f"XGBoost Test MAE: {metrics['mae']:.0f}")
    
    model.save('models/xgboost_model.json')
    
    return y_pred, model


def train_ensemble(predictions_dict, y_test, config):
    """Train ensemble."""
    logger.info("\n" + "="*50)
    logger.info("Training Ensemble")
    logger.info("="*50)
    
    ensemble = EnsembleModel(
        prophet_weight=config['models']['ensemble']['prophet_weight'],
        lstm_weight=config['models']['ensemble']['lstm_weight'],
        xgboost_weight=config['models']['ensemble']['xgboost_weight'],
        dynamic_weighting=False
    )
    
    y_pred, weights = ensemble.predict(predictions_dict, y_test)
    
    metrics = MetricsCalculator.calculate_all(y_test[:len(y_pred)], y_pred)
    logger.info(f"Ensemble Test MAE: {metrics['mae']:.0f}")
    logger.info(f"Ensemble weights: {weights}")
    
    ensemble.save_weights('models/ensemble_weights.json')
    
    return y_pred


def main():
    """Main training pipeline."""
    logger.info("Starting Model Training Pipeline")
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Loading data...")
    df = pd.read_csv('data/processed/features_full.csv', parse_dates=['date'])
    
    builder = FeatureBuilder()
    feature_cols = builder.get_feature_columns(df, target_col='new_cases')
    
    df_clean = df.dropna(subset=feature_cols + ['new_cases'])
    
    train_df, val_df, test_df = train_test_split_by_date(
        df_clean, 
        test_size=config['split']['test_size'],
        val_size=config['split']['val_size']
    )
    
    X_train = train_df[feature_cols].values
    y_train = train_df['new_cases'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['new_cases'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['new_cases'].values
    
    logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    predictions_dict = {}
    
    prophet_pred, prophet_model = train_prophet(train_df, val_df, test_df, config)
    predictions_dict['prophet'] = prophet_pred
    
    lstm_pred, lstm_model = train_lstm(X_train, y_train, X_val, y_val, X_test, feature_cols, config)
    predictions_dict['lstm'] = lstm_pred
    
    # XGBoost
    xgb_pred, xgb_model = train_xgboost(X_train, y_train, X_val, y_val, X_test, feature_cols, config)
    predictions_dict['xgboost'] = xgb_pred
    
    # Ensemble
    ensemble_pred = train_ensemble(predictions_dict, y_test, config)
    
    # Final summary
    logger.info("\n" + "="*50)
    logger.info("Training Complete!")
    logger.info("="*50)
    logger.info("Models saved in: models/")
    logger.info("  - prophet_model.pkl")
    logger.info("  - lstm_model.pt")
    logger.info("  - xgboost_model.json")
    logger.info("  - ensemble_weights.json")


if __name__ == "__main__":
    main()
