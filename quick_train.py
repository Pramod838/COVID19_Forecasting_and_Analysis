"""Quick model training demonstration."""

import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.evaluation.metrics import MetricsCalculator, train_test_split_by_date
from src.features.builders import FeatureBuilder

print('='*60)
print('MODEL TRAINING DEMONSTRATION')
print('='*60)

# Load data
features_df = pd.read_csv('data/processed/features_full.csv', parse_dates=['date'])

# Get features
builder = FeatureBuilder()
feature_cols = builder.get_feature_columns(features_df, target_col='new_cases')

# Split
train_df, val_df, test_df = train_test_split_by_date(features_df, test_size=0.15, val_size=0.1)

print(f"\nData Splits:")
print(f"  Train: {len(train_df)} samples ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
print(f"  Val:   {len(val_df)} samples ({val_df['date'].min().date()} to {val_df['date'].max().date()})")
print(f"  Test:  {len(test_df)} samples ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

# Train simple baseline (7-day moving average)
print("\n" + '='*60)
print("BASELINE: 7-Day Moving Average")
print('='*60)

y_test = test_df['new_cases'].values
baseline_pred = test_df['new_cases'].shift(1).rolling(window=7).mean().values
baseline_pred = np.nan_to_num(baseline_pred, nan=test_df['new_cases'].mean())

metrics = MetricsCalculator.calculate_all(y_test, baseline_pred)
print(f"MAE:  {metrics['mae']:,.0f}")
print(f"RMSE: {metrics['rmse']:,.0f}")
print(f"MAPE: {metrics['mape']:.1f}%")
print(f"MDA:  {metrics['mda']:.1f}%")

# Train Prophet
print("\n" + '='*60)
print("MODEL 1: Facebook Prophet")
print('='*60)

try:
    from src.models.prophet_model import ProphetModel
    
    prophet = ProphetModel()
    regressors = [c for c in ['workplaces', 'retail_recreation'] if c in train_df.columns]
    prophet.fit(train_df, target_col='new_cases', regressors=regressors)
    
    test_pred = prophet.predict_in_sample(test_df)
    y_pred_prophet = test_pred['yhat'].values
    
    metrics = MetricsCalculator.calculate_all(y_test, y_pred_prophet)
    print(f"MAE:  {metrics['mae']:,.0f}")
    print(f"RMSE: {metrics['rmse']:,.0f}")
    print(f"MAPE: {metrics['mape']:.1f}%")
    print(f"MDA:  {metrics['mda']:.1f}%")
    print("\n✓ Prophet model trained")
except Exception as e:
    print(f"Prophet error: {e}")

# Train XGBoost
print("\n" + '='*60)
print("MODEL 2: XGBoost")
print('='*60)

try:
    from src.models.xgboost_model import XGBoostModel
    
    X_train = train_df[feature_cols].values
    y_train = train_df['new_cases'].values
    X_test = test_df[feature_cols].values
    
    xgb = XGBoostModel(n_estimators=100, max_depth=5)
    xgb.fit(X_train, y_train, feature_names=feature_cols)
    y_pred_xgb = xgb.predict(X_test)
    
    metrics = MetricsCalculator.calculate_all(y_test, y_pred_xgb)
    print(f"MAE:  {metrics['mae']:,.0f}")
    print(f"RMSE: {metrics['rmse']:,.0f}")
    print(f"MAPE: {metrics['mape']:.1f}%")
    print(f"MDA:  {metrics['mda']:.1f}%")
    
    # Feature importance
    importance = xgb.get_feature_importance()
    print("\nTop 5 Features:")
    for i, row in importance.head(5).iterrows():
        print(f"  {row['feature']:25}: {row['importance']:.0f}")
    
    print("\n✓ XGBoost model trained")
except Exception as e:
    print(f"XGBoost error: {e}")

# Ensemble
print("\n" + '='*60)
print("ENSEMBLE MODEL")
print('='*60)

try:
    from src.models.ensemble import EnsembleModel
    
    # Create simple ensemble with available predictions
    ensemble = EnsembleModel(prophet_weight=0.4, lstm_weight=0.0, xgboost_weight=0.6)
    
    predictions_dict = {
        'prophet': y_pred_prophet if 'y_pred_prophet' in dir() else baseline_pred,
        'xgboost': y_pred_xgb if 'y_pred_xgb' in dir() else baseline_pred
    }
    
    y_pred_ensemble, weights = ensemble.predict(predictions_dict, y_test)
    
    metrics = MetricsCalculator.calculate_all(y_test[:len(y_pred_ensemble)], y_pred_ensemble)
    print(f"MAE:  {metrics['mae']:,.0f}")
    print(f"RMSE: {metrics['rmse']:,.0f}")
    print(f"MAPE: {metrics['mape']:.1f}%")
    print(f"MDA:  {metrics['mda']:.1f}%")
    print(f"\nEnsemble weights: {weights}")
    print("\n✓ Ensemble model trained")
except Exception as e:
    print(f"Ensemble error: {e}")

print("\n" + '='*60)
print("TRAINING COMPLETE!")
print('='*60)
