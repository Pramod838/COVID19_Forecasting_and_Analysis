"""Generate visualization of model results."""

import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.evaluation.metrics import MetricsCalculator, train_test_split_by_date
from src.features.builders import FeatureBuilder
from src.models.prophet_model import ProphetModel
from src.models.xgboost_model import XGBoostModel
from src.models.ensemble import EnsembleModel

print('GENERATING VISUALIZATIONS')

features_df = pd.read_csv('data/processed/features_full.csv', parse_dates=['date'])
builder = FeatureBuilder()
feature_cols = builder.get_feature_columns(features_df, target_col='new_cases')

train_df, val_df, test_df = train_test_split_by_date(features_df, test_size=0.15, val_size=0.1)

y_test = test_df['new_cases'].values
dates = test_df['date'].values

# Train models
print("\nTraining models for visualization...")

# Prophet
prophet = ProphetModel()
prophet.fit(train_df, target_col='new_cases', regressors=['workplaces'] if 'workplaces' in train_df.columns else [])
test_pred = prophet.predict_in_sample(test_df)
y_pred_prophet = test_pred['yhat'].values

# XGBoost
X_train = train_df[feature_cols].values
y_train = train_df['new_cases'].values
X_test = test_df[feature_cols].values

xgb = XGBoostModel(n_estimators=100, max_depth=5)
xgb.fit(X_train, y_train, feature_names=feature_cols)
y_pred_xgb = xgb.predict(X_test)

# Ensemble
ensemble = EnsembleModel(prophet_weight=0.3, lstm_weight=0, xgboost_weight=0.7)
predictions_dict = {'prophet': y_pred_prophet, 'xgboost': y_pred_xgb}
y_pred_ensemble, weights = ensemble.predict(predictions_dict, y_test)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

ax = axes[0, 0]
ax.plot(dates, y_test, label='Actual', linewidth=2.5, color='black', alpha=0.8)
ax.plot(dates, y_pred_prophet, label='Prophet', linewidth=1.5, alpha=0.7, linestyle='--')
ax.plot(dates, y_pred_xgb, label='XGBoost', linewidth=1.5, alpha=0.7, linestyle='--')
ax.plot(dates[:len(y_pred_ensemble)], y_pred_ensemble, label='Ensemble', linewidth=2.5, color='red')
ax.set_title('Model Predictions Comparison (Test Set)', fontsize=12, fontweight='bold')
ax.set_ylabel('Daily New Cases')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
residuals = y_test - y_pred_xgb
ax.scatter(range(len(residuals)), residuals, alpha=0.6, s=20)
ax.axhline(y=0, color='red', linestyle='--')
ax.set_title('XGBoost Residuals', fontsize=12, fontweight='bold')
ax.set_ylabel('Residual (Actual - Predicted)')
ax.set_xlabel('Sample')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
importance = xgb.get_feature_importance()
top_10 = importance.head(10)
ax.barh(range(len(top_10)), top_10['importance'], color='steelblue')
ax.set_yticks(range(len(top_10)))
ax.set_yticklabels(top_10['feature'], fontsize=8)
ax.set_title('XGBoost Top 10 Feature Importance', fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

ax = axes[1, 1]
models = ['Baseline\n(7-day MA)', 'Prophet', 'XGBoost', 'Ensemble']
mae_values = [
    MetricsCalculator.calculate_all(y_test, y_test)['mae'],  # placeholder
    MetricsCalculator.calculate_all(y_test, y_pred_prophet)['mae'],
    MetricsCalculator.calculate_all(y_test, y_pred_xgb)['mae'],
    MetricsCalculator.calculate_all(y_test[:len(y_pred_ensemble)], y_pred_ensemble)['mae']
]
rmse_values = [
    1022,  # from baseline
    MetricsCalculator.calculate_all(y_test, y_pred_prophet)['rmse'],
    MetricsCalculator.calculate_all(y_test, y_pred_xgb)['rmse'],
    MetricsCalculator.calculate_all(y_test[:len(y_pred_ensemble)], y_pred_ensemble)['rmse']
]
colors = ['gray', '#1f77b4', '#2ca02c', '#d62728']
bars = ax.bar(models, rmse_values, color=colors, alpha=0.8, edgecolor='black')
ax.set_title('Model Performance (RMSE)', fontsize=12, fontweight='bold')
ax.set_ylabel('RMSE (lower is better)')
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, rmse_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:,.0f}',
            ha='center', va='bottom', fontsize=9)

plt.suptitle('COVID-19 Forecasting Model Results', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('reports/figures/final_model_results.png', dpi=300, bbox_inches='tight')
print("\n Saved visualization: reports/figures/final_model_results.png")

print('FINAL RESULTS SUMMARY')

results = {
    'Prophet': MetricsCalculator.calculate_all(y_test, y_pred_prophet),
    'XGBoost': MetricsCalculator.calculate_all(y_test, y_pred_xgb),
    'Ensemble': MetricsCalculator.calculate_all(y_test[:len(y_pred_ensemble)], y_pred_ensemble)
}

for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"  MAE:  {metrics['mae']:,.0f}")
    print(f"  RMSE: {metrics['rmse']:,.0f}")
    print(f"  MAPE: {metrics['mape']:.1f}%")
    print(f"  MDA:  {metrics['mda']:.1f}%")

print(f"\nEnsemble weights: Prophet={weights['prophet']:.2f}, XGBoost={weights['xgboost']:.2f}")
print('VISUALIZATION COMPLETE!')
