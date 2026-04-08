
"""
Complete project execution - runs all notebooks and generates outputs.
This script executes the full pipeline without requiring Jupyter.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime

print(" "*20 + "COMPLETE PROJECT EXECUTION")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

Path('data/processed').mkdir(parents=True, exist_ok=True)
Path('data/external').mkdir(parents=True, exist_ok=True)
Path('reports/figures').mkdir(parents=True, exist_ok=True)
Path('models').mkdir(parents=True, exist_ok=True)


print("PART 1: DATA PROCESSING")

from src.data.cleaners import COVIDDataCleaner
from src.features.builders import FeatureBuilder

try:
    features_df = pd.read_csv('data/processed/features_full.csv', parse_dates=['date'])
    print(f"Loaded existing processed data: {features_df.shape}")
except:
    print("Processing raw data...")
    # Load raw
    covid_raw = pd.read_csv('data/raw/time_series_covid19_confirmed_global.csv')
    covid_deaths = pd.read_csv('data/raw/time_series_covid19_deaths_global.csv')
    
    # Process
    india_confirmed = covid_raw[covid_raw['Country/Region'] == 'India']
    india_deaths = covid_deaths[covid_deaths['Country/Region'] == 'India']
    
    date_cols = india_confirmed.columns[4:].tolist()
    covid_data = []
    for date_col in date_cols:
        date = pd.to_datetime(date_col)
        confirmed = india_confirmed[date_col].sum()
        deaths = india_deaths[date_col].sum() if date_col in india_deaths.columns else 0
        covid_data.append({'date': date, 'confirmed': confirmed, 'deaths': deaths})
    
    covid_df = pd.DataFrame(covid_data)
    covid_df['new_cases'] = covid_df['confirmed'].diff().fillna(0).clip(lower=0)
    
    # Load mobility
    mobility_df = pd.read_csv('data/raw/Global_Mobility_Report.csv', low_memory=False)
    india_mobility = mobility_df[mobility_df['country_region'] == 'India'].copy()
    india_mobility['date'] = pd.to_datetime(india_mobility['date'])
    
    mobility_cols = [c for c in india_mobility.columns if 'percent_change' in c]
    national_mobility = india_mobility.groupby('date')[mobility_cols].mean().reset_index()
    
    master_df = pd.merge(covid_df, national_mobility, on='date', how='outer').sort_values('date')
    for col in master_df.columns:
        if col != 'date':
            master_df[col] = master_df[col].ffill().bfill()
    
    builder = FeatureBuilder(lag_days=[1, 3, 7, 14], rolling_windows=[3, 7, 14])
    features_df = builder.build_all_features(master_df, target_col='new_cases', date_col='date').dropna()
    features_df.to_csv('data/processed/features_full.csv', index=False)
    print(f"✓ Data processed: {features_df.shape}")

print("PART 2: EXPLORATORY DATA ANALYSIS")

# Basic stats
print(f"\nDataset Overview:")
print(f"  Date range: {features_df['date'].min().date()} to {features_df['date'].max().date()}")
print(f"  Total records: {len(features_df)}")
print(f"  Peak daily cases: {features_df['new_cases'].max():,.0f}")
print(f"  Mean daily cases: {features_df['new_cases'].mean():,.0f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.plot(features_df['date'], features_df['confirmed'], linewidth=2, color='#1f77b4')
ax.set_title('Cumulative Confirmed Cases', fontweight='bold')
ax.set_ylabel('Cases (millions)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.bar(features_df['date'], features_df['new_cases'], alpha=0.7, color='#2ca02c', width=1)
if 'new_cases_7day_avg' in features_df.columns:
    ax.plot(features_df['date'], features_df['new_cases_7day_avg'], color='red', linewidth=2, label='7-day avg')
ax.set_title('Daily New Cases', fontweight='bold')
ax.set_ylabel('New Cases')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 0]
mobility_cols = [c for c in features_df.columns if 'workplaces' in c or 'retail' in c]
if mobility_cols:
    for col in mobility_cols[:2]:
        ax.plot(features_df['date'], features_df[col], label=col.replace('_', ' '), alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_title('Mobility Trends', fontweight='bold')
    ax.set_ylabel('Change from baseline (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

ax = axes[1, 1]
if 'new_cases_lag_1' in features_df.columns:
    ax.scatter(features_df['new_cases_lag_1'], features_df['new_cases'], alpha=0.5, s=20)
    ax.set_xlabel('Previous Day Cases (Lag-1)')
    ax.set_ylabel('Current Day Cases')
    ax.set_title('Autocorrelation: Lag-1 vs Current', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    corr = features_df['new_cases_lag_1'].corr(features_df['new_cases'])
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('COVID-19 Disease Dynamics - Exploratory Analysis', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('reports/figures/01_eda_summary.png', dpi=300, bbox_inches='tight')
print("\n EDA visualization saved: reports/figures/01_eda_summary.png")

print("PART 3: MODEL TRAINING & EVALUATION")

from src.evaluation.metrics import MetricsCalculator, train_test_split_by_date
from src.models.prophet_model import ProphetModel
from src.models.xgboost_model import XGBoostModel
from src.models.ensemble import EnsembleModel

# Prepare data
builder = FeatureBuilder()
feature_cols = builder.get_feature_columns(features_df, target_col='new_cases')

# Split
train_df, val_df, test_df = train_test_split_by_date(features_df, test_size=0.15, val_size=0.1)

print(f"\nData Split:")
print(f"  Train: {len(train_df)} samples ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
print(f"  Val:   {len(val_df)} samples")
print(f"  Test:  {len(test_df)} samples ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

y_test = test_df['new_cases'].values
dates_test = test_df['date'].values

# Model 1: Prophet
print("\n[1/3] Training Prophet model...")
prophet = ProphetModel()
prophet.fit(train_df, target_col='new_cases', regressors=[])
test_pred = prophet.predict_in_sample(test_df)
y_pred_prophet = test_pred['yhat'].values
metrics_prophet = MetricsCalculator.calculate_all(y_test, y_pred_prophet)
print(f"  MAE: {metrics_prophet['mae']:,.0f}, RMSE: {metrics_prophet['rmse']:,.0f}, MDA: {metrics_prophet['mda']:.1f}%")

# Model 2: XGBoost
print("\n[2/3] Training XGBoost model...")
X_train = train_df[feature_cols].values
y_train = train_df['new_cases'].values
X_test = test_df[feature_cols].values

xgb = XGBoostModel(n_estimators=100, max_depth=5)
xgb.fit(X_train, y_train, feature_names=feature_cols)
y_pred_xgb = xgb.predict(X_test)
metrics_xgb = MetricsCalculator.calculate_all(y_test, y_pred_xgb)
print(f"  MAE: {metrics_xgb['mae']:,.0f}, RMSE: {metrics_xgb['rmse']:,.0f}, MDA: {metrics_xgb['mda']:.1f}%")

# Model 3: Ensemble
print("\n[3/3] Training Ensemble model...")
ensemble = EnsembleModel(prophet_weight=0.3, lstm_weight=0, xgboost_weight=0.7)
predictions_dict = {'prophet': y_pred_prophet, 'xgboost': y_pred_xgb}
y_pred_ensemble, weights = ensemble.predict(predictions_dict, y_test)
metrics_ensemble = MetricsCalculator.calculate_all(y_test[:len(y_pred_ensemble)], y_pred_ensemble)
print(f"  MAE: {metrics_ensemble['mae']:,.0f}, RMSE: {metrics_ensemble['rmse']:,.0f}, MDA: {metrics_ensemble['mda']:.1f}%")
print(f"  Weights: Prophet={weights['prophet']:.2f}, XGBoost={weights['xgboost']:.2f}")

print("PART 4: GENERATING RESULTS VISUALIZATIONS")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Model comparison
ax = axes[0, 0]
ax.plot(dates_test, y_test, label='Actual', linewidth=2.5, color='black', alpha=0.8)
ax.plot(dates_test, y_pred_prophet, label=f'Prophet (MAE={metrics_prophet["mae"]:,.0f})', 
        linewidth=1.5, alpha=0.7, linestyle='--')
ax.plot(dates_test, y_pred_xgb, label=f'XGBoost (MAE={metrics_xgb["mae"]:,.0f})', 
        linewidth=1.5, alpha=0.7, linestyle='--')
ax.plot(dates_test[:len(y_pred_ensemble)], y_pred_ensemble, 
        label=f'Ensemble (MAE={metrics_ensemble["mae"]:,.0f})', linewidth=2.5, color='red')
ax.set_title('Model Predictions Comparison', fontweight='bold', fontsize=12)
ax.set_ylabel('Daily New Cases')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Feature importance
ax = axes[0, 1]
importance = xgb.get_feature_importance()
top_10 = importance.head(10)
ax.barh(range(len(top_10)), top_10['importance'], color='steelblue')
ax.set_yticks(range(len(top_10)))
ax.set_yticklabels(top_10['feature'], fontsize=8)
ax.set_title('Top 10 Feature Importance (XGBoost)', fontweight='bold', fontsize=12)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# Plot 3: Residuals
ax = axes[1, 0]
residuals = y_test - y_pred_xgb
ax.scatter(range(len(residuals)), residuals, alpha=0.6, s=20, color='steelblue')
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_title('XGBoost Residuals', fontweight='bold', fontsize=12)
ax.set_ylabel('Residual (Actual - Predicted)')
ax.set_xlabel('Sample Index')
ax.grid(True, alpha=0.3)

# Plot 4: Metrics comparison
ax = axes[1, 1]
models = ['Prophet', 'XGBoost', 'Ensemble']
mae_values = [metrics_prophet['mae'], metrics_xgb['mae'], metrics_ensemble['mae']]
colors = ['#1f77b4', '#2ca02c', '#d62728']
bars = ax.bar(models, mae_values, color=colors, alpha=0.8, edgecolor='black')
ax.set_title('Model Performance Comparison (MAE)', fontweight='bold', fontsize=12)
ax.set_ylabel('Mean Absolute Error (lower is better)')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, mae_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{val:,.0f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('COVID-19 Forecasting Model Results', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('reports/figures/final_results.png', dpi=300, bbox_inches='tight')
print(" Saved: reports/figures/final_results.png")

# PART 5: SUMMARY REPORT

print("PART 5: FINAL SUMMARY REPORT")

print("\n┌" + "─"*68 + "┐")
print("│" + "MODEL PERFORMANCE METRICS".center(68) + "│")
print(f"│ {'Model':<15} {'MAE':>12} {'RMSE':>12} {'MAPE':>12} {'MDA':>12} │")
print(f"│ {'Prophet':<15} {metrics_prophet['mae']:>12,.0f} {metrics_prophet['rmse']:>12,.0f} {metrics_prophet['mape']:>11.1f}% {metrics_prophet['mda']:>11.1f}% │")
print(f"│ {'XGBoost':<15} {metrics_xgb['mae']:>12,.0f} {metrics_xgb['rmse']:>12,.0f} {metrics_xgb['mape']:>11.1f}% {metrics_xgb['mda']:>11.1f}% │")
print(f"│ {'Ensemble':<15} {metrics_ensemble['mae']:>12,.0f} {metrics_ensemble['rmse']:>12,.0f} {metrics_ensemble['mape']:>11.1f}% {metrics_ensemble['mda']:>11.1f}% │")

print(f"\n BEST MODEL: {'XGBoost' if metrics_xgb['mae'] < metrics_ensemble['mae'] else 'Ensemble'}")
print(f"Ensemble Weights: Prophet={weights['prophet']:.2f}, XGBoost={weights['xgboost']:.2f}")

print(" "*25 + "EXECUTION COMPLETE!")
print("\nGenerated Outputs:")
print("reports/figures/01_eda_summary.png")
print("reports/figures/final_results.png")
print("data/processed/features_full.csv")
print("\nNext: Run 'jupyter notebook' to explore interactively!")
