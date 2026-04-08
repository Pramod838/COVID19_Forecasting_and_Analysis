"""Quick data processing pipeline."""

import sys
sys.path.insert(0, '.')
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print('='*60)
print('STEP 1: Loading and Cleaning COVID Data')
print('='*60)

# Load raw COVID data
covid_raw = pd.read_csv('data/raw/time_series_covid19_confirmed_global.csv')
covid_deaths = pd.read_csv('data/raw/time_series_covid19_deaths_global.csv')

# Process for India
india_confirmed = covid_raw[covid_raw['Country/Region'] == 'India']
india_deaths = covid_deaths[covid_deaths['Country/Region'] == 'India']

# Convert to time series
date_cols = india_confirmed.columns[4:].tolist()
covid_data = []
for date_col in date_cols:
    date = pd.to_datetime(date_col)
    confirmed = india_confirmed[date_col].sum()
    deaths = india_deaths[date_col].sum() if date_col in india_deaths.columns else 0
    covid_data.append({'date': date, 'confirmed': confirmed, 'deaths': deaths})

covid_df = pd.DataFrame(covid_data)
covid_df['new_cases'] = covid_df['confirmed'].diff().fillna(0).clip(lower=0)
covid_df['new_deaths'] = covid_df['deaths'].diff().fillna(0).clip(lower=0)

print(f'COVID data shape: {covid_df.shape}')
print(f'Date range: {covid_df["date"].min()} to {covid_df["date"].max()}')
print(f'Peak daily cases: {covid_df["new_cases"].max():,.0f}')

# Save
covid_df.to_csv('data/processed/covid_cleaned.csv', index=False)
print('Saved: data/processed/covid_cleaned.csv')

print()
print('='*60)
print('STEP 2: Processing Mobility Data')
print('='*60)

# Load mobility
mobility_df = pd.read_csv('data/raw/Global_Mobility_Report.csv', low_memory=False)
india_mobility = mobility_df[mobility_df['country_region'] == 'India'].copy()
india_mobility['date'] = pd.to_datetime(india_mobility['date'])

# Aggregate nationally
mobility_cols = ['retail_and_recreation_percent_change_from_baseline',
                'grocery_and_pharmacy_percent_change_from_baseline',
                'parks_percent_change_from_baseline',
                'transit_stations_percent_change_from_baseline',
                'workplaces_percent_change_from_baseline',
                'residential_percent_change_from_baseline']
mobility_cols = [c for c in mobility_cols if c in india_mobility.columns]

national_mobility = india_mobility.groupby('date')[mobility_cols].mean().reset_index()

# Rename columns
name_map = {
    'retail_and_recreation_percent_change_from_baseline': 'retail_recreation',
    'grocery_and_pharmacy_percent_change_from_baseline': 'grocery_pharmacy',
    'parks_percent_change_from_baseline': 'parks',
    'transit_stations_percent_change_from_baseline': 'transit',
    'workplaces_percent_change_from_baseline': 'workplaces',
    'residential_percent_change_from_baseline': 'residential'
}
national_mobility = national_mobility.rename(columns=name_map)

print(f'Mobility data shape: {national_mobility.shape}')
national_mobility.to_csv('data/processed/mobility_cleaned.csv', index=False)
print('Saved: data/processed/mobility_cleaned.csv')

print()
print('='*60)
print('STEP 3: Feature Engineering')
print('='*60)

from src.features.builders import FeatureBuilder

# Merge
master_df = pd.merge(covid_df, national_mobility, on='date', how='outer')
master_df = master_df.sort_values('date')

# Forward fill then backward fill
for col in master_df.columns:
    if col != 'date':
        master_df[col] = master_df[col].ffill().bfill()

# Build features
builder = FeatureBuilder(lag_days=[1, 3, 7, 14], rolling_windows=[3, 7, 14])
features_df = builder.build_all_features(master_df, target_col='new_cases', date_col='date')
features_df = features_df.dropna()

feature_cols = builder.get_feature_columns(features_df, target_col='new_cases')

print(f'Total features created: {len(feature_cols)}')
print(f'Final dataset shape: {features_df.shape}')

# Save
features_df.to_csv('data/processed/features_full.csv', index=False)
print('Saved: data/processed/features_full.csv')

print()
print('='*60)
print('DATA PIPELINE COMPLETE!')
print('='*60)
print(f'\nReady for modeling with {len(features_df)} samples and {len(feature_cols)} features')
