"""
Feature engineering pipeline for time series forecasting.

This module creates features from the merged dataset suitable for ML models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Build features for time series forecasting models."""
    
    def __init__(self, lag_days: List[int] = None, 
                 rolling_windows: List[int] = None):
        """
        Initialize feature builder.
        
        Args:
            lag_days: List of lag periods to create (e.g., [1, 3, 7, 14])
            rolling_windows: List of rolling window sizes (e.g., [3, 7, 14])
        """
        self.lag_days = lag_days or [1, 3, 7, 14]
        self.rolling_windows = rolling_windows or [3, 7, 14]
    
    def create_temporal_features(self, df: pd.DataFrame, 
                                 date_col: str = 'date') -> pd.DataFrame:
        """
        Create temporal features from date column.
        """
        df = df.copy()
        
        # Extract temporal features
        df['day_of_week'] = df[date_col].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = df[date_col].dt.day
        df['month'] = df[date_col].dt.month
        df['year'] = df[date_col].dt.year
        df['week_of_year'] = df[date_col].dt.isocalendar().week
        df['is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
        
        # Cyclical encoding for day of week and month
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Indian holidays (simplified - major ones)
        df['is_major_holiday'] = self._is_major_holiday(df[date_col]).astype(int)
        
        return df
    
    def _is_major_holiday(self, dates: pd.Series) -> pd.Series:
        """Mark major Indian holidays (simplified)."""
        # This is a simplified version - in production you'd use a proper holiday calendar
        holidays = []
        
        for date in dates:
            is_holiday = 0
            # Republic Day (Jan 26)
            if date.month == 1 and date.day == 26:
                is_holiday = 1
            # Independence Day (Aug 15)
            elif date.month == 8 and date.day == 15:
                is_holiday = 1
            # Gandhi Jayanti (Oct 2)
            elif date.month == 10 and date.day == 2:
                is_holiday = 1
            # Diwali (approximate - varies by year, using Oct-Nov period)
            elif date.month in [10, 11] and date.day in range(20, 31):
                is_holiday = 1
            
            holidays.append(is_holiday)
        
        return pd.Series(holidays, index=dates.index)
    
    def create_lag_features(self, df: pd.DataFrame, 
                           target_col: str = 'new_cases',
                           group_col: Optional[str] = None) -> pd.DataFrame:
        """
        Create lag features for time series.
        
        Args:
            df: DataFrame with time series data
            target_col: Column to create lags for
            group_col: If provided, create lags within each group (e.g., 'state')
        """
        df = df.copy()
        
        if group_col:
            # Create lags within each group
            for lag in self.lag_days:
                df[f'{target_col}_lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag)
        else:
            # Create lags for entire dataset
            for lag in self.lag_days:
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame,
                               target_col: str = 'new_cases',
                               group_col: Optional[str] = None) -> pd.DataFrame:
        """
        Create rolling window statistics.
        """
        df = df.copy()
        
        if group_col:
            grouped = df.groupby(group_col)[target_col]
            
            for window in self.rolling_windows:
                df[f'{target_col}_rolling_mean_{window}'] = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                df[f'{target_col}_rolling_std_{window}'] = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
                df[f'{target_col}_rolling_max_{window}'] = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
        else:
            for window in self.rolling_windows:
                df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(
                    window=window, min_periods=1).mean()
                df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(
                    window=window, min_periods=1).std()
                df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(
                    window=window, min_periods=1).max()
        
        return df
    
    def create_growth_features(self, df: pd.DataFrame,
                              target_col: str = 'new_cases',
                              group_col: Optional[str] = None) -> pd.DataFrame:
        """
        Create growth rate and trend features.
        """
        df = df.copy()
        
        if group_col:
            grouped = df.groupby(group_col)[target_col]
            
            # Day-over-day growth rate
            df[f'{target_col}_growth_rate'] = grouped.pct_change() * 100
            
            # Acceleration (change in growth rate)
            df[f'{target_col}_acceleration'] = grouped.transform(
                lambda x: x.pct_change().diff()
            ) * 100
            
            # Cumulative sum features
            df[f'{target_col}_cumulative'] = grouped.cumsum()
        else:
            df[f'{target_col}_growth_rate'] = df[target_col].pct_change() * 100
            df[f'{target_col}_acceleration'] = df[target_col].pct_change().diff() * 100
            df[f'{target_col}_cumulative'] = df[target_col].cumsum()
        
        # Clip extreme values
        df[f'{target_col}_growth_rate'] = df[f'{target_col}_growth_rate'].clip(-100, 500)
        df[f'{target_col}_acceleration'] = df[f'{target_col}_acceleration'].clip(-100, 100)
        
        return df
    
    def create_mobility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from mobility data.
        """
        df = df.copy()
        
        mobility_cols = ['retail_recreation', 'grocery_pharmacy', 'parks', 
                        'transit', 'workplaces', 'residential']
        
        # Calculate mobility index (composite score)
        available_cols = [col for col in mobility_cols if col in df.columns]
        if len(available_cols) >= 3:
            # Weighted average of mobility indicators
            # Negative impact: retail, transit, workplaces
            # Positive impact: residential (people staying home)
            weights = {
                'retail_recreation': -0.25,
                'transit': -0.25,
                'workplaces': -0.25,
                'residential': 0.25
            }
            
            df['mobility_index'] = 0
            for col, weight in weights.items():
                if col in df.columns:
                    df['mobility_index'] += df[col].fillna(0) * weight
        
        # Lagged mobility (mobility affects cases with delay)
        for col in available_cols:
            df[f'{col}_lag_7'] = df[col].shift(7)
            df[f'{col}_lag_14'] = df[col].shift(14)
        
        return df
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from weather data.
        """
        df = df.copy()
        
        # Temperature bins
        if 'temperature_mean' in df.columns:
            df['temp_category'] = pd.cut(df['temperature_mean'], 
                                        bins=[-50, 15, 25, 35, 50],
                                        labels=['cold', 'mild', 'warm', 'hot'])
            df['temp_category_encoded'] = df['temp_category'].cat.codes
        
        # Humidity bins
        if 'humidity_mean' in df.columns:
            df['humidity_high'] = (df['humidity_mean'] > 70).astype(int)
        
        # Precipitation indicator
        if 'precipitation' in df.columns:
            df['has_precipitation'] = (df['precipitation'] > 0).astype(int)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different feature types.
        """
        df = df.copy()
        
        # Mobility × Population density interaction
        if 'mobility_index' in df.columns and 'population_millions' in df.columns:
            df['mobility_pop_interaction'] = (
                df['mobility_index'] * np.log1p(df['population_millions'])
            )
        
        # Temperature × Weekday (weather affects mobility differently on weekdays)
        if 'temperature_mean' in df.columns and 'day_of_week' in df.columns:
            df['temp_weekday'] = df['temperature_mean'] * (df['day_of_week'] < 5).astype(int)
        
        # Lagged cases × mobility (cases respond to mobility changes)
        if 'new_cases_lag_7' in df.columns and 'workplaces_lag_7' in df.columns:
            df['cases_mobility_interaction'] = (
                df['new_cases_lag_7'] * df['workplaces_lag_7'] / 100
            )
        
        return df
    
    def build_all_features(self, df: pd.DataFrame, 
                          target_col: str = 'new_cases',
                          group_col: Optional[str] = None,
                          date_col: str = 'date') -> pd.DataFrame:
        """
        Build all features in one call.
        
        Args:
            df: Input DataFrame
            target_col: Target variable to create features for
            group_col: Column to group by (e.g., 'state' for state-level data)
            date_col: Date column name
        
        Returns:
            DataFrame with all features added
        """
        logger.info("Building temporal features...")
        df = self.create_temporal_features(df, date_col)
        
        logger.info("Building lag features...")
        df = self.create_lag_features(df, target_col, group_col)
        
        logger.info("Building rolling features...")
        df = self.create_rolling_features(df, target_col, group_col)
        
        logger.info("Building growth features...")
        df = self.create_growth_features(df, target_col, group_col)
        
        logger.info("Building mobility features...")
        df = self.create_mobility_features(df)
        
        logger.info("Building weather features...")
        df = self.create_weather_features(df)
        
        logger.info("Building interaction features...")
        df = self.create_interaction_features(df)
        
        # Handle any infinite or NaN values that might have been created
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].median())
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame, 
                           target_col: str = 'new_cases',
                           exclude_cols: List[str] = None) -> List[str]:
        """
        Get list of feature columns (excluding target and metadata).
        
        Args:
            df: DataFrame with features
            target_col: Target column to exclude
            exclude_cols: Additional columns to exclude
        
        Returns:
            List of feature column names
        """
        exclude_cols = exclude_cols or []
        default_excludes = [target_col, 'date', 'state', 'country', 'confirmed', 'deaths',
                           'temp_category', 'estimated_population']
        exclude_cols = list(set(exclude_cols + default_excludes))
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Only numeric features
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        return numeric_cols


def prepare_features_for_modeling(df: pd.DataFrame,
                                 target_col: str = 'new_cases',
                                 group_col: Optional[str] = None,
                                 drop_na: bool = True) -> tuple:
    """
    Prepare features for modeling by building all features and returning X, y.
    
    Args:
        df: Input DataFrame
        target_col: Target variable
        group_col: Grouping column (e.g., 'state')
        drop_na: Whether to drop rows with NaN values
    
    Returns:
        Tuple of (X, y, feature_names)
    """
    builder = FeatureBuilder()
    
    # Build features
    df_features = builder.build_all_features(df, target_col, group_col)
    
    # Get feature columns
    feature_cols = builder.get_feature_columns(df_features, target_col)
    
    # Drop NaN rows if requested
    if drop_na:
        df_features = df_features.dropna()
    
    X = df_features[feature_cols]
    y = df_features[target_col]
    
    return X, y, feature_cols
