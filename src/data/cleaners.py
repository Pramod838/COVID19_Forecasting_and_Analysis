"""
Data cleaning and preprocessing modules.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """Base class for data cleaning operations."""
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        
        Methods:
        - 'interpolate': Linear interpolation for time series
        - 'forward_fill': Forward fill
        - 'backward_fill': Backward fill
        - 'mean': Fill with mean
        """
        df = df.copy()
        
        if method == 'interpolate':
            # Interpolate numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
        elif method == 'forward_fill':
            df = df.ffill().bfill()
        elif method == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, columns: List[str], 
                       method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from specified columns.
        
        Methods:
        - 'iqr': Interquartile range method
        - 'zscore': Z-score method
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Clip values instead of removing (to maintain time series continuity)
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
            elif method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df


class COVIDDataCleaner(DataCleaner):
    """Cleaner for COVID-19 data."""
    
    @classmethod
    def clean(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Clean COVID-19 time series data."""
        df = df.copy()
        
        # Ensure date column exists and is datetime
        if 'date' not in df.columns:
            raise ValueError("DataFrame must have a 'date' column")
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Remove negative values (data corrections sometimes show as negatives)
        numeric_cols = ['confirmed', 'deaths', 'new_cases', 'new_deaths']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)
        
        # Handle missing values
        df = cls.handle_missing_values(df, method='interpolate')
        
        # Remove extreme outliers in new_cases (often data entry errors)
        if 'new_cases' in df.columns:
            df = cls.remove_outliers(df, ['new_cases'], method='iqr', threshold=3.0)
        
        return df
    
    @classmethod
    def add_derived_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to COVID-19 data."""
        df = df.copy()
        
        # Ensure we have new_cases calculated
        if 'new_cases' not in df.columns and 'confirmed' in df.columns:
            df['new_cases'] = df['confirmed'].diff().fillna(0).clip(lower=0)
        
        # 7-day rolling average
        if 'new_cases' in df.columns:
            df['new_cases_7day_avg'] = df['new_cases'].rolling(window=7, min_periods=1).mean()
        
        # Growth rate (as percentage)
        if 'confirmed' in df.columns:
            df['growth_rate'] = df['confirmed'].pct_change().fillna(0) * 100
            df['growth_rate'] = df['growth_rate'].clip(lower=-100, upper=500)
        
        # Case fatality rate
        if 'deaths' in df.columns and 'confirmed' in df.columns:
            df['case_fatality_rate'] = (df['deaths'] / df['confirmed'].clip(lower=1)) * 100
        
        return df


class MobilityDataCleaner(DataCleaner):
    """Cleaner for mobility data."""
    
    @classmethod
    def clean(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Clean mobility data."""
        df = df.copy()
        
        if df.empty:
            return df
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Handle missing states (national average or NULL)
        if 'state' in df.columns:
            # Fill missing states with 'National'
            df['state'] = df['state'].fillna('National')
        
        # Mobility columns to clean
        mobility_cols = ['retail_recreation', 'grocery_pharmacy', 'parks', 
                        'transit', 'workplaces', 'residential']
        
        # Handle missing values by interpolation
        for col in mobility_cols:
            if col in df.columns:
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
        
        return df
    
    @classmethod
    def aggregate_to_daily(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate mobility data to daily level by state."""
        if df.empty or 'state' not in df.columns:
            return df
        
        # Group by date and state
        agg_dict = {}
        mobility_cols = ['retail_recreation', 'grocery_pharmacy', 'parks', 
                        'transit', 'workplaces', 'residential']
        
        for col in mobility_cols:
            if col in df.columns:
                agg_dict[col] = 'mean'
        
        if not agg_dict:
            return df
        
        daily_df = df.groupby(['date', 'state']).agg(agg_dict).reset_index()
        
        return daily_df


class WeatherDataCleaner(DataCleaner):
    """Cleaner for weather data."""
    
    @classmethod
    def clean(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Clean weather data."""
        df = df.copy()
        
        if df.empty:
            return df
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Handle missing values
        df = cls.handle_missing_values(df, method='interpolate')
        
        return df


def clean_all_datasets(datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Clean all datasets in one call.
    
    Args:
        datasets: Dictionary with keys 'covid', 'mobility', 'weather'
    
    Returns:
        Dictionary with cleaned datasets
    """
    cleaned = {}
    
    if 'covid' in datasets:
        logger.info("Cleaning COVID-19 data...")
        covid_clean = COVIDDataCleaner.clean(datasets['covid'])
        covid_clean = COVIDDataCleaner.add_derived_features(covid_clean)
        cleaned['covid'] = covid_clean
        logger.info(f"COVID data shape: {covid_clean.shape}")
    
    if 'mobility' in datasets:
        logger.info("Cleaning mobility data...")
        mobility_clean = MobilityDataCleaner.clean(datasets['mobility'])
        mobility_clean = MobilityDataCleaner.aggregate_to_daily(mobility_clean)
        cleaned['mobility'] = mobility_clean
        logger.info(f"Mobility data shape: {mobility_clean.shape}")
    
    if 'weather' in datasets:
        logger.info("Cleaning weather data...")
        weather_clean = WeatherDataCleaner.clean(datasets['weather'])
        cleaned['weather'] = weather_clean
        logger.info(f"Weather data shape: {weather_clean.shape}")
    
    return cleaned
