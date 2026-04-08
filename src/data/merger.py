"""
Data merger module for integrating heterogeneous data sources.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataMerger:
    """Merge multiple data sources into unified dataset."""
    
    def __init__(self, processed_dir: str):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def merge_national_data(self, covid_df: pd.DataFrame, 
                           mobility_df: Optional[pd.DataFrame] = None,
                           weather_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Merge COVID-19, mobility, and weather data at national level.
        
        For national level:
        - COVID data is already national (India total)
        - Mobility needs to be averaged across all states
        - Weather needs to be averaged across all states
        """
        # Start with COVID data
        merged = covid_df.copy()
        
        # Add mobility (national average)
        if mobility_df is not None and not mobility_df.empty:
            logger.info("Merging mobility data...")
            
            # Get available mobility columns (may vary by dataset)
            expected_cols = ['retail_recreation', 'grocery_pharmacy', 'parks', 
                           'transit', 'workplaces', 'residential']
            mobility_cols = [col for col in expected_cols if col in mobility_df.columns]
            
            if mobility_cols:
                # Group by date only for national average
                mobility_national = mobility_df.groupby('date')[mobility_cols].mean().reset_index()
                
                # Merge
                merged = merged.merge(mobility_national, on='date', how='left')
            else:
                logger.warning("No recognized mobility columns found")
        
        # Add weather (national average)
        if weather_df is not None and not weather_df.empty:
            logger.info("Merging weather data...")
            
            # Get available weather columns
            expected_weather = ['temperature_mean', 'precipitation', 'humidity_mean']
            weather_cols = [col for col in expected_weather if col in weather_df.columns]
            
            if weather_cols:
                # Calculate national weather (mean across states)
                weather_national = weather_df.groupby('date')[weather_cols].mean().reset_index()
                
                # Merge
                merged = merged.merge(weather_national, on='date', how='left')
            else:
                logger.warning("No recognized weather columns found")
        
        # Sort and fill any remaining missing values
        merged = merged.sort_values('date').reset_index(drop=True)
        
        # Forward fill then backward fill for any remaining NAs
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'date':
                merged[col] = merged[col].ffill().bfill()
        
        return merged
    
    def merge_state_level_data(self, covid_df: pd.DataFrame,
                               mobility_df: pd.DataFrame,
                               weather_df: pd.DataFrame,
                               demo_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge datasets at state level.
        
        Note: Since we only have national COVID data from JHU, 
        we'll distribute it proportionally to states based on population
        for demonstration purposes.
        """
        logger.info("Creating state-level merged dataset...")
        
        # For state-level analysis, we need to estimate state-wise cases
        # using population proportions
        
        # Calculate population proportions
        total_pop = demo_df['population_millions'].sum()
        demo_df['pop_proportion'] = demo_df['population_millions'] / total_pop
        
        # Create a mapping of state names that might differ between datasets
        state_name_mapping = {
            'Maharashtra': 'Maharashtra',
            'Delhi': 'Delhi',
            'Karnataka': 'Karnataka',
            'Tamil Nadu': 'Tamil Nadu',
            'Kerala': 'Kerala',
            'Uttar Pradesh': 'Uttar Pradesh',
            'West Bengal': 'West Bengal',
            'Gujarat': 'Gujarat'
        }
        
        all_merged_states = []
        
        for _, state_row in demo_df.iterrows():
            state = state_row['state']
            pop_prop = state_row['pop_proportion']
            
            # Estimate state-level COVID (proportional to population)
            state_covid = covid_df.copy()
            state_covid['state'] = state
            state_covid['estimated_population'] = state_row['population_millions']
            
            # Scale metrics by population proportion
            state_covid['confirmed'] = (state_covid['confirmed'] * pop_prop).round()
            state_covid['deaths'] = (state_covid['deaths'] * pop_prop).round()
            state_covid['new_cases'] = (state_covid['new_cases'] * pop_prop).round()
            
            # Add demographics
            state_covid['population_millions'] = state_row['population_millions']
            
            # Add mobility
            if mobility_df is not None and not mobility_df.empty:
                state_mobility = mobility_df[mobility_df['state'] == state].copy()
                if not state_mobility.empty:
                    state_covid = state_covid.merge(
                        state_mobility.drop(columns=['state'], errors='ignore'),
                        on='date',
                        how='left'
                    )
            
            # Add weather
            if weather_df is not None and not weather_df.empty:
                state_weather = weather_df[weather_df['state'] == state].copy()
                if not state_weather.empty:
                    state_covid = state_covid.merge(
                        state_weather.drop(columns=['state'], errors='ignore'),
                        on='date',
                        how='left'
                    )
            
            all_merged_states.append(state_covid)
        
        # Combine all states
        merged = pd.concat(all_merged_states, ignore_index=True)
        
        # Sort
        merged = merged.sort_values(['state', 'date']).reset_index(drop=True)
        
        # Fill missing values within each state
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        for state in merged['state'].unique():
            state_mask = merged['state'] == state
            for col in numeric_cols:
                if col not in ['date', 'state']:
                    merged.loc[state_mask, col] = merged.loc[state_mask, col].ffill().bfill()
        
        return merged
    
    def create_master_dataset(self, datasets: Dict[str, pd.DataFrame],
                             level: str = 'national') -> pd.DataFrame:
        """
        Create master dataset by merging all sources.
        
        Args:
            datasets: Dictionary with 'covid', 'mobility', 'weather', 'demographics'
            level: 'national' or 'state'
        
        Returns:
            Merged DataFrame
        """
        covid_df = datasets.get('covid')
        mobility_df = datasets.get('mobility')
        weather_df = datasets.get('weather')
        demo_df = datasets.get('demographics')
        
        if covid_df is None:
            raise ValueError("COVID data is required")
        
        if level == 'national':
            merged = self.merge_national_data(covid_df, mobility_df, weather_df)
        else:
            if demo_df is None:
                raise ValueError("Demographics data required for state-level analysis")
            merged = self.merge_state_level_data(covid_df, mobility_df, weather_df, demo_df)
        
        # Save to processed directory
        output_path = self.processed_dir / f"master_dataset_{level}.csv"
        merged.to_csv(output_path, index=False)
        logger.info(f"Saved master dataset to {output_path}")
        
        return merged


def load_and_merge_all(raw_dir: str, processed_dir: str, 
                      external_dir: str, level: str = 'national') -> pd.DataFrame:
    """
    Convenience function to load all cleaned data and merge.
    
    This is used after running the downloaders and cleaners.
    """
    from .cleaners import clean_all_datasets
    
    # Load raw data
    covid_path = Path(processed_dir) / "covid_cleaned.csv"
    mobility_path = Path(processed_dir) / "mobility_cleaned.csv"
    weather_path = Path(external_dir) / "india_weather.csv"
    demo_path = Path(external_dir) / "india_demographics.csv"
    
    datasets = {}
    
    if covid_path.exists():
        datasets['covid'] = pd.read_csv(covid_path, parse_dates=['date'])
    
    if mobility_path.exists():
        datasets['mobility'] = pd.read_csv(mobility_path, parse_dates=['date'])
    
    if weather_path.exists():
        datasets['weather'] = pd.read_csv(weather_path, parse_dates=['date'])
    
    if demo_path.exists():
        datasets['demographics'] = pd.read_csv(demo_path)
    
    # Merge
    merger = DataMerger(processed_dir)
    master_df = merger.create_master_dataset(datasets, level=level)
    
    return master_df
