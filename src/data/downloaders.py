"""
Data downloaders for COVID-19, mobility, and demographic datasets.

This module handles downloading data from various sources including:
- JHU CSSE COVID-19 time series data
- Google Mobility Reports
- WorldPop demographic data
- OpenWeather historical climate data
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime, timedelta

import pandas as pd
import requests
from tqdm import tqdm
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDownloader:
    """Base class for data downloaders."""
    
    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self, url: str, filename: str) -> str:
        """Download file from URL to raw directory."""
        filepath = self.raw_dir / filename
        
        if filepath.exists():
            logger.info(f"{filename} already exists, skipping download.")
            return str(filepath)
        
        logger.info(f"Downloading {filename} from {url}...")
        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            logger.info(f"Downloaded {filename} successfully.")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            raise


class COVIDDataDownloader(DataDownloader):
    """Downloader for JHU CSSE COVID-19 data."""
    
    BASE_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
    
    def __init__(self, raw_dir: str, processed_dir: str):
        super().__init__(raw_dir, processed_dir)
        self.files = {
            'confirmed': 'time_series_covid19_confirmed_global.csv',
            'deaths': 'time_series_covid19_deaths_global.csv',
            'recovered': 'time_series_covid19_recovered_global.csv'
        }
    
    def download_all(self) -> Dict[str, str]:
        """Download all COVID-19 datasets."""
        paths = {}
        for data_type, filename in self.files.items():
            url = f"{self.BASE_URL}{filename}"
            paths[data_type] = self.download(url, filename)
        return paths
    
    def load_and_filter_india(self, data_type: str = 'confirmed') -> pd.DataFrame:
        """Load and filter data for India."""
        filepath = self.raw_dir / self.files[data_type]
        
        if not filepath.exists():
            self.download_all()
        
        df = pd.read_csv(filepath)
        
        # Filter for India
        india_df = df[df['Country/Region'] == 'India'].copy()
        
        # Check if we have state-level data (Province/State column)
        if 'Province/State' in india_df.columns:
            # Rename for consistency
            india_df = india_df.rename(columns={'Province/State': 'State'})
        
        return india_df
    
    def get_state_level_data(self) -> pd.DataFrame:
        """
        Get state-level COVID-19 data for India.
        Note: JHU data doesn't have Indian state-level breakdown, 
        so we'll create a national-level dataset with state estimates.
        """
        # Download all types
        paths = self.download_all()
        
        # Load confirmed cases
        df_confirmed = pd.read_csv(paths['confirmed'])
        df_deaths = pd.read_csv(paths['deaths'])
        
        # Filter for India
        india_confirmed = df_confirmed[df_confirmed['Country/Region'] == 'India'].copy()
        india_deaths = df_deaths[df_deaths['Country/Region'] == 'India'].copy()
        
        # Get date columns (they start from column 4)
        date_cols = india_confirmed.columns[4:].tolist()
        
        # Create national-level time series
        national_data = []
        for date_col in date_cols:
            date = pd.to_datetime(date_col)
            confirmed = india_confirmed[date_col].sum()
            deaths = india_deaths[date_col].sum()
            
            national_data.append({
                'date': date,
                'confirmed': confirmed,
                'deaths': deaths,
                'country': 'India'
            })
        
        result_df = pd.DataFrame(national_data)
        result_df = result_df.sort_values('date').reset_index(drop=True)
        
        # Calculate daily new cases
        result_df['new_cases'] = result_df['confirmed'].diff().fillna(0).clip(lower=0)
        result_df['new_deaths'] = result_df['deaths'].diff().fillna(0).clip(lower=0)
        
        return result_df


class MobilityDataDownloader(DataDownloader):
    """Downloader for Google Mobility Reports."""
    
    URL = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
    
    def __init__(self, raw_dir: str, processed_dir: str):
        super().__init__(raw_dir, processed_dir)
    
    def download_mobility(self) -> str:
        """Download Google Mobility Report."""
        return self.download(self.URL, "Global_Mobility_Report.csv")
    
    def load_and_filter_india(self) -> pd.DataFrame:
        """Load and filter mobility data for India."""
        filepath = self.raw_dir / "Global_Mobility_Report.csv"
        
        if not filepath.exists():
            self.download_mobility()
        
        logger.info("Loading mobility data (this may take a moment)...")
        
        # Use chunking for large file
        chunks = []
        chunk_iter = pd.read_csv(filepath, chunksize=100000, low_memory=False)
        
        for chunk in chunk_iter:
            india_chunk = chunk[chunk['country_region'] == 'India']
            if len(india_chunk) > 0:
                chunks.append(india_chunk)
        
        if not chunks:
            logger.warning("No India mobility data found!")
            return pd.DataFrame()
        
        df = pd.concat(chunks, ignore_index=True)
        
        # Clean and rename columns
        df['date'] = pd.to_datetime(df['date'])
        
        # Rename columns for clarity
        column_map = {
            'sub_region_1': 'state',
            'sub_region_2': 'district',
            'retail_and_recreation_percent_change_from_baseline': 'retail_recreation',
            'grocery_and_pharmacy_percent_change_from_baseline': 'grocery_pharmacy',
            'parks_percent_change_from_baseline': 'parks',
            'transit_stations_percent_change_from_baseline': 'transit',
            'workplaces_percent_change_from_baseline': 'workplaces',
            'residential_percent_change_from_baseline': 'residential'
        }
        
        df = df.rename(columns=column_map)
        
        # Select relevant columns
        mobility_cols = ['date', 'state', 'retail_recreation', 'grocery_pharmacy', 
                        'parks', 'transit', 'workplaces', 'residential']
        
        df = df[[col for col in mobility_cols if col in df.columns]]
        
        return df


class DemographicDataDownloader(DataDownloader):
    """Downloader for WorldPop demographic data."""
    
    def __init__(self, raw_dir: str, processed_dir: str, external_dir: str):
        super().__init__(raw_dir, processed_dir)
        self.external_dir = Path(external_dir)
        self.external_dir.mkdir(parents=True, exist_ok=True)
    
    def create_india_demographics(self) -> pd.DataFrame:
        """
        Create demographic data for Indian states.
        Using known population figures for major states.
        """
        # Population data (approximate 2020 figures in millions)
        state_populations = {
            'Maharashtra': 112.4,
            'Delhi': 19.0,
            'Karnataka': 64.1,
            'Tamil Nadu': 72.1,
            'Kerala': 33.4,
            'Uttar Pradesh': 199.8,
            'West Bengal': 91.3,
            'Gujarat': 60.4,
            'Rajasthan': 68.5,
            'Andhra Pradesh': 49.6,
            'Telangana': 35.2,
            'Madhya Pradesh': 72.6,
            'Bihar': 104.1,
            'Punjab': 27.7,
            'Haryana': 25.4,
            'Assam': 31.2,
            'Jharkhand': 32.4,
            'Chhattisgarh': 25.5,
            'Uttarakhand': 10.1,
            'Goa': 1.5,
            'Himachal Pradesh': 6.9,
            'Tripura': 4.0,
            'Meghalaya': 2.9,
            'Manipur': 2.9,
            'Nagaland': 2.0,
            'Arunachal Pradesh': 1.4,
            'Mizoram': 1.1,
            'Sikkim': 0.6
        }
        
        # Create DataFrame
        df = pd.DataFrame([
            {'state': state, 'population_millions': pop, 
             'population_density_estimate': pop * 0.5}  # Rough density proxy
            for state, pop in state_populations.items()
        ])
        
        # Save to external
        filepath = self.external_dir / "india_demographics.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"Saved demographics data to {filepath}")
        
        return df


class WeatherDataDownloader(DataDownloader):
    """
    Weather data downloader using Open-Meteo API (free, no key required).
    """
    
    def __init__(self, raw_dir: str, processed_dir: str, external_dir: str):
        super().__init__(raw_dir, processed_dir)
        self.external_dir = Path(external_dir)
        self.external_dir.mkdir(parents=True, exist_ok=True)
    
    # Approximate coordinates for major Indian cities (representing states)
    STATE_COORDINATES = {
        'Maharashtra': (19.08, 72.88),  # Mumbai
        'Delhi': (28.61, 77.21),
        'Karnataka': (12.97, 77.59),  # Bangalore
        'Tamil Nadu': (13.08, 80.27),  # Chennai
        'Kerala': (8.52, 76.94),  # Thiruvananthapuram
        'Uttar Pradesh': (26.85, 80.95),  # Lucknow
        'West Bengal': (22.57, 88.36),  # Kolkata
        'Gujarat': (23.03, 72.58),  # Ahmedabad
    }
    
    def download_weather_data(self, state: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download historical weather data for a state."""
        
        if state not in self.STATE_COORDINATES:
            logger.warning(f"No coordinates for {state}, skipping.")
            return pd.DataFrame()
        
        lat, lon = self.STATE_COORDINATES[state]
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'daily': ['temperature_2m_mean', 'precipitation_sum', 'relative_humidity_2m_mean'],
            'timezone': 'Asia/Kolkata'
        }
        
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'date': pd.to_datetime(data['daily']['time']),
                'temperature_mean': data['daily']['temperature_2m_mean'],
                'precipitation': data['daily']['precipitation_sum'],
                'humidity_mean': data['daily']['relative_humidity_2m_mean'],
                'state': state
            })
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to download weather for {state}: {e}")
            return pd.DataFrame()
    
    def download_all_states(self, start_date: str = "2020-01-01", 
                           end_date: str = None) -> pd.DataFrame:
        """Download weather data for all configured states."""
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        all_data = []
        
        for state in self.STATE_COORDINATES.keys():
            logger.info(f"Downloading weather data for {state}...")
            df = self.download_weather_data(state, start_date, end_date)
            if not df.empty:
                all_data.append(df)
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            
            # Save
            filepath = self.external_dir / "india_weather.csv"
            combined.to_csv(filepath, index=False)
            logger.info(f"Saved weather data to {filepath}")
            
            return combined
        
        return pd.DataFrame()


def main():
    """Main function to download all datasets."""
    
    # Load config
    config_path = Path("config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    raw_dir = config['data']['raw_dir']
    processed_dir = config['data']['processed_dir']
    external_dir = config['data']['external_dir']
    
    logger.info("="*50)
    logger.info("Starting data download pipeline")
    logger.info("="*50)
    
    # Download COVID-19 data
    logger.info("\n[1/4] Downloading COVID-19 data...")
    covid_downloader = COVIDDataDownloader(raw_dir, processed_dir)
    covid_data = covid_downloader.get_state_level_data()
    logger.info(f"COVID data shape: {covid_data.shape}")
    
    # Download mobility data
    logger.info("\n[2/4] Downloading Google Mobility data...")
    mobility_downloader = MobilityDataDownloader(raw_dir, processed_dir)
    mobility_data = mobility_downloader.load_and_filter_india()
    logger.info(f"Mobility data shape: {mobility_data.shape}")
    
    # Create demographic data
    logger.info("\n[3/4] Creating demographic data...")
    demo_downloader = DemographicDataDownloader(raw_dir, processed_dir, external_dir)
    demo_data = demo_downloader.create_india_demographics()
    logger.info(f"Demographics data shape: {demo_data.shape}")
    
    # Download weather data
    logger.info("\n[4/4] Downloading weather data...")
    weather_downloader = WeatherDataDownloader(raw_dir, processed_dir, external_dir)
    weather_data = weather_downloader.download_all_states()
    logger.info(f"Weather data shape: {weather_data.shape}")
    
    logger.info("\n" + "="*50)
    logger.info("Data download pipeline completed!")
    logger.info("="*50)
    
    return {
        'covid': covid_data,
        'mobility': mobility_data,
        'demographics': demo_data,
        'weather': weather_data
    }


if __name__ == "__main__":
    main()
