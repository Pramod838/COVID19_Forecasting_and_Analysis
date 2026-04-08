"""
Prophet model wrapper for COVID-19 forecasting.

Facebook Prophet is good for time series with:
- Strong seasonality
- Holiday effects
- Trend changepoints
- Multiple regressors
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging

logging.getLogger('prophet').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class ProphetModel:
    """Wrapper for Facebook Prophet model."""
    
    def __init__(self, 
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0,
                 holidays_prior_scale: float = 10.0,
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = False):
        """
        Initialize Prophet model.
        
        Args:
            changepoint_prior_scale: Flexibility of trend (higher = more flexible)
            seasonality_prior_scale: Strength of seasonality (higher = stronger)
            holidays_prior_scale: Strength of holiday effects
            yearly_seasonality: Include yearly seasonality
            weekly_seasonality: Include weekly seasonality
            daily_seasonality: Include daily seasonality
        """
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        
        self.model = None
        self.regressors = []
    
    def _create_holidays_df(self) -> pd.DataFrame:
        """Create holidays DataFrame for India."""
        holidays = []
        
        # Major Indian holidays 2020-2024
        holiday_dates = [
            # 2020
            ('2020-01-26', 'Republic Day'),
            ('2020-08-15', 'Independence Day'),
            ('2020-10-02', 'Gandhi Jayanti'),
            ('2020-11-14', 'Diwali'),
            # 2021
            ('2021-01-26', 'Republic Day'),
            ('2021-08-15', 'Independence Day'),
            ('2021-10-02', 'Gandhi Jayanti'),
            ('2021-11-04', 'Diwali'),
            # 2022
            ('2022-01-26', 'Republic Day'),
            ('2022-08-15', 'Independence Day'),
            ('2022-10-02', 'Gandhi Jayanti'),
            ('2022-10-24', 'Diwali'),
            # 2023
            ('2023-01-26', 'Republic Day'),
            ('2023-08-15', 'Independence Day'),
            ('2023-10-02', 'Gandhi Jayanti'),
            ('2023-11-12', 'Diwali'),
            # 2024
            ('2024-01-26', 'Republic Day'),
            ('2024-08-15', 'Independence Day'),
            ('2024-10-02', 'Gandhi Jayanti'),
            ('2024-11-01', 'Diwali'),
        ]
        
        for date, name in holiday_dates:
            holidays.append({
                'holiday': name,
                'ds': pd.to_datetime(date),
                'lower_window': -1,
                'upper_window': 1
            })
        
        return pd.DataFrame(holidays)
    
    def fit(self, df: pd.DataFrame, 
            target_col: str = 'new_cases',
            date_col: str = 'date',
            regressors: List[str] = None):
        """
        Fit Prophet model.
        
        Args:
            df: Training DataFrame
            target_col: Target variable column
            date_col: Date column
            regressors: List of additional regressor columns
        """
        try:
            from prophet import Prophet
        except ImportError:
            logger.error("Prophet not installed. Install with: pip install prophet")
            raise
        
        # Prepare data for Prophet (needs 'ds' and 'y' columns)
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df[date_col]),
            'y': df[target_col].values
        })
        
        # Add regressors
        self.regressors = regressors or []
        for reg in self.regressors:
            if reg in df.columns:
                prophet_df[reg] = df[reg].values
        
        # Initialize model
        holidays_df = self._create_holidays_df()
        
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            holidays=holidays_df
        )
        
        # Add regressors
        for reg in self.regressors:
            if reg in df.columns:
                self.model.add_regressor(reg)
        
        logger.info(f"Fitting Prophet with {len(prophet_df)} samples and {len(self.regressors)} regressors")
        
        # Fit
        self.model.fit(prophet_df)
        
        return self
    
    def predict(self, df: pd.DataFrame, 
                date_col: str = 'date',
                horizon: int = 30) -> pd.DataFrame:
        """
        Make predictions.
        
        Args:
            df: DataFrame with dates and regressors
            date_col: Date column
            horizon: Number of days to forecast ahead
        
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted yet!")
        
        # Create future dataframe
        last_date = pd.to_datetime(df[date_col]).max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        # Create future dataframe with regressors
        future_df = pd.DataFrame({'ds': future_dates})
        
        # Add regressors if needed (use last known values or averages)
        for reg in self.regressors:
            if reg in df.columns:
                # Use mean of last 7 days as projection
                future_df[reg] = df[reg].tail(7).mean()
        
        # Make predictions
        forecast = self.model.predict(future_df)
        
        return forecast
    
    def predict_in_sample(self, df: pd.DataFrame,
                         date_col: str = 'date') -> pd.DataFrame:
        """
        Make in-sample predictions (for training set evaluation).
        """
        if self.model is None:
            raise ValueError("Model not fitted yet!")
        
        # Create dataframe with historical dates
        future_df = pd.DataFrame({'ds': pd.to_datetime(df[date_col])})
        
        # Add regressors
        for reg in self.regressors:
            if reg in df.columns:
                future_df[reg] = df[reg].values
        
        # Predict
        forecast = self.model.predict(future_df)
        
        return forecast
    
    def cross_validate(self, df: pd.DataFrame,
                      target_col: str = 'new_cases',
                      date_col: str = 'date',
                      initial: str = '180 days',
                      period: str = '30 days',
                      horizon: str = '30 days') -> pd.DataFrame:
        """
        Cross-validate Prophet model.
        
        Uses Prophet's built-in cross-validation.
        """
        try:
            from prophet.diagnostics import cross_validation, performance_metrics
        except ImportError:
            logger.error("Prophet diagnostics not available")
            return pd.DataFrame()
        
        if self.model is None:
            raise ValueError("Model not fitted yet!")
        
        logger.info("Running cross-validation...")
        
        df_cv = cross_validation(
            self.model, 
            initial=initial,
            period=period,
            horizon=horizon,
            parallel="processes"
        )
        
        df_p = performance_metrics(df_cv)
        
        return df_p
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (coefficients for regressors).
        """
        if self.model is None:
            return {}
        
        importance = {}
        
        # Prophet stores coefficients in self.model.params['beta']
        if hasattr(self.model, 'params') and 'beta' in self.model.params:
            betas = self.model.params['beta'].mean(axis=0)
            
            # Map to regressor names
            for i, reg in enumerate(self.regressors):
                if i < len(betas):
                    importance[reg] = abs(betas[i])
        
        return importance
    
    def save(self, filepath: str):
        """Save model to file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'regressors': self.regressors,
                'params': {
                    'changepoint_prior_scale': self.changepoint_prior_scale,
                    'seasonality_prior_scale': self.seasonality_prior_scale,
                    'holidays_prior_scale': self.holidays_prior_scale
                }
            }, f)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load model from file."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(**data['params'])
        instance.model = data['model']
        instance.regressors = data['regressors']
        
        logger.info(f"Model loaded from {filepath}")
        return instance
