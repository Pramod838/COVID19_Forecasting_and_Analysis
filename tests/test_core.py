"""
Unit tests for core functionality.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.cleaners import COVIDDataCleaner, MobilityDataCleaner
from src.features.builders import FeatureBuilder
from src.evaluation.metrics import mean_absolute_percentage_error, mean_directional_accuracy


class TestDataCleaners(unittest.TestCase):
    """Test data cleaning functions."""
    
    def test_covid_cleaner_handles_negatives(self):
        """Test that negative values are handled."""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'confirmed': range(10),
            'deaths': range(10),
            'new_cases': [-5, 10, 20, -3, 30, 40, 50, 60, 70, 80]
        })
        
        cleaned = COVIDDataCleaner.clean(df)
        
        # Negative values should be clipped to 0
        self.assertTrue((cleaned['new_cases'] >= 0).all())
    
    def test_covid_cleaner_adds_features(self):
        """Test that derived features are added."""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'confirmed': range(10, 20),
            'deaths': range(10)
        })
        
        cleaned = COVIDDataCleaner.add_derived_features(df)
        
        self.assertIn('new_cases', cleaned.columns)
        self.assertIn('new_cases_7day_avg', cleaned.columns)


class TestFeatureBuilder(unittest.TestCase):
    """Test feature engineering."""
    
    def test_temporal_features_created(self):
        """Test temporal features are created."""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=30),
            'new_cases': range(30)
        })
        
        builder = FeatureBuilder()
        result = builder.create_temporal_features(df)
        
        self.assertIn('day_of_week', result.columns)
        self.assertIn('month', result.columns)
        self.assertIn('is_weekend', result.columns)
    
    def test_lag_features_created(self):
        """Test lag features are created."""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=30),
            'new_cases': range(30)
        })
        
        builder = FeatureBuilder(lag_days=[1, 7])
        result = builder.create_lag_features(df, target_col='new_cases')
        
        self.assertIn('new_cases_lag_1', result.columns)
        self.assertIn('new_cases_lag_7', result.columns)


class TestMetrics(unittest.TestCase):
    """Test evaluation metrics."""
    
    def test_mape_calculation(self):
        """Test MAPE calculation."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        # Expected: mean of |10/100|, |10/200|, |10/300| * 100
        expected = np.mean([10, 5, 3.333]) 
        self.assertAlmostEqual(mape, expected, places=1)
    
    def test_mda_calculation(self):
        """Test MDA calculation."""
        y_true = np.array([100, 110, 105, 120, 115])
        y_pred = np.array([100, 115, 100, 125, 110])
        
        # Direction: up, down, up, down
        # True: up, down, up, down
        # Pred: up, down, up, down -> all correct
        mda = mean_directional_accuracy(y_true, y_pred)
        
        self.assertEqual(mda, 100.0)


if __name__ == '__main__':
    unittest.main()
