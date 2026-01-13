"""Unit tests for feature engineering module."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Fix path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.features.engineer import (
    haversine_vectorized,
    engineer_features,
    engineer_features_production,
    prepare_for_training,
    SEATTLE_CENTER_LAT,
    SEATTLE_CENTER_LON,
)


@pytest.fixture
def sample_preprocessed_data():
    """Create sample preprocessed data for testing."""
    # Need at least 10 samples for 10-cluster KMeans
    n_samples = 15
    return pd.DataFrame({
        'BuildingType': ['Office'] * n_samples,
        'PrimaryPropertyType': ['Office'] * n_samples,
        'ZipCode': [98101 + i for i in range(n_samples)],
        'CouncilDistrictCode': list(range(1, n_samples + 1)),
        'Neighborhood': [f'Neighborhood_{i}' for i in range(n_samples)],
        'NumberofBuildings': [1] * n_samples,
        'NumberofFloors': [5 + i for i in range(n_samples)],
        'PropertyGFATotal': [30000.0 + i * 5000 for i in range(n_samples)],
        'PropertyGFAParking': [3000.0 + i * 500 for i in range(n_samples)],
        'PropertyGFABuilding(s)': [27000.0 + i * 4500 for i in range(n_samples)],
        'ListOfAllPropertyUseTypes': ['Office'] * n_samples,
        'LargestPropertyUseType': ['Office'] * n_samples,
        'LargestPropertyUseTypeGFA': [27000.0 + i * 4500 for i in range(n_samples)],
        'ENERGYSTARScore': [70.0 + i for i in range(n_samples)],
        'SiteEnergyUse(kBtu)': [150000.0 + i * 10000 for i in range(n_samples)],
        'BuildingAge': [15 + i for i in range(n_samples)],
        'SiteEnergyUse_log': [11.9 + i * 0.1 for i in range(n_samples)],
        'PropertyGFATotal_log': [10.3 + i * 0.05 for i in range(n_samples)],
        'Latitude': [47.6062 + i * 0.01 for i in range(n_samples)],
        'Longitude': [-122.3321 + i * 0.01 for i in range(n_samples)],
    })


def test_haversine_distance():
    """Test Haversine distance calculation."""
    # Distance from Seattle Center to itself should be 0
    dist = haversine_vectorized(
        SEATTLE_CENTER_LAT, SEATTLE_CENTER_LON,
        SEATTLE_CENTER_LAT, SEATTLE_CENTER_LON
    )
    assert abs(dist) < 0.01  # Nearly zero
    
    # Distance should be positive for different points
    dist = haversine_vectorized(
        47.6062, -122.3321,  # Seattle Center
        47.7, -122.4  # Nearby point
    )
    assert dist > 0


def test_engineer_features(sample_preprocessed_data):
    """Test feature engineering function."""
    df = engineer_features(sample_preprocessed_data, save_models=False)
    
    assert 'Distance_to_Center' in df.columns
    assert 'Neighborhood_Cluster' in df.columns
    assert 'Is_Downtown' in df.columns
    assert 'Rotated_Lat' in df.columns
    assert 'Rotated_Lon' in df.columns
    assert 'Surface_Cluster' in df.columns
    
    # Check data types
    assert pd.api.types.is_numeric_dtype(df['Distance_to_Center'])
    assert pd.api.types.is_numeric_dtype(df['Neighborhood_Cluster'])
    assert pd.api.types.is_numeric_dtype(df['Is_Downtown'])
    
    # Final shape should have 24 columns
    assert df.shape[1] == 24


def test_engineer_features_production(sample_preprocessed_data):
    """Test production feature engineering."""
    # First, create and save models
    df_train = engineer_features(sample_preprocessed_data, save_models=True)
    
    # Test production mode
    df_prod = engineer_features_production(sample_preprocessed_data)
    
    # Should have same columns as training
    assert set(df_train.columns) == set(df_prod.columns)
    assert df_prod.shape[1] == 24


def test_prepare_for_training(sample_preprocessed_data):
    """Test final training data preparation."""
    df = engineer_features(sample_preprocessed_data, save_models=False)
    X, y = prepare_for_training(df)
    
    # Check dimensions
    assert X.shape[0] == df.shape[0]
    assert y.shape[0] == df.shape[0]
    
    # Target should not be in features
    assert 'SiteEnergyUse(kBtu)' not in X.columns
    assert 'SiteEnergyUse_log' not in X.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
