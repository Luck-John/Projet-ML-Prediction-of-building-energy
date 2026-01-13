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
    return pd.DataFrame({
        'BuildingType': ['Office', 'Office', 'Retail'],
        'PrimaryPropertyType': ['Office', 'Office', 'Retail'],
        'ZipCode': [98101, 98102, 98103],
        'CouncilDistrictCode': [1, 2, 3],
        'Neighborhood': ['Downtown', 'Capitol Hill', 'Fremont'],
        'NumberofBuildings': [1, 1, 1],
        'NumberofFloors': [10, 5, 3],
        'PropertyGFATotal': [50000.0, 30000.0, 15000.0],
        'PropertyGFAParking': [5000.0, 3000.0, 1500.0],
        'PropertyGFABuilding(s)': [45000.0, 27000.0, 13500.0],
        'ListOfAllPropertyUseTypes': ['Office', 'Office', 'Retail'],
        'LargestPropertyUseType': ['Office', 'Office', 'Retail'],
        'LargestPropertyUseTypeGFA': [45000.0, 27000.0, 13500.0],
        'ENERGYSTARScore': [75.0, 80.0, 65.0],
        'SiteEnergyUse(kBtu)': [200000.0, 150000.0, 50000.0],
        'BuildingAge': [15, 20, 10],
        'SiteEnergyUse_log': [12.2, 11.9, 10.8],
        'PropertyGFATotal_log': [10.8, 10.3, 9.6],
        'Latitude': [47.6062, 47.6205, 47.6552],
        'Longitude': [-122.3321, -122.3212, -122.3554],
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
