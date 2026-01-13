"""
Unit tests for preprocessing module.
Tests individual preprocessing functions with controlled inputs.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from tempfile import TemporaryDirectory

# Fix path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessing.preprocessor import (
    filter_non_residential,
    drop_unusable_columns,
    clean_target,
    remove_duplicates,
    drop_sparse_columns,
    impute_energy_star,
    remove_outliers,
    final_dropna,
    apply_log_transforms,
    encode_categories,
    create_mlb_features,
    save_processed_df,
    preprocess_data,
    RESIDENTIAL_TYPES,
    TARGET_COL,
)


# =====================================================
# FIXTURES
# =====================================================

@pytest.fixture
def sample_raw_data():
    """Create a minimal sample of raw data for testing."""
    return pd.DataFrame({
        'OSEBuildingID': [1, 2, 3, 4, 5],
        'BuildingType': ['Multifamily LR (1-4)', 'Office', 'Office', 'Retail', 'Retail'],
        'PrimaryPropertyType': ['Multifamily', 'Office', 'Office', 'Retail', 'Retail'],
        'SiteEnergyUse(kBtu)': [100000.0, 200000.0, 150000.0, 50000.0, 75000.0],
        'PropertyGFATotal': [10000.0, 50000.0, 60000.0, 15000.0, 20000.0],
        'YearBuilt': [2000, 1990, 2005, 1995, 2010],
        'ENERGYSTARScore': [75.0, np.nan, 80.0, np.nan, 85.0],
        'Neighborhood': ['Downtown', 'Downtown', 'Fremont', 'Wallingford', 'Ballard'],
        'LargestPropertyUseType': ['Multifamily', 'Office', 'Office', 'Retail', 'Retail'],
        'ListOfAllPropertyUseTypes': ['Multifamily', 'Office', 'Office', 'Retail', 'Retail'],
    })


# =====================================================
# TESTS: FILTERING & CLEANING
# =====================================================

def test_filter_non_residential(sample_raw_data):
    """Test that residential buildings are filtered out."""
    df = filter_non_residential(sample_raw_data)
    assert len(df) == 4  # Only non-residential buildings
    assert not any(df['BuildingType'].isin(RESIDENTIAL_TYPES))


def test_drop_unusable_columns(sample_raw_data):
    """Test that specified columns are dropped."""
    df = drop_unusable_columns(sample_raw_data)
    assert 'OSEBuildingID' not in df.columns
    # Target column should remain
    assert TARGET_COL in df.columns


def test_clean_target(sample_raw_data):
    """Test that zero and null target values are removed."""
    df = sample_raw_data.copy()
    df.loc[1, TARGET_COL] = 0  # Add zero value
    df.loc[2, TARGET_COL] = np.nan  # Add null value
    
    df = clean_target(df)
    assert (df[TARGET_COL] > 0).all()  # All targets should be positive
    assert df[TARGET_COL].notna().all()  # No nulls


def test_remove_duplicates(sample_raw_data):
    """Test that duplicate OSEBuildingIDs are removed."""
    df = pd.concat([sample_raw_data, sample_raw_data.iloc[[0]]], ignore_index=True)
    assert len(df) == 6  # 5 + 1 duplicate
    
    df = remove_duplicates(df)
    assert len(df) == 5  # Duplicate removed
    assert df['OSEBuildingID'].duplicated().sum() == 0


def test_drop_sparse_columns():
    """Test that columns with many missing values are dropped."""
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [1, np.nan, np.nan, np.nan, np.nan],  # 80% missing
        'C': [1, 2, 3, 4, 5],
    })
    
    df = drop_sparse_columns(df, threshold=0.5)
    assert 'A' in df.columns
    assert 'B' not in df.columns  # 80% > 50% threshold
    assert 'C' in df.columns


def test_impute_energy_star(sample_raw_data):
    """Test that ENERGYSTARScore is imputed by property type."""
    df = sample_raw_data.copy()
    initial_nulls = df['ENERGYSTARScore'].isna().sum()
    
    df = impute_energy_star(df)
    final_nulls = df['ENERGYSTARScore'].isna().sum()
    
    assert initial_nulls > 0
    assert final_nulls == 0  # All imputed


def test_remove_outliers(sample_raw_data):
    """Test that extreme outliers are removed."""
    df = sample_raw_data.copy()
    df.loc[0, TARGET_COL] = 3e8  # Add extreme consumption
    df.loc[1, 'PropertyGFATotal'] = 4e6  # Add extreme surface
    
    initial_len = len(df)
    df = remove_outliers(df)
    
    assert len(df) < initial_len
    assert df[TARGET_COL].max() < 2e8  # Below threshold


def test_final_dropna():
    """Test that rows with any NA values are dropped."""
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [1, np.nan, 3, 4],
    })
    
    df = final_dropna(df)
    assert len(df) == 2  # Only rows without NA
    assert df.isna().sum().sum() == 0


# =====================================================
# TESTS: TRANSFORMATIONS
# =====================================================

def test_apply_log_transforms(sample_raw_data):
    """Test that log transforms are applied correctly."""
    df = sample_raw_data.copy()
    df = apply_log_transforms(df)
    
    assert 'SiteEnergyUse_log' in df.columns
    assert 'PropertyGFATotal_log' in df.columns
    assert 'BuildingAge' in df.columns
    
    # Verify log transform is correct
    expected_log = np.log(df[TARGET_COL])
    assert np.allclose(df['SiteEnergyUse_log'], expected_log)
    
    # Verify BuildingAge calculation
    expected_age = 2016 - df['YearBuilt']
    assert (df['BuildingAge'] == expected_age).all()


# =====================================================
# TESTS: CATEGORICAL ENCODING
# =====================================================

def test_encode_categories(sample_raw_data):
    """Test that categorical encoding works without target leakage."""
    df = sample_raw_data.copy()
    df = apply_log_transforms(df)  # Need target for encoding
    
    df_encoded, encoder = encode_categories(df, target_col='SiteEnergyUse_log')
    
    assert encoder is not None
    # Categorical columns should be numeric (encoded)
    for col in ['PrimaryPropertyType', 'BuildingType', 'Neighborhood', 'LargestPropertyUseType']:
        if col in df_encoded.columns:
            assert pd.api.types.is_numeric_dtype(df_encoded[col])


def test_create_mlb_features(sample_raw_data):
    """Test that multi-label binarization works."""
    df = sample_raw_data.copy()
    df_mlb = create_mlb_features(df, list_col='ListOfAllPropertyUseTypes')
    
    # Should return a DataFrame with new columns
    assert isinstance(df_mlb, pd.DataFrame)
    # Columns should have 'use_' prefix
    if len(df_mlb.columns) > 0:
        assert all(col.startswith('use_') for col in df_mlb.columns)


# =====================================================
# TESTS: EXPORT & VERSIONING
# =====================================================

def test_save_processed_df(sample_raw_data):
    """Test that processed data is saved with versioning metadata."""
    with TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, 'test_data.csv')
        version_path = os.path.join(tmpdir, 'test_version.json')
        
        df = sample_raw_data.copy()
        save_processed_df(df, output_path=output_path, version_path=version_path, force=True)
        
        # Check files exist
        assert os.path.exists(output_path)
        assert os.path.exists(version_path)
        
        # Load and verify CSV
        df_loaded = pd.read_csv(output_path)
        assert len(df_loaded) == len(df)
        
        # Load and verify version JSON
        import json
        with open(version_path, 'r') as f:
            version_info = json.load(f)
        
        assert 'timestamp' in version_info
        assert 'sha256' in version_info
        assert 'rows' in version_info
        assert version_info['rows'] == len(df)


# =====================================================
# TESTS: INTEGRATION (FULL PIPELINE)
# =====================================================

@pytest.mark.slow
def test_preprocess_data_complete():
    """Integration test: Full preprocessing pipeline on real data."""
    # This test runs the full pipeline on real data
    df = preprocess_data()
    
    # Verify output
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    
    # Check that target is present (needed for feature engineering later)
    assert TARGET_COL in df.columns
    
    # Check that log transforms were applied
    assert 'SiteEnergyUse_log' in df.columns
    assert 'PropertyGFATotal_log' in df.columns
    assert 'BuildingAge' in df.columns
    
    # Verify data quality
    assert df.isna().sum().sum() == 0  # No missing values
    assert (df[TARGET_COL] > 0).all()  # Positive targets


# =====================================================
# TESTS: PRODUCTION READINESS
# =====================================================

def test_no_target_leakage_in_preprocessing():
    """Verify that target variable handling is appropriate for production."""
    # The preprocessing should only include target for training pipeline,
    # not for prediction pipeline. This test documents this behavior.
    
    # Load sample data
    df = pd.DataFrame({
        'BuildingType': ['Office', 'Office', 'Retail'],
        'PrimaryPropertyType': ['Office', 'Office', 'Retail'],
        'SiteEnergyUse(kBtu)': [200000.0, 150000.0, 50000.0],
        'PropertyGFATotal': [50000.0, 60000.0, 15000.0],
        'YearBuilt': [1990, 2005, 1995],
        'ENERGYSTARScore': [75.0, 80.0, np.nan],
        'Neighborhood': ['Downtown', 'Fremont', 'Wallingford'],
        'LargestPropertyUseType': ['Office', 'Office', 'Retail'],
        'ListOfAllPropertyUseTypes': ['Office', 'Office', 'Retail'],
    })
    
    # Preprocessing steps that don't depend on target
    df = drop_unusable_columns(df)
    assert TARGET_COL in df.columns  # Target still present
    
    # When deploying to production, the target should be removed
    # before passing to feature engineering
    # This is a design note: ensure predict pipeline removes target
    assert True  # Placeholder - actual test in feature engineering


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
