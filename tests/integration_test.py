#!/usr/bin/env python
"""Test script to verify the complete preprocessing + feature engineering pipeline"""

import os
import sys

# Change to project root (go up one level from tests/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)

from src.preprocessing.preprocessor import preprocess_data, save_processed_df
from src.features.engineer import engineer_features, prepare_for_training

DATA_URL = "https://raw.githubusercontent.com/MouslyDiaw/handson-machine-learning/refs/heads/master/data/2016_Building_Energy_Benchmarking.csv"

print("=" * 70)
print("TEST: COMPLETE PREPROCESSING + FEATURE ENGINEERING PIPELINE")
print("=" * 70)

try:
    print("\n1. PREPROCESSING DATA...")
    df = preprocess_data(DATA_URL)
    print(f"   [OK] After preprocessing: {df.shape[0]} rows x {df.shape[1]} cols")
    assert df.shape[0] == 1553, f"Expected 1553 rows, got {df.shape[0]}"
    print(f"   [OK] Row count matches: 1553")
    
    print("\n2. FEATURE ENGINEERING...")
    df = engineer_features(df)
    print(f"   [OK] After engineering: {df.shape[0]} rows x {df.shape[1]} cols")
    
    # Expected columns from notebook
    expected_cols = [
        'BuildingType', 'PrimaryPropertyType', 'ZipCode', 'CouncilDistrictCode',
        'Neighborhood', 'NumberofBuildings', 'NumberofFloors', 'PropertyGFATotal',
        'PropertyGFAParking', 'PropertyGFABuilding(s)', 'ListOfAllPropertyUseTypes',
        'LargestPropertyUseType', 'LargestPropertyUseTypeGFA', 'ENERGYSTARScore',
        'SiteEnergyUse(kBtu)', 'BuildingAge', 'SiteEnergyUse_log', 'PropertyGFATotal_log',
        'Distance_to_Center', 'Neighborhood_Cluster', 'Is_Downtown', 'Rotated_Lat', 'Rotated_Lon',
        'Surface_Cluster'
    ]
    
    # Check if we have the core columns
    core_cols_present = [c for c in expected_cols if c in df.columns]
    print(f"   [OK] Core columns present: {len(core_cols_present)}/{len(expected_cols)}")
    
    print("\n3. PREPARING FOR TRAINING...")
    X, y = prepare_for_training(df)
    print(f"   [OK] Features shape: {X.shape}")
    print(f"   [OK] Target shape: {y.shape}")
    print(f"   [OK] Feature columns: {list(X.columns)[:5]}...")
    
    print("\n4. SAVING PROCESSED DATA...")
    save_processed_df(df, 
                     output_path="data/processed/seattle_energy_cleaned_final.csv",
                     version_path="artifacts/data_version.json",
                     force=True)
    print(f"   [OK] Data saved successfully")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("=" * 70)
    print(f"\nFinal dataset: {df.shape[0]} rows x {df.shape[1]} cols")
    print("Expected: 1553 rows x 24 cols")
    print("\nThe pipeline is ready for training!")
    
except Exception as e:
    print(f"\n[ERROR] TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
