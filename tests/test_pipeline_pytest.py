import os
import sys
import pytest

# Change to project root (go up one level from tests/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)

from src.preprocessing.preprocessor import preprocess_data, save_processed_df
from src.features.engineer import engineer_features, prepare_for_training

DATA_URL = "https://raw.githubusercontent.com/MouslyDiaw/handson-machine-learning/refs/heads/master/data/2016_Building_Energy_Benchmarking.csv"

@pytest.mark.integration
def test_pipeline_pytest():
    """Test the complete preprocessing + feature engineering pipeline (pytest version)"""
    df = preprocess_data(DATA_URL)
    assert df.shape[0] == 1553, f"Expected 1553 rows, got {df.shape[0]}"

    df = engineer_features(df)

    expected_cols = [
        'BuildingType', 'PrimaryPropertyType', 'ZipCode', 'CouncilDistrictCode',
        'Neighborhood', 'NumberofBuildings', 'NumberofFloors', 'PropertyGFATotal',
        'PropertyGFAParking', 'PropertyGFABuilding(s)', 'ListOfAllPropertyUseTypes',
        'LargestPropertyUseType', 'LargestPropertyUseTypeGFA', 'ENERGYSTARScore',
        'SiteEnergyUse(kBtu)', 'BuildingAge', 'SiteEnergyUse_log', 'PropertyGFATotal_log',
        'Distance_to_Center', 'Neighborhood_Cluster', 'Is_Downtown', 'Rotated_Lat', 'Rotated_Lon',
        'Surface_Cluster'
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"

    X, y = prepare_for_training(df)
    assert X.shape[0] == 1553, "Feature row count mismatch"
    assert y.shape[0] == 1553, "Target row count mismatch"

    # Save processed data (side effect, not asserted)
    save_processed_df(
        df,
        output_path="data/processed/seattle_energy_cleaned_final.csv",
        version_path="artifacts/data_version.json",
        force=True
    )
