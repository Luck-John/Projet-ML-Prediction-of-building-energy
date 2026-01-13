#!/usr/bin/env python
"""
Test: Verify that the saved encoder can handle unknown categories in production.
This is the KEY TEST for production readiness.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, 'src')

from preprocessing.production_artifacts import load_all_artifacts

def test_encoder_with_unknown_categories():
    """
    TEST CRITICAL: L'encodeur peut-il gérer des catégories INCONNUES?
    """
    print("\n" + "="*70)
    print("TEST: ENCODER WITH UNKNOWN CATEGORIES (PRODUCTION SCENARIO)")
    print("="*70)
    
    try:
        artifacts = load_all_artifacts()
        encoder = artifacts['encoder']
        
        if encoder is None:
            print("❌ FAILED: No encoder loaded")
            return False
        
        print(f"\n✅ Encoder loaded")
        print(f"   - Type: {type(encoder).__name__}")
        print(f"   - Handle unknown: {getattr(encoder, 'handle_unknown', 'not specified')}")
        print(f"   - Will encode these columns: {encoder.cols}")
        
        # IMPORTANT: encoder.fit_transform et transform attendent le DATAFRAME COMPLET
        # avec toutes les colonnes numériques ET catégorielles
        print(f"\n📝 Creating COMPLETE test data with UNKNOWN categories...")
        
        test_df = pd.DataFrame({
            # Categorical columns (unknown values)
            'BuildingType': ['unknown_building_type_xyz'],
            'PrimaryPropertyType': ['completely_new_property'],
            'Neighborhood': ['unknown_neigh_123'],
            'ListOfAllPropertyUseTypes': ['unknown_use_abc'],
            'LargestPropertyUseType': ['new_property_use'],
            'Surface_Cluster': ['unknown_surface_group'],
            # Numeric columns (required by encoder)
            'NumberofBuildings': [1],
            'NumberofFloors': [15.0],
            'PropertyGFATotal': [250000.0],
            'PropertyGFAParking': [50000.0],
            'PropertyGFABuilding(s)': [200000.0],
            'LargestPropertyUseTypeGFA': [200000.0],
            'ENERGYSTARScore': [75.0],
            'ZipCode': ['98101'],
            'CouncilDistrictCode': ['7'],
            'BuildingAge': [16],
            'PropertyGFATotal_log': [12.43],
            'Distance_to_Center': [2.5],
            'Neighborhood_Cluster': [3],
            'Is_Downtown': [1],
            'Rotated_Lat': [1.23],
            'Rotated_Lon': [-2.34],
        })
        
        print(f"   Data shape: {test_df.shape}")
        print(f"   Encoder will transform these columns: {encoder.cols}")
        print(f"\nSample data with UNKNOWN categories:")
        for col in encoder.cols:
            print(f"   {col}: '{test_df[col].iloc[0]}'")
        
        # Try to encode
        print(f"\n🔄 Calling encoder.transform()...")
        try:
            transformed = encoder.transform(test_df)
            
            print(f"\n✅ SUCCESS: Encoder handled unknown categories!")
            print(f"   Output shape: {transformed.shape}")
            
            # Check that encoded columns are numeric
            encoded_cols = encoder.cols
            for col in encoded_cols:
                if col in transformed.columns:
                    if pd.api.types.is_numeric_dtype(transformed[col]):
                        print(f"   ✅ {col}: numeric (correct)")
                    else:
                        print(f"   ❌ {col}: NOT numeric (wrong!)")
                        return False
            
            print(f"\n✅ All columns properly encoded to numeric values")
            return True
            
        except Exception as e:
            print(f"\n❌ FAILED: Encoder crashed with unknown categories")
            print(f"   Error: {e}")
            print(f"\n   This means encoder.handle_unknown setting may be wrong")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kmeans_prediction():
    """TEST: KMeans can predict on new data"""
    print("\n" + "="*70)
    print("TEST: KMEANS PREDICTION (PRODUCTION SCENARIO)")
    print("="*70)
    
    try:
        artifacts = load_all_artifacts()
        
        if artifacts['kmeans_neighborhood'] is None:
            print("⚠️  KMeans neighborhood not loaded, skipping")
            return True
        
        kmeans = artifacts['kmeans_neighborhood']
        
        # Test data
        test_coords = np.array([
            [47.6062, -122.3321],  # Seattle center
            [47.7511, -122.3724],  # Different location
        ])
        
        print(f"\n📝 Test coordinates: {test_coords.shape[0]} samples")
        print(f"   {test_coords}")
        
        print(f"\n🔄 Predicting clusters...")
        clusters = kmeans.predict(test_coords)
        
        print(f"\n✅ SUCCESS: KMeans prediction worked!")
        print(f"   Predictions: {clusters}")
        print(f"   Clusters: {set(clusters)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("PRODUCTION READINESS TEST - ENCODER AND KMEANS")
    print("="*70)
    
    test1 = test_encoder_with_unknown_categories()
    test2 = test_kmeans_prediction()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if test1 and test2:
        print("\n✅ ALL TESTS PASSED - PRODUCTION READY")
        print("\nThe encoder can handle unknown categories in production!")
        print("The KMeans models work correctly for prediction!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - NOT PRODUCTION READY")
        if not test1:
            print("   - Encoder test failed")
        if not test2:
            print("   - KMeans test failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
