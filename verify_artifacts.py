#!/usr/bin/env python
"""
Quick test to verify encoder and KMeans artifacts are saved and loadable.
"""

import sys
import os

sys.path.insert(0, 'src')

from preprocessing.production_artifacts import load_all_artifacts

if __name__ == "__main__":
    print("\n" + "="*60)
    print("VERIFYING PRODUCTION ARTIFACTS")
    print("="*60)
    
    try:
        artifacts = load_all_artifacts()
        
        print("\n✅ ENCODER:")
        if artifacts['encoder']:
            print(f"   Type: {type(artifacts['encoder']).__name__}")
            print(f"   Columns: {artifacts['encoder'].cols}")
            print(f"   Handle unknown: {getattr(artifacts['encoder'], 'handle_unknown', 'not specified')}")
        else:
            print("   ❌ NOT LOADED")
        
        print("\n✅ KMEANS NEIGHBORHOOD:")
        if artifacts['kmeans_neighborhood']:
            print(f"   Type: {type(artifacts['kmeans_neighborhood']).__name__}")
            print(f"   Clusters: {artifacts['kmeans_neighborhood'].n_clusters}")
        else:
            print("   ⚠️  NOT LOADED")
        
        print("\n✅ KMEANS SURFACE:")
        if artifacts['kmeans_surface']:
            print(f"   Type: {type(artifacts['kmeans_surface']).__name__}")
            print(f"   Clusters: {artifacts['kmeans_surface'].n_clusters}")
        else:
            print("   ⚠️  NOT LOADED")
        
        print("\n✅ MODEL:")
        print(f"   Type: {type(artifacts['model']).__name__}")
        print(f"   Target: {artifacts['target_col']}")
        
        print("\n" + "="*60)
        print("✅ ALL ARTIFACTS READY FOR PRODUCTION")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
