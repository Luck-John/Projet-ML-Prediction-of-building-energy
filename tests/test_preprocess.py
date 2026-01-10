import pandas as pd
from src.preprocessing.preprocessor import preprocess_data, preprocess_df


def test_preprocess_df_no_nans():
    df = pd.read_csv('data/processed/2016_Building_Energy_Benchmarking.csv', nrows=100)
    df_clean = preprocess_df(df)
    assert df_clean.isnull().sum().sum() == 0
    assert 'SiteEnergyUse(kBtu)' in df_clean.columns


def test_preprocess_from_path():
    df_clean2 = preprocess_data('data/processed/2016_Building_Energy_Benchmarking.csv')
    assert df_clean2.shape[0] > 0
    assert 'SiteEnergyUse(kBtu)' in df_clean2.columns
