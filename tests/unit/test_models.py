import os
import joblib


def test_model_artifact_exists():
    path = 'artifacts/model.joblib'
    assert os.path.exists(path), 'Model artifact not found, run training first.'


def test_model_contains_model_key():
    artefact = joblib.load('artifacts/model.joblib')
    assert isinstance(artefact, dict)
    assert 'model' in artefact
