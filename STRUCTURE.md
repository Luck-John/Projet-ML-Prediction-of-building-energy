# Machine Learning Project Structure

## Directories

```
Projet ML-Prediction of building energy/
├── src/                           # Source code
│   ├── preprocessing/            # Data preprocessing
│   │   ├── __init__.py
│   │   └── preprocessor.py
│   ├── features/                 # Feature engineering
│   │   ├── __init__.py
│   │   └── engineer.py
│   ├── models/                   # Model training and evaluation
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── compare_pipelines.py
│   └── __init__.py
├── tests/                         # Test suite
│   ├── unit/                     # Unit tests
│   │   ├── test_preprocessing.py
│   │   ├── test_features.py
│   │   └── test_models.py
│   ├── integration/              # Integration tests
│   │   ├── test_pipeline.py
│   │   └── test_end_to_end.py
│   ├── conftest.py              # Pytest fixtures
│   └── __init__.py
├── artifacts/                     # Model artifacts
│   ├── model.joblib             # Trained model
│   ├── model.pkl                # Model (pickle backup)
│   ├── best_params.joblib       # Best hyperparameters
│   ├── kmeans_neighborhood.joblib  # KMeans model for neighborhoods
│   ├── kmeans_neighborhood.pkl     # KMeans model (pickle)
│   ├── kmeans_surface.joblib       # KMeans model for surface
│   ├── kmeans_surface.pkl          # KMeans model (pickle)
│   └── data_version.json        # Data versioning metadata
├── data/                          # Data directory
│   ├── raw/                     # Raw data
│   └── processed/               # Processed data
├── notebooks/                     # Jupyter notebooks for EDA
│   └── energy_01_EDA.ipynb
├── mlruns/                        # MLflow runs directory
├── api/                           # API (FastAPI/Flask)
├── .github/
│   └── workflows/
│       └── ci.yml               # CI/CD pipeline
├── pytest.ini                     # Pytest configuration
├── .gitignore
├── README.md
└── requirements.txt              # Python dependencies
```

## Key Components

1. **src/**: Core ML code
   - preprocessing: Data cleaning and preparation
   - features: Feature engineering and selection
   - models: Training, evaluation, and comparison

2. **tests/**: Test suite
   - unit/: Individual component tests
   - integration/: End-to-end pipeline tests

3. **artifacts/**: Trained models and artifacts
   - Models saved in joblib format (with pickle backups)
   - KMeans clustering models for production

4. **MLflow Integration**:
   - Tracks experiments and model performance
   - Stores runs in mlruns/ directory
   - Version control for models and parameters

5. **CI/CD Pipeline**: 
   - .github/workflows/ci.yml for automated testing and deployment
