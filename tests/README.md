# Test Suite Documentation

## Overview

This project includes comprehensive unit and integration tests to ensure code quality and pipeline integrity.

## Test Structure

```
tests/
├── unit/                    # Unit tests (test individual components)
│   ├── test_preprocessing.py    # Preprocessing function tests
│   ├── test_features.py         # Feature engineering tests
│   └── test_models.py           # Model training/evaluation tests
├── integration/             # Integration tests (test full pipeline)
│   ├── test_pipeline.py         # Full preprocessing + feature engineering
│   └── test_end_to_end.py       # Complete pipeline with all steps
├── conftest.py             # Pytest fixtures and configuration
└── __init__.py
```

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run only unit tests
```bash
pytest tests/unit/ -v -m unit
```

### Run only integration tests
```bash
pytest tests/integration/ -v -m integration
```

### Run specific test file
```bash
pytest tests/unit/test_preprocessing.py -v
```

### Run with coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

## Test Categories

### Unit Tests (`tests/unit/`)

**test_preprocessing.py**
- Tests individual preprocessing functions
- Validates data cleaning, transformation, and outlier removal
- Ensures target variable handling for production vs training

**test_features.py**
- Tests feature engineering functions
- Validates geographic feature calculations (Haversine distance, clustering)
- Tests KMeans model saving and loading

**test_models.py**
- Tests model artifacts existence
- Validates model structure and parameters
- Tests model evaluation functions

### Integration Tests (`tests/integration/`)

**test_pipeline.py**
- Tests complete preprocessing pipeline
- Validates feature engineering with real data
- Checks pipeline consistency and data integrity

**test_end_to_end.py**
- Full end-to-end pipeline test
- Tests all steps from raw data to training-ready features
- Validates final output shape and content

## Key Test Features

### Fixtures (conftest.py)
- Sample datasets for testing
- Mock data generators
- Temporary directory management

### Mocking
- Mocked external data sources for isolated testing
- No dependency on external APIs during tests

### Assertions
- Data shape and type validation
- Value range checks
- Column existence and uniqueness

## Continuous Integration

Tests run automatically on:
- Push to master/main branch
- Pull requests
- Manual trigger via GitHub Actions

See `.github/workflows/ci.yml` for CI configuration.

## Best Practices

1. **Keep tests independent**: Each test should not depend on others
2. **Use fixtures**: Reuse common test data via fixtures
3. **Test behavior, not implementation**: Focus on what the function does
4. **Clear names**: Use descriptive test function names
5. **Fast tests**: Keep tests quick for rapid feedback

## Troubleshooting

### Tests fail locally but pass on CI
- Check Python version (should be 3.10+)
- Verify all dependencies installed
- Check PYTHONPATH is set correctly

### Import errors in tests
- Run: `python -m pytest` (not just `pytest`)
- Ensure `__init__.py` files exist in all packages

### Slow tests
- Mark with `@pytest.mark.slow`
- Run fast tests first: `pytest -m "not slow"`

## Coverage Goals

- Target: 80%+ code coverage
- Critical paths: 100% coverage
- Integration tests: All major workflows

## Adding New Tests

1. Create test file in `tests/unit/` or `tests/integration/`
2. Name file: `test_<module>.py`
3. Name functions: `test_<functionality>`
4. Add markers: `@pytest.mark.unit` or `@pytest.mark.integration`
5. Use fixtures from `conftest.py`
6. Run locally before committing
