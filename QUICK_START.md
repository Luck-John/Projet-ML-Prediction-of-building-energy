# 🚀 Quick Start Guide

> **Get started with the Building Energy Prediction API & Dashboard in 5 minutes**

---

## Option 1: Start the API (FastAPI)

### Step 1: Navigate to Project
```bash
cd "Projet ML-Prediction of building energy"
```

### Step 2: Activate Virtual Environment
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### Step 3: Start API Server
```bash
uvicorn src.api.main:app --reload
```

### Step 4: Access API Documentation
- **Interactive Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Step 5: Make a Prediction

**Using curl:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "PropertyGFATotal": 50000,
    "YearBuilt": 1990,
    "Latitude": 47.6,
    "Longitude": -122.3,
    "NumberofBuildings": 1
  }'
```

**Using Python:**
```python
import requests

response = requests.post('http://localhost:8000/predict', json={
    'PropertyGFATotal': 50000,
    'YearBuilt': 1990,
    'Latitude': 47.6,
    'Longitude': -122.3,
    'NumberofBuildings': 1
})

print(response.json())
```

**Response:**
```json
{
  "prediction_kBtu": 125456.78,
  "confidence": "🟢 Faible (High confidence)",
  "model_version": "StackingRegressor v1"
}
```

---

## Option 2: Start the Dashboard (Streamlit)

### Step 1: Navigate to Project
```bash
cd "Projet ML-Prediction of building energy"
```

### Step 2: Activate Virtual Environment
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### Step 3: Start Dashboard
```bash
streamlit run src/dashboard/app.py
```

### Step 4: Access Dashboard
Navigate to: **http://localhost:8501**

### Step 5: Use Dashboard

**Page 1 - 🔮 Prédiction:**
- Fill in building parameters in the form
- Click "Predict" button
- See prediction result with severity indicator

**Page 2 - 📊 Données:**
- View dataset overview (1,553 buildings)
- See statistical summaries
- Explore variable distributions

**Page 3 - 📈 Modèle:**
- View model architecture (Stacking Regressor)
- See performance metrics (MAPE, R², RMSE)
- Check feature importance & hyperparameters

**Page 4 - ℹ️ À Propos:**
- Project description
- Team collaborators with GitHub links
- Additional resources

---

## Option 3: Run Both (Recommended for Testing)

### Terminal 1: Start API
```bash
cd "Projet ML-Prediction of building energy"
.venv\Scripts\activate
uvicorn src.api.main:app --reload
# → http://localhost:8000/docs
```

### Terminal 2: Start Dashboard
```bash
cd "Projet ML-Prediction of building energy"
.venv\Scripts\activate
streamlit run src/dashboard/app.py
# → http://localhost:8501
```

Now you have:
- ✅ REST API at http://localhost:8000
- ✅ Dashboard at http://localhost:8501
- ✅ Full documentation at both interfaces

---

## 🧪 Quick Test: Verify Installation

### Check Python Version
```bash
python --version
# Should be 3.10+
```

### Check Packages Installed
```bash
pip list | grep -E "scikit-learn|fastapi|streamlit|xgboost|lightgbm"
```

### Check Model Exists
```bash
ls artifacts/model.joblib
# Should show: artifacts/model.joblib (24.6 MB)
```

### Run Unit Tests
```bash
pytest tests/ -v
# Should show: 5 passed
```

---

## 📊 API Endpoints Summary

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Check model status |
| `POST` | `/predict` | Make a prediction |
| `GET` | `/metrics` | View model performance |

### GET /health
```bash
curl http://localhost:8000/health
```

### POST /predict
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"PropertyGFATotal": 50000, "YearBuilt": 1990, "Latitude": 47.6, "Longitude": -122.3}'
```

### GET /metrics
```bash
curl http://localhost:8000/metrics
```

---

## 📍 Available Models

The model loaded is a **StackingRegressor** with:

```
Base Learners (trained with grid search):
  • ExtraTrees (n_estimators=500, max_depth=None)
  • XGBoost (n_estimators=300, learning_rate=0.05, max_depth=6)
  • LightGBM (n_estimators=300, learning_rate=0.05, num_leaves=50)
  • HistGradientBoosting (learning_rate=0.05, max_iter=200)

Meta-Learner:
  • LinearSVR(C=10)

Performance:
  • MAPE: 0.420
  • R²: 0.527
  • RMSE: 15,482 kBtu
  • MAE: 11,923 kBtu
```

---

## 🔄 Retrain the Model

If you want to rebuild the model:

```bash
python -m src.models.train
```

This will:
1. Load and preprocess data
2. Apply feature engineering
3. Run grid search on 4 base learners
4. Assemble StackingRegressor
5. Save `artifacts/model.joblib`

**Time:** ~5-15 minutes (depending on your machine)

---

## 📚 Need More Help?

### Check the Full Documentation
- **Main Guide:** [`README.md`](README.md)
- **Troubleshooting:** See "🐛 Troubleshooting" in README.md
- **File Guide:** [`ESSENTIAL_FILES.md`](ESSENTIAL_FILES.md)
- **Project Summary:** [`SUMMARY.md`](SUMMARY.md)

### Common Issues

**Port already in use?**
```bash
# Use different ports
uvicorn src.api.main:app --port 8001
streamlit run src/dashboard/app.py --server.port 8502
```

**ModuleNotFoundError?**
```bash
# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%cd%                       # Windows
```

**Dependencies missing?**
```bash
pip install -r requirements.txt
```

---

## 🎉 You're All Set!

Choose your interface:
- 🔌 **API** — For programmatic access (scripts, microservices)
- 📊 **Dashboard** — For interactive exploration (business users)
- 🧪 **Both** — For complete testing and development

Happy predicting! 🚀

---

**Need help?** Check the team:
- 🧑‍💻 [Malick Sene](https://github.com/malickseneisep2)
- 🧑‍💻 [Ameth Faye](https://github.com/ameth08faye)
- 🧑‍💻 [Hilda Edima](https://github.com/HildaEDIMA)
- 🧑‍💻 [Albert Zinaba](https://github.com/ZINABA-Albert)
