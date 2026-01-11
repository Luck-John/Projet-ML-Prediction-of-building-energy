# 📦 Fichiers Essentiels du Projet ML

## 🎯 **Pour Utiliser le Modèle en Production**

### Minimum Requis:
```
artifacts/
  └── model.joblib              ← LE MODÈLE FINAL (entrée principale)
```

**Usage:**
```python
import joblib
import numpy as np

# Charger le modèle
model_dict = joblib.load("artifacts/model.joblib")
model = model_dict['model']              # StackingRegressor
encoder = model_dict['encoder']          # TargetEncoder
best_params = model_dict['best_params']  # Info

# Prédire
X_new = ...  # DataFrame avec les features
y_pred_log = model.predict(X_new)
y_pred_real = np.exp(y_pred_log)  # Convert from log scale
```

---

## 🔧 **Pour Réentraîner le Modèle**

### Fichiers Utilisés:

#### 1. **Scripts Python** (src/)
```
src/models/train.py                    ← Lance l'entraînement complet
src/models/evaluate.py                 ← Évalue les performances
src/models/compare_pipelines.py        ← Compare les pipelines
src/preprocessing/preprocessor.py      ← Prétraitement des données
src/features/engineer.py               ← Ingénierie des features
```

#### 2. **Données** (data/)
```
data/processed/2016_Building_Energy_Benchmarking.csv  ← Données traitées
data/raw/2016_Building_Energy_Benchmarking.csv        ← Source brute
```

#### 3. **Configuration**
```
requirements.txt                       ← Dépendances Python
```

#### 4. **Tests**
```
tests/test_preprocess.py              ← Validation prétraitement
tests/test_models.py                  ← Validation modèle
tests/test_integration_metrics.py     ← Tests intégration
```

---

## 📋 **Structure Complète du Projet**

```
Projet ML-Prediction of building energy/
│
├── 📂 artifacts/                      ✅ MODÈLES & RÉSULTATS
│   ├── model.joblib                   ✅ Model FINAL (24.6 MB)
│   ├── best_params.joblib             ✅ Hyperparamètres
│   └── compare_report.joblib          ✅ Rapport comparaison
│
├── 📂 .github/                        ✅ CI/CD GITHUB ACTIONS
│   └── workflows/ci.yml               ✅ Pipeline automatique
│
├── 📂 data/                           ✅ DONNÉES
│   ├── processed/                     ✅ Données traitées (~396 KB)
│   │   └── 2016_Building_Energy_Benchmarking.csv
│   └── raw/                           ✅ Données brutes (~1.2 MB)
│       └── 2016_Building_Energy_Benchmarking.csv
│
├── 📂 notebooks/                      ✅ DOCUMENTATION NOTEBOOK
│   └── energy_01_analyse (11).ipynb   ✅ RÉFÉRENCE MODÈLE FINAL
│
├── 📂 src/                            ✅ CODE SOURCE PRINCIPAL
│   ├── preprocessing/
│   │   └── preprocessor.py            ✅ Nettoyage & encodage données
│   ├── features/
│   │   └── engineer.py                ✅ Création de features
│   └── models/
│       ├── train.py                   ✅ Entraînement stacking
│       ├── evaluate.py                ✅ Évaluation performances
│       └── compare_pipelines.py       ✅ Comparaison pipelines
│
├── 📂 tests/                          ✅ TESTS UNITAIRES
│   ├── test_preprocess.py             ✅ Valide preprocessor.py
│   ├── test_models.py                 ✅ Valide model.joblib
│   ├── test_integration_metrics.py    ✅ Valide comparaison
│   └── conftest.py                    ✅ Configuration pytest
│
├── 📂 .venv/                          ✅ VIRTUALENV (local only)
│   └── (ne pas git commit)
│
├── 📄 requirements.txt                ✅ Dépendances Python
├── 📄 pytest.ini                      ✅ Configuration pytest
├── 📄 README.md                       ✅ Documentation générale
├── 📄 .gitignore                      ✅ Fichiers à ignorer
├── 📄 METRICS_ALIGNMENT.md            ✅ Qualité métriques
└── 📄 CLEANUP_AUDIT.md                ✅ Audit nettoyage
```

---

## 🚀 **Commandes Essentielles**

### Installation
```bash
# Créer virtualenv
python -m venv .venv

# Activer (Windows)
.venv\Scripts\activate

# Installer dépendances
pip install -r requirements.txt
```

### Entraînement
```bash
# Lancer entraînement complet (produit artifacts/model.joblib)
python -m src.models.train

# Générer rapport comparaison
python -m src.models.compare_pipelines

# Lancer tests
python -m pytest tests/ -v
```

### CI/CD (GitHub Actions)
```
Automatique: à chaque push sur master
  1. Installe dépendances
  2. Lance entraînement
  3. Génère comparaison
  4. Lance tests
  5. Upload artifacts si succès
```

---

## 🎯 **Ce que Chaque Fichier Fait**

| Fichier | Rôle | Entrée | Sortie |
|---------|------|--------|--------|
| `preprocessor.py` | Nettoie & encode données | CSV brut | Données encodées |
| `engineer.py` | Crée features (distance, age, clusters) | DataFrame | DataFrame + features |
| `train.py` | Grid search + stacking + save model | Données | `model.joblib` |
| `evaluate.py` | Calcule métriques (MAE, RMSE, R², MAPE) | Prédictions | Métriques |
| `compare_pipelines.py` | Compare notebook vs script | Modèles | `compare_report.joblib` |
| `test_*.py` | Vérifie tout fonctionne | Code | PASS/FAIL |

---

## 📊 **Métriques Finales du Modèle**

```
Architecture: StackingRegressor
  Base learners: ExtraTrees, XGBoost, LightGBM, HistGradientBoosting
  Meta-learner: LinearSVR(C=10)

Performance (Test Set):
  MAPE: 0.4201 (21% erreur moyenne)
  R²:   0.527  (53% variance expliquée)
  MAE:  2,396,297 kBtu
  RMSE: 7,877,872 kBtu
```

---

## ✅ **Résumé - Fichiers à NE PAS TOUCHER**

```
✅ artifacts/model.joblib          - LE MODÈLE FINAL
✅ src/                             - Code production
✅ tests/                           - Validation
✅ data/processed/                  - Données d'entraînement
✅ .github/workflows/ci.yml         - Pipeline CI/CD
✅ requirements.txt                 - Dépendances
```

---

## 🎓 **Pour Créer un Dashboard ou API**

**Exemple avec FastAPI:**
```python
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Charger modèle au démarrage
model_dict = joblib.load("artifacts/model.joblib")
model = model_dict['model']
encoder = model_dict['encoder']

@app.post("/predict")
def predict(features: dict):
    X = pd.DataFrame([features])
    if encoder:
        X = encoder.transform(X)
    pred_log = model.predict(X)[0]
    return {"energy": float(np.exp(pred_log))}
```

C'est tout! Besoin de plus? 🚀
