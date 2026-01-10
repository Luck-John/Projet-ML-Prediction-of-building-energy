# Seattle Building Energy Prediction

## 📋 Vue d'ensemble

Ce projet prédait la consommation totale d'énergie (`SiteEnergyUse(kBtu)`) des bâtiments non-résidentiels de Seattle. L'objectif est également d'évaluer la pertinence de l'`ENERGYSTARScore` dans la modélisation.

**Dataset:** 2016 Building Energy Benchmarking Data (Seattle)

---

## 🏗️ Architecture du Projet

```
project_root/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Configuration CI/CD (GitHub Actions)
├── artifacts/                     # Modèles entraînés et données
│   ├── best_model_*.joblib
│   ├── multi_model_ranking_*.joblib
│   ├── X_train.joblib, X_test.joblib
│   └── y_train.joblib, y_test.joblib
├── configs/
│   └── params.yaml                # Paramètres de configuration
├── data/
│   ├── raw/                       # Données brutes
│   │   └── 2016_Building_Energy_Benchmarking.csv
│   └── processed/                 # Données nettoyées
│       └── 2016_Building_Energy_Benchmarking.csv
├── notebooks/
│   └── energy_01_analyse (5).ipynb   # Notebook de prototypage
├── src/
│   ├── api/
│   │   └── main.py                # API REST (FastAPI)
│   ├── features/
│   │   └── engineer.py            # Feature engineering
│   ├── models/
│   │   ├── train.py               # Entraînement MLflow
│   │   ├── evaluate.py            # Évaluation
│   │   ├── inference.py           # Inférence
│   │   ├── compare_pipelines.py   # Comparaison modèles
│   │   └── multi_evaluate.py      # Multi-scénarios
│   └── preprocessing/
│       └── preprocessor.py        # Nettoyage & prétraitement
├── tests/
│   ├── conftest.py                # Fixture pytest
│   ├── test_preprocess.py         # Tests prétraitement
│   ├── test_models.py             # Tests modèles
│   └── test_integration_metrics.py # Tests intégration
├── requirements.txt               # Dépendances Python
├── pytest.ini                     # Configuration pytest
└── README.md                      # Ce fichier
```

---

## 🚀 Installation & Setup

### 1. Cloner le projet
```bash
git clone https://github.com/votre-user/Projet-ML-Prediction-of-building-energy.git
cd Projet-ML-Prediction-of-building-energy
```

### 2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Ajouter le projet au PYTHONPATH
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
# ou
set PYTHONPATH=%cd%                       # Windows
```

---

## 📊 Données & Prétraitement

### Filtrage Appliqué
- **Bâtiments non-résidentiels uniquement** : Exclusion des multifamilial (1-4, 5-9, 10+)
- **Suppression des valeurs aberrantes** :
  - Consommation > 2×10⁸ kBtu (super-consommateurs)
  - Surface > 3×10⁶ sqft
- **Gestion des valeurs manquantes** :
  - `ENERGYSTARScore` : Imputation par médiane par type de bâtiment
  - Autres : Suppression des lignes

### Transformations
- **Log-transformation** : `SiteEnergyUse_log = log(SiteEnergyUse(kBtu))`
- **Target Encoding** : Variables catégorielles (smoothing=10)
- **Feature Engineering** :
  - Distance au centre-ville
  - Clustering spatial (10 clusters)
  - Indicateur centre-ville (< 2km)
  - Clustering par surface

---

## 🔬 Modèles & Scénarios

### Scénarios Testés
1. **Avec ENERGYSTARScore** : Utilisation complète du score
2. **Sans ENERGYSTARScore** : Exclusion du score (robustesse)

### Familles de Modèles
| Famille | Modèles |
|--------|---------|
| **Linéaire** | LinearRegression, Ridge, Lasso, ElasticNet, LinearSVR |
| **Arbre** | DecisionTree, KNN |
| **Ensemble** | RandomForest, ExtraTrees, XGBoost, LightGBM, HistGradientBoosting |

### Hyperparamètres Optimisés
- **GridSearchCV** (CV=5 pour linéaire, CV=3 pour ensemble)
- **Métrique** : RMSE négative
- **Parallélisation** : n_jobs=-1

---

## 📈 Métriques d'Évaluation

### Métriques Log-Scale
- **RMSE_Log** : Erreur RMS sur l'espace log
- **R²_Log** : Coefficient de détermination log

### Métriques Real-Scale (Prioritaires)
- **MAPE_Real** : Erreur Moyenne en Pourcentage Absolu (critère #1)
- **R²_Real** : Coefficient de détermination réel (critère #2)
- **RMSE_Real** : Erreur RMS réelle (kBtu)
- **MAE_Real** : Erreur Absolue Moyenne (kBtu)

---

## 🔧 Utilisation

### 1. Entraînement Simple
```bash
python -m src.models.train
```

### 2. Évaluation
```bash
python -m src.models.run_evaluation
```

### 3. Comparaison Multi-Modèles
```bash
python -m src.models.compare_pipelines
```

### 4. Tests Unitaires
```bash
pytest tests/ -v
```

### 5. Inférence
```python
from src.models.inference import predict

prediction = predict(X_new)
print(f"Prédiction : {prediction} kBtu")
```

---

## 🔍 MLflow Tracking

Tous les modèles sont automatiquement loggés dans **MLflow** :
```bash
mlflow ui  # Lancer le dashboard (http://127.0.0.1:5000)
```

**Éléments trackés:**
- Hyperparamètres
- Métriques (MAPE, R², RMSE, MAE)
- Modèle serialisé
- Artefacts (encoder, scaler)

---

## 🤖 CI/CD Pipeline

### GitHub Actions (`.github/workflows/ci.yml`)

Exécution automatique sur chaque `git push` :

1. ✅ **Setup Python** : 3.10
2. ✅ **Install Dependencies** : `pip install -r requirements.txt`
3. ✅ **Lint** : Vérification syntaxe avec flake8
4. ✅ **Run Tests** : `pytest tests/`
5. ✅ **Upload Artifacts** : Résultats tests

**Branches:** main, master
**Status:** Visible sur GitHub (badges, actions tab)

---

## 🧪 Structure des Tests

### `test_preprocess.py`
- ✅ Vérification suppression NaNs
- ✅ Présence cible
- ✅ Chargement depuis fichier

### `test_models.py`
- ✅ Existence artefacts modèles
- ✅ Structure correcte (dict avec clé 'model')

### `test_integration_metrics.py`
- ✅ Comparaison notebook vs refactor
- ✅ Tolérance 5% sur MAE

**Exécution locale:**
```bash
pytest tests/ -v --tb=short
```

---

## 📝 Configuration (`configs/params.yaml`)

```yaml
# Ajouter si nécessaire
data:
  raw_path: data/raw/2016_Building_Energy_Benchmarking.csv
  processed_path: data/processed/2016_Building_Energy_Benchmarking.csv

preprocessing:
  test_size: 0.2
  random_state: 42
  target_col: SiteEnergyUse_log

models:
  random_state: 42
  cv_folds: 5
```

---

## 🐛 Dépannage

### Erreur : `ModuleNotFoundError: No module named 'src'`
**Solution :**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Erreur : `ImportError` lors des tests
**Solution :**
```bash
# Assurez-vous que conftest.py ajoute le root au path
pytest tests/ -v
```

### MLflow ne sauvegarde pas
**Solution :**
```bash
# Vérifier l'expérience MLflow
mlflow experiments list
mlflow runs list --experiment-id 0
```

---

## 📚 Ressources

- **Data Source** : [Kaggle - Seattle Building Energy](https://www.kaggle.com/datasets)
- **Scikit-learn** : [Documentation](https://scikit-learn.org/)
- **MLflow** : [Documentation](https://mlflow.org/)
- **XGBoost** : [Documentation](https://xgboost.readthedocs.io/)
- **LightGBM** : [Documentation](https://lightgbm.readthedocs.io/)

---

## 👥 Contributeurs

- **MLOps Engineer / Code Quality** : Votre Nom

---

## 📄 Licence

Ce projet est sous licence MIT. Voir `LICENSE` pour détails.

---

## ✅ Checklist MLOps - Bloc 4

- ✅ Architecture Git structurée (src/, tests/, notebooks/, data/)
- ✅ Refactoring Notebook → Scripts Python
- ✅ MLflow tracking intégré
- ✅ Tests unitaires et intégration
- ✅ CI/CD avec GitHub Actions
- ✅ Gestion des imports (`__init__.py`)
- ✅ Pytest configuré
- ✅ Documentation complète (README)

---

**Dernière mise à jour :** Janvier 2026
