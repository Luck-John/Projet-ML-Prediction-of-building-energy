# Documentation des Tests

## 📋 Vue d'ensemble

Ce projet utilise **pytest** pour les tests unitaires et d'intégration. Les tests sont organisés en trois catégories :

1. **Tests de Prétraitement** (`test_preprocess.py`)
2. **Tests de Modèles** (`test_models.py`)
3. **Tests d'Intégration** (`test_integration_metrics.py`)

---

## 🧪 Structure des Tests

### `tests/test_preprocess.py`

Teste les fonctions de prétraitement et nettoyage des données.

#### Tests

| Test | Description | État |
|------|-------------|------|
| `test_preprocess_df_no_nans` | Vérifie suppression des NaN | ✅ PASSED |
| `test_preprocess_from_path` | Teste chargement depuis fichier CSV | ✅ PASSED |

**Lancer :**
```bash
pytest tests/test_preprocess.py -v
```

---

### `tests/test_models.py`

Teste l'existence et structure des artefacts modèles.

#### Tests

| Test | Description | État |
|------|-------------|------|
| `test_model_artifact_exists` | Vérifie existence `artifacts/model.joblib` | ⚠️ Attente d'entraînement |
| `test_model_contains_model_key` | Vérifie structure dict avec clé 'model' | ⚠️ Attente d'entraînement |

**Lancer :**
```bash
pytest tests/test_models.py -v
```

**Note :** Ces tests nécessitent un entraînement préalable. Lancer :
```bash
python -m src.models.train
```

---

### `tests/test_integration_metrics.py`

Teste la cohérence entre notebook et refactoring.

#### Tests

| Test | Description | État |
|------|-------------|------|
| `test_refactored_metrics_close_to_notebook` | Vérifie MAE < 5% écart entre versions | ⚠️ Attente de rapport |

**Lancer :**
```bash
pytest tests/test_integration_metrics.py -v
```

**Note :** Nécessite un rapport `artifacts/compare_report.joblib`. Lancer :
```bash
python -m src.models.compare_pipelines
```

---

## 🚀 Exécution des Tests

### Tous les tests
```bash
pytest tests/ -v
```

### Un fichier spécifique
```bash
pytest tests/test_preprocess.py -v
```

### Un test spécifique
```bash
pytest tests/test_preprocess.py::test_preprocess_df_no_nans -v
```

### Avec couverture de code
```bash
pip install pytest-cov
pytest tests/ --cov=src --cov-report=html
```

### Mode silencieux (minimal output)
```bash
pytest tests/ -q
```

---

## ⚙️ Configuration pytest

Le fichier `pytest.ini` configure :
- **Chemins des tests** : `testpaths = tests`
- **Patterns de fichiers** : `test_*.py`
- **Verbose par défaut** : `-v`
- **Short traceback** : `--tb=short`

---

## 🔧 Fixtures et Helpers

Le fichier `conftest.py` ajoute le chemin racine au Python path :

```python
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
```

Cela permet les imports relatifs comme :
```python
from src.preprocessing.preprocessor import preprocess_data
```

---

## 🐛 Dépannage

### Erreur : `ModuleNotFoundError: No module named 'src'`

**Solution :** 
```bash
# 1. Vérifier que conftest.py existe dans tests/
ls tests/conftest.py

# 2. Vérifier que __init__.py existent
ls src/__init__.py
ls src/preprocessing/__init__.py

# 3. Lancer pytest depuis la racine
cd /path/to/project
pytest tests/ -v
```

### Erreur : `FileNotFoundError: model.joblib`

**Solution :** Entraîner d'abord le modèle
```bash
python -m src.models.train
```

### Tests lents

**Optimisation :**
```bash
# Réduire verbosité
pytest tests/ -q

# Paralléliser avec pytest-xdist
pip install pytest-xdist
pytest tests/ -n auto
```

---

## 📊 Métriques de Couverture

Pour générer un rapport de couverture :
```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

**Objectif :** > 80% de couverture sur `src/`

---

## 🔄 Intégration CI/CD

Les tests s'exécutent automatiquement sur **GitHub Actions** :

Voir `.github/workflows/ci.yml` pour détails.

**Déclencheurs :**
- Push sur `main` ou `master`
- Pull request vers `main` ou `master`

---

## 📝 Bonnes Pratiques

### ✅ À faire
- Nommer les tests avec préfixe `test_`
- Utiliser des assertions explicites
- Tester cas normaux ET cas limites
- Garder tests rapides (< 1s par test)

### ❌ À éviter
- Tests qui dépendent l'un de l'autre
- Tests qui modifient les données
- Tests sans assertions
- Tests très longs (> 10s)

---

## 📚 Ressources

- **pytest Documentation** : https://docs.pytest.org/
- **pytest Fixtures** : https://docs.pytest.org/en/stable/how-to/fixtures.html
- **CI/CD avec pytest** : https://docs.pytest.org/en/stable/ci.html

---

**Dernière mise à jour :** Janvier 2026
