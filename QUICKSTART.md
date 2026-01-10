# 🚀 Guide de Démarrage Rapide

## ⚡ 5 Minuts Setup

### 1. Clone & Setup
```bash
git clone https://github.com/YOUR-USER/Projet-ML-Prediction-of-building-energy.git
cd Projet-ML-Prediction-of-building-energy
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Variables d'environnement
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 3. Vérifier l'installation
```bash
pytest tests/ -v  # Devrait passer les tests de preprocess
```

---

## 📊 Workflow Typique

### Scénario 1 : Entraîner un modèle
```bash
python -m src.models.train --experiment="baseline"
```

### Scénario 2 : Comparer modèles
```bash
python -m src.models.compare_pipelines
```

### Scénario 3 : Évaluer modèle
```bash
python -m src.models.run_evaluation
```

### Scénario 4 : Faire prédictions
```python
from src.models.inference import predict
import pandas as pd

# Charger données
X = pd.read_csv("data/processed/...csv")

# Prédire
predictions = predict(X)
print(predictions)
```

---

## 🧪 Tests & CI/CD

### Lancer tests localement
```bash
# Tous les tests
pytest tests/ -v

# Juste preprocessing (devrait passer)
pytest tests/test_preprocess.py -v

# Avec couverture
pytest tests/ --cov=src
```

### Vérifier CI/CD
```bash
# Les tests s'exécutent automatiquement au push
git push origin main
# Voir https://github.com/YOUR-USER/repo/actions
```

---

## 📊 MLflow Dashboard

```bash
mlflow ui
# Accès: http://127.0.0.1:5000
```

Voir :
- Tous les modèles entraînés
- Comparaison métriques
- Hyperparamètres
- Artefacts

---

## 🗂️ Structure Fichiers Clés

| Fichier | Purpose |
|---------|---------|
| `src/preprocessing/preprocessor.py` | Nettoyage données |
| `src/features/engineer.py` | Feature engineering |
| `src/models/train.py` | Entraînement |
| `src/models/evaluate.py` | Évaluation |
| `tests/test_preprocess.py` | Tests preprocess |
| `.github/workflows/ci.yml` | CI/CD pipeline |

---

## 🐛 Problèmes Courants

### ❌ `ModuleNotFoundError: No module named 'src'`
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%cd%                       # Windows
```

### ❌ `FileNotFoundError: model.joblib`
```bash
python -m src.models.train  # Entraîner d'abord
```

### ❌ Import pytest fails
```bash
pip install -q -r requirements.txt
pytest tests/ -v
```

---

## 📚 Documentation Complète

- **[README.md](README.md)** : Documentation complète
- **[ARCHITECTURE.md](ARCHITECTURE.md)** : Architecture système
- **[TESTING.md](TESTING.md)** : Guide tests détaillé

---

## 💡 Conseils

1. **Toujours vérifier que les tests passent avant de pusher**
   ```bash
   pytest tests/test_preprocess.py -v
   ```

2. **Utiliser branches pour nouvelles features**
   ```bash
   git checkout -b feature/my-feature
   git push origin feature/my-feature
   # Créer Pull Request
   ```

3. **Committer souvent avec messages clairs**
   ```bash
   git commit -m "feat: add new preprocessing step"
   ```

4. **Vérifier MLflow pour comparer performances**
   ```bash
   mlflow ui
   ```

---

## 🎯 Prochaines Étapes

1. ✅ Setup complet
2. ✅ Tests passent
3. 🔄 Entraîner modèles
4. 🔄 Comparer performances
5. 🔄 Push & CI/CD vérifie tout
6. 🔄 Déployer API

---

**Besoin d'aide ?** Voir les fichiers de documentation ou créer une Issue.
