# Smart Agriculture ML

Projet de machine learning pour recommander la meilleure culture à planter selon les conditions du sol et du climat.

Le modèle (Random Forest) prend en entrée 7 paramètres — azote, phosphore, potassium, température, humidité, pH et précipitations — et prédit la culture la plus adaptée.

## Structure

```
datasets/          → dataset crop recommendation (CSV)
models/            → modèle entraîné (.joblib)
notebooks/         → notebook d'exploration et d'entraînement
src/               → script Python du pipeline ML
app.py             → interface web Gradio pour tester le modèle
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

**Entraîner le modèle** : exécuter le notebook `notebooks/module1_crop_recommendation.ipynb` ou le script `src/module1_crop_recommendation.py`.

**Lancer l'app** :

```bash
python app.py
```

L'interface Gradio s'ouvre dans le navigateur. Il suffit de remplir les champs et le modèle renvoie la culture recommandée.

## Dataset

[Crop Recommendation Dataset](https://www.kaggle.com/datasets/abhaymaurya03/crop-recommendation) — 30 530 échantillons, 80 cultures, 7 features.
