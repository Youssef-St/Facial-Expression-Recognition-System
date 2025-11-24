# Système de Reconnaissance des Expressions Faciales en Temps Réel

## Fonctionnalités principales

- **Prétraitement des images** : conversion en niveaux de gris, redimensionnement à 48×48 pixels, normalisation.
- **Entraînement d’un modèle CNN** sur le dataset [FER2013](https://www.kaggle.com/datasets/msambare/fer2013).
- **Capture vidéo et détection de visage** via OpenCV.
- **Affichage en temps réel** de la prédiction sur le visage détecté.

## Structure du projet

```
facial-expression-recognition/
├── README.md
|
├── data/
│   └── fer2013.csv            # dataset FER2013
├── models/
│   └── emotion_model.h5       # modèle entraîné
├── src/
│   ├── train.py               # script d'entraînement du modèle
│   ├── model.py               # définition du CNN
│   ├── inference.py           # prédiction en temps réel via webcam
│   └── utils.py               # fonctions utilitaires
|
└── tests/
    └── test_model.py          # tests unitaires (optionnel)
```

## Installation


## Dépendances principales

- Python 3.x
- TensorFlow 2.x / Keras
- OpenCV
- NumPy
- pandas
- matplotlib

## Ressources

- [Dataset FER2013 sur Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- [Documentation TensorFlow](https://www.tensorflow.org/)
- [Documentation OpenCV](https://docs.opencv.org/)





