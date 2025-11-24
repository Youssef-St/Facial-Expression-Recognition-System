import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(48,48,1), num_classes=7):
    """
    Crée et retourne un modèle CNN pour la reconnaissance d'expressions faciales.
    
    Args:
        input_shape (tuple): La forme des images en entrée (hauteur, largeur, canaux).
        num_classes (int): Nombre de classes (émotions) à prédire.
    
    Returns:
        model (tf.keras.Model): Modèle CNN construit et prêt à l'entraînement.
    """
    model = models.Sequential([

        # Couche convolutionnelle n°1
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        # Détection locale de motifs sur l'image (par 32 filtres 3x3), ReLU = rectification de la sortie

        layers.BatchNormalization(),
        # Normalise les activations pour accélérer l'apprentissage et améliorer la stabilité
        layers.MaxPooling2D(2,2),
        # Réduction de la taille : garde les valeurs max d'une fenêtre 2x2, réduit la dimension spatiale
        layers.Dropout(0.25),
        # "Drop" (ignorer) 25% des neurones aléatoirement à chaque batch pour éviter le sur-apprentissage

        # Couche convolutionnelle n°2
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),
        # Même principe, mais plus de filtres (64) pour apprendre des caractéristiques plus complexes

        # Couche convolutionnelle n°3
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),
        # Encore plus de filtres (128), permet d'extraire des motifs encore plus abstraits

        layers.Flatten(),
        # Transforme l'image 3D finale en un vecteur 1D pour passer dans les couches denses (fully-connected)

        layers.Dense(128, activation='relu'),
        # "Dense" = tous les neurones connectés, permet de combiner toutes les caractéristiques extraites
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        # Dropout à 50%, aide à la généralisation sur les nouveaux exemples

        layers.Dense(num_classes, activation='softmax')
        # Couche finale : nombre de neurones = nombre de classes, softmax normalise les sorties pour donner une probabilité par émotion
    ])
    return model