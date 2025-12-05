import os

import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

def preprocess_face_array(pixels, size=(48,48)):
    """
    Prend un tableau 1D (pixels) du CSV, le transforme en image prétraitée pour le CNN.

    :param pixels: liste ou string de valeurs de pixels
    :param size: tuple, taille de sortie (par défaut 48x48)
    :return: tableau numpy shape (48,48,1) normalisé [0,1]
    """
    if isinstance(pixels, str):
        face = np.array([int(p) for p in pixels.split(' ')], dtype='float32')
    else:
        face = np.array(pixels, dtype='float32')
    face = face.reshape(48,48)
    face = cv2.resize(face, size)
    face = face / 255.0
    face = np.expand_dims(face, axis=-1)  # Pour la couche CNN
    return face

def preprocess_face_image(image, size=(48,48)):
    """
    Prend une image OpenCV (par exemple depuis webcam), la convertit en niveaux de gris,
    resize et la normalise.

    :param image: image numpy
    :param size: tuple, taille de sortie
    :return: image normalisée shape (1, 48, 48, 1) prête pour le modèle
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)
    norm_img = gray.astype('float32') / 255.0
    norm_img = np.expand_dims(norm_img, axis=(0, -1))  # batch et canal
    return norm_img

def encode_labels(labels, num_classes=7):
    """
    Encode les labels d'émotion en one-hot.

    :param labels: liste ou array des labels (ex: [0,2,5,1,...])
    :param num_classes: nombre de classes
    :return: array encodé one-hot, shape (N, num_classes)
    """
    return to_categorical(labels, num_classes)


