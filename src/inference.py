import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- 1. Charger le modèle entraîné ---
model = load_model('Models/emotion_model.h5')

# --- 2. Définir les labels des émotions ---
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- 3. Initialiser le détecteur de visages OpenCV (Haarcascade) ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 4. Accéder à la webcam ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Pour chaque visage détecté
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]                 # Extraire visage
        roi_gray = cv2.resize(roi_gray, (48, 48))     # Redimension 48x48
        roi = roi_gray.astype('float32') / 255.0      # Normalisation
        roi = np.expand_dims(roi, axis=0)             # Ajouter dimension batch
        roi = np.expand_dims(roi, axis=-1)            # Ajouter canal
        
        # --- 5. Prédiction ---
        preds = model.predict(roi)
        emotion = emotion_labels[np.argmax(preds)]
        
        # --- 6. Affichage cadre & émotion sur image ---
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (255,0,0), 2)
    
    cv2.imshow('Facial Expression Recognition', frame)
    
    # --- 7. Arrêt du programme : touche 'q' ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()