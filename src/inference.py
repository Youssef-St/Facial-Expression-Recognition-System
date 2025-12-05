import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from age_utils import age_from_distribution


EMOTION_MODEL_PATH = '../Models/emotion_model.h5'
AGE_MODEL_PATH = '../Models/age_model_best.keras'

print("Loading models...")

emotion_model = load_model(EMOTION_MODEL_PATH, compile=False)
age_model = load_model(AGE_MODEL_PATH, compile=False)

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if face_cascade.empty():
    raise IOError("Error: 'haarcascade_frontalface_default.xml' not found. "
                  "Make sure the file is in the same folder as this script.")

# --- 2. WEBCAM LOOP ---
cap = cv2.VideoCapture(0)

# Variable to store age to prevent flickering
saved_age = None

print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # If no faces are detected, reset saved age so we re-calculate for the next person
    if len(faces) == 0:
        saved_age = None

    for (x, y, w, h) in faces:
        # ---------------------------
        # 1. EMOTION DETECTION
        # ---------------------------
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_emotion = roi_gray.astype('float32') / 255.0
        roi_emotion = np.expand_dims(roi_emotion, axis=0)
        roi_emotion = np.expand_dims(roi_emotion, axis=-1)

        preds = emotion_model.predict(roi_emotion, verbose=0)
        emotion = emotion_labels[np.argmax(preds)]

        # ---------------------------
        # 2. AGE PREDICTION
        # ---------------------------
        if saved_age is None:
            # Extract face RGB
            face = frame[y:y + h, x:x + w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # Resize to 128x128 (Model Input Size)
            face_resized = cv2.resize(face_rgb, (128, 128))

            # Expand dims to make it a batch of 1: (1, 128, 128, 3)
            face_batch = np.expand_dims(face_resized, axis=0).astype('float32')

            # Preprocessing: MobileNet expects values between -1 and 1
            face_input = preprocess_input(face_batch)

            # Predict
            model_output = age_model.predict(face_input, verbose=0)

            # Get the first prediction vector (117,)
            age_distribution = model_output[0]

            # Calculate weighted age
            predicted_age = age_from_distribution(age_distribution)
            saved_age = int(predicted_age)

        # ---------------------------
        # 3. DRAW RESULTS
        # ---------------------------
        # Draw Box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Draw Emotion
        cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Draw Age
        cv2.putText(frame, f"Age: {saved_age}", (x, y + h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Facial Expression Recognition + Age', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()