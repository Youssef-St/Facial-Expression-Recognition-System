import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import drive
#hadi li flte7t ghir 7iidha ila mchiti l google collab
from age_utils import get_data_dataframe,custom_preprocessing

"""
7it resources 3ndi chwia na9ssin trainito fcollab ta3 google
Hadchi kolo runnih how ou age_utils fgoogle collab ha lien colab.research.google.com 
ou sir l Runtime > Change runtime type > Select T4 GPU 
b7al jupyter notebook 7et fcell lwla age_utils.py ou ftanya had lcode li fhad lfile ou matnsa tuploadi train 
data f drive 
ila mabghititch t trainih rah ghaykon fdak dossier Models fih lmodel mojoud khdem bhada age_model_best.keras
rah ghatl9ah zedto f inference.py y3ni ghir runi inference.py ou safi ghatl9ah tma
"""


drive.mount('/content/drive')

# --- CONFIG ---
#ghatzid dak dossier Data/age_Data/UTKFace (howa fach kayna trainning data) ldrive ta3ek 7et lpath hna :
DATA_PATH = "/content/drive/MyDrive/utkFace/UTKFace"
SAVE_DIR = "/content/drive/MyDrive/My_Projects/Models"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 40
NUM_CLASSES = 117
ALL_AGES = [str(i) for i in range(NUM_CLASSES)]

print("Scanning files...")
df = get_data_dataframe(DATA_PATH, NUM_CLASSES)
df['age'] = df['age'].astype(str)
print(f"Found {len(df)} images.")

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

train_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing,
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

val_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path',
    y_col='age',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=ALL_AGES,
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='image_path',
    y_col='age',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=ALL_AGES,
    shuffle=False
)


# --- 4. BUILD MODEL ---
def build_mobilenet_model(input_shape):
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)

    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='age_output')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


model = build_mobilenet_model((IMG_SIZE, IMG_SIZE, 3))

os.makedirs(SAVE_DIR, exist_ok=True)

checkpoint = ModelCheckpoint(
    os.path.join(SAVE_DIR, 'age_model_best.keras'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

print("Starting Training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[checkpoint, earlystop, reduce_lr]
)

# --- 6. FINE TUNING (Optional) ---
print("Fine-tuning...")
base_model = model.layers[1]
base_model.trainable = True

# Freeze first 100 layers, train the rest
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=[checkpoint, earlystop, reduce_lr]
)

model.save(os.path.join(SAVE_DIR, "age_model_final_v2.keras"))
print("Done.")