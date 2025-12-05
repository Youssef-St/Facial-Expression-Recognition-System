import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import drive
# hadi li flte7t ghir 7iidha ila mchiti l google collab
from age_utils import get_data_dataframe

"""
Hadchi kolo runnih how ou age_utils fgoogle collab ha lien colab.research.google.com 
ou sir l Runtime > Change runtime type > Select T4 GPU 
b7al jupyter notebook 7et fcell lwla age_utils.py ou ftanya had lcode li fhad lfile ou matnsa tuploadi train 
data f drive
"""

drive.mount('/content/drive')

# hada 7it resources 3ndi chwia na9ssin trainito fcollab ta3 google
# ghatzid dak dossier Data/age_Data/UTKFace (howa fach kayna trainning data) ldrive ta3ek 7et lpath hna :
DATA_PATH = "/content/drive/MyDrive/utkFace/UTKFace"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 40

# --- 1. PREPARE DATA ---
print("Scanning files...")
df = get_data_dataframe(DATA_PATH)
print(f"Found {len(df)} images.")

# Split into Train/Val/Test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)  # Create a validation set

# --- 2. DATA GENERATORS (Handles loading & augmentation) ---
# MobileNetV2 expects inputs [-1, 1], so we use its preprocess function
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

# Flow from dataframe
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path',
    y_col='age',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='raw'  # 'raw' for regression (returns the actual age number)
)

val_generator = test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='image_path',
    y_col='age',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='raw'
)


# --- 3. BUILD MODEL (Transfer Learning) ---
def build_mobilenet_model(input_shape):
    # Load pre-trained MobileNetV2, exclude the top classification layer
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze the base model initially (optional, but good for stability)
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(1, activation='linear', name='age_output')(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mae',  # Mean Absolute Error is best for age
        metrics=['mae']
    )
    return model


model = build_mobilenet_model((IMG_SIZE, IMG_SIZE, 3))
model.summary()

# --- 4. CALLBACKS ---
checkpoint = ModelCheckpoint(
    '/content/drive/MyDrive/My_Projects/Models/age_model_best.keras',  # Save to Drive
    monitor='val_mae',
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor='val_mae',
    patience=10,
    restore_best_weights=True
)

# Reduce LR if stuck (Helps get that error down further)
reduce_lr = ReduceLROnPlateau(
    monitor='val_mae',
    factor=0.2,
    patience=4,
    min_lr=1e-6
)

# --- 5. TRAIN ---
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[checkpoint, earlystop, reduce_lr]
)

# --- 6. OPTIONAL: FINE TUNING ---
# Unfreeze the last few layers of the base model for better accuracy
print("Fine-tuning...")
base_model = model.layers[1]
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='mae',
    metrics=['mae']
)

history_fine = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=[checkpoint, earlystop, reduce_lr]
)

save_dir = "/content/drive/MyDrive/My_Projects/Models"

# Create the directory if it doesn't exist (prevents "File not found" errors)
os.makedirs(save_dir, exist_ok=True)

# Save the model
model.save(os.path.join(save_dir, "age_model_final_v2.keras"))
print(f"Model saved to Google Drive at: {save_dir}")