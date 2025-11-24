import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_model

img_size = (48, 48)
batch_size = 32
num_classes = 7

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen  = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Ayoub Changer path ici!!!!!!
train_dir = "C:/Users/stito/OneDrive/Desktop/Project ML/Data/archive/train/"
test_dir = "C:/Users/stito/OneDrive/Desktop/Project ML/Data/archive/test/"

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)


test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

model = build_model(input_shape=(48,48,1), num_classes=num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('models/emotion_model.h5', monitor='val_accuracy', save_best_only=True)
earlystop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=50,
    validation_data=test_generator,
    callbacks=[checkpoint, earlystop]
)