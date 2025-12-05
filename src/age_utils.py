import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

"""
    hadi kat9ad wa7ed dataframe mn train data li 3ndna 7it smit kola image dayra b7al hka 'age_.._....jpg' y3ni kay extracti lage
    ou kaystocki dakchi blpath fcolone ou age fcolone 
"""
def get_data_dataframe(path, num_classes):
    file_paths = []
    ages = []
    for file_name in os.listdir(path):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                parts = file_name.split('_')
                if len(parts) < 1: continue

                age = int(parts[0])
                if age < 0 or age >= num_classes: continue

                file_paths.append(os.path.join(path, file_name))
                ages.append(age)
            except:
                continue

    df = pd.DataFrame({'image_path': file_paths, 'age': ages})
    return df

"""
    hadi kat9ad l image ou katzid liha wa7ed lpadding bach tresiza mzyan machi ghir tjbed ou safi
    y3ni ila kant limage sghira 3la (128, 128) katzid liha jnab blk7el bach tkberha b7al (frame) ou l3ekss ila kant kbira
"""
def resize_with_padding(image, target_size=(128, 128)):
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized_image = cv2.resize(image, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
    return canvas

#hadi hir kat3iit 3la lfct li lfo9 ou karje3 l image 3la chkel 'float32'
def custom_preprocessing(img, img_size = 128):
    img_padded = resize_with_padding(img.astype('uint8'), (img_size, img_size))
    return preprocess_input(img_padded.astype('float32'))


#hadi zedtha 3la 7ssab dak ta3 i3tik lage mn chi tsswira tuploadiha
def preprocess_face_image(image_source):
    img = None

    if isinstance(image_source, str):
        if os.path.exists(image_source):
            img = cv2.imread(image_source)
        else:
            print(f"Error: Path not found -> {image_source}")
            return None

    elif isinstance(image_source, np.ndarray):
        img = image_source
    if img is None:
        print("Error: Failed to load image data.")
        return None

    if img.size == 0:
        print("Error: Image has 0 size.")
        return None

    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_padded = resize_with_padding(img_rgb, (128, 128))

        img_batch = np.expand_dims(img_padded, axis=0)

        img_preprocessed = preprocess_input(img_batch.astype('float32'))

        return img_preprocessed

    except Exception as e:
        print(f"Preprocessing Error: {e}")
        return None


def age_from_distribution(pred):
    """
    Converts the softmax distribution (117 probabilities)
    into a single weighted average age.
    """
    ages = np.arange(117)
    return np.sum(pred * ages)


