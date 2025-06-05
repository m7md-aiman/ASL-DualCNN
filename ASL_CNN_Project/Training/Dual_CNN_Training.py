import os, glob, random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import pathlib

BASE_DIR = "/content/Train_Alphabet_128"
IMG_DIR = pathlib.Path(f"{BASE_DIR}/images")
LMK_DIR = pathlib.Path(f"{BASE_DIR}/landmarks")

label_names = sorted([d.name for d in IMG_DIR.glob("*") if d.is_dir()])
label_to_index = {name: idx for idx, name in enumerate(label_names)}
NUM_CLASSES = len(label_names)

img_paths = sorted(glob.glob(f"{IMG_DIR}/*/*.png"))
lmk_paths = [p.replace("images", "landmarks").replace(".png", ".npy") for p in img_paths]
labels = [label_to_index[pathlib.Path(p).parent.name] for p in img_paths]

# Shuffle and split
combined = list(zip(img_paths, lmk_paths, labels))
random.shuffle(combined)
split = int(0.8 * len(combined))
train_data, val_data = combined[:split], combined[split:]
train_imgs, train_lmks, train_labels = zip(*train_data)
val_imgs, val_lmks, val_labels = zip(*val_data)

def preprocess_image_pil(img_path):
    img_path = img_path.decode()  # ✅ decode directly
    img = Image.open(img_path).convert("L")
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = img.resize((50, 50))
    return np.array(img, dtype=np.float32) / 255.0


def preprocess_landmarks_np(lm_path):
    lm_path = lm_path.decode()
    lm = np.load(lm_path).astype(np.float32)
    lm -= np.mean(lm, axis=0)
    return lm[..., np.newaxis]


import tensorflow as tf

def salt_and_pepper_noise(image, amount=0.05):
    shape = tf.shape(image)
    total_pixels = tf.cast(shape[0] * shape[1], tf.float32)
    num_noisy = tf.cast(total_pixels * amount, tf.int32)

    # Generate random coordinates
    coords = tf.stack([
        tf.random.uniform([num_noisy], 0, shape[0], dtype=tf.int32),
        tf.random.uniform([num_noisy], 0, shape[1], dtype=tf.int32)
    ], axis=1)




data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.Resizing(60, 60),
    layers.RandomCrop(50, 50),
    layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.2)),
    layers.Lambda(lambda x: tf.clip_by_value(x + tf.random.uniform(tf.shape(x), -0.05, 0.05), 0.0, 1.0)),
])
def process_dual_input(img_path, lm_path, label, is_training=False):
    image = tf.numpy_function(preprocess_image_pil, [img_path], tf.float32)
    image.set_shape([50, 50])
    image = tf.expand_dims(image, -1)
    if is_training:
        image = data_augmentation(image)

    lm = tf.numpy_function(preprocess_landmarks_np, [lm_path], tf.float32)
    lm.set_shape([21, 3, 1])
    label = tf.one_hot(label, NUM_CLASSES)
    return (image, lm), label

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64

train_ds = tf.data.Dataset.from_tensor_slices((list(train_imgs), list(train_lmks), list(train_labels)))
val_ds = tf.data.Dataset.from_tensor_slices((list(val_imgs), list(val_lmks), list(val_labels)))

train_ds = train_ds.map(lambda x, y, z: process_dual_input(x, y, z, is_training=True), num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(lambda x, y, z: process_dual_input(x, y, z, is_training=False), num_parallel_calls=AUTOTUNE)

train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def build_paper_style_dual_model(img_shape=(50, 50, 1), num_classes=NUM_CLASSES):
    img_input = Input(shape=img_shape, name="image_input")
    x = layers.Conv2D(64, 3, padding="same", activation='relu')(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(512, 3, padding="same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(64, 3, padding="same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(32, 3, padding="same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)

    lm_input = Input(shape=(21, 3, 1), name="landmark_input")
    y = layers.Conv2D(50, 3, padding="same", activation="relu")(lm_input)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2D(25, 3, padding="same", activation="relu")(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)
    y = layers.Flatten()(y)

    z = layers.Concatenate()([x, y])
    z = layers.Dense(512, activation='relu')(z)
    z = layers.Dropout(0.5)(z)
    output = layers.Dense(num_classes, activation='softmax')(z)

    return models.Model(inputs=[img_input, lm_input], outputs=output)



model = build_lightweight_dual_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
]

history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks)



