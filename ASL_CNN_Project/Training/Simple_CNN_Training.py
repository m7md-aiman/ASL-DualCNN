import os, glob, random, pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from PIL import Image, ImageEnhance

# === Paths ===
BASE_DIR = "/content/Train_Alphabet_128"
IMG_DIR = pathlib.Path(f"{BASE_DIR}/images")

# === Label setup ===
label_names = sorted([d.name for d in IMG_DIR.glob("*") if d.is_dir()])
label_to_index = {name: idx for idx, name in enumerate(label_names)}
NUM_CLASSES = len(label_names)

img_paths = sorted(glob.glob(f"{IMG_DIR}/*/*.png"))
labels = [label_to_index[pathlib.Path(p).parent.name] for p in img_paths]

# === Shuffle & Split ===
combined = list(zip(img_paths, labels))
random.shuffle(combined)
split = int(0.8 * len(combined))
train_data, val_data = combined[:split], combined[split:]
train_imgs, train_labels = zip(*train_data)
val_imgs, val_labels = zip(*val_data)


# === Image Preprocessing ===
def preprocess_image_pil(img_path):
    img_path = img_path.decode()
    img = Image.open(img_path).convert("L")
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = img.resize((50, 50))
    return np.array(img, dtype=np.float32) / 255.0


# === Augmentation ===
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


def process_image_only(img_path, label, is_training=False):
    image = tf.numpy_function(preprocess_image_pil, [img_path], tf.float32)
    image.set_shape([50, 50])
    image = tf.expand_dims(image, -1)
    if is_training:
        image = data_augmentation(image)
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


# === Dataset Creation ===
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64

train_ds = tf.data.Dataset.from_tensor_slices((list(train_imgs), list(train_labels)))
val_ds = tf.data.Dataset.from_tensor_slices((list(val_imgs), list(val_labels)))

train_ds = train_ds.map(lambda x, y: process_image_only(x, y, is_training=True), num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: process_image_only(x, y, is_training=False), num_parallel_calls=AUTOTUNE)

train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


# === Build Simple 3-CNN Layer Model ===
def build_simple_image_model(img_shape=(50, 50, 1), num_classes=NUM_CLASSES):
    inputs = Input(shape=img_shape)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)


# === Compile & Train ===
model = build_simple_image_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint("image_model_best.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
]

history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks)
