import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import numpy as np

# Constants
IMG_SIZE = 128
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE
DATA_DIR = pathlib.Path("/content/SignLanguageTranslator/datasets/asl_alphabet_train")

# Step 1.1: Map folder names to label indices
label_names = sorted([item.name for item in DATA_DIR.glob("*") if item.is_dir()])
label_to_index = {name: idx for idx, name in enumerate(label_names)}
NUM_CLASSES = len(label_names)
print("Labels:", label_to_index)

# Step 1.2: Get image paths and labels
all_image_paths = list(DATA_DIR.glob("*/*.jpg")) + list(DATA_DIR.glob("*/*.png"))
all_image_paths = [str(p) for p in all_image_paths]
all_image_labels = [label_to_index[pathlib.Path(p).parent.name] for p in all_image_paths]

# Step 1.3: Shuffle and split
combined = list(zip(all_image_paths, all_image_labels))
np.random.shuffle(combined)
split_index = int(0.8 * len(combined))
train_data, val_data = combined[:split_index], combined[split_index:]

train_paths, train_labels = zip(*train_data)
val_paths, val_labels = zip(*val_data)

# Step 1.4: Preprocessing function
def process_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0
    return img, tf.one_hot(label, NUM_CLASSES)

# Step 1.5: Build tf.data.Dataset
train_ds = tf.data.Dataset.from_tensor_slices((list(train_paths), list(train_labels)))
val_ds = tf.data.Dataset.from_tensor_slices((list(val_paths), list(val_labels)))

train_ds = train_ds.map(process_image, num_parallel_calls=AUTOTUNE).shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_ds = val_ds.map(process_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

for images, labels in train_ds.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

from tensorflow.keras import layers

# Step 2.1: Define augmentation layer
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),         # ±~15°
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(0.1, 0.1)
])

# Step 2.2: New train image processing function with augment option
def process_image_aug(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    img = data_augmentation(img)
    return img, tf.one_hot(label, NUM_CLASSES)

# Replace train_ds with augmented version
train_ds = tf.data.Dataset.from_tensor_slices((list(train_paths), list(train_labels)))
val_ds = tf.data.Dataset.from_tensor_slices((list(val_paths), list(val_labels)))

train_ds = train_ds.map(process_image_aug, num_parallel_calls=AUTOTUNE).shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_ds = val_ds.map(process_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Preview 8 augmented training samples
plt.figure(figsize=(12, 6))
for images, labels in train_ds.take(1):
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].numpy())
        label_index = tf.argmax(labels[i]).numpy()
        plt.title(label_names[label_index])
        plt.axis("off")
plt.tight_layout()
plt.show()

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load ResNet50 base (frozen)
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 128, 3)
)
base_model.trainable = False  # freeze for phase 1

# Custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=outputs)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Compile for phase 1
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks_phase1 = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ModelCheckpoint("resnet50_asl_phase1.keras", save_best_only=True, monitor='val_accuracy'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# Train Phase 1
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks_phase1
)

# Unfreeze last ~50 layers
for layer in model.layers[-50:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks for fine-tuning
callbacks_phase2 = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ModelCheckpoint("resnet50_asl_finetuned.keras", save_best_only=True, monitor='val_accuracy'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# Train Phase 2
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=callbacks_phase2
)

# Continue Phase 2 fine-tuning for more epochs
history3 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=35,  # Continue from epoch 20 → up to 35
    initial_epoch=20,  # avoids restarting
    callbacks=callbacks_phase2
)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

callbacks_resnet_phase4 = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ModelCheckpoint("resnet50_asl_phase4.keras", save_best_only=True, monitor='val_accuracy'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

history_resnet4 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks_resnet_phase4
)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

# Recompile with lower LR
model.compile(
    optimizer=Adam(learning_rate=1e-6),
    loss=CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)

# Callbacks for safe training
callbacks_resnet_phase5 = [
    EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    ModelCheckpoint("resnet50_asl_phase5.keras", save_best_only=True, monitor='val_accuracy'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

# Train
history_resnet5 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks_resnet_phase5
)






