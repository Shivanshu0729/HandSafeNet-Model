import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ============================================================
# PATH CONFIGURATION
# ============================================================
# Get base directory of project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Path to dataset folders
HAND_DIR = os.path.join(DATASET_DIR, "hand")
NOHAND_DIR = os.path.join(DATASET_DIR, "no-hand")

IMG_SIZE = 128  # All images resized to 128x128

print("HAND DIR:", HAND_DIR)
print("NO HAND DIR:", NOHAND_DIR)

# ============================================================
# DATA LOADING FUNCTION
# ============================================================
def load_images(folder, label):
    """
    Loads all images from a folder, preprocesses them,
    and assigns the given label.
    """
    X, y = [], []
    for file in os.listdir(folder):
        if not file.lower().endswith((".jpg", ".png")):
            continue

        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is None:
            continue

        # Preprocessing steps
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X.append(img)
        y.append(label)

    return np.array(X), np.array(y)

# ============================================================
# LOAD DATASET
# ============================================================
print("Loading hand images...")
X_hand, y_hand = load_images(HAND_DIR, 1)

print("Loading no-hand images...")
X_no, y_no = load_images(NOHAND_DIR, 0)

# Merge dataset
X = np.concatenate([X_hand, X_no], axis=0)
y = np.concatenate([y_hand, y_no], axis=0)

print("Total images:", len(X))

# Normalize pixel values
X = X.astype("float32") / 255.0

# ============================================================
# SPLIT DATA INTO TRAIN / VALIDATION / TEST
# ============================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

# ============================================================
# DATA AUGMENTATION (Improves model generalization)
# ============================================================
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.25,
    height_shift_range=0.25,
    zoom_range=0.35,
    shear_range=0.25,
    brightness_range=[0.4, 1.6],
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# ============================================================
# CNN MODEL ARCHITECTURE
# ============================================================
model = Sequential([
    # Block 1
    Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Block 2
    Conv2D(64, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Block 3
    Conv2D(128, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Block 4
    Conv2D(256, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Fully connected layers
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),

    # Output layer
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0007),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# TRAINING CONFIGURATION
# ============================================================
save_path = os.path.join(BASE_DIR, "model", "hand_model.h5")
os.makedirs(os.path.dirname(save_path), exist_ok=True)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True, verbose=1)
]

print("Training model...")

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=45,
    callbacks=callbacks
)

# ============================================================
# MODEL EVALUATION
# ============================================================
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)
print("Model saved at:", save_path)
