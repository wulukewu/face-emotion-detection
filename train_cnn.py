# %%
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import random

# Deep Learning Imports (Keras/TensorFlow)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Config
DATA_PATH = "Data"
IMG_SIZE = (64, 64)
MODEL_PATH = "emotion_model_cnn.h5"

# Check if GPU is available (Optional, makes training faster)
print(f"TensorFlow Version: {tf.__version__}")
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"✅ GPU Detected: {gpu_devices}")
else:
    print("⚠️ No GPU detected. Training might be slower on CPU.")

# %%
images = []
labels = []
label_map = {}
label_idx = 0

print("Loading images...")
for emotion_folder in os.listdir(DATA_PATH):
    emotion_path = os.path.join(DATA_PATH, emotion_folder)
    if not os.path.isdir(emotion_path): continue
    
    if emotion_folder not in label_map:
        label_map[emotion_folder] = label_idx
        label_idx += 1
    
    current_label = label_map[emotion_folder]
    
    for img_file in os.listdir(emotion_path):
        full_path = os.path.join(emotion_path, img_file)
        # Read as Grayscale
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, IMG_SIZE)
            images.append(img_resized)
            labels.append(current_label)

# Convert to Numpy Arrays
X_raw = np.array(images)
y_raw = np.array(labels)
inv_label_map = {v: k for k, v in label_map.items()}

print(f"Loaded {len(X_raw)} images.")
print(f"Labels: {label_map}")

# %%
# 1. Normalization: 
# Neural Networks like numbers between 0 and 1. 
# Pixel values are 0-255, so we divide by 255.
X = X_raw / 255.0

# 2. Reshaping for Convolution:
# A standard image shape is (Height, Width).
# A CNN expects (Height, Width, Channels). 
# Since it is Grayscale, we have 1 channel. We must add that '1' explicitly.
X = X.reshape(-1, 64, 64, 1)

# 3. One-Hot Encoding Labels:
# Random Forest liked labels like 0, 1, 2.
# Neural Networks output probabilities (e.g., [0.1, 0.8, 0.1]).
# So we convert label '1' to [0, 1, 0].
y = to_categorical(y_raw, num_classes=len(label_map))

print("Preprocessing Complete.")
print(f"Input Shape: {X.shape} (Note the extra '1' at the end)")
print(f"Label Example (One-Hot): Label {y_raw[0]} becomes {y[0]}")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y_raw, random_state=42
)

# %%
model = Sequential([
    # Layer 1: Find basic edges/lines (32 filters)
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)), # Shrink the image (keep strongest features)

    # Layer 2: Find shapes/curves (64 filters)
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Layer 3: Find complex facial features (128 filters)
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Flatten: Turn the 3D cube of numbers into a flat list
    Flatten(),

    # Dense: Think about what the features mean
    Dense(128, activation='relu'),
    
    # Dropout: Close 50% of neurons randomly during training (prevents memorization)
    Dropout(0.5),

    # Output: Give probability for each emotion
    Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("CNN Model Created:")
model.summary()

# %%
print("\nStarting Training... (Grab a coffee ☕)")

history = model.fit(
    X_train, y_train,
    epochs=15,             # How many times to read the whole dataset
    batch_size=32,         # How many images to read at once
    validation_data=(X_test, y_test) # Check accuracy on test data after every epoch
)

print("Training Finished!")

# %%
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Is the model getting smarter?')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss (Error)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Is the error going down?')
plt.xlabel('Epochs')
plt.ylabel('Loss (Error)')
plt.legend()

plt.show()

# %%
# 1. Get predictions
y_pred_probs = model.predict(X_test, batch_size=32) # Returns probabilities [0.1, 0.9, ...]
y_pred_classes = np.argmax(y_pred_probs, axis=1) # Convert to [1]
y_true_classes = np.argmax(y_test, axis=1)       # Convert to [1]

# 2. Accuracy Score
test_acc = np.mean(y_pred_classes == y_true_classes)
print(f"\nFinal Test Accuracy: {test_acc:.2%}")

# 3. Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=inv_label_map.values(),
            yticklabels=inv_label_map.values())
plt.title("Confusion Matrix (CNN)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# %%
# Note: Keras models save as .h5 files, not .joblib (usually)

model.save(MODEL_PATH)

# We still need the label map for the App
joblib.dump(inv_label_map, "label_map.joblib")

print(f"Model saved as {MODEL_PATH}")
print("To use this in the app, you will need to modify app.py to load .h5 files!")