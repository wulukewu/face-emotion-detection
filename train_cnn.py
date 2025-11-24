import tensorflow as tf

import os
os.environ["OMP_NUM_THREADS"] = "1"
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Docker
import matplotlib.pyplot as plt
import joblib

# Keras / TensorFlow Imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# CONFIGURATION
DATA_PATH = "Data"
IMG_SIZE = (64, 64)
MODEL_PATH = "emotion_model_cnn.h5"

print(f"TensorFlow Version: {tf.__version__}")
print("Loading images for CNN...")

# --- Data Loading ---
images = []
labels = []
label_map = {}
label_idx = 0

# Ensure Data folder exists
if not os.path.exists(DATA_PATH):
    print(f"Error: {DATA_PATH} not found.")
    exit()

# Loop through folders (Angry, Happy, etc.)
for emotion_folder in os.listdir(DATA_PATH):
    emotion_path = os.path.join(DATA_PATH, emotion_folder)
    if not os.path.isdir(emotion_path): continue
    
    # Assign a number to the emotion (Happy=0, Sad=1...)
    if emotion_folder not in label_map:
        label_map[emotion_folder] = label_idx
        label_idx += 1
    
    current_label = label_map[emotion_folder]
    
    # Loop through images
    for img_file in os.listdir(emotion_path):
        full_path = os.path.join(emotion_path, img_file)
        # Read as Grayscale
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, IMG_SIZE)
            images.append(img_resized)
            labels.append(current_label)

# Convert to Numpy Arrays (The format ML models need)
X = np.array(images)
y = np.array(labels)

# --- Deep Learning Preprocessing (Key Steps!) ---

# A. Normalization: Scale pixel values from 0-255 to 0-1
X = X / 255.0

# B. Reshape: CNN needs input shape (height, width, channels)
# Grayscale images have 1 channel, so reshape to (64, 64, 1)
X = X.reshape(-1, 64, 64, 1)

# C. One-Hot Encoding: Convert labels 0, 1, 2 to [1,0,0], [0,1,0]...
y_onehot = to_categorical(y, num_classes=len(label_map))

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, stratify=y, random_state=42
)

print(f"Dataset Shape: {X.shape}")
print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")

# --- Build Keras CNN Model ---
# Sequential: Layers are stacked one after another
model = Sequential([
    # First Convolutional Layer: Extract basic features (lines, edges)
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),  # Reduce image size, keep important features

    # Second Convolutional Layer: Extract complex features (eyes, mouth shape)
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Third Convolutional Layer
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Flatten Layer: Convert 3D feature maps to 1D array for Dense layers
    Flatten(),

    # Fully Connected Layer (Dense): Perform classification
    Dense(128, activation='relu'),
    
    # Dropout: Randomly drop 50% of neurons to prevent overfitting
    Dropout(0.5),

    # Output Layer: Output probability for each emotion (Softmax sums to 1)
    Dense(len(label_map), activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()  # Display model architecture

# --- Start Training ---
print("\nStarting Training... (This may take a few minutes)")
history = model.fit(
    X_train, y_train,
    epochs=15,             # Train for 15 rounds
    batch_size=32,         # Process 32 images at a time
    validation_data=(X_test, y_test)  # Validate with test set
)

# --- Evaluation and Save ---
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nFinal Test Accuracy: {test_acc:.2%}")

# Save model (Keras format)
model.save(MODEL_PATH)

# Save label mapping (App needs to know 0=Angry or Happy)
inv_label_map = {v: k for k, v in label_map.items()}
joblib.dump(inv_label_map, "label_map.joblib")

print(f"Model saved to {MODEL_PATH}")

# --- Plot Training Curves ---
plt.figure(figsize=(12, 5))

# Left plot: Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Model Accuracy (CNN)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Right plot: Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Save the figure instead of showing it (for Docker compatibility)
plt.savefig('training_history_cnn.png')
print("Training history plot saved to training_history_cnn.png")