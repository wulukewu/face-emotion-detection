# %%
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import random

# Scikit-Learn for ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import the helper file
import feature_extractor as fe

# CONFIGURATION
DATA_PATH = "Data"
IMG_SIZE = (64, 64)

print("Libraries loaded successfully.")

# %%
images = []
labels = []
label_map = {}
label_idx = 0

print(f"Loading images from {DATA_PATH}...")

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
X_raw = np.array(images)
y = np.array(labels)
inv_label_map = {v: k for k, v in label_map.items()}

print(f"Loaded {len(X_raw)} images.")
print(f"Image Shape: {X_raw.shape}") # Should be (Num_Images, 64, 64)
print(f"Labels: {label_map}")

# %%
# Create a simple dataframe just for plotting
df_plot = pd.DataFrame({'Label': [inv_label_map[i] for i in y]})

plt.figure(figsize=(8, 5))
sns.countplot(data=df_plot, x='Label', palette='viridis')
plt.title("How many images do we have per emotion?")
plt.ylabel("Count")
plt.show()

# %%
# Pick a random image
rand_idx = random.randint(0, len(X_raw) - 1)
sample_img = X_raw[rand_idx]
sample_label = inv_label_map[y[rand_idx]]

# Get HOG visualization (Edges/Shapes)
_, hog_vis = fe._extract_hog_single(sample_img, visualize=True)

# Get LBP Histogram (Texture)
lbp_hist = fe._extract_lbp_single(sample_img)

# Plot them
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Raw Image
axes[0].imshow(sample_img, cmap='gray')
axes[0].set_title(f"1. Raw Input (Gray)\nEmotion: {sample_label}")
axes[0].axis('off')

# 2. HOG Features
axes[1].imshow(hog_vis, cmap='gray')
axes[1].set_title("2. HOG Features\n(Model sees: Edges/Shape)")
axes[1].axis('off')

# 3. LBP Features
axes[2].bar(range(len(lbp_hist)), lbp_hist)
axes[2].set_title("3. LBP Histogram\n(Model sees: Texture Info)")

plt.show()

# %%
print("Starting Feature Extraction... (This may take a moment)")
start_t = time.time()

# 1. Extract Pixel Features (Just flattening the image)
X_pixels = fe.extract_pixel_features(X_raw)

# 2. Extract LBP (Texture)
X_lbp = fe.extract_lbp_features(X_raw)

# 3. Extract HOG (Shape)
X_hog = fe.extract_hog_features(X_raw)

# 4. Combine LBP + HOG (The Best Version)
X_combined = np.concatenate([X_lbp, X_hog], axis=1)

print(f"\nDone! Total time: {time.time() - start_t:.2f}s")
print(f"Combined Feature Shape: {X_combined.shape}")

# %%
import warnings
# Silence the scary math warnings
warnings.filterwarnings('ignore') 

print("Running Comparison Experiment...")

feature_sets = {
    "Raw Pixels": X_pixels,
    "LBP Only": X_lbp,
    "HOG Only": X_hog,
    "Combined (HOG+LBP)": X_combined
}

results = []

for name, X_data in feature_sets.items():
    # Split Data 80% Train / 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, stratify=y)
    
    # Scale Data (Important for ML!)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    # Quick Train
    model = LogisticRegression(max_iter=500)
    t0 = time.time()
    model.fit(X_train_sc, y_train)
    train_time = time.time() - t0
    
    # Score
    acc = accuracy_score(y_test, model.predict(X_test_sc))
    results.append({"Method": name, "Accuracy": acc, "Time": train_time})
    print(f"Finished {name} -> Accuracy: {acc:.2%}")

# Show Table
df_results = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
print("\n--- FINAL RESULTS ---")
print(df_results)

# Reset warnings
warnings.filterwarnings('default')

# %%
print("Training Final Random Forest Model...")

# 1. Use the Best Data
X_final = X_combined
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)

# 2. Scale
final_scaler = StandardScaler()
X_train_scaled = final_scaler.fit_transform(X_train)
X_test_scaled = final_scaler.transform(X_test)

# 3. Train Random Forest (Stronger than Logistic Regression)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 4. Evaluate
y_pred = rf_model.predict(X_test_scaled)
print(f"Final Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred, target_names=inv_label_map.values()))

# 5. Save Files
joblib.dump(rf_model, "emotion_model.joblib")
joblib.dump(final_scaler, "feature_scaler.joblib")
joblib.dump(inv_label_map, "label_map.joblib")
print("Model saved! You can now run 'streamlit run app.py'")
