import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import our custom feature extractors
import feature_extractor as fe

# --- 1. Configuration ---
DATA_PATH = "Data" 
IMG_SIZE = (64, 64)

# --- 2. Data Loading & EDA ---

def load_data(path, img_size):
    """
    Loads images and labels from structured folders.
    """
    images = []
    labels = []
    label_map = {}
    label_idx = 0
    
    print(f"Loading data from {path}...")
    
    for emotion_folder in os.listdir(path):
        emotion_path = os.path.join(path, emotion_folder)
        
        if not os.path.isdir(emotion_path):
            continue
            
        if emotion_folder not in label_map:
            label_map[emotion_folder] = label_idx
            label_idx += 1
            
        label = label_map[emotion_folder]
        
        for img_file in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Could not read {img_path}")
                    continue
                img_resized = cv2.resize(img, img_size)
                images.append(img_resized)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
    print(f"Data loading complete. Total {len(images)} images.")
    inv_label_map = {v: k for k, v in label_map.items()}
    return np.array(images), np.array(labels), inv_label_map

def plot_data_distribution(labels, inv_label_map):
    """
    Visualize the distribution of samples per class.
    """
    label_counts = Counter(labels)
    label_names = [inv_label_map[i] for i in label_counts.keys()]
    counts = list(label_counts.values())
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=label_names, y=counts)
    plt.title("Sample Distribution per Emotion Class")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.savefig("data_distribution.png") # Save the plot
    print("Data distribution plot saved as data_distribution.png")
    # plt.show() # Uncomment if running in an interactive environment

# --- 3. Main Training & Evaluation Pipeline ---

def main():
    # --- Part 1: Load and Explore Data ---
    X_raw_images, y_labels, inv_label_map = load_data(DATA_PATH, IMG_SIZE)
    if X_raw_images.size == 0:
        print(f"No images loaded. Check DATA_PATH: {DATA_PATH}")
        return
        
    print(f"Label map created: {inv_label_map}")
    plot_data_distribution(y_labels, inv_label_map)

    # --- Part 2: Feature Extraction ---
    print("\n--- Extracting Features ---")
    X_pixels = fe.extract_pixel_features(X_raw_images)
    print(f"Pixel features shape: {X_pixels.shape}")
    
    X_lbp = fe.extract_lbp_features(X_raw_images)
    print(f"LBP features shape: {X_lbp.shape}")
    
    X_hog = fe.extract_hog_features(X_raw_images)
    print(f"HOG features shape: {X_hog.shape}")
    
    X_combined = np.concatenate([X_lbp, X_hog], axis=1)
    print(f"LBP + HOG combined features shape: {X_combined.shape}")
    
    feature_sets = {
        "Pixels": X_pixels,
        "LBP": X_lbp,
        "HOG": X_hog,
        "LBP + HOG": X_combined
    }
    
    # --- Part 3: Model Comparison ---
    print("\n--- Starting Model Iterative Comparison (using LogisticRegression) ---")
    results = {}
    model_lr = LogisticRegression(max_iter=1000, random_state=42)

    for name, X in feature_sets.items():
        print(f"\nTraining model with: {name} features (Shape: {X.shape})")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_labels, test_size=0.25, random_state=42, stratify=y_labels
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        start_time = time.time()
        model_lr.fit(X_train_scaled, y_train)
        end_time = time.time()
        
        y_pred = model_lr.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        results[name] = {
            "accuracy": acc,
            "train_time": end_time - start_time,
            "n_features": X.shape[1]
        }
        print(f"Done. Accuracy: {acc:.4f}, Training time: {end_time - start_time:.2f}s")

    print("\n--- Feature Engineering Impact Summary ---")
    results_df = pd.DataFrame(results).T.sort_values(by="accuracy", ascending=False)
    print(results_df)
    
    # --- Part 4: Train and Save Best Model ---
    print("\n--- Training and Saving Best Model (RandomForest w/ LBP+HOG) ---")
    
    # Use the best features (LBP + HOG)
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_labels, test_size=0.25, random_state=42, stratify=y_labels
    )
    
    # Create and fit the final scaler *on the training data*
    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train)
    X_test_scaled = final_scaler.transform(X_test)
    
    # Create and fit the final model
    final_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    final_model.fit(X_train_scaled, y_train)
    
    # Evaluate final model
    y_pred_final = final_model.predict(X_test_scaled)
    final_acc = accuracy_score(y_test, y_pred_final)
    print(f"Final Model Accuracy: {final_acc:.4f}")
    
    print("Final Model Classification Report:")
    print(classification_report(y_test, y_pred_final, target_names=inv_label_map.values()))
    
    # Save the model, scaler, and label map
    joblib.dump(final_model, "emotion_model.joblib")
    joblib.dump(final_scaler, "feature_scaler.joblib")
    joblib.dump(inv_label_map, "label_map.joblib")
    
    print("\nModel, scaler, and label map saved successfully!")
    print("You are now ready to run 'streamlit run app.py'")

if __name__ == "__main__":
    main()