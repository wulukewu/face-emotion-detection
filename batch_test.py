# %%
import os
import cv2
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm # Progress bar

# Import helper
import feature_extractor as fe

# CONFIG
TEST_DATA_PATH = "TestData"
IMG_SIZE = (64, 64)

# %%
print("Loading Models & Resources...")

resources = {"rf": {}, "cnn": {}}

# Load Face Detector
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load Random Forest
try:
    resources["rf"]["model"] = joblib.load("emotion_model.joblib")
    resources["rf"]["scaler"] = joblib.load("feature_scaler.joblib")
    label_map = joblib.load("label_map.joblib")
    
    # Invert map (0 -> "Happy")
    if isinstance(list(label_map.keys())[0], str):
        inv_map = {v: k for k, v in label_map.items()}
    else:
        inv_map = label_map
    resources["rf"]["inv_map"] = inv_map
    print("✅ Random Forest Loaded")
except Exception as e:
    print(f"❌ RF Error: {e}")

# Load CNN
try:
    if os.path.exists("emotion_model_cnn.h5"):
        resources["cnn"]["model"] = load_model("emotion_model_cnn.h5")
        resources["cnn"]["inv_map"] = inv_map # Share map
        print("✅ CNN Loaded")
    else:
        print("⚠️ CNN model file not found.")
except Exception as e:
    print(f"❌ CNN Error: {e}")

# %%
def get_face_roi(img_gray):
    """
    Simulates the App's detection logic.
    Returns: (cropped_face, found_bool)
    """
    # Use same settings as App: scale 1.1, neighbors 3
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    
    if len(faces) > 0:
        # Take the largest face
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        (x, y, w, h) = faces[0]
        return img_gray[y:y+h, x:x+w], True
    else:
        # Fallback: Return whole image
        return img_gray, False

def predict_rf(face_img):
    try:
        features = fe.preprocess_and_extract_features_single(face_img)
        features_scaled = resources["rf"]["scaler"].transform(features)
        pred_idx = resources["rf"]["model"].predict(features_scaled)[0]
        return resources["rf"]["inv_map"][pred_idx]
    except:
        return "Error"

def predict_cnn(face_img):
    try:
        img_resized = cv2.resize(face_img, IMG_SIZE)
        img_norm = img_resized.astype("float32") / 255.0
        img_input = img_norm.reshape(1, 64, 64, 1)
        probs = resources["cnn"]["model"].predict(img_input, verbose=0)[0]
        pred_idx = np.argmax(probs)
        return resources["cnn"]["inv_map"][pred_idx]
    except:
        return "Error"

# %%
def run_evaluation(use_opencv_cropping):
    """Runs a full evaluation pass, with an option to use OpenCV cropping."""
    print(f"\n--- Running Evaluation (OpenCV Cropping: {use_opencv_cropping}) ---")
    results = []
    
    # Verify path
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: Folder '{TEST_DATA_PATH}' does not exist.")
        return pd.DataFrame()

    # Get total count for progress bar
    total_files = sum([len(files) for r, d, files in os.walk(TEST_DATA_PATH)])
    
    with tqdm(total=total_files, desc=f"Mode: {'OpenCV' if use_opencv_cropping else 'Full Image'}") as pbar:
        for emotion_folder in os.listdir(TEST_DATA_PATH):
            folder_path = os.path.join(TEST_DATA_PATH, emotion_folder)
            if not os.path.isdir(folder_path): continue
            
            true_label = emotion_folder # The folder name is the truth
            
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                
                # 1. Read Image
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None: 
                    pbar.update(1)
                    continue
                
                face_roi = None
                face_found = "N/A"

                if use_opencv_cropping:
                    # 2. Pipeline: Detect Face
                    face_roi, face_found = get_face_roi(img)
                else:
                    # 2. Pipeline: Just resize the whole image
                    face_roi = cv2.resize(img, IMG_SIZE)
                
                # 3. Predict
                pred_rf = predict_rf(face_roi)
                pred_cnn = predict_cnn(face_roi)
                
                results.append({
                    "Image": img_file,
                    "True_Label": true_label,
                    "Face_Detected": face_found,
                    "Pred_RF": pred_rf,
                    "Pred_CNN": pred_cnn
                })
                pbar.update(1)

    # Convert to DataFrame
    df = pd.DataFrame(results)
    print("\nTesting Complete.")
    return df

def analyze_and_print_results(df, title):
    """Takes a results dataframe and prints all statistics and reports."""
    print(f"\n--- 📊 {title} ---")

    # Filter out any errors during prediction
    df_clean = df[(df["Pred_RF"] != "Error") & (df["Pred_CNN"] != "Error")].copy()
    if len(df_clean) == 0:
        print("No successful predictions to analyze.")
        return

    # Calculate Accuracies
    acc_rf = accuracy_score(df_clean["True_Label"], df_clean["Pred_RF"])
    acc_cnn = accuracy_score(df_clean["True_Label"], df_clean["Pred_CNN"])
    
    print(f"Total Images Tested: {len(df_clean)}")
    # Only show face detection rate if it's relevant
    if df_clean["Face_Detected"].dtype == 'bool':
        face_det_rate = df_clean["Face_Detected"].mean()
        print(f"Face Detection Success Rate: {face_det_rate:.2%}")

    print(f"Random Forest Accuracy: {acc_rf:.2%}")
    print(f"CNN Accuracy: {acc_cnn:.2%}")

    # --- Per-Emotion Accuracy (Classification Report) ---
    print("\n--- 🌲 Random Forest Report ---")
    print(classification_report(df_clean["True_Label"], df_clean["Pred_RF"]))

    print("\n--- 🧠 CNN Report ---")
    print(classification_report(df_clean["True_Label"], df_clean["Pred_CNN"]))

print(f"Starting Batch Test on '{TEST_DATA_PATH}'...")

# --- Run Both Evaluations ---
df_with_cv = run_evaluation(use_opencv_cropping=True)
df_without_cv = run_evaluation(use_opencv_cropping=False)

# --- Analyze and Print Results ---
if not df_with_cv.empty:
    analyze_and_print_results(df_with_cv, "Pipeline Statistics (With OpenCV Cropping)")

if not df_without_cv.empty:
    analyze_and_print_results(df_without_cv, "Pipeline Statistics (Without OpenCV Cropping / Full Image)")