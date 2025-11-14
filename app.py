# filename: app.py

import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

# Import our custom feature extractor
import feature_extractor as fe

IMG_SIZE = (64, 64)

# --- 1. Load Model and Preprocessors ---
try:
    model = joblib.load("emotion_model.joblib")
    scaler = joblib.load("feature_scaler.joblib")
    label_map = joblib.load("label_map.joblib")
    print("Model and supporting files loaded successfully.")
except FileNotFoundError:
    st.error("Error: Could not find model files.")
    st.error("Please run 'python train_model.py' first to generate the model files.")
    st.stop()
    
# --- 2. Streamlit Interface ---
st.title("Facial Emotion Recognition Demo 📸")
st.write(f"Based on LBP + HOG Feature Engineering & Random Forest Model")
st.write(f"Recognizes: {list(label_map.values())}")

uploaded_file = st.file_uploader("Upload a facial image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display the uploaded image
    image_pil = Image.open(uploaded_file)
    # FIX: Use use_container_width instead of use_column_width
    st.image(image_pil, caption="Uploaded Image", use_container_width=True)
    
    # 2. Convert to Grayscale directly
    img_cv_pil = np.array(image_pil)
    
    # Convert PIL image (RGB or RGBA) to Grayscale
    if len(img_cv_pil.shape) == 2:
        # Image is already grayscale
        img_gray = img_cv_pil
    elif img_cv_pil.shape[2] == 4:
        # Image is RGBA
        img_gray = cv2.cvtColor(img_cv_pil, cv2.COLOR_RGBA2GRAY)
    elif img_cv_pil.shape[2] == 3:
        # Image is RGB
        img_gray = cv2.cvtColor(img_cv_pil, cv2.COLOR_RGB2GRAY)

    st.subheader("Processing...")
    
    try:
        # 3. Extract features using the imported function
        features = fe.preprocess_and_extract_features_single(img_gray, img_size=IMG_SIZE)
        
        # 4. Standardize (using the loaded scaler)
        features_scaled = scaler.transform(features)
        
        # 5. Predict
        prediction_idx = model.predict(features_scaled)[0]
        prediction_label = label_map[prediction_idx]
        
        # 6. Get prediction probabilities
        probabilities = model.predict_proba(features_scaled)[0]
        
        st.subheader("Prediction Result 🎉")
        st.metric(label="Predicted Emotion", value=prediction_label)
        
        st.subheader("Prediction Probabilities")
        # Display probabilities in a nice way
        prob_data = {label_map[i]: prob for i, prob in enumerate(probabilities)}
        st.dataframe(prob_data, use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("This might be because the image is unusual. Please try another.")