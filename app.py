import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

import feature_extractor as fe

IMG_SIZE = (64, 64)

# --- 1. Load Model, Preprocessors, and Face Detector ---
try:
    model = joblib.load("emotion_model.joblib")
    scaler = joblib.load("feature_scaler.joblib")
    label_map = joblib.load("label_map.joblib")
    print("Model loaded.")
except FileNotFoundError:
    st.error("Error: Model files not found.")
    st.stop()

# Load OpenCV's pre-trained Haar Cascade for face detection
try:
    # cv2.data.haarcascades points to the folder where opencv keeps xml files
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise IOError("Failed to load Haar Cascade xml file.")
except Exception as e:
    st.error(f"Error loading Face Detector: {e}")
    st.stop()

# --- 2. Define the Real-time Video Processor ---

class EmotionTransformer(VideoTransformerBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 1. Get the frame from the webcam
        img_bgr = frame.to_ndarray(format="bgr24")
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 2. Detect faces in the frame
        # scaleFactor=1.1, minNeighbors=5 are standard parameters
        faces = face_cascade.detectMultiScale(img_gray, 1.1, 5)
        
        # 3. Loop through each detected face
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            try:
                # --- CRITICAL STEP ---
                # Crop the face region (Region of Interest - ROI)
                face_roi = img_gray[y:y+h, x:x+w]
                
                # Run prediction ONLY on the cropped face
                # Note: The feature extractor handles resizing to (64, 64)
                features = fe.preprocess_and_extract_features_single(face_roi, img_size=IMG_SIZE)
                features_scaled = scaler.transform(features)
                
                # Predict
                prediction_idx = model.predict(features_scaled)[0]
                prediction_label = label_map[prediction_idx]
                
                # Draw the label above the face
                cv2.putText(
                    img_bgr,
                    prediction_label,
                    (x, y - 10), # Position above the box
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (36, 255, 12), # Color (Green)
                    2
                )
            except Exception as e:
                pass # If prediction fails for a frame, just ignore

        # 4. Return the annotated frame
        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# --- 3. Streamlit Interface ---
st.title("Live Facial Emotion Recognition 📸")
st.write("Now with Face Detection! (Blue box = Face detected)")

webrtc_streamer(
    key="emotion-detection",
    video_transformer_factory=EmotionTransformer,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False}
)

st.write(f"Recognizes: {list(label_map.values())}")

# (Optional: We can keep the file uploader as a separate feature)
st.divider()
st.subheader("Or, test with a single file:")
uploaded_file = st.file_uploader("Upload a facial image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption="Uploaded Image", use_container_width=True)
    
    img_cv_pil = np.array(image_pil)
    if len(img_cv_pil.shape) == 2:
        img_gray = img_cv_pil
    elif img_cv_pil.shape[2] == 4:
        img_gray = cv2.cvtColor(img_cv_pil, cv2.COLOR_RGBA2GRAY)
    elif img_cv_pil.shape[2] == 3:
        img_gray = cv2.cvtColor(img_cv_pil, cv2.COLOR_RGB2GRAY)

    try:
        features = fe.preprocess_and_extract_features_single(img_gray, img_size=IMG_SIZE)
        features_scaled = scaler.transform(features)
        prediction_idx = model.predict(features_scaled)[0]
        prediction_label = label_map[prediction_idx]
        
        st.subheader("Prediction Result 🎉")
        st.metric(label="Predicted Emotion", value=prediction_label)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")