import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

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

# --- 2. Define the Real-time Video Processor ---

class EmotionTransformer(VideoTransformerBase):
    def __init__(self):
        # We can store the last predicted label to make the text smoother
        self.last_label = "Waiting..."

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        This function is called by streamlit-webrtc for every frame.
        """
        # 1. Convert the frame to an OpenCV-compatible format (BGR)
        img_bgr = frame.to_ndarray(format="bgr24")
        
        # 2. Convert to grayscale for our model
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        try:
            # 3. Run our prediction pipeline (from feature_extractor.py)
            features = fe.preprocess_and_extract_features_single(img_gray, img_size=IMG_SIZE)
            
            # 4. Standardize
            features_scaled = scaler.transform(features)
            
            # 5. Predict
            prediction_idx = model.predict(features_scaled)[0]
            prediction_label = label_map[prediction_idx]
            
            # (Optional: Get probabilities for display)
            # probabilities = model.predict_proba(features_scaled)[0]
            # prob_max = np.max(probabilities)
            # self.last_label = f"{prediction_label} ({prob_max*100:.1f}%)"
            
            self.last_label = f"Emotion: {prediction_label}"

        except Exception as e:
            # If feature extraction fails (e.g., no face), just skip
            # print(f"Error during prediction: {e}") 
            pass

        # 6. Draw the result back onto the *color* frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            img_bgr,
            self.last_label,
            (10, 50),  # Position (x, y)
            font,
            1,  # Font scale
            (255, 255, 255),  # Color (White)
            2,  # Thickness
            cv2.LINE_AA,
        )

        # 7. Convert the modified frame back to the format Streamlit expects
        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# --- 3. Streamlit Interface ---
st.title("Live Facial Emotion Recognition 📸")
st.write("Based on LBP + HOG Feature Engineering & Random Forest Model")
st.write("Click 'START' to begin. Your browser will ask for camera permission.")

# This is the main component that runs the webcam
webrtc_streamer(
    key="emotion-detection",
    video_transformer_factory=EmotionTransformer,
    rtc_configuration={  # This is needed for deployment
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    }
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