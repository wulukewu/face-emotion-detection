import streamlit as st
import cv2
import numpy as np
import joblib
import av
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Import the helper
import feature_extractor as fe

# --- 1. Page Configuration ---
st.set_page_config(page_title="Emotion AI", page_icon="🧠", layout="wide")

# --- Sidebar for Model Selection ---
with st.sidebar:
    st.header("⚙️ Model Selection")
    model_selection = st.radio("Choose a model:", ("Random Forest", "CNN"))
    st.divider()

# --- 2. Load Resources (Cached for Speed) ---
@st.cache_resource
def load_resources():
    """Load all models and resources."""
    resources = {
        "rf": {},
        "cnn": {}
    }
    try:
        # Load Random Forest resources
        resources["rf"]["model"] = joblib.load("emotion_model.joblib")
        resources["rf"]["scaler"] = joblib.load("feature_scaler.joblib")
        label_map_rf = joblib.load("label_map.joblib")
        if isinstance(list(label_map_rf.keys())[0], str):
             resources["rf"]["inv_map"] = {v: k for k, v in label_map_rf.items()}
        else:
             resources["rf"]["inv_map"] = label_map_rf

        # Load CNN resources
        if os.path.exists("emotion_model_cnn.h5"):
            resources["cnn"]["model"] = load_model("emotion_model_cnn.h5")
        else:
            st.error("⚠️ Cannot find emotion_model_cnn.h5. Please run train_cnn.py first.")
        
        # The label map should be the same, but we load it again for safety
        label_map_cnn = joblib.load("label_map.joblib")
        if isinstance(list(label_map_cnn.keys())[0], str):
             resources["cnn"]["inv_map"] = {v: k for k, v in label_map_cnn.items()}
        else:
             resources["cnn"]["inv_map"] = label_map_cnn

    except Exception as e:
        st.error(f"Error loading model files: {e}")
    return resources

resources = load_resources()

# --- Load Face Detector ---
try:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        st.error("Error: Could not load Haar Cascade XML.")
except Exception as e:
    st.error(f"Error loading Face Detector: {e}")

# --- 3. Prediction Functions ---
def predict_emotion_rf(face_img):
    if "rf" not in resources or not all(k in resources["rf"] for k in ["model", "scaler", "inv_map"]):
        return "Error", 0.0, {}
    features = fe.preprocess_and_extract_features_single(face_img)
    features_scaled = resources["rf"]["scaler"].transform(features)
    probs = resources["rf"]["model"].predict_proba(features_scaled)[0]
    best_idx = np.argmax(probs)
    label_map = resources["rf"]["inv_map"]
    best_label = label_map[best_idx]
    best_conf = probs[best_idx]
    prob_dict = {label_map[i]: probs[i] for i in range(len(probs))}
    return best_label, best_conf, prob_dict

def predict_emotion_cnn(face_img):
    if "cnn" not in resources or not all(k in resources["cnn"] for k in ["model", "inv_map"]):
        return "Error", 0.0, {}
    img_resized = cv2.resize(face_img, (64, 64))
    img_norm = img_resized.astype("float32") / 255.0
    img_input = img_norm.reshape(1, 64, 64, 1)
    with tf.device('/cpu:0'):
        probs = resources["cnn"]["model"].predict(img_input, verbose=0)[0]
    best_idx = np.argmax(probs)
    label_map = resources["cnn"]["inv_map"]
    best_label = label_map[best_idx]
    best_conf = probs[best_idx]
    prob_dict = {label_map[i]: float(probs[i]) for i in range(len(probs))}
    return best_label, best_conf, prob_dict

# --- 4. Video Processor Class ---
class EmotionProcessor(VideoProcessorBase):
    def __init__(self, model_type):
        self.model_type = model_type

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = img_gray[y:y+h, x:x+w]
            
            if self.model_type == "CNN":
                label, conf, _ = predict_emotion_cnn(face_roi)
                color = (0, 0, 255) # Red for CNN
            else: # Random Forest
                label, conf, _ = predict_emotion_rf(face_roi)
                color = (0, 255, 0) # Green for RF

            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(img, (x, y-30), (x+w, y), color, -1)
            text = f"{label} ({int(conf*100)}%)"
            cv2.putText(img, text, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. Main UI ---
st.title(f"🧠 Face Emotion Detection: {model_selection}")

tab1, tab2 = st.tabs(["📸 Live Webcam", "📂 Upload Image"])

with tab1:
    st.subheader("Live Feed")
    webrtc_streamer(
        key=f"webrtc-{model_selection}", # Key must be unique per model
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: EmotionProcessor(model_selection),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with tab2:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key=f"uploader-{model_selection}")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        col_img, col_stats = st.columns(2)
        with col_img:
            st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        if len(faces) == 0:
            st.warning("⚠️ No face detected. Analyzing full image.")
            face_roi = img_gray
        else:
            st.success(f"✅ {len(faces)} face(s) detected!")
            (x, y, w, h) = faces[0]
            face_roi = img_gray[y:y+h, x:x+w]

        if model_selection == "CNN":
            label, conf, prob_dict = predict_emotion_cnn(face_roi)
            chart_color = "#FF4B4B"
        else:
            label, conf, prob_dict = predict_emotion_rf(face_roi)
            chart_color = "#00CC96"

        with col_stats:
            st.subheader("Analysis Results")
            st.metric(label="Predicted Emotion", value=label, delta=f"{conf*100:.1f}% Confidence")
            st.markdown("---")
            df_probs = pd.DataFrame(list(prob_dict.items()), columns=["Emotion", "Probability"])
            df_probs["Probability"] *= 100
            df_probs = df_probs.set_index("Emotion")
            st.bar_chart(df_probs, color=chart_color)
