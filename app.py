import streamlit as st
import cv2
import numpy as np
import joblib
import av
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import os

# Mac M3 Configuration (Prevent Mutex Lock Crash)
os.environ["OMP_NUM_THREADS"] = "1"

# TensorFlow must be imported after setting environment variables
import tensorflow as tf
from tensorflow.keras.models import load_model

# Page Configuration
st.set_page_config(page_title="Face Emotion Detection", page_icon="🧠", layout="wide")

# Load Model Resources (Cached)
@st.cache_resource
def load_cnn_model():
    """
    Load CNN model and label mapping
    """
    resources = {}
    try:
        # Load CNN model
        if os.path.exists("emotion_model_cnn.h5"):
            resources['cnn_model'] = load_model("emotion_model_cnn.h5")
        else:
            st.error("⚠️ Cannot find emotion_model_cnn.h5, please run train_cnn.py first")
            return None

        # Load label mapping
        if os.path.exists("label_map.joblib"):
            label_map = joblib.load("label_map.joblib")
            if label_map and len(label_map) > 0 and isinstance(list(label_map.keys())[0], str):
                 resources['inv_map'] = {v: k for k, v in label_map.items()}
            else:
                 resources['inv_map'] = label_map 
        else:
            st.error("⚠️ Cannot find label_map.joblib")
            return None
             
        return resources
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None

# Execute loading
resources = load_cnn_model()
label_map = resources['inv_map'] if resources else {}

# Load face detector
try:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
except Exception as e:
    st.error(f"Error loading Face Detector: {e}")

# Core Prediction Function
def predict_emotion(face_img):
    """
    Predict emotion using CNN model
    """
    if not resources or 'cnn_model' not in resources or resources['cnn_model'] is None:
        return "Error", 0.0, {}

    img_resized = cv2.resize(face_img, (64, 64))
    # Normalize (divide by 255) - Important step!
    img_norm = img_resized.astype("float32") / 255.0
    img_input = img_norm.reshape(1, 64, 64, 1)
    
    # Predict (Mac M3 fix: Force CPU usage)
    with tf.device('/cpu:0'):
        probs = resources['cnn_model'].predict(img_input, verbose=0)[0]

    # Post-processing
    best_idx = np.argmax(probs)
    best_label = label_map[best_idx]
    best_conf = probs[best_idx]
    prob_dict = {label_map[i]: float(probs[i]) for i in range(len(probs))}
    
    return best_label, best_conf, prob_dict

# Video Processing Class (Background Thread)
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                try:
                    face_roi = img_gray[y:y+h, x:x+w]
                    label, conf, _ = predict_emotion(face_roi)
                    
                    color = (0, 255, 0)
                    if label in ['Angry', 'Fear', 'Sad']: 
                        color = (0, 0, 255)
                    elif label == 'Happy': 
                        color = (0, 255, 255)
                    
                    cv2.rectangle(img, (x, y-30), (x+w, y), color, -1)
                    text = f"{label} ({int(conf*100)}%)"
                    cv2.putText(img, text, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                    
                except Exception as inner_e:
                    print(f"Prediction Error: {inner_e}")
                    pass

            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            print(f"Frame Processing Error: {e}")
            return frame

# Main UI

st.title("🧠 Face Emotion Detection System")
st.markdown("### Deep Learning CNN Model")

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Information")
    st.info("**Deep Learning (CNN)**")
    st.caption("✅ End-to-End Feature Learning\n✅ 3-Layer Convolutional Network\n⚠️ Running on CPU (Mac Optimization)")

# Tab Interface
tab1, tab2 = st.tabs(["📸 Live Webcam", "📂 Upload Image"])

# TAB 1: Live Webcam
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Real-time Analysis")
        webrtc_streamer(
            key="emotion-live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=EmotionProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    with col2:
        st.write("### Technical Details")
        st.markdown("""
        - **Input:** 64x64 Normalized Pixels
        - **Architecture:** 3-Layer CNN
        - **Backend:** TensorFlow (CPU Mode)
        - **Output:** Softmax Probabilities
        """)

# TAB 2: Image Upload
with tab2:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
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
            st.warning("⚠️ No specific face detected. Analyzing full image area.")
            face_roi = img_gray
        else:
            st.success(f"✅ Face detected!")
            (x, y, w, h) = faces[0] 
            face_roi = img_gray[y:y+h, x:x+w]

        try:
            label, conf, prob_dict = predict_emotion(face_roi)
            
            with col_stats:
                st.subheader("Results")
                
                emoji_map = {"Happy": "😄", "Sad": "😢", "Angry": "😡", "Fear": "😱", "Surprise": "😲", "Neutral": "😐"}
                emoji = emoji_map.get(label, "😐")
                
                st.metric(label="Predicted Emotion", value=f"{emoji} {label}", delta=f"{conf*100:.1f}% Confidence")
                
                st.markdown("---")
                df_probs = pd.DataFrame(list(prob_dict.items()), columns=["Emotion", "Probability"])
                df_probs["Probability"] = df_probs["Probability"] * 100 
                df_probs = df_probs.set_index("Emotion")
                
                st.bar_chart(df_probs, color="#FF4B4B")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")