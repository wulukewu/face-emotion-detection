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
st.set_page_config(page_title="Face Emotion Detection", page_icon="🧠", layout="wide")

# --- Sidebar for Model Selection ---
with st.sidebar:
    st.header("⚙️ Model Selection")
    model_selection = st.radio("Choose a model:", ("Random Forest", "CNN"))
    st.divider()
    st.markdown("### Visualization Settings")
    show_hud = st.checkbox("Show Detailed HUD", value=True)

# --- 2. Load Resources (Cached) ---
@st.cache_resource
def load_resources():
    resources = {"rf": {}, "cnn": {}}
    try:
        # Load Random Forest
        resources["rf"]["model"] = joblib.load("emotion_model.joblib")
        resources["rf"]["scaler"] = joblib.load("feature_scaler.joblib")
        label_map_rf = joblib.load("label_map.joblib")
        # Ensure map is {0: "Happy"}
        if isinstance(list(label_map_rf.keys())[0], str):
             resources["rf"]["inv_map"] = {v: k for k, v in label_map_rf.items()}
        else:
             resources["rf"]["inv_map"] = label_map_rf

        # Load CNN
        if os.path.exists("emotion_model_cnn.h5"):
            resources["cnn"]["model"] = load_model("emotion_model_cnn.h5")
        
        # Load CNN Label Map (Reusing RF map logic usually works if trained same way, but safe to reload)
        label_map_cnn = joblib.load("label_map.joblib")
        if isinstance(list(label_map_cnn.keys())[0], str):
             resources["cnn"]["inv_map"] = {v: k for k, v in label_map_cnn.items()}
        else:
             resources["cnn"]["inv_map"] = label_map_cnn

    except Exception as e:
        st.error(f"Error loading files: {e}")
    return resources

resources = load_resources()

# --- Load Face Detector ---
try:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
except Exception as e:
    st.error(f"Error loading Face Detector: {e}")

# --- 3. Prediction Functions ---
def predict_emotion_rf(face_img):
    if "rf" not in resources or "model" not in resources["rf"]: return "Error", 0.0, {}
    features = fe.preprocess_and_extract_features_single(face_img)
    features_scaled = resources["rf"]["scaler"].transform(features)
    probs = resources["rf"]["model"].predict_proba(features_scaled)[0]
    best_idx = np.argmax(probs)
    label_map = resources["rf"]["inv_map"]
    return label_map[best_idx], probs[best_idx], {label_map[i]: probs[i] for i in range(len(probs))}

def predict_emotion_cnn(face_img):
    if "cnn" not in resources or "model" not in resources["cnn"]: return "Error", 0.0, {}
    img_resized = cv2.resize(face_img, (64, 64))
    img_norm = img_resized.astype("float32") / 255.0
    img_input = img_norm.reshape(1, 64, 64, 1)
    with tf.device('/cpu:0'):
        probs = resources["cnn"]["model"].predict(img_input, verbose=0)[0]
    best_idx = np.argmax(probs)
    label_map = resources["cnn"]["inv_map"]
    return label_map[best_idx], probs[best_idx], {label_map[i]: float(probs[i]) for i in range(len(probs))}

# --- 4. The HUD Drawer (New Function) ---
def draw_hud(img, x, y, w, h, prob_dict, best_label):
    """
    Draws a Heads-Up Display (HUD) with probability bars next to the face.
    """
    height, width, _ = img.shape
    
    # 1. Determine HUD position (Left or Right of face)
    # If face is on the right side, draw HUD on the left, and vice versa
    if x > width // 2:
        hud_x = x - 170 # Draw left
    else:
        hud_x = x + w + 10 # Draw right
        
    hud_y = y
    hud_w = 160
    hud_h = 20 + (len(prob_dict) * 20)
    
    # 2. Draw Semi-Transparent Background
    overlay = img.copy()
    cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (0, 0, 0), -1)
    alpha = 0.6 # Transparency factor
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # 3. Draw Header
    cv2.putText(img, "Emotion Stats", (hud_x + 5, hud_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 4. Draw Bars for each emotion
    y_offset = 35
    for emotion, score in prob_dict.items():
        # Text Label
        text_color = (0, 255, 0) if emotion == best_label else (200, 200, 200)
        cv2.putText(img, f"{emotion[:4]}", (hud_x + 5, hud_y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        
        # Bar Background
        cv2.rectangle(img, (hud_x + 50, hud_y + y_offset - 5), (hud_x + 150, hud_y + y_offset), (50, 50, 50), -1)
        
        # Bar Fill (Length depends on score)
        bar_len = int(score * 100)
        fill_color = (0, 255, 0) if emotion == best_label else (0, 150, 255) # Green for winner, Orange for others
        cv2.rectangle(img, (hud_x + 50, hud_y + y_offset - 5), (hud_x + 50 + bar_len, hud_y + y_offset), fill_color, -1)
        
        # Percentage Text
        cv2.putText(img, f"{int(score*100)}%", (hud_x + 125, hud_y + y_offset + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        y_offset += 20

# --- 5. Video Processor Class ---
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        # We access global variables (st.session_state is harder in threads)
        # So we pass arguments via the factory or check globals
        pass

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect Face
        faces = face_cascade.detectMultiScale(img_gray, 1.1, 4, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = img_gray[y:y+h, x:x+w]
            
            # Predict based on global selection
            if model_selection == "CNN":
                label, conf, prob_dict = predict_emotion_cnn(face_roi)
                box_color = (0, 165, 255) # Orange for CNN
            else:
                label, conf, prob_dict = predict_emotion_rf(face_roi)
                box_color = (0, 255, 0) # Green for RF

            # Draw Standard Box
            cv2.rectangle(img, (x, y), (x+w, y+h), box_color, 2)
            
            # Draw Label Top
            text = f"{label} {int(conf*100)}%"
            cv2.rectangle(img, (x, y-25), (x+w, y), box_color, -1)
            cv2.putText(img, text, (x+5, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            
            # DRAW THE DETAILED HUD (If enabled)
            if show_hud:
                try:
                    draw_hud(img, x, y, w, h, prob_dict, label)
                except Exception as e:
                    print(f"HUD Draw Error: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 6. Main UI Layout ---
st.title(f"🧠 Face Emotion Detection: {model_selection}")

tab1, tab2 = st.tabs(["📸 Live Webcam", "📂 Upload Image"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Live Feed")
        webrtc_streamer(
            key="emotion-live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=EmotionProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    with col2:
        st.info("💡 **Feature:**\nThe live feed now includes a **Real-Time HUD**.\n\nIt shows the confidence for *all* emotions, not just the winner. If 'Happy' and 'Neutral' are fighting, you will see both bars jumping!")

with tab2:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="uploader")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        col_img, col_stats = st.columns(2)
        with col_img:
            st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

        faces = face_cascade.detectMultiScale(img_gray, 1.1, 3, minSize=(30, 30))
        
        if len(faces) == 0:
            st.warning("⚠️ No specific face detected. Analyzing full image.")
            face_roi = img_gray
        else:
            st.success(f"✅ {len(faces)} face(s) detected!")
            (x, y, w, h) = faces[0]
            face_roi = img_gray[y:y+h, x:x+w]

        if model_selection == "CNN":
            label, conf, prob_dict = predict_emotion_cnn(face_roi)
            color_code = "#FFA500"
        else:
            label, conf, prob_dict = predict_emotion_rf(face_roi)
            color_code = "#00CC96"

        with col_stats:
            st.subheader("Analysis Results")
            st.metric(label="Predicted Emotion", value=label, delta=f"{conf*100:.1f}% Confidence")
            
            st.markdown("### Probability Distribution")
            df_probs = pd.DataFrame(list(prob_dict.items()), columns=["Emotion", "Probability"])
            df_probs["Probability"] *= 100
            df_probs = df_probs.set_index("Emotion")
            st.bar_chart(df_probs, color=color_code)