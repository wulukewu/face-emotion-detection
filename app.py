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
import altair as alt

# Import the helper
import feature_extractor as fe

# --- 1. CONFIGURATION & COLORS ---
st.set_page_config(page_title="Face Emotion Detection", page_icon="🧠", layout="wide")

# Colors (BGR for OpenCV, RGB for Streamlit/Altair)
EMOTION_COLORS = {
    "Happy":    (0, 255, 255),    # Yellow
    "Sad":      (255, 0, 0),      # Blue
    "Fear":     (255, 0, 255),    # Purple
    "Angry":    (0, 0, 255),      # Red
    "Surprise": (0, 255, 0),      # Green
    "Neutral":  (200, 200, 200)   # Gray
}

def get_color(label):
    """Helper to get BGR color for OpenCV"""
    clean_label = label.strip().capitalize()
    return EMOTION_COLORS.get(clean_label, (200, 200, 200))

def bgr_to_hex(bgr_tuple):
    """Converts BGR (OpenCV) to Hex (Altair/Streamlit)"""
    b, g, r = bgr_tuple
    return f"#{r:02x}{g:02x}{b:02x}"

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Model Selection")
    # This variable determines which model is active
    model_selection = st.radio("Choose a model:", ("Random Forest", "CNN"))
    
    st.divider()
    st.markdown("### Visualization Settings")
    show_hud = st.checkbox("Show Detailed HUD", value=True)
    st.markdown("""
    **Color Legend:**
    - 🟡 **Happy**
    - 🔵 **Sad**
    - 🟣 **Fear**
    - 🔴 **Angry**
    - 🟢 **Surprise**
    """)

# --- 2. Load Resources ---
@st.cache_resource
def load_resources():
    resources = {"rf": {}, "cnn": {}}
    try:
        # Load Random Forest
        resources["rf"]["model"] = joblib.load("emotion_model.joblib")
        resources["rf"]["scaler"] = joblib.load("feature_scaler.joblib")
        label_map_rf = joblib.load("label_map.joblib")
        if isinstance(list(label_map_rf.keys())[0], str):
             resources["rf"]["inv_map"] = {v: k for k, v in label_map_rf.items()}
        else:
             resources["rf"]["inv_map"] = label_map_rf

        # Load CNN
        if os.path.exists("emotion_model_cnn.h5"):
            resources["cnn"]["model"] = load_model("emotion_model_cnn.h5")
        
        # Load CNN Label Map
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

# --- 4. HUD Drawer ---
def draw_hud(img, x, y, w, h, prob_dict, best_label):
    height, width, _ = img.shape
    
    # Position HUD based on face location
    if x > width // 2: hud_x = x - 170 
    else: hud_x = x + w + 10 
        
    hud_y = y
    hud_w = 160
    hud_h = 20 + (len(prob_dict) * 20)
    
    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    # Header
    cv2.putText(img, "Emotion Stats", (hud_x + 5, hud_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Bars
    y_offset = 35
    for emotion, score in prob_dict.items():
        bar_color = get_color(emotion)
        
        # Highlight winner text
        text_color = (255, 255, 255) if emotion == best_label else (180, 180, 180)
        cv2.putText(img, f"{emotion[:4]}", (hud_x + 5, hud_y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        
        # Bar Background
        cv2.rectangle(img, (hud_x + 50, hud_y + y_offset - 5), (hud_x + 150, hud_y + y_offset), (50, 50, 50), -1)
        
        # Bar Fill
        bar_len = int(score * 100)
        cv2.rectangle(img, (hud_x + 50, hud_y + y_offset - 5), (hud_x + 50 + bar_len, hud_y + y_offset), bar_color, -1)
        
        # Percentage Text
        cv2.putText(img, f"{int(score*100)}%", (hud_x + 125, hud_y + y_offset + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        y_offset += 20

# --- 5. Video Processor ---
# We pass the 'model_name' into the class to ensure the thread knows which one to use
class EmotionProcessor(VideoProcessorBase):
    def __init__(self, model_name):
        self.model_name = model_name

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(img_gray, 1.1, 4, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = img_gray[y:y+h, x:x+w]
            
            # Use self.model_name (passed from the factory)
            if self.model_name == "CNN":
                label, conf, prob_dict = predict_emotion_cnn(face_roi)
            else:
                label, conf, prob_dict = predict_emotion_rf(face_roi)

            box_color = get_color(label)

            cv2.rectangle(img, (x, y), (x+w, y+h), box_color, 2)
            cv2.rectangle(img, (x, y-25), (x+w, y), box_color, -1)
            text = f"{label} {int(conf*100)}%"
            cv2.putText(img, text, (x+5, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            
            if show_hud:
                try:
                    draw_hud(img, x, y, w, h, prob_dict, label)
                except Exception:
                    pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 6. Main UI ---
st.title(f"🧠 Face Emotion Detection: {model_selection}")

tab1, tab2 = st.tabs(["📸 Live Webcam", "📂 Upload Image"])

with tab1:
    st.subheader("Live Feed")
    # By adding model_selection to the key, Streamlit reloads the widget
    # whenever you switch models in the sidebar.
    webrtc_streamer(
        key=f"emotion-live-{model_selection}", 
        mode=WebRtcMode.SENDRECV,
        # Pass the model name to the processor factory
        video_processor_factory=lambda: EmotionProcessor(model_selection),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

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
        else:
            label, conf, prob_dict = predict_emotion_rf(face_roi)

        with col_stats:
            st.subheader("Analysis Results")
            st.metric(label="Predicted Emotion", value=label, delta=f"{conf*100:.1f}% Confidence")
            
            st.markdown("### Probability Distribution")
            
            # Prepare Data for Altair
            df_probs = pd.DataFrame(list(prob_dict.items()), columns=["Emotion", "Probability"])
            df_probs["Probability"] *= 100
            
            # Colors
            winner_bgr = get_color(label)
            winner_hex = bgr_to_hex(winner_bgr)
            
            # Custom Chart (Winner Solid, Others Faded)
            chart = alt.Chart(df_probs).mark_bar().encode(
                x=alt.X('Emotion', sort='-y'),
                y=alt.Y('Probability'),
                color=alt.value(winner_hex),
                opacity=alt.condition(
                    alt.datum.Emotion == label,
                    alt.value(1.0),
                    alt.value(0.3)
                ),
                tooltip=['Emotion', 'Probability']
            ).properties(height=300)
            
            st.altair_chart(chart, use_container_width=True)