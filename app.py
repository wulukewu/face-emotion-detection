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

# --- 2. Load Resources (Cached) ---
@st.cache_resource
def load_resources():
    resources = {"rf": {}, "cnn": {}}
    try:
        # Load Random Forest
        if os.path.exists("emotion_model.joblib"):
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
            # Share label map logic if applicable, or load separate
            # Assuming shared map for simplicity if not re-saved separately
            if "inv_map" in resources["rf"]:
                resources["cnn"]["inv_map"] = resources["rf"]["inv_map"]
            else:
                 # Fallback load if RF missing
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

# --- 3. Prediction Logic ---
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
    if x > width // 2: hud_x = x - 170 
    else: hud_x = x + w + 10 
    hud_y = y
    hud_w = 160
    hud_h = 20 + (len(prob_dict) * 20)
    
    overlay = img.copy()
    cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    cv2.putText(img, "Emotion Stats", (hud_x + 5, hud_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset = 35
    for emotion, score in prob_dict.items():
        bar_color = get_color(emotion)
        text_color = (255, 255, 255) if emotion == best_label else (180, 180, 180)
        cv2.putText(img, f"{emotion[:4]}", (hud_x + 5, hud_y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        cv2.rectangle(img, (hud_x + 50, hud_y + y_offset - 5), (hud_x + 150, hud_y + y_offset), (50, 50, 50), -1)
        bar_len = int(score * 100)
        cv2.rectangle(img, (hud_x + 50, hud_y + y_offset - 5), (hud_x + 50 + bar_len, hud_y + y_offset), bar_color, -1)
        cv2.putText(img, f"{int(score*100)}%", (hud_x + 125, hud_y + y_offset + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        y_offset += 20

# --- 5. Video Processor ---
class EmotionProcessor(VideoProcessorBase):
    def __init__(self, model_name, show_hud_flag):
        self.model_name = model_name
        self.show_hud_flag = show_hud_flag

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, 1.1, 4, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = img_gray[y:y+h, x:x+w]
            if self.model_name == "CNN":
                label, conf, prob_dict = predict_emotion_cnn(face_roi)
            else:
                label, conf, prob_dict = predict_emotion_rf(face_roi)

            box_color = get_color(label)
            cv2.rectangle(img, (x, y), (x+w, y+h), box_color, 2)
            cv2.rectangle(img, (x, y-25), (x+w, y), box_color, -1)
            text = f"{label} {int(conf*100)}%"
            cv2.putText(img, text, (x+5, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            
            if self.show_hud_flag:
                try:
                    draw_hud(img, x, y, w, h, prob_dict, label)
                except Exception:
                    pass
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
#              PAGE NAVIGATION
# ==========================================

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["📖 Project Overview", "🚀 Live Demo"])

st.sidebar.divider()

# Only show Model Selection in Sidebar if we are in Demo Mode
if page == "🚀 Live Demo":
    st.sidebar.header("⚙️ Settings")
    model_selection = st.sidebar.radio("Choose Model:", ("Random Forest", "CNN"))
    show_hud_checkbox = st.sidebar.checkbox("Show Detailed HUD", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Color Legend")
    st.sidebar.markdown("""
    - 🟡 **Happy**
    - 🔵 **Sad**
    - 🟣 **Fear**
    - 🔴 **Angry**
    - 🟢 **Surprise**
    """)
else:
    # Default variables to prevent errors if referenced
    model_selection = "Random Forest"
    show_hud_checkbox = True

# ==========================================
#           PAGE 1: PROJECT OVERVIEW
# ==========================================
if page == "📖 Project Overview":
    st.title("🧠 Face Emotion Detection Project")
    st.markdown("### Comparing Traditional ML (Random Forest) vs. Deep Learning (CNN)")
    
    # --- 1. Dataset ---
    st.info("""
    **📚 Dataset Source:** [Kaggle: Human Face Emotions](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions)  
    A collection of images labeled with 5 emotions: **Happy, Sad, Fear, Angry, Surprise**.
    """)

    # --- 2. Methodology ---
    st.subheader("🛠️ Methodologies Compared")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌲 Random Forest")
        st.write("**Type:** Traditional Machine Learning")
        st.markdown("""
        Instead of reading raw pixels directly, we use **Feature Engineering** to tell the computer what to look for:
        * **HOG (Edges/Shapes):** Good for detecting the shape of the mouth and eyes.
        * **LBP (Texture):** Good for detecting skin texture changes (like furrowed brows).
        """)
    
    with col2:
        st.markdown("#### 🧠 CNN (Deep Learning)")
        st.write("**Type:** Convolutional Neural Network")
        st.markdown("""
        The model automatically learns features through layers:
        * **Conv2D Layers:** 32 → 64 → 128 Filters. Captures simple lines first, then complex facial structures.
        * **MaxPooling:** Reduces image size to focus on key features.
        * **Dropout:** Randomly ignores neurons to prevent memorizing the training data.
        """)

    # --- 3. Results ---
    st.subheader("📊 Performance Results")
    st.markdown("The use of **OpenCV Face Cropping** significantly improved accuracy for both models.")
    
    metrics_data = {
        "Model": ["Random Forest", "CNN"],
        "Accuracy (Full Image)": ["~18%", "~24%"],
        "Accuracy (Face Cropped)": ["42.15%", "45.45%"]
    }
    st.table(pd.DataFrame(metrics_data))
    
    st.warning("""
    ### 📉 Analysis: Why "Sad" & "Fear" are hard to detect?
    
    In this experiment, both models struggled to distinguish between **“Sad”** and **“Fear,”** often misclassifying them as **“Angry.”** 
    
    This bias mainly stems from two factors:
    1. **Imbalanced Dataset:** The models had limited exposure to “Sad” and “Fear” samples during training.
    2. **Facial Similarity:** These negative emotions share highly similar facial muscle movements (e.g., lowered eyebrows), making them difficult to separate.
    
    **🚀 Future Work:**  
    Techniques such as **data augmentation** or **balancing the sample sizes** could help improve performance in recognizing these subtle emotional differences.
    """)

# ==========================================
#           PAGE 2: LIVE DEMO
# ==========================================
elif page == "🚀 Live Demo":
    st.title(f"🚀 Live Demo: {model_selection}")
    st.caption("Select a model in the sidebar to start.")

    tab_cam, tab_upload = st.tabs(["📸 Webcam Feed", "📂 Upload Image"])

    # --- Tab 1: Webcam ---
    with tab_cam:
        st.write("Click 'Start' to open your webcam.")
        # We need a unique key for the streamer to force reload when model changes
        webrtc_streamer(
            key=f"streamer-{model_selection}", 
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: EmotionProcessor(model_selection, show_hud_checkbox),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    # --- Tab 2: Upload ---
    with tab_upload:
        uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            # Read file
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            col_img, col_res = st.columns(2)
            
            with col_img:
                st.image(img_rgb, caption="Original Image", use_container_width=True)

            # Detect Face
            faces = face_cascade.detectMultiScale(img_gray, 1.1, 3, minSize=(30, 30))
            
            if len(faces) == 0:
                st.warning("⚠️ No specific face detected. Analyzing full image.")
                face_roi = img_gray
            else:
                st.success(f"✅ {len(faces)} face(s) detected.")
                (x, y, w, h) = faces[0]
                face_roi = img_gray[y:y+h, x:x+w]
                # Draw box on preview
                cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                with col_img:
                    st.image(img_rgb, caption="Face Detected", use_container_width=True)

            # Predict
            if model_selection == "CNN":
                label, conf, prob_dict = predict_emotion_cnn(face_roi)
            else:
                label, conf, prob_dict = predict_emotion_rf(face_roi)

            # Show Results
            with col_res:
                st.subheader(f"Prediction: {label}")
                st.metric(label="Confidence", value=f"{conf*100:.1f}%")
                
                # Bar Chart
                df_probs = pd.DataFrame(list(prob_dict.items()), columns=["Emotion", "Probability"])
                df_probs["Probability"] *= 100
                
                winner_hex = bgr_to_hex(get_color(label))
                
                chart = alt.Chart(df_probs).mark_bar().encode(
                    x=alt.X('Emotion', sort='-y'),
                    y=alt.Y('Probability'),
                    color=alt.value(winner_hex),
                    opacity=alt.condition(
                        alt.datum.Emotion == label,
                        alt.value(1.0),
                        alt.value(0.3)
                    )
                ).properties(height=300)
                
                st.altair_chart(chart, use_container_width=True)