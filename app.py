import streamlit as st
import cv2
import numpy as np
import joblib
import av
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Import the helper
import feature_extractor as fe

# --- 1. Page Configuration ---
st.set_page_config(page_title="Emotion AI", page_icon="🧠", layout="wide")

# --- 2. Load Resources (Cached for Speed) ---
@st.cache_resource
def load_model_resources():
    try:
        model = joblib.load("emotion_model.joblib")
        scaler = joblib.load("feature_scaler.joblib")
        label_map = joblib.load("label_map.joblib")
        
        # Ensure label map is {0: "Happy", 1: "Sad"}
        if isinstance(list(label_map.keys())[0], str):
             inv_map = {v: k for k, v in label_map.items()}
        else:
             inv_map = label_map 
             
        return model, scaler, inv_map
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

model, scaler, label_map = load_model_resources()

# --- Load Face Detector with Error Handling ---
try:
    # We use the path from cv2.data
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        st.error("Error: Could not load Haar Cascade XML. Check your opencv installation.")
except Exception as e:
    st.error(f"Error loading Face Detector: {e}")

# --- 3. Helper Function for Prediction ---
def predict_emotion(face_img):
    """
    Input: Gray scale face image (cropped)
    Output: Best Label, Confidence Score, Probability Dictionary
    """
    # 1. Preprocess
    # Note: feature_extractor expects the raw grayscale crop
    features = fe.preprocess_and_extract_features_single(face_img)
    
    # 2. Scale
    features_scaled = scaler.transform(features)
    
    # 3. Predict Probabilities
    probs = model.predict_proba(features_scaled)[0]
    
    # 4. Get Best Match
    best_idx = np.argmax(probs)
    best_label = label_map[best_idx]
    best_conf = probs[best_idx]
    
    # 5. Create Dictionary for Charts
    prob_dict = {label_map[i]: probs[i] for i in range(len(probs))}
    
    return best_label, best_conf, prob_dict

# --- 4. Live Webcam Processor ---
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Flip horizontally for mirror effect
        img = cv2.flip(img, 1)
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # TWEAKED: minNeighbors=4 (Less strict than 5), scaleFactor=1.1
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Draw Green Box
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            try:
                # Crop & Predict
                face_roi = img_gray[y:y+h, x:x+w]
                label, conf, _ = predict_emotion(face_roi)
                
                # Draw Label background
                cv2.rectangle(img, (x, y-30), (x+w, y), (0, 255, 0), -1)
                
                # Draw Text
                text = f"{label} {int(conf*100)}%"
                cv2.putText(img, text, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            except Exception as e:
                print(f"Prediction Error: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. Main UI Layout ---

st.title("🧠 Face Emotion Detection")
st.markdown("Machine Learning Final Project")

tab1, tab2 = st.tabs(["📸 Live Webcam", "📂 Upload Image"])

# --- TAB 1: LIVE WEBCAM ---
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
        st.info("💡 **Troubleshooting:**\n1. If no box appears, try moving closer/further.\n2. Ensure light is shining ON your face, not behind you.")

# --- TAB 2: UPLOAD IMAGE ---
with tab2:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Convert file to opencv image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        col_img, col_stats = st.columns(2)
        
        with col_img:
            st.image(img_rgb, caption="Uploaded Image", use_container_width=True)
            
        # TWEAKED: minNeighbors=3 (Even less strict for static images)
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        
        # --- FALLBACK LOGIC ---
        # If no face is found, we take the WHOLE image
        if len(faces) == 0:
            st.warning("⚠️ No specific face detected. Analyzing the entire image area as a fallback.")
            face_roi = img_gray # Use the whole image
            # Draw a box around the whole image so user knows what happened
            h, w = img_gray.shape
            # Just to be safe, we don't draw on the displayed image since it's already shown
        else:
            # Process the largest face found
            st.success(f"✅ Face detected!")
            (x, y, w, h) = faces[0] 
            face_roi = img_gray[y:y+h, x:x+w]

        # Predict
        try:
            label, conf, prob_dict = predict_emotion(face_roi)
            
            with col_stats:
                st.subheader("Analysis Results")
                st.metric(label="Predicted Emotion", value=label, delta=f"{conf*100:.1f}% Confidence")
                
                st.markdown("---")
                st.write("### Probability Distribution")
                
                # Create DataFrame for Chart
                df_probs = pd.DataFrame(list(prob_dict.items()), columns=["Emotion", "Probability"])
                df_probs["Probability"] = df_probs["Probability"] * 100 
                df_probs = df_probs.set_index("Emotion")
                
                # Display Chart
                st.bar_chart(df_probs, color="#00CC96")
                
                with st.expander("See Raw Data"):
                    st.dataframe(df_probs)
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    st.write(f"**Model:** Random Forest")
    st.write(f"**Features:** LBP + HOG")
    st.divider()