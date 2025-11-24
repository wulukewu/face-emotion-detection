import streamlit as st
import cv2
import numpy as np
import joblib
import av
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import os

# --- 1. Mac M3 關鍵設定 (防止 Mutex Lock 崩潰) ---
os.environ["OMP_NUM_THREADS"] = "1"

# --- 2. TensorFlow 必須在設定完環境變數後匯入 ---
import tensorflow as tf
from tensorflow.keras.models import load_model

# 匯入你的特徵提取工具
import feature_extractor as fe

# --- 3. 全域配置 (解決 ScriptRunContext 警告的關鍵) ---
# 我們用這個字典來在 UI 和 背景執行緒 之間傳遞設定
if "system_config" not in st.session_state:
    st.session_state.system_config = {"model_type": "Traditional ML (HOG+RF)"}

# 定義一個全域變數引用，讓背景執行緒也能讀到
SYSTEM_CONFIG = {"model_type": "Traditional ML (HOG+RF)"}

# --- 4. 頁面設定 ---
st.set_page_config(page_title="Emotion AI Dual-Core", page_icon="🧠", layout="wide")

# --- 5. 載入模型資源 (快取加速) ---
@st.cache_resource
def load_all_models():
    """
    一次載入所有模型資源
    """
    resources = {}
    try:
        # A. 載入傳統機器學習模型
        if os.path.exists("emotion_model.joblib"):
            resources['rf_model'] = joblib.load("emotion_model.joblib")
            resources['scaler'] = joblib.load("feature_scaler.joblib")
        else:
            st.error("⚠️ 找不到 emotion_model.joblib，請先執行 train_model.py")
            return None
        
        # B. 載入深度學習模型 (CNN)
        if os.path.exists("emotion_model_cnn.h5"):
            resources['cnn_model'] = load_model("emotion_model_cnn.h5")
        else:
            st.warning("⚠️ 找不到 emotion_model_cnn.h5 (CNN 模型)，請先執行 train_cnn.py")
            resources['cnn_model'] = None

        # C. 載入標籤對應表
        if os.path.exists("label_map.joblib"):
            label_map = joblib.load("label_map.joblib")
            if isinstance(list(label_map.keys())[0], str):
                 resources['inv_map'] = {v: k for k, v in label_map.items()}
            else:
                 resources['inv_map'] = label_map 
        else:
            st.error("⚠️ 找不到 label_map.joblib")
            return None
             
        return resources
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None

# 執行載入
resources = load_all_models()
label_map = resources['inv_map'] if resources else {}

# 載入人臉偵測器
try:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
except Exception as e:
    st.error(f"Error loading Face Detector: {e}")

# --- 6. 核心預測函式 (包含 Mac M3 修復) ---
def predict_emotion(face_img, model_type):
    """
    根據使用者選擇，將圖片送往不同的模型
    """
    if not resources:
        return "Error", 0.0, {}

    # A. 傳統機器學習路徑
    if model_type == "Traditional ML (HOG+RF)":
        features = fe.preprocess_and_extract_features_single(face_img)
        features_scaled = resources['scaler'].transform(features)
        probs = resources['rf_model'].predict_proba(features_scaled)[0]

    # B. 深度學習 (CNN) 路徑
    else:
        if resources['cnn_model'] is None:
            return "No Model", 0.0, {}

        img_resized = cv2.resize(face_img, (64, 64))
        # Normalize (除以 255) - 這一步超級重要！
        img_norm = img_resized.astype("float32") / 255.0
        img_input = img_norm.reshape(1, 64, 64, 1)
        
        # 預測 (Mac M3 關鍵修復：強制用 CPU)
        with tf.device('/cpu:0'):
            probs = resources['cnn_model'].predict(img_input, verbose=0)[0]

    # 後處理
    best_idx = np.argmax(probs)
    best_label = label_map[best_idx]
    best_conf = probs[best_idx]
    prob_dict = {label_map[i]: float(probs[i]) for i in range(len(probs))}
    
    return best_label, best_conf, prob_dict

# --- 7. 影像處理類別 (背景執行緒) ---
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
                    
                    # 【修正】直接讀取全域變數
                    current_model = SYSTEM_CONFIG["model_type"]
                    
                    label, conf, _ = predict_emotion(face_roi, model_type=current_model)
                    
                    color = (0, 255, 0)
                    if label in ['Angry', 'Fear', 'Sad']: color = (0, 0, 255)
                    elif label == 'Happy': color = (0, 255, 255)
                    
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

# --- 8. 主介面 UI ---

st.title("🧠 Face Emotion Detection System")
st.markdown("### Scikit-Learn vs TensorFlow comparison")

# --- 側邊欄 ---
with st.sidebar:
    st.header("⚙️ Model Settings")
    
    model_choice = st.radio(
        "Choose AI Core:",
        ("Traditional ML (HOG+RF)", "Deep Learning (CNN)"),
        index=0
    )
    
    # 【修正】更新全域變數
    SYSTEM_CONFIG["model_type"] = model_choice
    
    st.divider()
    st.info(f"**Current Engine:**\n{model_choice}")
    
    if model_choice == "Traditional ML (HOG+RF)":
        st.caption("✅ Fast Inference\n✅ Explicit Features (LBP/HOG)\n❌ Less Robust to lighting")
    else:
        st.caption("✅ Deep Learning\n✅ End-to-End Feature Learning\n⚠️ Running on CPU (Mac Optimization)")

# --- 分頁介面 ---
tab1, tab2 = st.tabs(["📸 Live Webcam", "📂 Upload Image"])

# --- TAB 1: 即時攝影機 ---
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
        st.write(f"**Active Model:** {model_choice}")
        if model_choice == "Deep Learning (CNN)":
            st.markdown("""
            - **Input:** 64x64 Normalized Pixels
            - **Architecture:** 3-Layer CNN
            - **Backend:** TensorFlow (CPU Mode)
            """)
        else:
            st.markdown("""
            - **Input:** HOG (Shape) + LBP (Texture)
            - **Algorithm:** Random Forest Classifier
            - **Backend:** Scikit-Learn
            """)

# --- TAB 2: 圖片上傳 ---
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
            label, conf, prob_dict = predict_emotion(face_roi, model_type=model_choice)
            
            with col_stats:
                st.subheader(f"Results ({model_choice})")
                
                emoji_map = {"Happy": "😄", "Sad": "😢", "Angry": "😡", "Fear": "😱", "Surprise": "😲", "Neutral": "😐"}
                emoji = emoji_map.get(label, "😐")
                
                st.metric(label="Predicted Emotion", value=f"{emoji} {label}", delta=f"{conf*100:.1f}% Confidence")
                
                st.markdown("---")
                df_probs = pd.DataFrame(list(prob_dict.items()), columns=["Emotion", "Probability"])
                df_probs["Probability"] = df_probs["Probability"] * 100 
                df_probs = df_probs.set_index("Emotion")
                
                chart_color = "#FF4B4B" if model_choice == "Deep Learning (CNN)" else "#00CC96"
                st.bar_chart(df_probs, color=chart_color)
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")