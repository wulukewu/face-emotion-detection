import tensorflow as tf

import os
os.environ["OMP_NUM_THREADS"] = "1"
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Docker
import matplotlib.pyplot as plt
import joblib

# Keras / TensorFlow Imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# CONFIGURATION
DATA_PATH = "Data"
IMG_SIZE = (64, 64)
MODEL_PATH = "emotion_model_cnn.h5"  # 存成 .h5 格式，這是 Keras 的標準

print(f"TensorFlow Version: {tf.__version__}")
print("Loading images for CNN...")

# --- 1. 資料載入 (與原本類似，但不做 HOG/LBP) ---
images = []
labels = []
label_map = {}
label_idx = 0

# 確保 Data 資料夾存在
if not os.path.exists(DATA_PATH):
    print(f"Error: {DATA_PATH} not found.")
    exit()

for emotion_folder in os.listdir(DATA_PATH):
    emotion_path = os.path.join(DATA_PATH, emotion_folder)
    if not os.path.isdir(emotion_path): continue
    
    # 建立標籤對應表
    if emotion_folder not in label_map:
        label_map[emotion_folder] = label_idx
        label_idx += 1
    
    current_label = label_map[emotion_folder]
    
    for img_file in os.listdir(emotion_path):
        full_path = os.path.join(emotion_path, img_file)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE) # 讀取灰階
        if img is not None:
            img_resized = cv2.resize(img, IMG_SIZE)
            images.append(img_resized)
            labels.append(current_label)

# 轉換為 Numpy Array
X = np.array(images)
y = np.array(labels)

# --- 2. 深度學習專用預處理 (關鍵步驟！) ---

# A. 正規化 (Normalization): 將像素值從 0-255 縮放到 0-1 之間
X = X / 255.0

# B. Reshape: CNN 需要輸入形狀為 (長, 寬, 通道數)
# 灰階圖只有 1 個通道，所以要變成 (64, 64, 1)
X = X.reshape(-1, 64, 64, 1)

# C. One-Hot Encoding: 將標籤 0, 1, 2 變成 [1,0,0], [0,1,0]...
y_onehot = to_categorical(y, num_classes=len(label_map))

# 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, stratify=y, random_state=42
)

print(f"Dataset Shape: {X.shape}")
print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")

# --- 3. 搭建 Keras CNN 模型 (期末簡報重點) ---
# Sequential: 代表層是一層一層堆疊下去的
model = Sequential([
    # 第一層卷積：提取基礎特徵 (如線條、邊緣)
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)), # 縮減圖片尺寸，保留重要特徵

    # 第二層卷積：提取更複雜的特徵 (如眼睛、嘴巴形狀)
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # 第三層卷積
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # 攤平層：將立體的特徵圖拉平成一維陣列，準備進全連接層
    Flatten(),

    # 全連接層 (Dense)：進行分類
    Dense(128, activation='relu'),
    
    # Dropout: 隨機丟棄 50% 神經元，防止過擬合 (Overfitting)
    Dropout(0.5),

    # 輸出層：有幾個情緒就輸出幾個機率值 (Softmax 讓總和為 1)
    Dense(len(label_map), activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary() # 顯示模型架構 (可以截圖放在簡報)

# --- 4. 開始訓練 ---
print("\nStarting Training... (這可能會花幾分鐘)")
history = model.fit(
    X_train, y_train,
    epochs=15,             # 訓練 15 輪
    batch_size=32,         # 每次看 32 張圖
    validation_data=(X_test, y_test) # 用測試集驗證成效
)

# --- 5. 評估與存檔 ---
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nFinal Test Accuracy: {test_acc:.2%}")

# 儲存模型 (Keras 格式)
model.save(MODEL_PATH)

# 儲存標籤對應表 (App 預測時需要知道 0 是 Angry 還是 Happy)
inv_label_map = {v: k for k, v in label_map.items()}
joblib.dump(inv_label_map, "label_map.joblib") # 覆蓋舊的沒關係，只要確保一致

print(f"Model saved to {MODEL_PATH}")

# --- 6. 繪製訓練曲線 (簡報素材) ---
plt.figure(figsize=(12, 5))

# 左圖：準確率
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Model Accuracy (CNN)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 右圖：損失值
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Save the figure instead of showing it (for Docker compatibility)
plt.savefig('training_history_cnn.png')
print("Training history plot saved to training_history_cnn.png")