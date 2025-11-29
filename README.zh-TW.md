# 🧠 臉部情緒辨識

[![English](https://img.shields.io/badge/Language-English-red)](README.md)

這是一個期末專題報告，比較了傳統機器學習 (Random Forest) 與深度學習 (CNN) 在即時臉部情緒辨識上的表現。

## 📊 專題簡介
本專案旨在探索兩種不同的方法將人類臉部情緒分類為 5 種類別：**Happy (開心)、Sad (悲傷)、Fear (恐懼)、Angry (生氣) 以及 Surprise (驚訝)**。

### 1. 資料集
* **來源:** [Human Face Emotions (Kaggle)](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions)
* **內容:** 包含來自 Kaggle, Google Images, Bing 等多種來源的臉部影像。
* **前處理:** 影像被轉換為灰階並縮放至 **64x64 像素**。

### 2. 使用方法比較

#### 🌲 Random Forest (隨機森林)
我們不直接輸入原始像素，而是透過特徵工程提取影像重點：
* **HOG (圖形邊緣特徵):** 捕捉臉部的 **形狀與邊緣 (Edges/Shape)**。
* **LBP (圖形紋理特徵):** 捕捉臉部的 **紋理資訊 (Texture Info)**。
* **結論:** 結合 HOG + LBP 的準確率最高。

#### 🧠 CNN (卷積神經網路)
透過深度學習進行自動化的 **特徵學習 (Feature Learning)**。
* **模型架構:**
    * **3 個卷積層 (Convolutional Blocks):** 分別使用 32, 64, 128 個濾鏡，從簡單線條提取至複雜五官。
    * **池化層 (MaxPooling):** 縮小圖片尺寸，保留最顯著特徵並減少運算量。
    * **Dropout (0.5):** 訓練時隨機關閉 50% 神經元，防止過擬合 (Overfitting)。

### 3. 實驗結果與觀察
* **前處理的重要性:** 加入 OpenCV 進行臉部裁切後，Random Forest 準確率由 18% 提升至 **42.15%**；CNN 由 24% 提升至 **45.45%**。
* **類別偏差 (Class Bias):** 模型較難區分 "Sad" 與 "Fear"，常將其誤判為 "Angry"。
* **過擬合 (Overfitting):** CNN 模型在訓練約 5-10 個 Epoch 後，Loss 值容易開始上升，需注意過擬合現象。

## 🛠️ 安裝與使用

### 環境需求
* Python 3.9+
* Docker (選用)

### 本地執行
1.  安裝套件:
    ```bash
    pip install -r requirements.txt
    ```
2.  啟動應用程式:
    ```bash
    streamlit run app.py
    ```

### 使用 Docker 執行
```bash
docker-compose up --build