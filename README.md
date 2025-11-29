# 🧠 Face Emotion Detection

[![Chinese (Traditional)](https://img.shields.io/badge/Language-Chinese%20(Traditional)-blue)](README.zh-TW.md)

A machine learning project comparing Traditional ML (Random Forest) and Deep Learning (CNN) for real-time face emotion recognition.

## 📊 Project Overview
This project explores two different approaches to classify human facial emotions into 5 categories: **Happy, Sad, Fear, Angry, and Surprise**.

### 1. Dataset
* **Source:** [Human Face Emotions (Kaggle)](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions)
* **Content:** A collection of emotion-labeled images suitable for Computer Vision.
* **Preprocessing:** Images are converted to Grayscale and resized to **64x64 pixels**.

### 2. Methodologies Compared

#### 🌲 Random Forest (Traditional ML)
Instead of feeding raw pixels, we use feature engineering to help the model "see":
* **HOG (Histogram of Oriented Gradients):** Captures **Shape & Edges**.
* **LBP (Local Binary Patterns):** Captures **Texture Information**.
* **Result:** Combining HOG + LBP yielded the best results for this model.

#### 🧠 CNN (Deep Learning)
A Convolutional Neural Network that performs **Feature Learning** automatically.
* **Architecture:**
    * **3 Convolutional Blocks:** 32, 64, and 128 filters to capture features from simple lines to complex facial parts.
    * **MaxPooling:** Reduces dimensions and keeps significant features.
    * **Dropout (0.5):** Prevents overfitting by randomly disabling neurons during training.

### 3. Results & Observations
* **Face Detection:** Using OpenCV to crop faces significantly improved accuracy from ~18% to **42.15%** (RF) and ~24% to **45.45%** (CNN).
* **Class Bias:** The model struggles most with "Sad" and "Fear," often confusing them with "Angry" due to similar facial features.
* **Overfitting:** The CNN model tends to overfit after 5-10 epochs, requiring careful monitoring of validation loss.

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.9+
* Docker (Optional)

### Run Locally
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the app:
    ```bash
    streamlit run app.py
    ```

### Run with Docker
```bash
docker-compose up --build