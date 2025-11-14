# filename: feature_extractor.py

import numpy as np
import cv2
from skimage.feature import local_binary_pattern, hog

# --- Single-Image Helper Functions (for Prediction) ---
# These are the core logic, operating on one image at a time.

def _extract_lbp_single(img):
    """
    Extracts LBP histogram for a single grayscale image.
    """
    radius = 3
    n_points = 8 * radius
    n_bins = n_points + 2
    
    lbp_img = local_binary_pattern(img, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp_img.ravel(),
                             bins=np.arange(0, n_bins + 1),
                             range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def _extract_hog_single(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Extracts HOG features for a single grayscale image.
    """
    hog_features = hog(img, pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block,
                       visualize=False,
                       feature_vector=True)
    return hog_features

# --- Batch Processing Functions (for Training) ---
# These apply the single-image functions to an entire dataset.

def extract_pixel_features(images):
    """
    Flatten (n, 64, 64) image array to (n, 4096) feature vector.
    """
    n_samples = images.shape[0]
    return images.reshape(n_samples, -1)

def extract_lbp_features(images):
    """
    Extracts LBP features for a batch of images.
    """
    print("Extracting LBP features for batch...")
    features = [_extract_lbp_single(img) for img in images]
    return np.array(features)

def extract_hog_features(images, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Extracts HOG features for a batch of images.
    """
    print("Extracting HOG features for batch...")
    features = [_extract_hog_single(img, pixels_per_cell, cells_per_block) for img in images]
    return np.array(features)

# --- Prediction Pipeline Function (for App) ---

def preprocess_and_extract_features_single(image_gray_raw, img_size=(64, 64)):
    """
    Runs the full preprocessing and feature extraction pipeline on a single
    RAW GRAYSCALE (CV2) image.
    This is used by the Streamlit app.
    """
    
    # 1. Resize
    # We now resize the raw grayscale image passed from the app
    img_resized = cv2.resize(image_gray_raw, img_size)
    
    # 2. Extract LBP and HOG
    lbp_features = _extract_lbp_single(img_resized)
    hog_features = _extract_hog_single(img_resized)
    
    # 3. Combine features and reshape for prediction (1, n_features)
    combined_features = np.concatenate([lbp_features, hog_features])[np.newaxis, :]
    
    return combined_features