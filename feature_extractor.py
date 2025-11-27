import numpy as np
import cv2
from skimage.feature import local_binary_pattern, hog

# --- 1. Single Image Extractors (The Core Logic) ---

def _extract_lbp_single(img):
    """Extracts LBP histogram (Texture) for a single image."""
    radius = 3
    n_points = 8 * radius
    n_bins = n_points + 2
    
    # Ensure image is integer type for LBP
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
        
    lbp_img = local_binary_pattern(img, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp_img.ravel(),
                             bins=np.arange(0, n_bins + 1),
                             range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def _extract_hog_single(img, visualize=False):
    """Extracts HOG features (Shape)."""
    return hog(img, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2),
               visualize=visualize,
               feature_vector=True)

# --- 2. App Pipeline Function ---

def preprocess_and_extract_features_single(image_gray_raw, img_size=(64, 64)):
    """
    Takes a raw cropped face, resizes it, and returns the Combined LBP+HOG vector.
    Used by app.py.
    """
    # 1. Resize to match training size
    img_resized = cv2.resize(image_gray_raw, img_size)
    
    # 2. Extract LBP
    lbp_features = _extract_lbp_single(img_resized)
    
    # 3. Extract HOG
    hog_features = _extract_hog_single(img_resized)
    
    # 4. Combine
    # We add [np.newaxis, :] to make it shape (1, N) for the model
    combined_features = np.concatenate([lbp_features, hog_features])[np.newaxis, :]
    
    return combined_features

# --- 3. Batch Functions (Used by train_model.py) ---

def extract_pixel_features(images):
    n_samples = images.shape[0]
    return images.reshape(n_samples, -1)

def extract_lbp_features(images):
    print("  > Extracting LBP features...")
    features = [_extract_lbp_single(img) for img in images]
    return np.array(features)

def extract_hog_features(images):
    print("  > Extracting HOG features...")
    features = [_extract_hog_single(img, visualize=False) for img in images]
    return np.array(features)