import numpy as np
import cv2
from skimage.feature import local_binary_pattern, hog

# --- Single Image Functions (Used by App & Visualization) ---

def _extract_lbp_single(img):
    """Extracts LBP histogram (Texture) for a single image."""
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

def _extract_hog_single(img, visualize=False):
    """Extracts HOG features (Shape). Can return image for visualization."""
    return hog(img, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2),
               visualize=visualize,
               feature_vector=True)

# --- Batch Functions (Used by Training) ---

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