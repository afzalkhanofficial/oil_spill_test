import sys
import numpy as np
sys.modules['numpy._core'] = np.core

from flask import Flask, request, jsonify
import cv2
from joblib import load as joblib_load
from skimage.feature import hog
import os
import base64

app = Flask(__name__)

# ----------------------------
# Parameters & Global Settings
# ----------------------------
MODEL_PATH = 'rf_oil_spill_model.pkl'
IMG_SIZE = (128, 128)
# Refined HOG parameters
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)
# Confidence threshold (e.g., 60%)
CONF_THRESHOLD = 0.6

# Load the pre-trained model using joblib
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
model = joblib_load(MODEL_PATH)

# ----------------------------
# Utility Functions
# ----------------------------
def extract_features(img, img_size=IMG_SIZE, pixels_per_cell=PIXELS_PER_CELL, cells_per_block=CELLS_PER_BLOCK):
    """Convert image to grayscale (if needed), resize it, and extract HOG features."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, img_size)
    features = hog(
        img,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=False,
        feature_vector=True
    )
    return features

def generate_heatmap(image, model, window_size=128, stride=32):
    """Slide a window over the image and compute oil spill probability for each patch."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = image.shape
    heatmap_rows = (rows - window_size) // stride + 1
    heatmap_cols = (cols - window_size) // stride + 1
    heatmap = np.zeros((heatmap_rows, heatmap_cols))
    
    for i, r in enumerate(range(0, rows - window_size + 1, stride)):
        for j, c in enumerate(range(0, cols - window_size + 1, stride)):
            patch = image[r:r+window_size, c:c+window_size]
            feat = hog(
                patch,
                pixels_per_cell=PIXELS_PER_CELL,
                cells_per_block=CELLS_PER_BLOCK,
                visualize=False,
                feature_vector=True
            )
            feat = feat.reshape(1, -1)
            prob = model.predict_proba(feat)[0][1]
            heatmap[i, j] = prob
            
    heatmap_resized = cv2.resize(heatmap, (cols, rows), interpolation=cv2.INTER_CUBIC)
    return heatmap_resized

def encode_image_to_base64(image):
    """Encode an image (numpy array) as a base64 string for JSON transport."""
    _, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

# ----------------------------
# Flask Endpoints
# ----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image file'}), 400

    # Resize and extract features for prediction
    img_resized = cv2.resize(img, IMG_SIZE)
    features = extract_features(img_resized)
    features = features.reshape(1, -1)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(max(probabilities))

    result = {
        'prediction': 'Oil spill detected' if prediction == 1 else 'No oil spill detected',
        'probability': confidence,
        'confidence_warning': confidence < CONF_THRESHOLD
    }

    # Generate heatmap overlay if image dimensions allow
    if img.shape[0] >= IMG_SIZE[1] and img.shape[1] >= IMG_SIZE[0]:
        heatmap = generate_heatmap(img, model, window_size=IMG_SIZE[0], stride=32)
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)
        result['heatmap_overlay'] = encode_image_to_base64(overlay)
    else:
        result['heatmap_overlay'] = None

    # Generate HOG visualization from the resized image
    hog_features, hog_image = hog(
        cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY),
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK,
        visualize=True,
        feature_vector=True
    )
    result['hog_image'] = encode_image_to_base64(hog_image)

    return jsonify(result)

# ----------------------------
# Main entry point
# ----------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
