from flask import Flask, request, jsonify
import cv2
import numpy as np
import pickle
from skimage.feature import hog
import os
import base64
import io
from matplotlib import pyplot as plt

app = Flask(__name__)

# ----------------------------
# Parameters & Global Settings
# ----------------------------
MODEL_PATH = 'rf_oil_spill_model.pkl'
IMG_SIZE = (128, 128)
# Refined HOG parameters
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)
# Allowed image extensions (for logging/validation if needed)
ALLOWED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp')
# Confidence threshold (e.g., 60%)
CONF_THRESHOLD = 0.6

# Load the pre-trained model from pickle file
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# ----------------------------
# Utility Functions
# ----------------------------
def extract_features(img, img_size=IMG_SIZE, pixels_per_cell=PIXELS_PER_CELL, cells_per_block=CELLS_PER_BLOCK):
    """
    Convert image to grayscale if needed, resize it, and extract HOG features.
    """
    # If image has more than one channel, convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, img_size)
    features = hog(img,
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block,
                   visualize=False,
                   feature_vector=True)
    return features

def generate_heatmap(image, model, window_size=128, stride=32):
    """
    Slide a window over the image, compute oil spill probability for each patch,
    and return an interpolated heatmap.
    """
    # Ensure image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = image.shape
    heatmap_rows = (rows - window_size) // stride + 1
    heatmap_cols = (cols - window_size) // stride + 1
    heatmap = np.zeros((heatmap_rows, heatmap_cols))
    
    for i, r in enumerate(range(0, rows - window_size + 1, stride)):
        for j, c in enumerate(range(0, cols - window_size + 1, stride)):
            patch = image[r:r+window_size, c:c+window_size]
            feat = hog(patch, pixels_per_cell=PIXELS_PER_CELL, cells_per_block=CELLS_PER_BLOCK,
                       visualize=False, feature_vector=True)
            feat = feat.reshape(1, -1)
            prob = model.predict_proba(feat)[0][1]  # probability for oil spill class
            heatmap[i, j] = prob
            
    # Resize heatmap to original image dimensions
    heatmap_resized = cv2.resize(heatmap, (cols, rows), interpolation=cv2.INTER_CUBIC)
    return heatmap_resized

def encode_image_to_base64(image):
    """
    Encode an image (numpy array) as a base64 string for JSON transport.
    """
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
    # Decode image from bytes; support multiple formats
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image file'}), 400

    # Extract HOG features for prediction from a resized copy
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

    # If the uploaded image is larger than the training size, generate a heatmap.
    if img.shape[0] >= IMG_SIZE[1] and img.shape[1] >= IMG_SIZE[0]:
        heatmap = generate_heatmap(img, model, window_size=IMG_SIZE[0], stride=32)
        # Apply a colormap for visualization
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        # Overlay the heatmap on the original image (convert image to BGR if needed)
        overlay = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)
        # Encode overlay image to base64
        result['heatmap_overlay'] = encode_image_to_base64(overlay)
    else:
        result['heatmap_overlay'] = None

    return jsonify(result)

# ----------------------------
# Main entry point
# ----------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
