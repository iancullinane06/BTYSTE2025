from flask import Flask, request, jsonify, render_template  
import os
import numpy as np
import rasterio
import tensorflow as tf
from keras.models import load_model
from dotenv import load_dotenv

# Set parameters
IMG_HEIGHT = 244
IMG_WIDTH = 244
IMG_CHANNELS = 6

def dice_coefficient(y_true, y_pred, smooth=1):
    """
    Dice Coefficient for binary segmentation.
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Accuracy metric for final testing
def f2_score(y_true, y_pred, beta=2):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    true_positive = tf.keras.backend.sum(y_true_f * y_pred_f)
    precision = true_positive / (tf.keras.backend.sum(y_pred_f) + tf.keras.backend.epsilon())
    recall = true_positive / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.epsilon())
    f2 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + tf.keras.backend.epsilon())
    return f2

def pixel_accuracy(y_true, y_pred):
    """
    Pixel Accuracy metric.
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)

    # Compare predictions and true values
    correct_predictions = tf.keras.backend.sum(tf.keras.backend.cast(tf.keras.backend.equal(y_true_f, tf.keras.backend.round(y_pred_f)), dtype='float32'))
    
    # Divide by the total number of pixels
    total_pixels = tf.keras.backend.sum(tf.keras.backend.ones_like(y_true_f))
    return correct_predictions / total_pixels


def mean_iou(y_true, y_pred, num_classes=2):
    y_pred = tf.keras.backend.round(y_pred)
    iou_scores = []
    for i in range(num_classes):
        intersection = tf.keras.backend.sum(tf.keras.backend.cast(y_true == i, tf.float32) * tf.keras.backend.cast(y_pred == i, tf.float32))
        union = tf.keras.backend.sum(tf.keras.backend.cast(y_true == i, tf.float32) + tf.keras.backend.cast(y_pred == i, tf.float32)) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_scores.append(iou)
    return tf.keras.backend.mean(tf.keras.backend.stack(iou_scores))


def dice_loss(y_true, y_pred):
    """
    Dice loss function, which is 1 - Dice Coefficient.
    """
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """
    Combined loss: Dice loss + Binary Cross-Entropy
    """
    return dice_loss(y_true, y_pred) + tf.keras.losses.binary_crossentropy(y_true, y_pred)


app = Flask(__name__)
load_dotenv()

# Load the model with a check
model_path = os.getenv('RHODODENDRON-DEEPLAB-MODEL')
if model_path is None or not os.path.exists(model_path):
    raise ValueError("Model path not set or does not exist.")

model = load_model(model_path, custom_objects={
    'combined_loss': combined_loss,
    'dice_coefficient': dice_coefficient,
    'f2_score': f2_score,
    'pixel_accuracy': pixel_accuracy,
    'mean_iou': mean_iou
})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_image', methods=['POST'])
def load_image():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    # If the user does not select a file, the browser may submit an empty part without a filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Process the uploaded file
    with rasterio.open(file) as src:
        image = src.read()  # Read image data

    # You might want to perform additional processing on the image here

    # Return processed image info or confirmation
    return jsonify({"status": "Image loaded successfully", "shape": image.shape})

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content response

@app.route('/run_inference', methods=['POST'])
def run_inference():
    data = request.json
    image_data = np.array(data['image_data'])
    
    # Check the shape of image_data
    if image_data.shape != (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        return jsonify({"error": "Input image must have shape (244, 244, 6)."}), 400
    
    prediction = model.predict(image_data)
    binary_prediction = (prediction > 0.5).astype(np.int32)
    
    return jsonify({"prediction": binary_prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)