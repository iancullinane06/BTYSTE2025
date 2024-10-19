from flask import Flask, request, jsonify, send_file
import os
import numpy as np
import rasterio
import tensorflow as tf
from keras.models import load_model
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from EcoLytix.Packages import *

app = Flask(__name__)

# Define global variables for your paths
model_path = None
output_dir = os.path.join('inferenced')
os.makedirs(output_dir, exist_ok=True)

def load_model_with_custom_objects(model_path):
    return load_model(model_path, custom_objects={
    'combined_loss': combined_loss,
    'dice_coefficient': dice_coefficient,
    'f2_score': f2_score,
    'pixel_accuracy': pixel_accuracy,
    'mean_iou': mean_iou
})

@app.route('/upload_model', methods=['POST'])
def upload_model():
    global model_path
    model = request.files['model']
    model_path = os.path.join(output_dir, model.filename)
    model.save(model_path)
    return jsonify({"message": "Model uploaded successfully!"})

@app.route('/upload_shapefile', methods=['POST'])
def upload_shapefile():
    shapefile = request.files['shapefile']
    shapefile_path = os.path.join(output_dir, shapefile.filename)
    shapefile.save(shapefile_path)
    return jsonify({"message": "Shapefile uploaded successfully!"})

@app.route('/run_inference', methods=['POST'])
def run_inference():
    # Here, you'll load the model and run your inference, as you did in PyQt5.
    threshold = request.json['threshold']  # Get the threshold from the front end
    model = load_model_with_custom_objects(model_path)
    # Call your inference function similar to your PyQt5 logic.
    return jsonify({"message": "Inference complete!"})

if __name__ == '__main__':
    app.run(debug=True)
