import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from EcoLytix.Packages import *

import numpy as np
import rasterio
import geopandas as gpd
import tensorflow as tf
from keras.models import load_model
from dotenv import load_dotenv
from rasterio.windows import Window
from rasterio.features import shapes
import fiona
from fiona.crs import from_epsg
from shapely.geometry import shape, mapping
import threading  # For handling long-running tasks without freezing the UI
from tkinter import ttk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Directories and model path
data_dir = os.getenv('RHODODENDRON-DATASET-PRESPLIT')
out_dir = os.getenv('RHODODENDRON-DATASET-OUTPUT')
model_path = os.getenv('RHODODENDRON-DEEPLAB-MODEL')

# Paths to raster and shapefile
raster_path = os.path.join(data_dir, 'Coillte_Multispectral.tif')
output_dir = os.path.join(out_dir, 'inferenced')

threshold = 0.5

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Parameters
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 244, 244, 6
grid_size = 244

# Load the model
model = load_model(model_path, custom_objects={'combined_loss': combined_loss, 'dice_coefficient': dice_coefficient, 'f2_score': f2_score, 'pixel_accuracy': pixel_accuracy, 'mean_iou': mean_iou})

# Create Tkinter window
root = tk.Tk()
root.title("Inference stitcher")
root.geometry("800x600")

# Set window icon (you need to specify your image path)
icon_path = os.getenv('ECOLYTIX-LOGO-ICON')  # Replace with the correct icon path
root.iconbitmap(icon_path)

# Apply dark theme
style = ttk.Style()
style.theme_use('clam')
style.configure('TFrame', background='#333')
style.configure('TLabel', background='#333', foreground='white')
style.configure('TProgressbar', troughcolor='#555', background='#00cc66')

# Create frame and progress bar
frame = ttk.Frame(root, padding=10)
frame.pack(fill=tk.BOTH, expand=True)
progress = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=600, mode='determinate')
progress.pack(pady=10)

# Initialize figure for predictions
fig, ax = plt.subplots(figsize=(5, 5))
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Function to process the raster and run inference with progress update
def process_and_inference_raster():
    with rasterio.open(raster_path) as src:
        crs = src.crs
        transform = src.transform
        raster_height, raster_width = src.height, src.width
        stitched_result = np.zeros((raster_height, raster_width), dtype=np.float32)

        num_steps = (raster_width // grid_size) * (raster_height // grid_size)
        current_step = 0

        for i in range(0, raster_width, grid_size):
            for j in range(0, raster_height, grid_size):
                window_width = min(grid_size, raster_width - i)
                window_height = min(grid_size, raster_height - j)
                window = Window(i, j, window_width, window_height)
                cell_image = src.read(window=window)

                # Preprocess and run inference
                processed_image = preprocess_image(cell_image)
                prediction = model.predict(processed_image)[0].squeeze()
                binary_prediction = (prediction >= threshold).astype(int)

                # Update stitched result
                stitched_result[j:j+window_height, i:i+window_width] = binary_prediction[:window_height, :window_width]

                # Update plot and progress
                ax.clear()
                ax.imshow(stitched_result, cmap='gray')
                canvas.draw()
                current_step += 1
                progress['value'] = (current_step / num_steps) * 100
                root.update_idletasks()

        # Save results
        output_path = os.path.join(output_dir, 'stitched_prediction.tif')
        write_stitched_result(stitched_result, src, output_path)

        shapefile_output_path = os.path.join(output_dir, 'prediction_shapefile.shp')
        raster_to_shapefile(stitched_result, transform, crs.to_epsg(), shapefile_output_path)
        display_shapefile(shapefile_output_path)

# Function to preprocess the image for inference
def preprocess_image(cell_image):
    num_channels = cell_image.shape[0]
    cell_image_resized = np.zeros((num_channels, grid_size, grid_size), dtype=cell_image.dtype)
    window_height, window_width = cell_image.shape[1], cell_image.shape[2]
    cell_image_resized[:, :window_height, :window_width] = cell_image

    # Normalize the image
    for channel in range(num_channels):
        channel_image = cell_image_resized[channel]
        cell_min = channel_image.min()
        cell_max = channel_image.max()
        if cell_min != cell_max:
            cell_image_resized[channel] = (channel_image - cell_min) / (cell_max - cell_min) * 255
        else:
            cell_image_resized[channel] = np.zeros_like(channel_image)

    cell_image_resized = np.clip(cell_image_resized, 0, 255).astype(np.uint8)
    cell_image_resized = tf.image.resize(cell_image_resized.transpose(1, 2, 0), (IMG_HEIGHT, IMG_WIDTH))
    return np.expand_dims(cell_image_resized / 255.0, axis=0)

# Function to write stitched result to a file
def write_stitched_result(stitched_result, src, output_path):
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=stitched_result.shape[0],
        width=stitched_result.shape[1],
        count=1,
        dtype=stitched_result.dtype,
        crs=src.crs,
        transform=src.transform
    ) as dst:
        dst.write(stitched_result, 1)

# Function to convert raster to shapefile
def raster_to_shapefile(raster_array, transform, crs, shapefile_path):
    shapes_gen = shapes(raster_array, transform=transform)
    schema = {'geometry': 'Polygon', 'properties': {'class': 'int'}}

    with fiona.open(shapefile_path, 'w', 'ESRI Shapefile', schema=schema, crs=from_epsg(crs)) as shp:
        for geom, value in shapes_gen:
            if int(value) == 1:  # Only write shapes for rhododendron areas
                shp.write({
                    'geometry': mapping(shape(geom)),
                    'properties': {'class': 1}  # Class 1 for rhododendron
                })

# Function to display the shapefile
def display_shapefile(shapefile_path):
    gdf = gpd.read_file(shapefile_path)
    fig, ax = plt.subplots()
    gdf.plot(ax=ax, color='blue')
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Run the inference process in a separate thread to avoid freezing the UI
def run_inference():
    threading.Thread(target=process_and_inference_raster).start()

# Add a button to start the process
start_button = ttk.Button(frame, text="Start Inference", command=run_inference)
start_button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
