import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from EcoLytix.Packages import *

import numpy as np
import rasterio
import geopandas as gpd
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from dotenv import load_dotenv
from rasterio.windows import Window
from rasterio.features import shapes
import fiona
from fiona.crs import from_epsg
from shapely.geometry import shape, mapping
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

model = load_model(model_path, custom_objects={'combined_loss': combined_loss, 'dice_coefficient': dice_coefficient})

# Enable interactive mode
plt.ion()

# Function to process the raster, run inference, and generate shapefile
def process_and_inference_raster():
    with rasterio.open(raster_path) as src:
        crs = src.crs
        transform = src.transform
        raster = src.read()
        raster_height, raster_width = raster.shape[1], raster.shape[2]
        num_channels = raster.shape[0]

        # Initialize stitched result
        stitched_result = np.zeros((raster_height, raster_width), dtype=np.float32)

        # Initialize figure for dynamic updates
        fig, axes, text = initialize_plot(num_channels)

        for i in range(0, raster_width, grid_size):
            for j in range(0, raster_height, grid_size):
                window_width = min(grid_size, raster_width - i)
                window_height = min(grid_size, raster_height - j)
                window = Window(i, j, window_width, window_height)
                cell_image = src.read(window=window)

                # Resize the cell image if it's smaller than the grid size (edge case)
                cell_image_resized = np.zeros((num_channels, grid_size, grid_size), dtype=cell_image.dtype)
                cell_image_resized[:, :window_height, :window_width] = cell_image

                # Normalize and preprocess the image for inference
                processed_image = preprocess_image(cell_image_resized)

                # Run inference
                prediction = model.predict(processed_image)[0]
                prediction = prediction.squeeze()

                # Update the existing plot with new data and tile location
                update_plot(axes, text, cell_image_resized, prediction, (i, j))

                # Apply threshold to convert the prediction to binary
                binary_prediction = (prediction >= threshold).astype(int)

                # Resize the binary prediction to fit back into the main result
                prediction_resized = binary_prediction[:window_height, :window_width]
                stitched_result[j:j+window_height, i:i+window_width] = prediction_resized

        # Write stitched result and create a vector shapefile
        output_path = os.path.join(output_dir, 'stitched_prediction.tif')
        write_stitched_result(stitched_result, src, output_path)

        # Convert binary raster to shapefile and display
        shapefile_output_path = os.path.join(output_dir, 'prediction_shapefile.shp')
        raster_to_shapefile(stitched_result, transform, crs.to_epsg(), shapefile_output_path)
        display_shapefile(shapefile_output_path)
        
    # Enable interactive mode
    plt.ion()

# Initialize a single figure and axes for the input channels and prediction, including text for the location
def initialize_plot(num_channels):
    fig, axes = plt.subplots(2, num_channels, figsize=(20, 5))

    for channel in range(num_channels):
        axes[0, channel].imshow(np.zeros((grid_size, grid_size)), cmap='gray')
        axes[0, channel].set_title(f'Input Channel {channel + 1}')
        axes[0, channel].axis('off')

    axes[1, 0].imshow(np.zeros((grid_size, grid_size)), cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Prediction')
    axes[1, 0].axis('off')

    # Create a text annotation for displaying tile location
    text = fig.text(0.5, 0.95, '', ha='center', fontsize=12)

    plt.tight_layout()
    plt.show()

    return fig, axes, text

# Update the existing plot with new data, including the location text
def update_plot(axes, text, cell_image, prediction, location):
    num_channels = cell_image.shape[0]

    # Update input channels
    for channel in range(num_channels):
        axes[0, channel].images[0].set_data(cell_image[channel, :, :])

    # Update prediction
    axes[1, 0].images[0].set_data(prediction)

    # Update the tile location text
    text.set_text(f"Processing tile at location: {location}")

    plt.draw()
    plt.pause(0.5)  # Pause to allow update

# Preprocess image for inference
def preprocess_image(cell_image):
    # Normalize each channel separately if it has a valid range
    for channel in range(cell_image.shape[0]):
        channel_image = cell_image[channel]
        cell_min = channel_image.min()
        cell_max = channel_image.max()
        if cell_min != cell_max:
            cell_image[channel] = (channel_image - cell_min) / (cell_max - cell_min) * 255
        else:
            cell_image[channel] = np.zeros_like(channel_image)

    # Clip the values to 0-255 and convert to uint8
    cell_image = np.clip(cell_image, 0, 255).astype(np.uint8)
    
    # Resize and normalize the image for the model
    cell_image = tf.image.resize(cell_image.transpose(1, 2, 0), (IMG_HEIGHT, IMG_WIDTH))  # Resize to model input
    cell_image = cell_image / 255.0  # Normalize
    return np.expand_dims(cell_image, axis=0)


# Save the stitched result with the original raster's spatial information
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

# Convert the thresholded raster to shapefile
def raster_to_shapefile(raster_array, transform, crs, shapefile_path):
    shapes_gen = shapes(raster_array, transform=transform)

    # Prepare shapefile schema
    schema = {
        'geometry': 'Polygon',
        'properties': {'class': 'int'}
    }

    # Write to shapefile
    with fiona.open(shapefile_path, 'w', 'ESRI Shapefile', schema=schema, crs=from_epsg(crs)) as shp:
        for geom, value in shapes_gen:
            shp.write({
                'geometry': mapping(shape(geom)),
                'properties': {'class': int(value)}
            })

# Display the shapefile using Tkinter
def display_shapefile(shapefile_path):
    root = tk.Tk()
    root.title("Shapefile Viewer")

    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Plot the shapefile
    fig, ax = plt.subplots()
    gdf.plot(ax=ax, color='blue')

    # Display it in a Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    root.mainloop()

# Run the raster processing and inference
process_and_inference_raster()
