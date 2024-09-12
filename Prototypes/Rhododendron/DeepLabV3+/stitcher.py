import os
import numpy as np
import rasterio
from shapely.geometry import box
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

# Load model
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def combined_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred) + tf.keras.losses.binary_crossentropy(y_true, y_pred)

model = load_model(model_path, custom_objects={'combined_loss': combined_loss, 'dice_coefficient': dice_coefficient})

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

                # Debugging: Check range of prediction values
                print(f"Prediction range for tile at ({i}, {j}): {prediction.min()} - {prediction.max()}")

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

# Preprocess image for inference
def preprocess_image(cell_image):
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
