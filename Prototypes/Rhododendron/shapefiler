import numpy as np
import rasterio
from shapely.geometry import box
from skimage.io import imsave
import os
import geopandas as gpd
from dotenv import load_dotenv
from tkinter import Tk, Label, Frame
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import queue

# Load environment variables
load_dotenv()

data_dir = os.getenv('RHODODENDRON-DATASET-PRESPLIT')
out_dir = os.getenv('RHODODENDRON-DATASET-OUTPUT')

# Paths to files and directories
raster_path = os.path.join(data_dir, 'Coillte_Multispectral.tif')
shapefile_path = os.path.join(data_dir, 'Rhodo_mapping.shp')
map_raster_path = os.path.join(data_dir, 'Rhodo_label.tif')
output_image_dir = os.path.join(out_dir, 'images')
output_map_dir = os.path.join(out_dir, 'maps')

grid_size = 244
crop_size = 122  # Half of the grid size
total_images = 0
rhododendron_images = 0
non_rhododendron_images = 0

# Create output directories if they don't exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_map_dir, exist_ok=True)

def load_shapefile():
    global gdf, crs
    try:
        with rasterio.open(raster_path) as src:
            crs = src.crs  # Get CRS from raster

        gdf = gpd.read_file(shapefile_path)
        gdf = gdf.to_crs(crs)  # Ensure CRS match
        print(f"Successfully loaded shapefile with {len(gdf)} features.")
    except Exception as e:
        print(f"Error reading the shapefile: {e}")
        raise

def is_artifact(cell_image, extreme_threshold=0.8):
    """
    Determine if the image is an artifact based on extreme values.
    
    Parameters:
        cell_image (numpy array): The image data with multiple channels.
        extreme_threshold (float): Threshold proportion for detecting artifacts.
    
    Returns:
        bool: True if the image is an artifact, False otherwise.
    """

    height, width = cell_image.shape
    total_pixels = height * width
    num_extreme = np.sum((cell_image == 0) | (cell_image == 254))
    
    # Calculate the proportion of extreme values
    proportion_extreme = num_extreme / total_pixels
    
    # Print diagnostic information
    print(f"Number of extreme values: {num_extreme}")
    print(f"Total number of values: {total_pixels}")
    print(f"Proportion of extreme values: {proportion_extreme}")
    print(f"Threshold: {extreme_threshold}")
    
    return proportion_extreme > extreme_threshold


def process_images():
    global total_images, rhododendron_images, non_rhododendron_images

    try:
        with rasterio.open(raster_path) as src:
            crs = src.crs  # Ensure CRS is captured
            transform = src.transform
            raster = src.read()
            num_channels, raster_height, raster_width = raster.shape
            print(f"Raster has {num_channels} channels, dimensions: {raster_width}x{raster_height}")

            # Load the map raster
            with rasterio.open(map_raster_path) as map_src:
                global map_data
                map_data = map_src.read(1)  # Assuming single band raster

            # Define the bounding box of the raster
            raster_bounds = src.bounds

            # Process the entire grid normally
            image_count = 0
            for i in range(0, raster_width, grid_size):
                for j in range(0, raster_height, grid_size):
                    # Define the grid cell boundaries
                    minx = raster_bounds.left + i * transform[0]
                    maxx = minx + grid_size * transform[0]
                    maxy = raster_bounds.top - j * abs(transform[4])
                    miny = maxy - grid_size * abs(transform[4])

                    cell_bounds = box(minx, miny, maxx, maxy)

                    # Extract the grid cell image
                    window = rasterio.windows.Window(i, j, grid_size, grid_size)
                    cell_image = src.read(window=window)

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

                    # Check if the cell image is an artifact
                    if is_artifact(np.mean(cell_image, axis=0)):
                        print(f"Skipping artifact image at ({i}, {j})")
                        continue  # Skip saving this image

                    # Display the current cell image channels
                    display_channels(cell_image, frame, image_count)

                    # Check for rhododendron presence
                    intersects = gdf.intersects(cell_bounds).any()
                    
                    # Print debugging information
                    print(f"Cell bounds: {cell_bounds}")
                    print(f"Intersection check: {'rhododendron' if intersects else 'non_rhododendron'}")

                    # Save the image if it's not an artifact
                    file_name = f"{i}_{j}.tif"
                    output_path = os.path.join(output_image_dir, file_name)
                    print(f"Saving image to: {output_path}")  # Debug print
                    transposed_image = cell_image.transpose(1, 2, 0)
                    imsave(output_path, transposed_image, plugin="tifffile", photometric='minisblack', planarconfig='contig')

                    # Save the corresponding map
                    map_output_path = os.path.join(output_map_dir, file_name)
                    print(f"Saving map to: {map_output_path}")  # Debug print
                    map_cell = map_data[j:j+grid_size, i:i+grid_size]
                    imsave(map_output_path, map_cell, plugin="tifffile", photometric='minisblack', planarconfig='contig')

                    total_images += 1
                    if intersects:
                        rhododendron_images += 1
                    else:
                        non_rhododendron_images += 1
                    image_count += 1

            # Crop the entire image
            crop_window = rasterio.windows.Window(crop_size, crop_size, raster_width - 2*crop_size, raster_height - 2*crop_size)
            cropped_raster = src.read(window=crop_window)

            # Define new raster bounds after cropping
            cropped_minx = raster_bounds.left + crop_size * transform[0]
            cropped_maxx = cropped_minx + (raster_width - 2*crop_size) * transform[0]
            cropped_maxy = raster_bounds.top - crop_size * abs(transform[4])
            cropped_miny = cropped_maxy - (raster_height - 2*crop_size) * abs(transform[4])

            cropped_raster_bounds = (cropped_minx, cropped_miny, cropped_maxx, cropped_maxy)

            # Re-divide the cropped image into grid cells
            for i in range(0, raster_width - 2*crop_size, grid_size):
                for j in range(0, raster_height - 2*crop_size, grid_size):
                    # Define the grid cell boundaries
                    minx = cropped_raster_bounds[0] + i * transform[0]
                    maxx = minx + grid_size * transform[0]
                    maxy = cropped_raster_bounds[3] - j * abs(transform[4])
                    miny = maxy - grid_size * abs(transform[4])

                    cell_bounds = box(minx, miny, maxx, miny)

                    # Extract the grid cell image from the cropped raster
                    window = rasterio.windows.Window(crop_size + i, crop_size + j, grid_size, grid_size)
                    cell_image = cropped_raster[:, j:j+grid_size, i:i+grid_size]

                    # Check if the cell image is an artifact
                    if is_artifact(np.mean(cell_image, axis=0)):
                        print(f"Skipping artifact cropped image at ({i}, {j})")
                        continue  # Skip saving this image

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

                    # Display the current cell image channels
                    display_channels(cell_image, frame, image_count)

                    # Check for rhododendron presence
                    intersects = gdf.intersects(cell_bounds).any()
                    
                    # Print debugging information
                    print(f"Cell bounds: {cell_bounds}")
                    print(f"Intersection check: {'rhododendron' if intersects else 'non_rhododendron'}")

                    # Save the image if it's not an artifact
                    file_name = f"cropped_{i}_{j}.tif"
                    output_path = os.path.join(output_image_dir, file_name)
                    print(f"Saving cropped image to: {output_path}")  # Debug print
                    transposed_image = cell_image.transpose(1, 2, 0)
                    imsave(output_path, transposed_image, plugin="tifffile", photometric='minisblack', planarconfig='contig')

                    # Save the corresponding map
                    map_output_path = os.path.join(output_map_dir, file_name)
                    print(f"Saving cropped map to: {map_output_path}")  # Debug print
                    map_cell = map_data[crop_size + j:crop_size + j + grid_size, crop_size + i:crop_size + i + grid_size]
                    imsave(map_output_path, map_cell, plugin="tifffile", photometric='minisblack', planarconfig='contig')

                    total_images += 1
                    image_count += 1

    except Exception as e:
        print(f"Error processing raster: {e}")
        raise

    print("Processing complete.")
    print(f"Total images: {total_images}")
    print(f"Rhododendron images: {rhododendron_images}")
    print(f"Non-rhododendron images: {non_rhododendron_images}")

def display_channels(cell_image, frame, image_count):
    global label_rgb, labels
    
    # Merge the first three channels into an RGB image
    red_channel = cell_image[0]
    green_channel = cell_image[1]
    blue_channel = cell_image[2]
    
    rgb_image = np.stack((red_channel, green_channel, blue_channel), axis=-1)
    
    # Check if the image is an artifact (only 0 and 255 values)
    is_artifact = np.all((rgb_image == 0) | (rgb_image == 255))
    
    # If artifact, set background to grey
    if is_artifact:
        frame.configure(bg='grey')
        rgb_image[:, :, :] = 128  # Set all values to grey
    else:
        frame.configure(bg='white')
    
    # Convert to PIL Image and Tkinter-compatible format
    rgb_image_pil = Image.fromarray(rgb_image, 'RGB')
    rgb_imgtk = ImageTk.PhotoImage(image=rgb_image_pil)
    
    # Update Tkinter labels
    label_rgb.config(image=rgb_imgtk)
    label_rgb.image = rgb_imgtk  # Keep a reference to avoid garbage collection
    
    # Normalize and prepare the remaining channels
    channels = []
    for channel_index in range(3, 6):
        channel_data = cell_image[channel_index]
        channel_min = channel_data.min()
        channel_max = channel_data.max()
        if channel_min != channel_max:
            normalized_channel = (channel_data - channel_min) / (channel_max - channel_min) * 255
        else:
            normalized_channel = np.zeros_like(channel_data)  # Handle channels with no variation
        normalized_channel = np.clip(normalized_channel, 0, 255).astype(np.uint8)
        channels.append(normalized_channel)

    # Convert arrays to PIL images
    channel_images_pil = [Image.fromarray(ch, 'L') for ch in channels]

    # Convert images to Tkinter-compatible format
    channel_imgtk = [ImageTk.PhotoImage(image=ch) for ch in channel_images_pil]

    # Update the labels with new images
    if image_count % 5 == 0:  # Update every 5 images
        for idx, img in enumerate(channel_imgtk):
            labels[idx].config(image=img)
            labels[idx].image = img  # Keep a reference to avoid garbage collection

        # Update the display
        root.update_idletasks()

def update_gui():
    if not image_queue.empty():
        cell_image, image_count = image_queue.get()
        display_channels(cell_image, frame, image_count)
    root.after(100, update_gui)

# Set up Tkinter window
root = Tk()
root.title("Image Viewer")

# Create a dark theme style
style = ttk.Style()
style.theme_use('clam')
style.configure('TLabel', background='white', foreground='black')

# Create a frame for image display
frame = Frame(root)
frame.pack()

# Create labels for displaying images
label_rgb = Label(frame)
label_rgb.pack(side='left', padx=5)

labels = [Label(frame) for _ in range(3)]
for label in labels:
    label.pack(side='left', padx=5)

# Queue for inter-thread communication
image_queue = queue.Queue()

# Load shapefile
load_shapefile()

# Start image processing in a separate thread
processing_thread = threading.Thread(target=process_images, daemon=True)
processing_thread.start()

# Start the GUI update loop
root.after(100, update_gui)
root.mainloop()
