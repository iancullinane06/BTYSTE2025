import os
import sys
import numpy as np
import rasterio
import geopandas as gpd
import tensorflow as tf
from keras.models import load_model
from rasterio.windows import Window
from rasterio.features import shapes
import fiona
from fiona.crs import from_epsg
from shapely.geometry import shape, mapping
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QSlider, QFileDialog, QProgressBar, QMessageBox, QSizePolicy)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QIcon
import matplotlib.pyplot as plt


# Parameters
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 244, 244, 6
grid_size = 244
threshold = 0.5
selected_shapefile = None
model_path = None  # This will be set dynamically from file dialog
accent_colour = "#00adfe"

output_dir = os.path.join('inferenced')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


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


def load_model_with_custom_objects(model_path):
    return load_model(model_path, custom_objects={
    'combined_loss': combined_loss,
    'dice_coefficient': dice_coefficient,
    'f2_score': f2_score,
    'pixel_accuracy': pixel_accuracy,
    'mean_iou': mean_iou
})

class InferenceThread(QThread):
    progress_update = pyqtSignal(int)
    eta_update = pyqtSignal(str)
    shapefile_update = pyqtSignal(str)  # New signal for shapefile updates
    
    def __init__(self, raster_path, output_dir, threshold, model):
        super().__init__()
        self.raster_path = raster_path
        self.output_dir = output_dir
        self.threshold = threshold
        self.model = model

    def run(self):
        with rasterio.open(self.raster_path) as src:
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
                    processed_image = self.preprocess_image(cell_image)
                    prediction = self.model.predict(processed_image)[0].squeeze()
                    binary_prediction = (prediction >= self.threshold).astype(int)

                    # Update stitched result
                    stitched_result[j:j+window_height, i:i+window_width] = binary_prediction[:window_height, :window_width]

                    # Update progress
                    current_step += 1
                    progress = int((current_step / num_steps) * 100)
                    self.progress_update.emit(progress)
                    self.eta_update.emit("ETA: Calculating...")  # Placeholder for ETA

                    # Emit shapefile update periodically
                    if current_step % (num_steps // 10) == 0:  # Update every 10% progress
                        self.save_and_emit_shapefile(stitched_result, src)

            # Final save
            output_path = os.path.join(self.output_dir, 'stitched_prediction.tif')
            self.write_stitched_result(stitched_result, src, output_path)

            shapefile_output_path = os.path.join(self.output_dir, 'prediction_shapefile.shp')
            self.raster_to_shapefile(stitched_result, transform, crs.to_epsg(), shapefile_output_path)
            self.shapefile_update.emit(shapefile_output_path)  # Emit final shapefile path

    def save_and_emit_shapefile(self, stitched_result, src):
        temp_shapefile_path = os.path.join(self.output_dir, 'temp_shapefile.shp')
        self.raster_to_shapefile(stitched_result, src.transform, src.crs.from_epsg(src.crs.to_epsg()), temp_shapefile_path)
        self.shapefile_update.emit(temp_shapefile_path)

    def display_shapefile(self, shapefile_path):
        gdf = gpd.read_file(shapefile_path)
        fig, ax = plt.subplots()
        gdf.plot(ax=ax, color='blue')
        plt.close(fig)
        img = self.fig2img(fig)
        self.display_image(img)
        
    def preprocess_image(self, cell_image):
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

    def write_stitched_result(self, stitched_result, src, output_path):
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

    def raster_to_shapefile(self, raster_array, transform, crs, shapefile_path):
        # Check if crs is valid and extract the EPSG code, else fallback to CRS WKT or other format
        epsg_code = crs if isinstance(crs, int) else crs.to_epsg()

        if epsg_code is None:
            # If no EPSG is available, use the CRS WKT string
            fiona_crs = crs.to_wkt()
        else:
            fiona_crs = from_epsg(epsg_code)

        shapes_gen = shapes(raster_array, transform=transform)
        schema = {
            'geometry': 'Polygon',
            'properties': {'class': 'int'}
        }

        with fiona.open(shapefile_path, 'w', 'ESRI Shapefile', schema=schema, crs=fiona_crs) as shp:
            for geom, value in shapes_gen:
                if int(value) == 1:  # Only write shapes for rhododendron presence
                    shp.write({
                        'geometry': mapping(shape(geom)),
                        'properties': {'class': 1}
                    })


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Inference Stitcher')
        self.setFixedSize(1000, 600)
        self.setStyleSheet("background-color: #111;")  # Set background color to grey

        # Main layout
        main_layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Top layout (Left and Right frames)
        top_layout = QHBoxLayout()
        main_layout.addLayout(top_layout)

        # Left frame layout
        left_frame = QWidget()
        left_frame.setStyleSheet("background-color: #444; border-radius: 15px; text-align: center;")
        left_layout = QVBoxLayout()
        left_frame.setLayout(left_layout)
        top_layout.addWidget(left_frame, 1)  # Stretch factor of 1

        # Right frame layout
        self.right_frame = QWidget()
        self.right_frame.setStyleSheet("background-color: #444; border-radius: 10px; text-align: center;")
        self.right_layout = QVBoxLayout()
        self.right_frame.setLayout(self.right_layout)
        top_layout.addWidget(self.right_frame, 1)  # Stretch factor of 1

        # Logo and description
        logo_label = QLabel()
        pixmap = QPixmap('assets/logo.png')  # Path to your logo (relative)
        logo_label.setPixmap(pixmap.scaled(QSize(400, 100), Qt.KeepAspectRatio))
        left_layout.addWidget(logo_label)

        description_label = QLabel("This application performs inference on a raster dataset ...")
        description_label.setStyleSheet("color: white;")
        description_label.setWordWrap(True)
        left_layout.addWidget(description_label)

        # Threshold slider
        threshold_label = QLabel("Threshold:")
        threshold_label.setStyleSheet("color: white;")
        left_layout.addWidget(threshold_label)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 10)
        self.threshold_slider.setValue(5)
        self.threshold_slider.setStyleSheet(f"QSlider::groove:horizontal {{ background: #555; height: 10px; }} "
                                             f"QSlider::handle:horizontal {{ background: {accent_colour}; border-radius: 5px; width: 15px; height: 15px; }}")
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        left_layout.addWidget(self.threshold_slider)

        self.threshold_value_label = QLabel("0.5")
        self.threshold_value_label.setStyleSheet("color: white;")
        left_layout.addWidget(self.threshold_value_label)

        # Shapefile button
        shapefile_button = QPushButton("Select Shapefile")
        shapefile_button.setStyleSheet(f"background-color: {accent_colour}; color: white; border-radius: 5px; padding: 5px; width: 50%;")
        shapefile_button.clicked.connect(self.open_shapefile)
        left_layout.addWidget(shapefile_button)

        # Model selection button
        model_button = QPushButton("Select Model")
        model_button.setStyleSheet(f"background-color: {accent_colour}; color: white; border-radius: 5px; padding: 5px; width: 50%;")
        model_button.clicked.connect(self.select_model)
        left_layout.addWidget(model_button)

        # Start inference button
        start_button = QPushButton("Start Inference")
        start_button.setStyleSheet(f"background-color: {accent_colour}; color: white; border-radius: 5px; padding: 5px; width: 50%;")
        start_button.clicked.connect(self.run_inference)
        left_layout.addWidget(start_button)

        # Current coordinate label
        self.current_coord_label = QLabel("Current Coordinate: ")
        self.current_coord_label.setStyleSheet("color: white;")
        left_layout.addWidget(self.current_coord_label)

        # Right frame - Preview area
        self.preview_label = QLabel("Preview")
        self.preview_label.setStyleSheet("color: white; font-size: 20px;")
        self.right_layout.addWidget(self.preview_label)

        # Bottom frame with progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(f"QProgressBar {{ border-radius: 5px; background: #555; }} QProgressBar::chunk {{ background: {accent_colour}; }}")
        main_layout.addWidget(self.progress_bar)

    def update_threshold_label(self):
        """Update the threshold value displayed."""
        value = self.threshold_slider.value() / 10.0
        self.threshold_value_label.setText(f"{value:.1f}")

    def open_shapefile(self):
        """Handle shapefile selection."""
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Shapefile", "", "Shapefiles (*.shp)", options=options)
        if file_path:
            global selected_shapefile
            selected_shapefile = file_path
            self.update_file_indicator(True)

    def select_model(self):
        file_dialog = QFileDialog()
        model_file, _ = file_dialog.getOpenFileName(self, "Select Model", "", "Model Files (*.h5)")
        if model_file:
            global model_path
            model_path = model_file
            QMessageBox.information(self, "Model Loaded", f"Model selected: {os.path.basename(model_file)}")



def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()