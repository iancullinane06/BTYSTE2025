![EcoLytix-Icon](https://github.com/user-attachments/assets/541a3711-02e7-458d-87d9-c497837a6bc5)


This is the repository for my project for BTYSTE (the BT Young Scientist & Technology Exhibition). The goal of this project is to leverage machine learning techniques for environmental monitoring and analysis using image data. Specifically, I focus on the tasks of semantic segmentation and classification to identify specific plant species and detect forest fires. The datasets utilized in this project include those generously supplied by Coillte, the ADE20K dataset, and several others.

## Objectives

- **Semantic Segmentation:**  
  - **Early Stages:**  
    I experimented with semantic segmentation to perform statistical analysis on pixel values to identify tree species.
    This involved using the ADE20k dataset to segment trees then to use an SVM on those values. It was a poor idea however and didnèt work well.
    An instance segmentation model would have been necessary.
  - **Later Stages:**  
    I am currently developing models to manipulate semantic segmentation for the identification of rhododendron in multispectral image scans.
    I'm using semantic segmentation in the context of aerial images now nad producing masks that are inherently the plant it's trying to recognise.

- **Classification with ResNet:**  
  - I implemented ResNet models with 50 and 101 layers to classify whether a forest fire is present in an image. This work aims to enhance early detection and monitoring of forest fires. My model had issues with overfitting but overall a rather comprehensive model.
  I might try develop a semantic segmentation model for this next

## Datasets

- **Coillte Dataset:** Provided forestry data, including various tree species and multispectral images. This was the push I needed to start using shapefiles, so `.tiff` images with 6 colour bands predominantly.
- **ADE20K Dataset:** A large dataset used for semantic segmentation, containing diverse images with annotated objects.
- **Other Datasets:** Various datasets used for training and validating models.

## Methods & Prototypes

The project is structured around two main components: semantic segmentation and classification.

### Semantic Segmentation

- **Approach:**  
  Using deep learning models to identify and segment specific regions in images based on pixel analysis. This allows me to recognise different plants with ease and produce shapefiles that bound the regions I'm interested in.
- **Tools:**  
  - Leveraging pre-trained models and fine-tuning them for the specific task of identifying rhododendrons in multispectral images.
  - Performing pixel-level statistical analysis to enhance model accuracy in species identification.
  - DeepLabV3Plus for the model architecture.
  - Leveraging the Tensorflow framework and Keras as a high level interface to train my models.
  

### Classification with ResNet

- **Approach:**  
  Implementing ResNet models with 50 and 101 layers to classify images based on the presence of forest fires.
- **Tools:**  
  - Using pre-trained ResNet models and fine-tuning them with custom datasets to improve classification performance.

## Python Libraries Used

Ensure you have these libraries installed. You can download them using `pip3`.

- **OpenCV:** `opencv-python` for video processing
- **Pandas:** `pandas` for data manipulation
- **NumPy:** `numpy` for numerical computations
- **Scikit-Learn:** `sklearn` for building AI models
- **Seaborn:** `seaborn` for creating heatmaps
- **Matplotlib:** `matplotlib` for data visualization
- **Joblib:** `joblib` for saving and loading .pkl files
- **PyTorch:** `torch` for loading pre-trained models in early stages
- **TensorFlow:** `tensorflow` for building and training models in later stages
- **Keras:** `keras` as a high-level API for TensorFlow models
- **Python Dotenv:** `python-dotenv` for managing environment variables
- **Tifffile:** `tiffile` for handling TIFF files, especially useful for multispectral images
- **TKinter** `tkinter` for building early UI's
- **PyQt5** `PyQt5` for building later UI's
- **Threading** `threading` for running tkinter window separate
- **Rasterio** `rasterio` to load raster files
- **GeoPandas** `geopandas` to handle shapefiles
- **Shapely** `shapely` to save 
- **Fiona** `fiona`

## Installation

To set up the project environment, you can create a virtual environment and install the required libraries:

```bash
# Create a virtual environment
python3 -m venv skysci-env

# Activate the virtual environment
source skysci-env/bin/activate  # On Windows use `skysci-env\Scripts\activate`

# Install the required libraries
pip3 install opencv-python mediapipe pandas numpy scikit-learn seaborn matplotlib joblib torch tensorflow keras python-dotenv tifffile
