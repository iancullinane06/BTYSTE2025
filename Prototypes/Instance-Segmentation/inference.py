import os
import sys
import numpy as np
import skimage.io
import tree_config
import tree_dataset
from mrcnn import model as modellib, visualize

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
WEIGHTS_PATH = "path/to/your/trained/model.h5"

# Configuration
class InferenceConfig(tree_config.TreeConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model in inference mode
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)

# Load trained weights
model.load_weights(WEIGHTS_PATH, by_name=True)

# Load a random image from the dataset
image = skimage.io.imread("path/to/an/image.jpg")

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            ['BG', 'tree'], r['scores'])
