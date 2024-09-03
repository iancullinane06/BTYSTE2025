import os
import sys
import numpy as np
import tree_config
import tree_dataset
from mrcnn import model as modellib
from mrcnn.config import Config
from dotenv import load_dotenv

class TreeConfig(Config):
    NAME = "tree"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + tree
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9

from mrcnn import model as modellib, utils

# Load environment variables
load_dotenv()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(os.getenv("LOG-DIR"), "logs")

# Path to trained weights file
WEIGHTS_PATH = os.getenv("TREE-MODEL")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join("mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Configuration
config = tree_config.TreeConfig()
config.display()

# Dataset
dataset_train = tree_dataset.TreeDataset()
dataset_train.load_tree("path/to/your/dataset", "train")
dataset_train.prepare()

dataset_val = tree_dataset.TreeDataset()
dataset_val.load_tree("path/to/your/dataset", "val")
dataset_val.prepare()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

# Load pre-trained weights (except for the output layers)
model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# Train the model
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=30,
            layers='heads')
