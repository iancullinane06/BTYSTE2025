import os
import sys
from mrcnn.config import Config

class TreeConfig(Config):
    NAME = "tree"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + tree
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9

# Add the Mask R-CNN directory to the path
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)

from mrcnn import model as modellib, utils

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
