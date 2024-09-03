import os
import numpy as np
import skimage.io
from mrcnn.utils import Dataset

class TreeDataset(Dataset):
    def load_tree(self, dataset_dir, subset):
        self.add_class("tree", 1, "tree")
        # Add images
        for i, filename in enumerate(os.listdir(os.path.join(dataset_dir, subset))):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                self.add_image(
                    "tree",
                    image_id=i,
                    path=os.path.join(dataset_dir, subset, filename),
                    annotation_path=os.path.join(dataset_dir, subset, filename.replace(".jpg", ".json"))
                )

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotation_path = info['annotation_path']
        # Load the mask using your preferred method, e.g., reading polygons from a JSON file
        # masks = ...
        # class_ids = ...
        return masks, class_ids

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info["path"]
