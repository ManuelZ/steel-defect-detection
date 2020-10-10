"""
Steel Defect Detection with Mask-RCNN - Inference
Script that displays an image with a predicted mask of defects.
------------------------------------------------------------
Usage:
    # Show an image and its predicted defects-mask as an overlay
    python inference.py --image E:\Datasets\steel-defects-detection\train_images\00b989e78.jpg
"""

# Built-in imports
import argparse
import datetime

# External imports
from mrcnn.model import MaskRCNN
import skimage
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from config import SteelConfig
from config import MODEL_DIR # Where trained weights were saved


# Parse command line arguments
parser = argparse.ArgumentParser(description='Detect defects in steel')
parser.add_argument(
    '--image',
    required=True,
    metavar="path to image",
    help='Image where to detect defects'
)
args = parser.parse_args()


class InferenceConfig(SteelConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

steel_config = InferenceConfig()

model = MaskRCNN(
    mode="inference",
    config=steel_config,
    model_dir=str(MODEL_DIR)
)

print(f"Running on {args.image}")

image = skimage.io.imread(args.image)

# Detect objects
r = model.detect([image], verbose=1)[0]

# Extract masks
mask = r['masks']

# Plot image with a transparent overlay of the mask
fig,axs = plt.subplots(1, 1, figsize=(30,30))
plt.imshow(image, cmap='gray')
plt.imshow(mask[:,:,0], cmap='jet', alpha=0.5)
plt.show()
