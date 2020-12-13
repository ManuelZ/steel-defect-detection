"""
Steel Defect Detection with Mask-RCNN - Inference
Script that displays an image with a predicted mask of defects.
------------------------------------------------------------
Usage:
    # Show an image and its predicted defects-mask as an overlay
    python inference.py --image ff6e35e0a.jpg
"""

# Built-in imports
import argparse
import datetime

# External imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import skimage
from mrcnn.model import MaskRCNN
from mrcnn import visualize
from matplotlib.patches import Patch


# Local imports
from config import SteelConfig
from config import InferenceConfig
from config import CSV_FILE
from config import MODEL_DIR # Where trained weights were saved
from train import run_length_encoded_to_mask
from config import TRAIN_IMAGES_DIR
from config import TEST_IMAGES_DIR
from config import DEFECTS
from config import CLASS_NAMES


# Parse command line arguments
parser = argparse.ArgumentParser(description='Detect defects in steel')
parser.add_argument(
    '--image',
    required = True,
    metavar  = "path to image",
    help     = 'Image where to detect defects'
)
args = parser.parse_args()


###############################################################################
# Load image 
###############################################################################

im_path = str(TRAIN_IMAGES_DIR / args.image)
print(f"Running on image '{args.image}'.")
original_im = cv2.imread(im_path)
original_im = cv2.cvtColor(original_im, cv2.COLOR_BGR2RGB)
im          = original_im.copy()


###############################################################################
# Detection 
###############################################################################

steel_config = InferenceConfig()

model = MaskRCNN(
    mode      = "inference",
    config    = steel_config,
    model_dir = str(MODEL_DIR)
)

# Run the detection pipeline
# images: List of images, potentially of different sizes.
# Returns a list of dicts, one dict per image. The dict contains:
#     rois      : [N, (y1, x1, y2, x2)] detection bounding boxes
#     class_ids : [N] int class IDs
#     scores    : [N] float probability scores for the class IDs
#     masks     : [H, W, N] instance binary masks
results = model.detect(images=[im], verbose=1)
r = results[0]

###############################################################################
# Visualization 
###############################################################################

visualize.display_instances(
    image       = im,
    boxes       = r['rois'],
    masks       = r['masks'],
    class_ids   = r['class_ids'], 
    scores      = r['scores'],
    class_names = CLASS_NAMES
)
