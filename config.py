
from pathlib import Path
from mrcnn.config import Config

ROOT_DIR = Path('E:\Datasets\steel-defects-detection')

# Directory to save logs and trained model
MODEL_DIR = ROOT_DIR / 'models'

# Local path to trained weights file
COCO_MODEL_PATH = ROOT_DIR / "mask_rcnn_coco.h5"

# Path towards the directory with the train images
TRAIN_IMAGES_DIR = ROOT_DIR/'train_images'

# Path towards the directory with the test images
TEST_IMAGES_DIR = ROOT_DIR/'test_images'

CSV_FILE = ROOT_DIR/'train.csv'


class SteelConfig(Config):
    """
    """
    NAME = 'steel'
    BACKBONE = 'resnet50'
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100