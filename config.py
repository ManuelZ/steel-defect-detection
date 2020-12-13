
# Built-in imports
from pathlib import Path

# External imports
from mrcnn.config import Config


# Directory where the data and pre-trained weight files are located
ROOT_DIR = Path('/root/steel-defect-detection/data')

# Directory to save logs and trained model
MODEL_DIR = ROOT_DIR / 'models'

# Local path to trained weights file
COCO_MODEL_PATH = ROOT_DIR / "mask_rcnn_coco.h5"

# Path towards the directory with the train images
TRAIN_IMAGES_DIR = ROOT_DIR / 'train_images'

# Path towards the directory with the test images
TEST_IMAGES_DIR = ROOT_DIR / 'test_images'

# CSV file with all the training masks and information
CSV_FILE = ROOT_DIR / 'train.csv'


class SteelConfig(Config):
    """
    Standard way of configuring Mask-RCNN: subclass Config.
    """

    # A name is required
    NAME = 'steel'
    
    # Either 'resnet50' or 'resnet101'
    # With 8GB of RAM and a GTX1060 only resnet50 works, otherwise it
    # just quits without even starting.
    # With 16 GB pf RAM both resnet50 and resnet101 work
    BACKBONE = 'resnet50'
    
    # For GPUs with not so much memory, use 1
    IMAGES_PER_GPU = 1 

    # Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Number of training steps per epoch
    # Should actually be (number of examples // batch size)
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    VALIDATION_STEPS = 50

    # Parameter added by me
    NUM_EPOCHS = 5

    LEARNING_RATE = 0.0001
