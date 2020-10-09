# Built-in imports
import sys

# External imports
import numpy as np
import pandas as pd
from mrcnn.config import Config
from mrcnn.utils import Dataset
from mrcnn.model import MaskRCNN
from mrcnn import visualize
from sklearn.model_selection import train_test_split

# Local imports
from config import ROOT_DIR
from config import MODEL_DIR # New trained models will be auto-saved here
from config import COCO_MODEL_PATH
from config import TRAIN_IMAGES_DIR
from config import TEST_IMAGES_DIR


def run_length_encoded_to_mask(encoded_pixels, h, w):
    """ Transform a line of Run Length Encoded pixels into a binary mask """
    
    encoded_pixels = np.asarray(encoded_pixels.split(), dtype=np.int)
    starts = encoded_pixels[0::2]
    lengths = encoded_pixels[1::2]

    linear_mask = np.zeros(w*h, dtype=np.uint8)
    for start,length in zip(starts, lengths):
        linear_mask[start:start+length] = 1
    
    mask = linear_mask.reshape(w,h).transpose()

    return mask


class SteelConfig(Config):
    """
    """
    BACKBONE = 'resnet50'
    NAME = 'steel'
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


class SteelDataset(Dataset):
    """
    """

    def __init__(self, df):
        
        # Call the original Dataset' __init__
        super().__init__(self)
        
        self.df = df


    def load_dataset(self, images_ids, imgs_folder):
        """
        """

        for i in range(1,5):
            self.add_class(source='', class_id=i, class_name=f'type_{i}')
    
        for im_id in images_ids:
            file_name = im_id
            file_path = imgs_folder / file_name
            assert file_path.exists(), "File doesn't exists."
                        
            self.add_image(source='', 
                           image_id=file_name, 
                           path=str(file_path)
            )


    def load_mask(self, image_id):
        """
        Load instance masks for the given image.

        Returns:
            masks    : An array of shape [height, width, instance count] 
                       with a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        
        # Each row corresponds to one defect class, if more than one is available
        subdf = self.df.loc[self.df.ImageId == image_id, :].copy()
        subdf.reset_index(drop=True, inplace=True)

        # Images' size
        w,h = (1600, 256)

        # Collection of masks with shape:
        # [height, width, instance_count]
        mask_tensor = np.zeros([h, w, subdf.shape[0]], dtype=np.uint8)

        for i, row in subdf.iterrows():
            encoded_pixels = row['EncodedPixels']
            mask = run_length_encoded_to_mask(encoded_pixels, h, w)
            mask_tensor[:, :, i] = mask

        class_ids = subdf['ClassId'].to_numpy()

        return mask_tensor.astype(np.bool), class_ids


if __name__ == '__main__':
    
    train_images = str(ROOT_DIR/'train_images')
    test_images = str(ROOT_DIR/'test_images')
    
    data = pd.read_csv(str(ROOT_DIR/'train.csv'))

    # Keep only the images with defects
    df_defects = data.dropna(subset=['EncodedPixels'], axis=0).copy()

    image_ids = df_defects.ImageId.unique()
    train_ids, val_ids = train_test_split(image_ids, test_size=0.3, random_state=42)

    data_train = SteelDataset(df_defects)
    data_train.load_dataset(train_ids, TRAIN_IMAGES_DIR)
    data_train.prepare()

    data_val = SteelDataset(df_defects)
    data_train.load_dataset(val_ids, TRAIN_IMAGES_DIR) # val images are in the train dir
    data_val.prepare()

    config = SteelConfig()
    model = MaskRCNN(mode="training", config=config, model_dir=str(MODEL_DIR))
    
    # Exclude the last layers because they require a matching number of classes
    # ^ Original comment at:
    # https://github.com/matterport/Mask_RCNN/blob/master/samples/balloon/balloon.py
    model.load_weights(str(COCO_MODEL_PATH),
                       by_name=True,
                       exclude=[
                           "mrcnn_class_logits", 
                           "mrcnn_bbox_fc", 
                           "mrcnn_bbox", 
                           "mrcnn_mask", 
                       ])

    model.train(
        data_train,
        data_val,
        learning_rate=config.LEARNING_RATE,
        epochs=30, 
        layers="heads"
    )