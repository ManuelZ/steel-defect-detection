"""
Steel Defect Detection with Mask-RCNN - Training
Script that trains Mask-RCNN on the Kaggle's Steel Defect Detection dataset to
identify four types of steel defects.
------------------------------------------------------------
Usage:
    # Train with COCO pre-trained weights
    python train.py --model default
    
    # Train with the last available trained weights
    python train.py --model last


To see the training scores:
    tensorboard --logdir=D:\Datasets\steel-defects-detection\models

And navigate to:
    localhost:6006
"""


# Built-in imports
import sys
import argparse

# External imports
import numpy as np
import pandas as pd
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
from config import CSV_FILE
from config import SteelConfig


def run_length_encoded_to_mask(encoded_pixels, h, w):
    """ Transform a line of Run Length Encoded pixels into a binary mask """

    # These integers will be used to fill the linear mask (by slicing an array
    # like: myarray[start:start+length]), they should have a large enough max 
    # value so that their sum doesn't overflow, hence uint32
    encoded_pixels = np.asarray(encoded_pixels.split(), dtype=np.uint32)
    starts = encoded_pixels[0::2]
    lengths = encoded_pixels[1::2]

    linear_mask = np.zeros(w*h, dtype=np.uint8)
    for start,length in zip(starts, lengths):
        linear_mask[start:start+length] = 1
    
    mask = linear_mask.reshape(w,h).transpose()

    return mask


class SteelDataset(Dataset):

    def __init__(self, df):
        
        # Call the original Dataset' __init__
        super().__init__(self)
        
        self.df = df


    def load_dataset(self, images_ids, imgs_folder):

        for i in range(1,5):
            self.add_class(source='SDD', class_id=i, class_name=f'type_{i}')
    
        for im_id in images_ids:
            file_name = im_id
            file_path = imgs_folder / file_name
            assert file_path.exists(), "File doesn't exists."
            
            self.add_image(
                source='SDD', 
                image_id=file_name, # is stored in self.image_info[INDEX]['id']
                path=str(file_path)
            )


    def load_mask(self, image_index):
        """
        Load instance masks for the given image.

        Args:
            image_index: An int that points to an element of a list (the 
                         add_image method of the Dataset class stores images'
                         information in an internal list).

        Returns:
            masks    : An array of shape [height, width, instance count] 
                       with a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """

        image_id = self.image_info[image_index]['id']

        # Dataframe where each row represents one defect of the same image
        subdf = self.df.loc[self.df.ImageId == image_id, :].copy()
        subdf.reset_index(drop=True, inplace=True)

        # Images' size
        h,w = (256, 1600)

        # Collection of masks with shape:
        # [height, width, instance_count]
        mask_tensor = np.zeros([h, w, subdf.shape[0]], dtype=np.uint8)

        for i, row in subdf.iterrows():
            encoded_pixels = row['EncodedPixels']
            mask = run_length_encoded_to_mask(encoded_pixels, h, w)
            mask_tensor[:, :, i] = mask

        class_ids = subdf['ClassId'].to_numpy()
        
        return mask_tensor, class_ids


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on steel data'
    )

    parser.add_argument('--model', required=True,
                        help="'last' or 'default'"
    )
    args = parser.parse_args()

    data = pd.read_csv(str(CSV_FILE))

    # Keep only the images with defects
    df_defects = data.dropna(subset=['EncodedPixels'], axis=0).copy()

    image_ids = df_defects.ImageId.unique()
    train_ids, val_ids = train_test_split(image_ids, test_size=0.3, random_state=42)

    data_train = SteelDataset(df_defects)
    data_train.load_dataset(train_ids, TRAIN_IMAGES_DIR)
    data_train.prepare()

    data_val = SteelDataset(df_defects)
    data_val.load_dataset(val_ids, TRAIN_IMAGES_DIR) # val images are in the train dir
    data_val.prepare()

    steel_config = SteelConfig()

    model = MaskRCNN(
        mode      = "training",
        config    = steel_config,
        model_dir = str(MODEL_DIR)
    )
    
    if args.model.lower() == "last":
        model_path = model.find_last() # Find last trained weights
    elif args.model.lower() == "default":
        model_path = str(COCO_MODEL_PATH)
    
    # Exclude the last layers because they require a matching number of classes
    # ^ Original comment at:
    # https://github.com/matterport/Mask_RCNN/blob/master/samples/balloon/balloon.py
    model.load_weights(
        filepath = model_path,
        by_name  = True,
        exclude  = [
            "mrcnn_class_logits", 
            "mrcnn_bbox_fc", 
            "mrcnn_bbox", 
            "mrcnn_mask"
        ]
    )

    model.train(
        train_dataset = data_train,
        val_dataset   = data_val,
        learning_rate = steel_config.LEARNING_RATE,
        epochs        = steel_config.NUM_EPOCHS, 
        layers        = "heads"
    )