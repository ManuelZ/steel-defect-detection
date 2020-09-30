import numpy as np

def defects_mask(encoded_pixels, imshape):
    """ Transform a line of encoded pixels into a mask """
    
    h,w = imshape[:2]
    
    encoded_pixels = np.asarray(encoded_pixels.split(), dtype=np.int)
    starts = encoded_pixels[0::2]
    lengths = encoded_pixels[1::2]

    linear_mask = np.zeros(w*h, dtype=np.uint8)
    for start,length in zip(starts, lengths):
        linear_mask[start:start+length] = 1
    
    mask = linear_mask.reshape(w,h).transpose()
    return mask