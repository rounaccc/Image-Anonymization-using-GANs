import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import re
from keras.preprocessing.image import img_to_array


# to get the files in proper order
def sorted_alphanumeric(data):  
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',key)]
    return sorted(data,key = alphanum_key)

# defining the size of the image
def preprocess(SIZE, path, batch_size):
    # SIZE = 128
    _img = []
    # path = '../input/face-mask-lite-dataset/without_mask'
    files = os.listdir(path)
    files = sorted_alphanumeric(files)
    for i in tqdm(files):    
            if i == 'seed9090.png':
                break
            else:    
                img = cv2.imread(path + '/'+i,1)
                # open cv reads images in BGR format so we have to convert it to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #resizing image
                img = cv2.resize(img, (SIZE, SIZE))
                img = (img - 127.5) / 127.5
                imh = img.astype(float)
                _img.append(img_to_array(img))

    # batch_size = 32
    dataset=tf.data.Dataset.from_tensor_slices(np.array(_img)).batch(batch_size)
    return dataset