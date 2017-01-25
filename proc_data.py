import csv
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.optimizers import Adam
from keras.models import Model
#from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
#from keras.callbacks import ModelCheckpoint
import keras.backend as K

import config as cf



def load_data(csv_driving_data, verbose=False):
    """
    Loads data as numpy array from csv driving_log file.
    :param csv_driving_data: path to Udacity csv driving_log file
    return: data from csv file as numpy array. Don't use panda
            frame data because of performance issue.
    """
    data_frame = pd.DataFrame.from_csv(csv_driving_data, index_col=None)
    data = data_frame.values
    header = data_frame.columns.values
    assert(header[cf.CENTER] == 'center')
    assert(header[cf.LEFT] == 'left')
    assert(header[cf.RIGHT] == 'right')
    assert(header[cf.STEERING] == 'steering')
    if verbose:
        print('data shape: ', data.shape)
        print('driving_log header: ', header)
        print(data[0:2])
    return data
    

    
def train_validation_split(data, test_size=0.2):
    """
    Splits data, loaded from csv driving_log file, into training and validation sets
    param data: numpy array (or list) from Udacity csv driving_log
    return: train & validation sets:
    """
    data_train, data_val = train_test_split(data, test_size=test_size, random_state=0)
    return data_train, data_val

            
    
def preprocess_image_train(bgrImg, steering, verbose=False):
    """
    Preprocessing:
        1. random left/right/center camera
        2. horz/vert random shift
        3. rand brightness (Y channel)
        4. add shadows ????
        5. crop image to modelinput size
    """
    
    cropImg = bgrImg;
    
    if verbose:
        plt.figure(1)
        plt.imshow(cv2.cvtColor(cropImg, code=cv2.COLOR_BGR2RGB))
        plt.show()
        
    return cropImg.astype('float32')

    
    
def preprocess_image(frm_bgr):
    """
    Prepare image for model: crop
    
    Normalize here or in keras lambda func
    """    
    return None

    
def generate_train_from_data_batch(data,
                                   batch_size=32,
                                   augment=True,
                                   steering_bias=0.0,
                                   pb_thresh=0.1):
    return None
    