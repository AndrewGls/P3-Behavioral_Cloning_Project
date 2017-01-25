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
from  proc_data import load_data, train_validation_split



if __name__ == '__main__':

    data_train, data_val = train_validation_split(load_data(cf.DRIVING_LOG, verbose=True))
