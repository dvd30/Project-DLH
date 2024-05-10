import os
import numpy as np
import matplotlib.image as mp_img
import matplotlib.pyplot as mp_plt
import tensorflow as tf
import matplotlib.pyplot as plt
import random as r
from tensorflow import keras
import math
from tensorflow.keras.optimizers import Adam
from keras import backend as keras


image_path = '/mnt/DATA/ee21b025/EE21B013/AISD/Project-DLH/image'
mask_path = '/mnt/DATA/ee21b025/EE21B013/AISD/Project-DLH/mask'


def image_train_test_val_split(path):
    train_data = r"/mnt/DATA/ee21b025/EE21B013/AISD/Project-DLH/train image"
    test_data = r"/mnt/DATA/ee21b025/EE21B013/AISD/Project-DLH/test image"
    validation_data = r"/mnt/DATA/ee21b025/EE21B013/AISD/Project-DLH/validation image"
    if not os.path.exists(train_data) and not os.path.exists(test_data) and not os.path.exists(validation_data):
        os.makedirs(train_data)
        os.makedirs(test_data)
        os.makedirs(validation_data)
    else:
        print("Data Splitted")
        return
    test_folders = ['0073410', '0072723', '0226290', '0537908', '0538058', '0091415', '0538780', '0073540', '0226188', 
                    '0226258', '0226314', '0091507', '0226298', '0538975', '0226257', '0226142', '0072681', '0091538',
                    '0538983', '0537961', '0091646', '0072765', '0226137', '0091621', '0091458', '0021822', '0538319', 
                    '0226133', '0091657', '0537925', '0073489', '0538502', '0091476', '0226136', '0538532','0073312', 
                    '0539025', '0226309', '0226307']
    validation_folder = ['0226125', '0072691', '0538425', '0226199', '0226261', '0091383', '0021092',
                         '0537990', '0226299', '0073060', '0538505', '0073424', '0091534']
    for folder in os.listdir(path):
        if folder in test_folders:
            os.rename(os.path.join(path,folder),os.path.join(test_data,folder))
        elif folder in validation_folder:
            os.rename(os.path.join(path,folder),os.path.join(validation_data,folder))
        else:
            os.rename(os.path.join(path,folder),os.path.join(train_data,folder))
            
            
def mask_train_test_val_split(path):
    train_data = r"/mnt/DATA/ee21b025/EE21B013/AISD/Project-DLH/train mask"
    test_data = r"/mnt/DATA/ee21b025/EE21B013/AISD/Project-DLH/test mask"
    validation_data = r"/mnt/DATA/ee21b025/EE21B013/AISD/Project-DLH/validation mask"
    if not os.path.exists(train_data) and not os.path.exists(test_data) and not os.path.exists(validation_data):
        os.makedirs(train_data)
        os.makedirs(test_data)
        os.makedirs(validation_data)
    else:
        print("Data Splitted")
        return
    test_folders = ['0073410', '0072723', '0226290', '0537908', '0538058', '0091415', '0538780', '0073540', '0226188', 
                    '0226258', '0226314', '0091507', '0226298', '0538975', '0226257', '0226142', '0072681', '0091538',
                    '0538983', '0537961', '0091646', '0072765', '0226137', '0091621', '0091458', '0021822', '0538319', 
                    '0226133', '0091657', '0537925', '0073489', '0538502', '0091476', '0226136', '0538532','0073312', 
                    '0539025', '0226309', '0226307']
    validation_folder = ['0091383', '0021092', '0537990', '0226299', '0073060', '0538505',
                    '0073424', '0091534', '0226125', '0072691', '0538425', '0226199', '0226261']
    for folder in os.listdir(path):
        if folder in test_folders:
            os.rename(os.path.join(path,folder),os.path.join(test_data,folder))
        elif folder in validation_folder:
            os.rename(os.path.join(path,folder),os.path.join(validation_data,folder))
        else:
            os.rename(os.path.join(path,folder),os.path.join(train_data,folder))


def round(data):
    data = np.round(data/(np.amax(data))*5)
    return data


def load_data(path):
    data = []
    for folder in os.listdir(path):
        for file in os.listdir(os.path.join(path,folder)):
            img = mp_img.imread(os.path.join(path,folder,file))
            if img is not None:
                img = img[::4,::4]
                data.append(img)
    return data


def one_hot_encoded_mask(data):
    array = np.zeros((data.shape[0],data.shape[1],data.shape[2],6))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                if data[i][j][k] == 0:
                    array[i][j][k][0] = 1
                elif data[i][j][k] == 1:
                    array[i][j][k][1] = 1
                elif data[i][j][k] == 2:
                    array[i][j][k][2] = 1
                elif data[i][j][k] == 3:
                    array[i][j][k][3] = 1
                elif data[i][j][k] == 4:
                    array[i][j][k][4] = 1
                elif data[i][j][k] == 5:
                    array[i][j][k][5] = 1
    return array


image_train_test_val_split(image_path)
mask_train_test_val_split(mask_path)
image_train_data = load_data('/mnt/DATA/ee21b025/EE21B013/AISD/Project-DLH/train image')
mask_train_data = load_data('/mnt/DATA/ee21b025/EE21B013/AISD/Project-DLH/train mask')
image_test_data = load_data('/mnt/DATA/ee21b025/EE21B013/AISD/Project-DLH/test image')
mask_test_data = load_data('/mnt/DATA/ee21b025/EE21B013/AISD/Project-DLH/test mask')
image_validation_data = load_data('/mnt/DATA/ee21b025/EE21B013/AISD/Project-DLH/validation image')
mask_validation_data = load_data('/mnt/DATA/ee21b025/EE21B013/AISD/Project-DLH/validation mask')
image_train = np.array(image_train_data)
mask_train = round(np.array(mask_train_data))
image_test = np.array(image_test_data)
mask_test = round(np.array(mask_test_data))
image_validation = np.array(image_validation_data)
mask_validation = round(np.array(mask_validation_data))
final_mask_train = one_hot_encoded_mask(mask_train)
final_mask_test = one_hot_encoded_mask(mask_test)
final_mask_validation = one_hot_encoded_mask(mask_validation)
# final_mask_train = mask_train
# final_mask_test = mask_test
# final_mask_validation = mask_validation