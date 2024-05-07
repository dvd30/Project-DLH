import os
import numpy as np
import matplotlib.image as mp_img
import matplotlib.pyplot as mp_plt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random as r
import math
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Input, UpSampling2D,BatchNormalization
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from keras import backend as keras


image_path = 'image\image'
mask_path = 'mask\mask'


def image_train_test_val_split(path):
    train_data = r"C:/Users/divya/OneDrive - iittp.ac.in/Attachments/Documents/IIT/DLH/Project DLH/train image"
    test_data = r"C:/Users/divya/OneDrive - iittp.ac.in/Attachments/Documents/IIT/DLH/Project DLH/test image"
    validation_data = r"C:/Users/divya/OneDrive - iittp.ac.in/Attachments/Documents/IIT/DLH/Project DLH/validation image"
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
                    '0539025', '0226309', '0226307', '0091383', '0021092', '0537990', '0226299', '0073060', '0538505',
                    '0073424', '0091534']
    validation_folder = ['0226125', '0072691', '0538425', '0226199', '0226261']
    for folder in os.listdir(path):
        if folder in test_folders:
            os.rename(os.path.join(path,folder),os.path.join(test_data,folder))
        elif folder in validation_folder:
            os.rename(os.path.join(path,folder),os.path.join(validation_data,folder))
        else:
            os.rename(os.path.join(path,folder),os.path.join(train_data,folder))
            
            
def mask_train_test_val_split(path):
    train_data = r"C:/Users/divya/OneDrive - iittp.ac.in/Attachments/Documents/IIT/DLH/Project DLH/train mask"
    test_data = r"C:/Users/divya/OneDrive - iittp.ac.in/Attachments/Documents/IIT/DLH/Project DLH/test mask"
    validation_data = r"C:/Users/divya/OneDrive - iittp.ac.in/Attachments/Documents/IIT/DLH/Project DLH/validation mask"
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
                    '0539025', '0226309', '0226307', '0091383', '0021092', '0537990', '0226299', '0073060', '0538505',
                    '0073424', '0091534']
    validation_folder = ['0226125', '0072691', '0538425', '0226199', '0226261']
    for folder in os.listdir(path):
        if folder in test_folders:
            os.rename(os.path.join(path,folder),os.path.join(test_data,folder))
        elif folder in validation_folder:
            os.rename(os.path.join(path,folder),os.path.join(validation_data,folder))
        else:
            os.rename(os.path.join(path,folder),os.path.join(train_data,folder))


def load_data(path):
    data = []
    for folder in os.listdir(path):
        for file in os.listdir(os.path.join(path,folder)):
            img = mp_img.imread(os.path.join(path,folder,file))
            if img is not None:
                data.append(img)
    return data



def unet(input_size = (512,512,1)):
    inputs = Input(input_size)
    #down convolution and max-pooling
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    # up convolution
    up6 = Conv2DTranspose(512,2,strides=(2,2),padding='same')(drop5)
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    up7 = Conv2DTranspose(256,2,strides=(2,2),padding='same')(conv6)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    up8 = Conv2DTranspose(128,2,strides=(2,2),padding='same')(conv7)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    up9 = Conv2DTranspose(64,2,strides=(2,2),padding='same')(conv8)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs, conv10)
    model.compile(optimizer = Adam(), loss='binary_crossentropy', metrics=['dice_coef', 'accuracy'])
    model.summary()
    return model


image_train_test_val_split(image_path)
mask_train_test_val_split(mask_path)
image_train_data = load_data('C:/Users/divya/OneDrive - iittp.ac.in/Attachments/Documents/IIT/DLH/Project DLH/train image')
mask_train_data = load_data('C:/Users/divya/OneDrive - iittp.ac.in/Attachments/Documents/IIT/DLH/Project DLH/train mask')
image_test_data = load_data('C:/Users/divya/OneDrive - iittp.ac.in/Attachments/Documents/IIT/DLH/Project DLH/test image')
mask_test_data = load_data('C:/Users/divya/OneDrive - iittp.ac.in/Attachments/Documents/IIT/DLH/Project DLH/test mask')
image_validation_data = load_data('C:/Users/divya/OneDrive - iittp.ac.in/Attachments/Documents/IIT/DLH/Project DLH/validation image')
mask_validation_data = load_data('C:/Users/divya/OneDrive - iittp.ac.in/Attachments/Documents/IIT/DLH/Project DLH/validation mask')
image_train = np.array(image_train_data)
mask_train = np.array(mask_train_data)
image_test = np.array(image_test_data)
mask_test = np.array(mask_test_data)
image_validation = np.array(image_validation_data)
mask_validation = np.array(mask_validation_data)
print(len(image_train))
print(len(mask_train))
print(len(image_test))
print(len(mask_test))
print(len(image_validation))
print(len(mask_validation))
unet_model = unet()
unet_model.fit(image_train, mask_train, batch_size=1, epochs=40, verbose=1, shuffle=True, validation_data=(image_test, mask_test))
unet_model.evaluate(image_test, mask_test)
prediction  = unet_model.predict(image_test_data)
print(prediction)
print("Model Trained")
