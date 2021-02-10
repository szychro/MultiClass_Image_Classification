import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import models
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Conv2D, Conv3D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, Input
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing import image
from tqdm import tqdm
import random
from tensorflow.keras.preprocessing.image import img_to_array
import seaborn as sns
import nibabel as nib
from scipy import ndimage
from sklearn.preprocessing import LabelEncoder




batch_size = 16
target_size = (110, 110, 110)
#train_path = '/home/szychro/PycharmProjects/Researchproject/mnist_png/training/'
CATEGORIES = ['AD', 'NC'] #, 'NC'
"""
def yield_image():
    for category in CATEGORIES:
        load_path = os.path.join('/home/szychro/PycharmProjects/pythonProject/data/trimmed_110-110-110', category)
        if category == 'AD':
            for img in tqdm(os.listdir(load_path)):
                image_AD = nib.load(os.path.join(load_path, img))
                yield image_AD
        else:
            for img in tqdm(os.listdir(load_path)):
                image_NC = nib.load(os.path.join(load_path, img))
                yield image_NC

X_AD, X_NC = yield_image()
"""

def load_images():
    train_images = [] # X
    #train_labels = [] # y

    counter = 0
    for category in CATEGORIES:
        load_path = os.path.join('/home/szychro/PycharmProjects/pythonProject/data/trimmed_110-110-110', category)
        #pos = CATEGORIES.index(category)
        for img in tqdm(os.listdir(load_path)):
            image = nib.load(os.path.join(load_path, img))
            image = image.get_fdata()
            #image = np.array(image)
            #image = np.resize(image, target_size)
            #image = np.expand_dims(image, axis=3)
            #image /= 255.

            train_images.append(image)
            #train_labels.append(CATEGORIES[pos])
            counter= counter+1

    train_images = np.ndarray((counter, 110, 110, 110, 1), dtype=np.float32)
    #train_images = np.array(train_images, dtype="float") #or image.dataobj,get_fdata()
    #train_images = np.array(image.get_data_dtype()) #or asarray


    #train_labels = np.array(train_labels)

    return train_images

def load_labels():
    #train_images = [] # X
    train_labels = [] # y

    for category in CATEGORIES:
        load_path = os.path.join('/home/szychro/PycharmProjects/pythonProject/data/trimmed_110-110-110', category)
        pos = CATEGORIES.index(category)
        for img in tqdm(os.listdir(load_path)):
            #image = nib.load(os.path.join(load_path, img))
            #image = image.get_fdata()
            #image = np.array(image)
            #image = np.resize(image, target_size)
            #image = np.expand_dims(image, axis=3)
            #image /= 255.

            #train_images.append(image)
            train_labels.append(CATEGORIES[pos])

    #train_images = np.ndarray((counter, 110, 110, 110, 1), dtype=np.float32)
    #train_images = np.array(train_images, dtype="float") #or image.dataobj,get_fdata()
    #train_images = np.array(image.get_data_dtype()) #or asarray


    train_labels = np.array(train_labels)

    return train_labels

X = load_images()
y = load_labels()

print(X.shape)
#print(y)
#print((y.shape))

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

train_X, val_X, train_y, val_y = train_test_split(X, encoded_y, test_size=0.25, random_state=None)
print(train_X.shape)

ntrain = len(train_X)
nval = len(val_X)



def get_model():


    inputs = Input(shape=( 110, 110, 110,1))


    x = Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = BatchNormalization()(x)

    x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = BatchNormalization()(x)

    #x = Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    #x = BatchNormalization()(x)


    x = Dense(256, activation="relu")(x)
    #x = Dropout(0.3)(x)
    x = Flatten()(x)

    outputs = Dense(1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# Build model.
model = get_model()
model.summary()

model.compile(optimizer=Adam(lr=1e-7), loss='binary_crossentropy', metrics=['accuracy']) #(lr=1e-7)

history = model.fit(train_X, train_y,
                      steps_per_epoch=ntrain // batch_size,
                      epochs=10,
                      validation_data=(val_X, val_y),
                      validation_steps=nval // batch_size
                      )




target_dir = '/home/szychro/PycharmProjects/pythonProject/results/checkpoints/'
model.save(target_dir + 'CNN_Alz_10epochs.h5')
model.save_weights(target_dir + 'CNN_Alz_10epochs.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))


fig = plt.figure(figsize=(20, 10))
plt.plot(epochs, acc, 'r', label="Training Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc='lower right')
plt.show()
fig.savefig('/home/szychro/PycharmProjects/pythonProject/results/graphs/Accuracy_function_CNN_Alz_10epochs.jpg')

fig2 = plt.figure(figsize=(20, 10))
plt.plot(epochs, loss, 'r', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and validation loss')
fig2.savefig('/home/szychro/PycharmProjects/Researchproject/mnist_png_results/Loss_function_CNN_Alz_10epochs.jpg')
