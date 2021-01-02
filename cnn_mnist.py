import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras import models
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing import image
from tqdm import tqdm
import random
from tensorflow.keras.preprocessing.image import img_to_array
import seaborn as sns


batch_size = 32
shape = (28, 28)
#train_path = '/home/szychro/PycharmProjects/Researchproject/mnist_png/training/'
CATEGORIES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#print('Folders :', os.listdir(train_path))

def load_data():
    train_images = [] # X
    train_labels = [] # y

    for category in CATEGORIES:
        load_path = os.path.join('/home/szychro/PycharmProjects/Researchproject/mnist_png/training/', category)
        pos = CATEGORIES.index(category)
        for img in tqdm(os.listdir(load_path)):
            image = os.path.join(load_path, img)
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            image_res = cv2.resize(image, shape)
            image_to_add = img_to_array(image_res)

            train_images.append(image_res)
            train_labels.append(CATEGORIES[pos])

    train_images = np.array(train_images, dtype="float")
    train_labels = np.array(train_labels)

    return train_images, train_labels


X, y = load_data()

print(X.shape)
#print(y.shape)
#print(y)

X = X.reshape(-1, 28, 28, 1)
print(X.shape)

#X = np.array(X)
#print(X.shape)
#y = np.array(y)
train_y_one_hot = to_categorical(y)



train_X, val_X, train_y, val_y = train_test_split(X, train_y_one_hot, test_size=0.25, random_state=None)

#print(train_X.shape)
#print(val_X.shape)
#print(train_y.shape)
#print(val_y.shape)

ntrain = len(train_X)
nval = len(val_X)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer=Adam(lr=1e-7), loss='categorical_crossentropy', metrics=['accuracy'])
"""
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(train_X, train_y,
                                     batch_size=batch_size,
                                     shuffle=True)
                                     #target_size=(shape),
                                     #class_mode='categorical')

val_generator = val_datagen.flow(val_X, val_y,
                                 batch_size=batch_size)
                                 #target_size=(shape),
                                 #class_mode='categorical'


history = model.fit_generator(train_generator,
                              steps_per_epoch=ntrain // batch_size,
                              epochs=25,
                              validation_data=val_generator,
                              validation_steps=nval // batch_size
                              )
"""

history = model.fit(x=train_X, y=train_y, validation_data=(val_X, val_y),
                    epochs=200, batch_size=batch_size, shuffle=True,
                    steps_per_epoch=ntrain // batch_size,
                    validation_steps=nval // batch_size
                    )


target_dir = '/home/szychro/PycharmProjects/Researchproject/mnist_png_results/mnist_pred/'
model.save(target_dir + 'CNN_newmodel_200epochs.h5')
model.save_weights(target_dir + 'CNN_newmodel_200epochs.h5')

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
fig.savefig('/home/szychro/PycharmProjects/Researchproject/mnist_png_results/Accuracy_function_newCNN_256_200epochs.jpg')

fig2 = plt.figure(figsize=(20, 10))
plt.plot(epochs, loss, 'r', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and validation loss')
fig2.savefig('/home/szychro/PycharmProjects/Researchproject/mnist_png_results/Loss_function_newCNN_256_200epochs.jpg')
