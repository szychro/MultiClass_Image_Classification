import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

img_width = 28
img_height = 28
batch_size = 32


TRAINING_DIR = '/home/szychro/PycharmProjects/Researchproject/mnist_png/training'
TESTING_DIR = '/home/szychro/PycharmProjects/Researchproject/mnist_png/testing'


train_datagen = ImageDataGenerator(rescale=1 / 255.0, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    target_size=(img_height, img_width),
                                                    shuffle=True,
                                                    subset='training'
                                                    )

test_DIR = '/home/szychro/PycharmProjects/Researchproject/mnist_png/testing'


validation_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                         batch_size=batch_size,
                                                         class_mode='categorical',
                                                         target_size=(img_height, img_width),
                                                         subset='validation'
                                                         )

test_datagen = ImageDataGenerator(rescale=1 / 255.0)

test_generator = test_datagen.flow_from_directory(TESTING_DIR,
                                                         batch_size=batch_size,
                                                         class_mode='categorical',
                                                         target_size=(img_height, img_width),
                                                         shuffle=False
                                                         )

callbacks = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode=',min')
#mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model = tf.keras.models.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit_generator(train_generator,
                              epochs=300,
                              verbose=1,
                              validation_data=validation_generator,
                              )


target_dir = '/home/szychro/PycharmProjects/Researchproject/mnist_png_results/mnist_pred/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
model.save(target_dir + 'CNN_model_256_300epochs.h5')
model.save_weights(target_dir + 'CNN_weights_256_300epochs.h5')


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
fig.savefig('/home/szychro/PycharmProjects/Researchproject/mnist_png_results/Accuracy_curve_CNN_256_300epochs.jpg')

fig2 = plt.figure(figsize=(20, 10))
plt.plot(epochs, loss, 'r', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and validation loss')
fig2.savefig('/home/szychro/PycharmProjects/Researchproject/mnist_png_results/Loss_curve_CNN_256_300epochs.jpg')

