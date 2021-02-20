import os
from glob import glob
from tqdm import tqdm
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Conv3D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, Input




def yield_dict(load_path, counter):
    inputs = []
    labels = []
    countcount = 0
    while True:
        for category in CATEGORIES:
            load_path2 = os.path.join(load_path, category)
            pos = CATEGORIES.index(category)
            for img in tqdm(os.listdir(load_path2)):
                image = nib.load(os.path.join(load_path2, img))
                image = image.get_fdata()
                image = np.resize(image, target_size)
                inputs.append(image)
                labels.append(CATEGORIES[pos])
                countcount +=1
                if countcount > counter:
                    X = np.array(inputs, dtype="float")
                    y = np.array(labels)
                    yield (X,y)
                    inputs=[]
                    labels=[]
                    countcount=0
                    break

CATEGORIES = ['AD', 'NC']
dict = {}
x=32
n=0
batch_size=32
target_size = (110,110,110)
loadpath='/home/szychro/PycharmProjects/pythonProject/data/trimmed_110-110-110'

#for i in (0,n):
for INPUT, LABELS in yield_dict(loadpath, x):
    dict["Image"] = INPUT
    dict["Label"] = LABELS
    print(dict["Image"].shape)


encoder = LabelEncoder()
encoder.fit(dict["Label"])
encoded_y = encoder.transform(dict["Label"])

train_X, val_X, train_y, val_y = train_test_split(dict["Image"], encoded_y, test_size=0.25, random_state=None)

ntrain = len(train_X)
nval = len(val_X)

def get_model():


    inputs = Input(shape=(110,110,110,1))


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
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

model = get_model()
model.summary()

model.compile(optimizer=Adam(lr=1e-7), loss='binary_crossentropy', metrics=['accuracy']) #(lr=1e-7)

#model.fit(train_X, train_y, batch_size=32, epochs=5)
history = model.fit_generator((train_X, train_y),
                              batch_size=batch_size,
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