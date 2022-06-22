"""
Sequential Model Implementation
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import ReduceLROnPlateau
import cv2
import os
import pandas as pd

#           DOWNLOAD DATASET ---------------> https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
#           Getting testing, training and validation dataset

labels=['PNEUMONIA','NORMAL']
img_size=150
def get_data(data_dir):
    data= []
    for label in labels:
        path=os.path.join(data_dir, label)
        class_num=labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr= cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr=cv2.resize(img_arr,(img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                    print(e)
    return np.array(data)

#           Splitting x-labels(features) and y-labels(labels) of training, testing and validation dataset

train = get_data('../input/chest-xray-pneumonia/chest_xray/train')
val = get_data('../input/chest-xray-pneumonia/chest_xray/val')
test = get_data('../input/chest-xray-pneumonia/chest_xray/test')


x_train= []
y_train = []

x_val = []
y_val = []

x_test = []
y_test = []

for feature, label in train:
    x_train.append(feature)
    y_train.append( label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

    positives = []
    negatives = []
    for i in range(len(y_train)):
        if y_train[i]:
            positives.append(x_train[i])
        else:
            negatives.append(x_train[i])

    plt.bar(labels, [len(negatives), len(positives)], color=["green", "blue"])
    plt.title("Cases count in training data set")
    plt.ylabel("Count")
    plt.show()

    plt.imshow(positives[0])
    plt.title("Pneumonia", cmap="gray")
    plt.show()
    plt.imshow(negatives[4], cmap="gray")
    plt.title("Normal")
    plt.show()

    #           Grayscale Normalization

    x_train = np.array(x_train) / 255
    x_val = np.array(x_val) / 255
    x_test = np.array(x_test) / 255

    #           Resize data(x-labels) for deep learning
    x_train = x_train.reshape(-1, img_size, img_size, 1)
    y_train = np.array(y_train)

    x_val = x_val.reshape(-1, img_size, img_size, 1)
    y_val = np.array(y_val)

    x_test = x_test.reshape(-1, img_size, img_size, 1)
    y_test = np.array(y_test)

    #           Resize data(y-labels) for deep learning

    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    #           Performing Data Augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.2,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

#           Building the model

model = Sequential()
model.add(Conv2D(32, (3,3), strides=1, padding='same', activation='relu', input_shape=(150,150,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides=2, padding='same'))

model.add(Conv2D(64, (3,3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides=2, padding='same'))

model.add(Conv2D(64, (3,3), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides=2, padding='same'))

model.add(Conv2D(128, (3,3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides=2, padding='same'))

model.add(Conv2D(256, (3,3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides=2, padding='same'))

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#--------------------------------------------------------------------------------------

learning_rate_reduction= ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose='1', factor=0.3, min_lr=0.000001)

#-----------------------------------------------------------------------------------

history= model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=15, validation_data=datagen.flow(x_val, y_val), callbacks=learning_rate_reduction)
#       Number of epochs can be significantly increased for more powerful computing systems

#--------------------------------------------------------------------------------------

print("Loss of the model is : ",model.evaluate(x_test, y_test)[0])
print("Accuracy of the model is : ", model.evaluate(x_test, y_test)[1]*100, "%")

#---------------------------------------------------------------------------------------

predictions= model.predict(x_test)
for i in range(len(predictions)):
    predictions[i] = 1 if predictions[i]>0.5 else 0
    
#------------------------------------------------------------------------------------------

print(classification_report(y_test, predictions, target_names = ['Pneumonia[class 0]', 'Normal[class 1]']))

#----------------------------------------------------------------------------------------

c_matrix=confusion_matrix(y_test, predictions)
c_matrix=pd.DataFrame(c_matrix, index=['0','1'], columns=['0','1'])
c_matrix

#       This is an intial commit changes could be made.
