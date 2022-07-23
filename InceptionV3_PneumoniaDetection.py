import os
os.listdir('../input/chest-xray-pneumonia/chest_xray/train/')
os.listdir('../input/chest-xray-pneumonia/chest_xray/val/')
os.listdir('../input/chest-xray-pneumonia/chest_xray/test/')

train_dir='../input/chest-xray-pneumonia/chest_xray/train/'
test_dir='../input/chest-xray-pneumonia/chest_xray/test/'
val_dir='../input/chest-xray-pneumonia/chest_xray/val/'

# train
os.listdir(train_dir)
train_n = train_dir+'NORMAL/'
train_p = train_dir+'PNEUMONIA/'

#Normal pic
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
print(len(os.listdir(train_n)))
rand_norm= np.random.randint(0,len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ',norm_pic)

norm_pic_address = train_n+norm_pic

#Pneumonia
rand_p = np.random.randint(0,len(os.listdir(train_p)))

sic_pic =  os.listdir(train_p)[rand_norm]
sic_address = train_p+sic_pic
print('pneumonia picture title:', sic_pic)

# Load the images
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)

#Let's plt these images
f = plt.figure(figsize= (10,6))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')

import cv2
import matplotlib.pyplot as plt

for dire in os.listdir(train_dir):
    path = os.path.join(train_dir, dire)
    for img in os.listdir(path):
        paths = os.path.join(path, img)

        img_array = cv2.imread(paths, cv2.IMREAD_COLOR)
        plt.imshow(img_array)
        break

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.python.keras.layers import   Dropout, MaxPooling2D, ZeroPadding2D,BatchNormalization

train_generator_2=ImageDataGenerator(preprocessing_function=preprocess_input,featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = False,  # randomly flip images
        vertical_flip=False)
training_generator_2=train_generator_2.flow_from_directory(train_dir,target_size=(224,224),batch_size=4,class_mode='binary')

val_generator_2=ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator_2=val_generator_2.flow_from_directory(val_dir,target_size=(224,224),batch_size=12,class_mode='binary')

from tensorflow.keras.applications import InceptionV3

inception=InceptionV3(input_shape=[224,224,3],weights='imagenet',include_top=False)

inception.summary()

#train the layers of the model
for layers in inception.layers[:50]:
    layers.trainable=False

x = Flatten()(inception.output)
prediction = Dense(1, activation='sigmoid')(x)
# create a model object
model = Model(inputs=inception.input, outputs=prediction)

model.summary()

from keras import optimizers
model.compile(optimizer=optimizers.Adam(lr=0.001, decay=0.005),
              loss='binary_crossentropy',
              metrics=['accuracy'])
#CallBacks Function
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
early_stop=EarlyStopping(monitor="val_acc",
                         patience=10,
                         mode="auto",)
Learning_rate_reduction=ReduceLROnPlateau(monitor='val_acc',patience=2,verbose=1,factor=0.5,min_lr=0.001)

callbacks=[early_stop,Learning_rate_reduction]

history = model.fit_generator(training_generator_2,validation_data = validation_generator_2,epochs = 10, verbose = 1,callbacks=callbacks)

accuracy_3=history.history['accuracy']
loss_3=history.history['loss']
val_accuracy_3=history.history['val_accuracy']
val_loss_3=history.history['val_loss']

epochs = range(len(accuracy_3))
epochs

import matplotlib.pyplot as plt
plt.plot(epochs,accuracy_3,'r',label='training_accuracy')
plt.plot(epochs,val_accuracy_3,'g',label='val_accuracy')
plt.legend()
plt.show()

#plotting training values
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

#accuracy plot
plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.figure()
#loss plot
plt.plot(epochs, loss, color='pink', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# predicting an image

from keras.preprocessing import image
import numpy as np
image_path = "/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/PNEUMONIA/person96_bacteria_464.jpeg"
new_img = image.load_img(image_path, target_size=(224, 224))
img = image.img_to_array(new_img)
img = np.expand_dims(img, axis=0)
img = img/255

print("Following is our prediction:")
prediction = model.predict(img)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
d = prediction.flatten()
j = d.max()
for index,item in enumerate(d):
    if item == j:
        class_name = li[index]
plt.figure(figsize = (4,4))
plt.imshow(new_img)
plt.axis('off')
plt.title(class_name)
plt.show()