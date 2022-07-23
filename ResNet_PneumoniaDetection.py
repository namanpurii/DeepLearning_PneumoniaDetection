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

#Plotting the images
f = plt.figure(figsize= (10,6))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_generator=ImageDataGenerator(rescale=1/255,featurewise_center=False,  # set input mean to 0 over the dataset
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
training_generator=train_generator.flow_from_directory(train_dir,target_size=(224,224),batch_size=4,class_mode='binary')

val_generator=ImageDataGenerator(rescale=1/255.0)
validation_generator=val_generator.flow_from_directory(val_dir,target_size=(224,224),batch_size=12,class_mode='binary')

test_generator=ImageDataGenerator(rescale=1/255.0)
test_generator=test_generator.flow_from_directory(test_dir,target_size=(224,224,3),batch_size=12,class_mode='binary')

from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D,BatchNormalization
resnet=ResNet50(input_shape=[224,224,3],weights='imagenet',include_top=False)

resnet.summary()

for layers in resnet.layers[:50]:
    layers.trainable=False #as the weights are already trained

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
x = Flatten()(resnet.output)#adding the flatten layer

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
prediction = Dense(1, activation='sigmoid')(x)
# create a model object
model = Model(inputs=resnet.input, outputs=prediction)

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='binary_crossentropy',metrics=['acc'])

history = model.fit_generator(training_generator,validation_data = validation_generator,epochs = 10, verbose = 1)#fitting the model

accuracy=history.history['acc']
loss=history.history['loss']
val_accuracy=history.history['val_acc']
val_loss=history.history['val_loss']

epochs = range(len(accuracy))
epochs

import matplotlib.pyplot as plt
plt.plot(epochs,accuracy,'r',label='training_accuracy')
plt.plot(epochs,val_accuracy,'g',label='val_accuracy')
plt.legend()
plt.show()#plot for training and validation accuracy

from keras.applications.resnet50 import preprocess_input#rescalting impage using preprocess_input function

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

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau

early_stopper = EarlyStopping( patience = 10)
# checkpointer = ModelCheckpoint( monitor = 'val_loss', save_best_only = True, mode = 'auto')
learning_rate_reduction=ReduceLROnPlateau(monitor='val_acc',patience=2,verbose=1,factor=0.5,min_lr=0.001)

hist=model.fit_generator(training_generator_2,validation_data = validation_generator_2,epochs = 10, verbose = 1)


import matplotlib.pyplot as plt
plt.plot(epochs,accuracy_2,'r',label='training_accuracy')
plt.plot(epochs,val_accuracy_2,'g',label='val_accuracy')
plt.legend()
plt.show()#again plotting the validation and train accuracy curvem, this time model has increased accuracy

 #predicting an image

from keras.preprocessing import image
import numpy as np
image_path = "/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/NORMAL/person96_bacteria_464.jpeg"
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
