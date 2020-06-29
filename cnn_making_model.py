# -*- coding: utf-8 -*-
import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.utils as Utils
from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.utils import plot_model
import os
import matplotlib.pyplot as plot
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint
from IPython.display import SVG
import matplotlib.gridspec as gridspec
import pandas as pd


def get_images(directory):
    Images = []
    Labels = []
    imageNames = []
    label = 0

    for labels in os.listdir(directory):
        if labels == 'Audi':
            label = 0
        elif labels == 'BMW':
            label = 1
        elif labels == 'Bently':
            label = 2
        elif labels == 'Honda':
            label = 3
        elif labels == 'Ray':
            label = 4
        elif labels == 'Tico':
            label = 5

        for image_file in os.listdir(directory+labels):
            before_image = cv2.imread(directory+labels+r'/'+image_file)

            #이미지 읽은것 중에서 None타입을 뺄수있게
            if type(before_image) != type(None):
                image = before_image

            image = cv2.resize((image),(150,150))

            imageNames.append(image_file)
            Images.append(image)
            Labels.append(label)

    return shuffle(Images,Labels,imageNames,random_state=817328462)

def get_classlabel(class_code):
    labels = {0:'Audi', 1:'BMW', 2:'Bently', 3:'Honda', 4:'Ray', 5:'Tico'}
    return labels[class_code]

PATH = os.getcwd()
# Define data path
data_path = PATH + '/new_train_data/'
os.system("cd new_12_29_train_data; find . -name '.DS_Store' -type f -delete")

Images, Labels, imageNames = get_images(data_path) #Extract the training images from the folders.
Images = np.array(Images) #Converting the list of images to numpy array.
Labels = np.array(Labels)

print("Shape of Images:",Images.shape)
print("Shape of Labels:",Labels.shape)
print("Image names:", imageNames)


f,ax = plot.subplots(5,5)
f.subplots_adjust(0,0,3,3)
for i in range(0,5,1):
    for j in range(0,5,1):
        rnd_number = randint(0,len(Images)-1)
        ax[i,j].imshow(Images[rnd_number])
        ax[i,j].set_title(get_classlabel(Labels[rnd_number]))
        ax[i,j].axis('off')


model = Models.Sequential()
model.add(Layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))
model.add(Layers.Conv2D(180,kernel_size=(3,3),activation='relu'))
model.add(Layers.MaxPool2D(5,5))
model.add(Layers.Conv2D(180,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(140,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(100,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(50,kernel_size=(3,3),activation='relu'))
model.add(Layers.MaxPool2D(5,5))
model.add(Layers.Flatten())
model.add(Layers.Dense(180,activation='relu'))
model.add(Layers.Dense(100,activation='relu'))
model.add(Layers.Dense(50,activation='relu'))
model.add(Layers.Dropout(rate=0.5))
model.add(Layers.Dense(9,activation='softmax'))

model.compile(optimizer=Optimizer.Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()

trained = model.fit(Images,Labels,epochs=20,validation_split=0.2)

plot.plot(trained.history['accuracy'])
plot.plot(trained.history['val_accuracy'])
plot.title('Model accuracy')
plot.ylabel('Accuracy')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
plot.show()

plot.plot(trained.history['loss'])
plot.plot(trained.history['val_loss'])
plot.title('Model loss')
plot.ylabel('Loss')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
plot.show()

data_path = PATH + '/new_test_data/'
os.system("cd new_test_data; find . -name '.DS_Store' -type f -delete")
test_images,test_labels, imageNames = get_images(data_path)
test_images = np.array(test_images)
test_labels = np.array(test_labels)
model.evaluate(test_images,test_labels, verbose=1)

model.save('your_model.h5')
