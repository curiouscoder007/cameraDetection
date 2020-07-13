################################################################################################################
#This file contains two classes.
#class saveFiles: This program is designed to run from a pickle file in order to save time. This class helps to
#                 create the pickle files from the training/test dataset
#class trainCNNModel: This class contains functions to load the data from pickle files, get a CNN model,
#                     fit the model, train/save the model and test the Model
#Note: some of the filenames have been hardcoded for ease of use. The same directory structure has to be followed
#       for the program to work.
################################################################################################################


import tensorflow as tf
import pathlib
import imageio
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import keras
from keras.utils.np_utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

GLOBAL_CLASSES = {'HTC-1-M7': 0,
                      'iPhone-4s': 1,
                      'iPhone-6': 2,
                      'LG-Nexus-5x': 3,
                      'Motorola-Droid-Maxx': 4,
                      'Motorola-Nexus-6': 5,
                      'Motorola-X': 6,
                      'Samsung-Galaxy-Note3': 7,
                      'Samsung-Galaxy-S4': 8,
                      'Sony-NEX-7': 9}
def reshape_input_cnn(X):
    X = np.stack(X.values) #convert a pandas series of arrays into one single array
    print(X.shape)
    return X

def reshape_target(Y):
    Y = to_categorical(Y,num_classes=10)
    return Y

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [192, 192])
  image /= 255.0  # normalize to [0,1] range
  return image

class saveFiles:
    #Initialize which files to be saved. Options are 'train' and 'test'
    def __init__(self,type):
        self.type = type

    #Function to load the image files , resize them and save them as df in a pickle file.
    def savePickle(self,filename):
        path = os.path.join(os.getcwd(),'Data',self.type)
        files = ['*\*.jpg','*.tif'] #test folder has tif, train folder has more folders which contain jpg
        training_paths = pathlib.Path(path).glob(files)
        imglist = [str(x) for x in training_paths]
        img_df = pd.DataFrame(columns=['location', 'image'])
        i = 0
        for imglocation in imglist:
            img_raw = tf.io.read_file(imglocation)
            img = preprocess_image(img_raw)
            img_df.set_value(i, 'location', imglocation)
            img_df.set_value(i, 'image', img)
            i += 1

        with open(os.path.join(path,filename), 'wb') as f:
            pickle.dump(img_df, f)


class trainCNNModel:

    def __init__(self,filename,merge=True,f1='train_images1.pkl',f2='train_images1.pkl'):
        self.picklefile = os.path.join(os.getcwd(),'Data','train',filename)
        self.picklefile1 = os.path.join(os.getcwd(),'Data','train',f1)
        self.picklefile2 = os.path.join(os.getcwd(), 'Data', 'train', f2)
        self.Xtrain,self.Xtest,self.Ytrain,self.Ytest = [],[],[],[]
        self.merge = merge

    def loadData(self):
        if self.merge:
            traindf = pd.read_pickle(self.picklefile1)
            df2 = pd.read_pickle(self.picklefile2)
            traindf = pd.concat([traindf,df2])
            del df2
        else:
            traindf = pd.read_pickle(self.picklefile)
        #get the camera class from the location
        traindf['camera'] = traindf['location'].apply(lambda x: x.split('\\')[-2])
        #convert classes to numbers
        traindf['class'] = traindf['camera'].apply(lambda x: GLOBAL_CLASSES[x])
        traindf['image_array'] = traindf['image'].apply(lambda x: x.numpy())
        self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = train_test_split(traindf['image_array'], traindf['class'].values, test_size=.33,
                                                        random_state=42)

    def get_cnnmodel(self,x):
        model = Sequential()
        shape = x[0].shape
        model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=shape))
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))
        return model

    def fitmodel(self,xtrain, ytrain, xval, yval, epochs,weights=False):
        earlystopping_monitor = EarlyStopping(monitor='val_accuracy', min_delta=.1, patience=10, verbose=1)
        datagen = ImageDataGenerator(
            featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True)
        trainingsize = 6400  # keep the number as a multiple of batchsize to get same number of images each time
        batchsize = 64
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath='model_weights_cnn_v2.h5',
            save_weights_only=True,
            monitor='accuracy',
            mode='max',
            save_best_only=True)

        model = self.get_cnnmodel(xtrain)
        optimizer = Adam(0.001)  # default learning rate is .001
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        if weights:
            model.load_weights(os.path.join(os.getcwd(),'model','model_weights_cnn_v2.h5'))

        model.fit_generator(datagen.flow(xtrain, ytrain, batchsize), steps_per_epoch=5120, epochs=epochs,
                            validation_data=datagen.flow(xval, yval, batchsize), validation_steps=1280, verbose=1,
                            callbacks=[model_checkpoint_callback])
        return model

    def saveModel(self,filename,epochs,weights):
        print(self.Xtrain.shape)
        self.Xtrain = reshape_input_cnn(self.Xtrain)
        self.Ytrain = reshape_target(self.Ytrain)
        xtrain, xval, ytrain, yval = train_test_split(self.Xtrain, self.Ytrain, test_size=.2, random_state=42)
        model = self.fitmodel(xtrain,ytrain,xval,yval,epochs=epochs,weights=weights)
        self.model = model
        self.model.save(os.path.join(os.getcwd(),'model',filename))

    def getModel(self):
        return self.model


    def testModel(self,filename):
        model = keras.models.load_model(os.path.join(os.getcwd(),'model',filename))
        y_pred = model.predict(reshape_input_cnn(self.Xtest))
        # print(y_pred)
        y_pred = [np.argmax(y) for y in y_pred]
        score = accuracy_score(self.Ytest, y_pred)
        report = classification_report(self.Ytest, y_pred)
        return score, report


