###
#Contains class cnnModel: functions to decode image from bytes file, resize/normalize it,
#                          function to get the class name from the predicted value and a function to predict the target
#
######

import tensorflow as tf
import os
import numpy as np
import keras
import trainModel

class cnnModel:

    def __init__(self,bytes):
        self.image = tf.io.decode_jpeg(bytes,channels=3)

    #resize the image to 192,192. While this isnt optimal, this is done for easier training
    def preprocessimage(self):
        self.image = tf.image.resize(self.image, [192, 192])
        #print(self.image.shape)
        self.image /= 255.0  # normalize to [0,1] range

    #The model requires input in [x,192,192,3] format. here x=1
    def reshape_input_cnn(self):
        self.image = np.expand_dims(self.image,0)  # convert a pandas series of arrays into one single array

    #uses the global variable from training to find the inverse of class values
    def classDef(self,target):
        #class_dict is the initial class definition that was used while training Model

        inv_class = {v: k for k, v in trainModel.GLOBAL_CLASSES.items()}
        self.camera = inv_class[np.argmax(target)]

    #Predict the function from the loaded model. Value sent back to app.py
    def predict(self):
        modelPath = os.path.join(os.getcwd(),'model')
        model = keras.models.load_model(modelPath + '\\cnn_model_v2.h5')
        self.preprocessimage()
        self.reshape_input_cnn()
        target = model.predict(self.image)
        self.classDef(target)
        return self.camera


