# cameraDetection

## Introduction
The camera detection model is a Kaggle project which aims to identify a camera based on an image taken from it. 
The details of the project is listed here : https://www.kaggle.com/c/sp-society-camera-model-identification/overview   
This project aims to solve this problem at a very basic level using a simple CNN model.

## Requirements
The required libraries are mentioned in the requirements.txt and can be installed with 
pip install -r requirements.txt . (Best to use tensorflow-gpu for the training of the model)

## Execution
The application web service can be started with app.py which functions as the controller. 

To Train a model with the input data , use the train.py and to test and get score/classification report, test.py can be used.

Some parts of the program has been hardcoded for the easier execution of the program at this point. 

## Folders:
  Data: Extract the train and test dataset in this folder. This program also uses pre-saved pickle files which are saved in the same folders \
  Model: The model and the weights files(used to start with improved weights) are stored in this folder \
  Data Exploration: More detailed analysis and comments about the process of developing the CNN model in multiple files \
  Static: standard Flask setup for .css files \
  templates: standard Flake setup for html files 
  tests: Unit test files for checking the correctness of the programs
  
## Model
  The program uses a simple sequential model with 3 conv layers and flattens the output from them. The output is then passed through a dense layer which helps to classify the image based on the camera used. 
  
## Future Scope
  This model is pretty basic as of now. In the kaggle LB, this scored a mere 24% accuracy. In training these images, they are resized to 192 x 192. Instead cropping them at the center, as similar to the test dataset provided by kaggle might improve accuracy. Similary, using transfer learning for the model, will greatly improve the final score. 
 
  
