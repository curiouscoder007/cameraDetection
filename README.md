# cameraDetection

## Introduction
The camera detection model is a Kaggle project which aims to identify a camera based on an image taken from it. 
The details of the project is listed here : https://www.kaggle.com/c/sp-society-camera-model-identification/overview   
This project aims to solve this problem at a very basic level using a simple CNN model.

## Requirements
The required libraries are mentioned in the requirements.txt and can be installed with 
pip install -r requirements.txt

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
  
 
  
