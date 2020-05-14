# Dog Breeds Identification using CNN Transfer Learning

### Table of Contents

1. [Project Overview](#overview)
2. [File Descriptions](#files)
3. [Results](#results)

## Project Overview<a name="overview"></a>

This project is a part of the Udacity Data Science Nanodegree program. The goal in this project is to classify images of dogs according
to their breed. To achieve this goal a neural network model with Convolutional Neural Network (CNN) transfer learning was build 
and trained using 8351 dog images of 133 breeds. CNNs are a type of deep neural networks and are mostly used to classify image data. 
The architecture of a CNN consists of an input and an output layer as well as hidden layers. Transfer learning is used when a previous
developed and trained model can be used for a different task. To detect human faces in images `OpenCV's` implementation of `Haar feature-based 
cascade classifier` was used and for dog detection in images a pre-trained `ResNet-50 model` was used. This pre-trained 
model was trained on `ImageNet` which contains more than 10 million URLs, each linking to an image. Given an image the algorithm will
first detect if it is a dog or a human. If it can't detect any of boths the output will be `No human or dog detected' otherwise the
algorithm will predict the breed of the dog or the most resembling dog breed for a human.


## File Descriptions <a name="files"></a>

Below are main foleders/files for this project:
1. haarcascades
    - haarcascade_frontalface_alt.xml:  a pre-trained face detector provided by OpenCV
2. bottleneck_features
    - DogVGG19Data.npz: pre-computed the bottleneck features for VGG-19
3. saved_models
    - weights.best.VGG19.hdf5: saved model weights with best validation loss
4. dog_app.ipynb: a notebook used to build and train the dog breeds classification model 
5. extract_bottleneck_features.py: functions to compute bottleneck features given a tensor converted from an image
6. images: a few images to test the model

## Results<a name="results"></a>

1. The model was able to reach an accuracy of `75.6%` on test data.

Blog post: https://medium.com/


