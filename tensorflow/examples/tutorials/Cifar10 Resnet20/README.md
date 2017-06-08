# Cifar10 - Resnet 20 layers Tutorial (by [Valery Kovtun](mailto:kovalera@gmail.com))

This tutorial goes over the basics of image acquisition, processing and analysis, by using the 20 layer Residual Network.

##Goals of the tutorial-
* Learn about the basics of classification
* Learn to understand and implement the ResNET in Tensorflow
* Visualizing models in Tensorboard

## Main Dependencies:

* TensorFlow v1.0
* Numpy 1.12
* h5py 2.6

## Important Note
In order to make explanations more clear, I have split all the definitions from the main class, to make the Jupyter Notebook code work,  
I have made a self inheritance, this is only from convenience considerations only, please do not repeat that if you are reimplementing 
this code! 

# Less Important Note
In order to enjoy the whole amount of this tutorial and the strength of Tensorflow, it is recommended to copy this code into python files
and play around as much as possible with everything in it, this way you can really understand the workings of the system.