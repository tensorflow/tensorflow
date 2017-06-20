# TF Learn Examples

Learn is a high-level API for TensorFlow that allows you to create,
train, and use deep learning models easily. See the [Quickstart tutorial](https://www.tensorflow.org/get_started/tflearn)
for an introduction to the API.

To run most of these examples, you need to install the `scikit learn` library (`sudo pip install sklearn`).
Some examples use the `pandas` library for data processing (`sudo pip install pandas`).

## Basics

* [Deep Neural Network Regression with Boston Data](boston.py)
* [Deep Neural Network Classification with Iris Data](iris.py)
* [Building a Custom Model](iris_custom_model.py)
* [Building a Model Using Different GPU Configurations](iris_run_config.py)

## Techniques

* [Improving Performance Using Early Stopping with Iris Data](iris_val_based_early_stopping.py)
* [Using skflow with Pipeline](iris_with_pipeline.py)
* [Deep Neural Network with Customized Decay Function](iris_custom_decay_dnn.py)

## Specialized Models
* [Building a Random Forest Model](random_forest_mnist.py)
* [Building a Wide & Deep Model](wide_n_deep_tutorial.py)
* [Building a Residual Network Model](resnet.py)

## Text classification

* [Text Classification Using Recurrent Neural Networks on Words](text_classification.py)
* [Text Classification Using Convolutional Neural Networks on Words](text_classification_cnn.py)
* [Text Classification Using Recurrent Neural Networks on Characters](text_classification_character_rnn.py)
* [Text Classification Using Convolutional Neural Networks on Characters](text_classification_character_cnn.py)
