# Examples of Using skflow

Scikit Flow is high level API that allows to create,
train and use deep learning models easily with well
known Scikit Learn API.

To run these examples, you need to have `scikit learn` library installed (`sudo pip install sklearn`).
Some examples use the `pandas` library for data processing (`sudo pip install pandas`).

## Basics

* [Deep Neural Network Regression with Boston Data](boston.py)
* [Convolutional Neural Networks with Digits Data](digits.py)
* [Deep Neural Network Classification with Iris Data](iris.py)
* [Deep Neural Network Autoencoder with Iris Data](dnn_autoencoder_iris.py)
* [Building A Custom Model](iris_custom_model.py)
* [Accessing Weights and Biases in A Custom Model](mnist_weights.py)
* [Building A Model Using Different GPU Configurations](iris_run_config.py)
* [Example of Saving and Restoring Models](iris_save_restore.py)
* [Multi-output Deep Neural Network regression](multioutput_regression.py)


## Techniques

* [Improving Performance Using Early Stopping with Iris Data](iris_val_based_early_stopping.py)
* [Using skflow with Pipeline](iris_with_pipeline.py)
* [Building A Custom Model Using Multiple GPUs](multiple_gpu.py)
* [Grid search and Deep Neural Network Classification](iris_gridsearch_cv.py)
* [Deep Neural Network with Customized Decay Function](iris_custom_decay_dnn.py)
* [Out-of-core Data Classification Using Dask](out_of_core_data_classification.py)
* [Handling Large HDF5 Dataset](hdf5_classification.py)

## Image classification

* [Convolutional Neural Networks on MNIST Data](mnist.py)
* [Recurrent Neural Networks on MNIST Data](mnist_rnn.py)
* [Deep Residual Networks on MNIST Data](resnet.py)


## Text classification

* [Text Classification Using Recurrent Neural Networks on Words](text_classification.py)
(See also [Simplified Version Using Built-in RNN Model](text_classification_builtin_rnn_model.py) using built-in parameters)
* [Text Classification Using Convolutional Neural Networks on Words](text_classification_cnn.py)
* [Text Classification Using Recurrent Neural Networks on Characters](text_classification_character_rnn.py)
* [Text Classification Using Convolutional Neural Networks on Characters](text_classification_character_cnn.py)


## Language modeling

* [Character level language modeling](language_model.py)


## Text sequence to sequence

* [Character level neural language translation](neural_translation.py)
* [Word level neural language translation](neural_translation_word.py)
