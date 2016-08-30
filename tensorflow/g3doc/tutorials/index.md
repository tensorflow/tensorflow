# Tutorials

## Basic Neural Networks

The first few Tensorflow tutorials guide you through training and testing a
simple neural network to classify handwritten digits from the MNIST database of
digit images.

### MNIST For ML Beginners

If you're new to machine learning, we recommend starting here.  You'll learn
about a classic problem, handwritten digit classification (MNIST), and get a
gentle introduction to multiclass classification.

[View Tutorial](../tutorials/mnist/beginners/index.md)


### Deep MNIST for Experts

If you're already familiar with other deep learning software packages, and are
already familiar with MNIST, this tutorial will give you a very brief primer
on TensorFlow.

[View Tutorial](../tutorials/mnist/pros/index.md)

### TensorFlow Mechanics 101

This is a technical tutorial, where we walk you through the details of using
TensorFlow infrastructure to train models at scale.  We use MNIST as the
example.

[View Tutorial](../tutorials/mnist/tf/index.md)


## Easy ML with tf.contrib.learn

### tf.contrib.learn Quickstart

A quick introduction to tf.contrib.learn, a high-level API for TensorFlow.
Build, train, and evaluate a neural network with just a few lines of
code.

[View Tutorial](../tutorials/tflearn/index.md)

### Overview of Linear Models with tf.contrib.learn

An overview of tf.contrib.learn's rich set of tools for working with linear
models in TensorFlow.

[View Tutorial](../tutorials/linear/overview.md)

### Linear Model Tutorial

This tutorial walks you through the code for building a linear model using
tf.contrib.learn.

[View Tutorial](../tutorials/wide/index.md)

### Wide and Deep Learning Tutorial

This tutorial shows you how to use tf.contrib.learn to jointly train a linear
model and a deep neural net to harness the advantages of each type of model.

[View Tutorial](../tutorials/wide_and_deep/index.md)

### Logging and Monitoring Basics with tf.contrib.learn

This tutorial shows you how to use TensorFlow’s logging capabilities and the
Monitor API to audit the in-progress training of a neural network.

[View Tutorial](../tutorials/monitors/index.md)

### Building Input Functions with tf.contrib.learn

This tutorial introduces you to creating input functions in tf.contrib.learn,
and walks you through implementing an `input_fn` to train a neural network
for predicting median house values.

[View Tutorial](../tutorials/input_fn/index.md)

## TensorFlow Serving

### TensorFlow Serving

An introduction to TensorFlow Serving, a flexible, high-performance system for
serving machine learning models, designed for production environments.

[View Tutorial](../tutorials/tfserve/index.md)


## Image Processing

### Convolutional Neural Networks

An introduction to convolutional neural networks using the CIFAR-10 data set.
Convolutional neural nets are particularly tailored to images, since they
exploit translation invariance to yield more compact and effective
representations of visual content.

[View Tutorial](../tutorials/deep_cnn/index.md)

### Image Recognition

How to run object recognition using a convolutional neural network
trained on ImageNet Challenge data and label set.

[View Tutorial](../tutorials/image_recognition/index.md)

### Deep Dream Visual Hallucinations

Building on the Inception recognition model, we will release a TensorFlow
version of the [Deep Dream](https://github.com/google/deepdream) neural network
visual hallucination software.

[View Tutorial](https://www.tensorflow.org/code/tensorflow/examples/tutorials/deepdream/deepdream.ipynb)


## Language and Sequence Processing

### Vector Representations of Words

This tutorial motivates why it is useful to learn to represent words as vectors
(called *word embeddings*). It introduces the word2vec model as an efficient
method for learning embeddings. It also covers the high-level details behind
noise-contrastive training methods (the biggest recent advance in training
embeddings).

[View Tutorial](../tutorials/word2vec/index.md)

### Recurrent Neural Networks

An introduction to RNNs, wherein we train an LSTM network to predict the next
word in an English sentence.  (A task sometimes called language modeling.)

[View Tutorial](../tutorials/recurrent/index.md)

### Sequence-to-Sequence Models

A follow on to the RNN tutorial, where we assemble a sequence-to-sequence model
for machine translation.  You will learn to build your own English-to-French
translator, entirely machine learned, end-to-end.

[View Tutorial](../tutorials/seq2seq/index.md)

### SyntaxNet: Neural Models of Syntax

An introduction to SyntaxNet, a Natural Language Processing framework for
TensorFlow.

[View Tutorial](../tutorials/syntaxnet/index.md)


## Non-ML Applications

### Mandelbrot Set

TensorFlow can be used for computation that has nothing to do with machine
learning.  Here's a naive implementation of Mandelbrot set visualization.

[View Tutorial](../tutorials/mandelbrot/index.md)

### Partial Differential Equations

As another example of non-machine learning computation, we offer an example of
a naive PDE simulation of raindrops landing on a pond.

[View Tutorial](../tutorials/pdes/index.md)
