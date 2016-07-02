#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
This example builds rnn network for mnist data.
Borrowed structure from here: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/recurrent_network.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import metrics, preprocessing

import tensorflow as tf
from tensorflow.contrib import learn

# Parameters
learning_rate = 0.1
training_steps = 3000
batch_size = 128

# Network Parameters
n_input = 28  # MNIST data input (img shape: 28*28)
n_steps = 28  # timesteps
n_hidden = 128  # hidden layer num of features
n_classes = 10  # MNIST total classes (0-9 digits)

### Download and load MNIST data.
mnist = learn.datasets.load_dataset('mnist')

X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

# It's useful to scale to ensure Stochastic Gradient Descent will do the right thing
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


def rnn_model(X, y):
  X = tf.reshape(X, [-1, n_steps, n_input])  # (batch_size, n_steps, n_input)
  # # permute n_steps and batch_size
  X = tf.transpose(X, [1, 0, 2])
  # # Reshape to prepare input to hidden activation
  X = tf.reshape(X, [-1, n_input])  # (n_steps*batch_size, n_input)
  # # Split data because rnn cell needs a list of inputs for the RNN inner loop
  X = tf.split(0, n_steps, X)  # n_steps * (batch_size, n_input)

  # Define a GRU cell with tensorflow
  lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
  # Get lstm cell output
  _, encoding = tf.nn.rnn(lstm_cell, X, dtype=tf.float32)

  return learn.models.logistic_regression(encoding, y)


classifier = learn.TensorFlowEstimator(model_fn=rnn_model, n_classes=n_classes,
                                       batch_size=batch_size,
                                       steps=training_steps,
                                       learning_rate=learning_rate)

classifier.fit(X_train, y_train, logdir="/tmp/mnist_rnn")
score = metrics.accuracy_score(y_test, classifier.predict(X_test))
print('Accuracy: {0:f}'.format(score))
