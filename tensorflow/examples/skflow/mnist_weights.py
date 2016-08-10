# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""This demonstrates one way to access the weights of a custom skflow model.

It is otherwise identical to the standard MNIST convolutional code.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib import learn

### Download and load MNIST data.

mnist = learn.datasets.load_dataset('mnist')

### Linear classifier.

feature_columns = learn.infer_real_valued_columns_from_input(mnist.train.images)
classifier = learn.TensorFlowLinearClassifier(
    feature_columns=feature_columns, n_classes=10, batch_size=100, steps=1000,
    learning_rate=0.01)
classifier.fit(mnist.train.images, mnist.train.labels)
score = metrics.accuracy_score(
    mnist.test.labels, classifier.predict(mnist.test.images))
print('Accuracy: {0:f}'.format(score))

### Convolutional network


def max_pool_2x2(tensor_in):
  return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                        padding='SAME')


def conv_model(X, y):
  # pylint: disable=invalid-name,missing-docstring
  # reshape X to 4d tensor with 2nd and 3rd dimensions being image width and
  # height final dimension being the number of color channels
  X = tf.reshape(X, [-1, 28, 28, 1])
  # first conv layer will compute 32 features for each 5x5 patch
  with tf.variable_scope('conv_layer1'):
    h_conv1 = learn.ops.conv2d(X, n_filters=32, filter_shape=[5, 5],
                               bias=True, activation=tf.nn.relu)
    h_pool1 = max_pool_2x2(h_conv1)
  # second conv layer will compute 64 features for each 5x5 patch
  with tf.variable_scope('conv_layer2'):
    h_conv2 = learn.ops.conv2d(h_pool1, n_filters=64, filter_shape=[5, 5],
                               bias=True, activation=tf.nn.relu)
    h_pool2 = max_pool_2x2(h_conv2)
    # reshape tensor into a batch of vectors
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
  # densely connected layer with 1024 neurons
  h_fc1 = learn.ops.dnn(
      h_pool2_flat, [1024], activation=tf.nn.relu, dropout=0.5)
  return learn.models.logistic_regression(h_fc1, y)

# Training and predicting
classifier = learn.TensorFlowEstimator(
    model_fn=conv_model, n_classes=10, batch_size=100, steps=20000,
    learning_rate=0.001)
classifier.fit(mnist.train.images, mnist.train.labels)
score = metrics.accuracy_score(
    mnist.test.labels, classifier.predict(mnist.test.images))
print('Accuracy: {0:f}'.format(score))

# Examining fitted weights

## General usage is classifier.get_tensor_value('foo')
## 'foo' must be the variable scope of the desired tensor followed by the
## graph path.

## To understand the mechanism and figure out the right scope and path, you can
## do logging. Then use TensorBoard or a text editor on the log file to look at
## available strings.

## First Convolutional Layer
print('1st Convolutional Layer weights and Bias')
print(classifier.get_tensor_value('conv_layer1/convolution/filters:0'))
print(classifier.get_tensor_value('conv_layer1/convolution/bias:0'))

## Second Convolutional Layer
print('2nd Convolutional Layer weights and Bias')
print(classifier.get_tensor_value('conv_layer2/convolution/filters:0'))
print(classifier.get_tensor_value('conv_layer2/convolution/bias:0'))

## Densely Connected Layer
print('Densely Connected Layer weights')
print(classifier.get_tensor_value('dnn/layer0/Linear/Matrix:0'))

## Logistic Regression weights
print('Logistic Regression weights')
print(classifier.get_tensor_value('logistic_regression/weights:0'))
