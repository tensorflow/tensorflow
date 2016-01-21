"""Various high level TF models."""
#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
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

from __future__ import division, print_function, absolute_import

import tensorflow as tf

from skflow.ops import mean_squared_error_regressor, softmax_classifier, dnn

from tensorflow.models.rnn import rnn_cell


def linear_regression(X, y):
    """Creates linear regression TensorFlow subgraph.

    Args:
        X: tensor or placeholder for input features.
        y: tensor or placeholder for target.

    Returns:
        Predictions and loss tensors.
    """
    with tf.variable_scope('linear_regression'):
        tf.histogram_summary('linear_regression.X', X)
        tf.histogram_summary('linear_regression.y', y)
        y_shape = y.get_shape()
        if len(y_shape) == 1:
            output_shape = 1
        else:
            output_shape = y_shape[1]
        weights = tf.get_variable('weights', [X.get_shape()[1], output_shape])
        bias = tf.get_variable('bias', [output_shape])
        tf.histogram_summary('linear_regression.weights', weights)
        tf.histogram_summary('linear_regression.bias', bias)
        return mean_squared_error_regressor(X, y, weights, bias)


def logistic_regression(X, y, class_weight=None):
    """Creates logistic regression TensorFlow subgraph.

    Args:
        X: tensor or placeholder for input features,
           shape should be [batch_size, n_features].
        y: tensor or placeholder for target,
           shape should be [batch_size, n_classes].
        class_weight: tensor, [n_classes], where for each class
                      it has weight of the class. If not provided
                      all ones are used.

    Returns:
        Predictions and loss tensors.
    """
    with tf.variable_scope('logistic_regression'):
        tf.histogram_summary('logistic_regression.X', X)
        tf.histogram_summary('logistic_regression.y', y)
        weights = tf.get_variable('weights', [X.get_shape()[1],
                                              y.get_shape()[-1]])
        bias = tf.get_variable('bias', [y.get_shape()[-1]])
        tf.histogram_summary('logistic_regression.weights', weights)
        tf.histogram_summary('logistic_regression.bias', bias)
        return softmax_classifier(X, y, weights, bias,
                                  class_weight=class_weight)


def get_dnn_model(hidden_units, target_predictor_fn):
    """Returns a function that creates a DNN TensorFlow subgraph with given
    params.

    Args:
        hidden_units: List of values of hidden units for layers.
        target_predictor_fn: Function that will predict target from input
                             features. This can be logistic regression,
                             linear regression or any other model,
                             that takes X, y and returns predictions and loss tensors.

    Returns:
        A function that creates the subgraph.
    """
    def dnn_estimator(X, y):
        """DNN estimator with target predictor function on top."""
        layers = dnn(X, hidden_units)
        return target_predictor_fn(layers, y)
    return dnn_estimator


def get_rnn_model(rnn_size, target_predictor_fn, cell_type='gru'):
    def rnn_estimator(X, y):
        if cell_type == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif cell_type == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif cell_type == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise ValueError("cell_type {} is not supported. ".format(cell_type))
        cell = rcell_fn(rnn_size)
        _, encoding = rnn.rnn(cell, X, dtype=tf.float32)
        return target_predictor_fn(encoding[-1], y)
