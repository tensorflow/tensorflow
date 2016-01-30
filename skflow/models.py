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
from tensorflow.models.rnn import rnn, rnn_cell

from skflow.ops import mean_squared_error_regressor, softmax_classifier, dnn


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


def get_rnn_model(rnn_size, cell_type, num_layers, input_op_fn,
                  bidirection, target_predictor_fn):
    """Returns a function that creates a RNN TensorFlow subgraph with given
    params.

    Args:
        rnn_size: The size for rnn cell, e.g. size of your word embeddings.
        cell_type: The type of rnn cell, including rnn, gru, and lstm.
        num_layers: The number of layers of the rnn model.
        input_op_fn: Function that will transform the input tensor, such as
                     creating word embeddings, byte list, etc. This takes
                     an argument X for input and returns transformed X.
        bidirection: Whether this is a bidirectional rnn.
        target_predictor_fn: Function that will predict target from input
                             features. This can be logistic regression,
                             linear regression or any other model,
                             that takes X, y and returns predictions and loss tensors.

    Returns:
        A function that creates the subgraph.
    """
    def rnn_estimator(X, y):
        """RNN estimator with target predictor function on top."""
        X = input_op_fn(X)
        if cell_type == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif cell_type == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif cell_type == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise ValueError("cell_type {} is not supported. ".format(cell_type))
        if bidirection:
            # forward direction cell
            rnn_fw_cell = cell_fn(rnn_size)
            # backward direction cell
            rnn_bw_cell = cell_fn(rnn_size)
            encoding = rnn.bidirectional_rnn(rnn_fw_cell, rnn_bw_cell)
        else:
            cell = rnn_cell.MultiRNNCell([cell_fn(rnn_size)] * num_layers)
            _, encoding = rnn.rnn(cell, X, dtype=tf.float32)
        return target_predictor_fn(encoding[-1], y)
    return rnn_estimator
