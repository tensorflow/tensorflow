#  Copyright 2015 Google Inc. All Rights Reserved.
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

import tensorflow as tf

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
        weights = tf.get_variable('weights', [X.get_shape()[1], 1])
        bias = tf.get_variable('bias', [1])
        return mean_squared_error_regressor(X, y, weights, bias)


def logistic_regression(X, y):
    """Creates logistic regression TensorFlow subgraph.

    Args:
        X: tensor or placeholder for input features.
        y: tensor or placeholder for target.

    Returns:
        Predictions and loss tensors.
    """
    with tf.variable_scope('logistic_regression'):
        weights = tf.get_variable('weights', [X.get_shape()[1], y.get_shape()[1]])
        bias = tf.get_variable('bias', [y.get_shape()[1]])
        return softmax_classifier(X, y, weights, bias)


def get_dnn_model(hidden_units, target_predictor_fn):
    """Returns a function that creates a DNN TensorFlow subgraph with given
    params.

    Args:
        hidden_units: List of values of hidden units for layers.
        target_predictor_fn: Function that will predict target from input
        features. This can be logistic regression, linear regression or any
        other model, that takes X, y and returns predictions and loss tensors.

    Returns:
        A funcition that creates the subgraph.
    """
    def dnn_classifier(X, y):
        layers = dnn(X, hidden_units)
        return target_predictor_fn(layers, y)
    return dnn_classifier

