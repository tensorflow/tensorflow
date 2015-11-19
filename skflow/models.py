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

from skflow.ops import mean_squared_error_regressor, softmax_classifier


class LinearRegression(object):
    """Linear Regression TensorFlow model."""

    def __init__(self, input_shape, graph):
        with graph.as_default():
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.inp = tf.placeholder(tf.float32, [None, input_shape], name="input")
            self.out = tf.placeholder(tf.float32, [None], name="output")
            self.weights = tf.get_variable("weights", [input_shape, 1])
            self.bias = tf.get_variable("bias", [1])
            self.predictions, self.loss = mean_squared_error_regressor(
                self.inp, self.out, self.weights, self.bias)


class LogisticRegression(object):
    """Logistic Regression TensorFlow model."""

    def __init__(self, n_classes, input_shape, graph):
        with graph.as_default():
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.inp = tf.placeholder(tf.float32, [None, input_shape], name="input")
            self.out = tf.placeholder(tf.float32, [None, n_classes], name="output")
            self.weights = tf.get_variable("weights", [input_shape, n_classes])
            self.bias = tf.get_variable("bias", [n_classes])
            self.predictions, self.loss = softmax_classifier(
                self.inp, self.out, self.weights, self.bias)

