"""TensorFlow Ops for loss computation."""
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


def mean_squared_error_regressor(tensor_in, labels, weights, biases, name=None):
    """Returns prediction and loss for mean squared error regression."""
    with tf.op_scope([tensor_in, labels], name, "mean_squared_error_regressor"):
        predictions = tf.nn.xw_plus_b(tensor_in, weights, biases)
        diff = predictions - labels
        loss = tf.reduce_mean(tf.mul(diff, diff))
        return predictions, loss


def softmax_classifier(tensor_in, labels, weights, biases, name=None):
    """Returns prediction and loss for softmax classifier."""
    with tf.op_scope([tensor_in, labels], name, "softmax_classifier"):
        logits = tf.nn.xw_plus_b(tensor_in, weights, biases)
        xent = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                       labels,
                                                       name="xent_raw")
        loss = tf.reduce_mean(xent, name="xent")
        predictions = tf.nn.softmax(logits, name=name)
        return predictions, loss
