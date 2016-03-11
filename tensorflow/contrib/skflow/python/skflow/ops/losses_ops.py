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
        if len(labels.get_shape()) == 1:
            labels = tf.reshape(labels, [-1, 1])
        diff = labels - predictions
        loss = tf.reduce_mean(tf.mul(diff, diff))
        return predictions, loss


def softmax_classifier(tensor_in, labels, weights, biases, class_weight=None, name=None):
    """Returns prediction and loss for softmax classifier.

    Args:
        tensor_in: Input tensor, [batch_size, feature_size], features.
        labels: Tensor, [batch_size, n_classes], labels of the output classes.
        weights: Tensor, [batch_size, feature_size], linear transformation matrix.
        biases: Tensor, [batch_size], biases.
        class_weight: Tensor, optional, [n_classes], weight for each class.
                      If not given, all classes are supposed to have weight
                      one.

    Returns:
        Prediction and loss tensors.
    """
    with tf.op_scope([tensor_in, labels], name, "softmax_classifier"):
        logits = tf.nn.xw_plus_b(tensor_in, weights, biases)
        if class_weight:
            logits = tf.mul(logits, class_weight)
        xent = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                       labels,
                                                       name="xent_raw")
        loss = tf.reduce_mean(xent, name="xent")
        predictions = tf.nn.softmax(logits, name=name)
        return predictions, loss

