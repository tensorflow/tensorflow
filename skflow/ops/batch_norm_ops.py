"""TensorFlow ops for Batch Normalization."""
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
from tensorflow.python import control_flow_ops


def batch_normalize(X, epsilon=1e-5, scale_after_normalization=True):
    """Batch Normalization

    Args:
        X: Input Tensor
        epsilon : A float number to avoid being divided by 0.
        scale_after_normalization: Whether to scale after normalization.
    """
    shape = X.get_shape().as_list()

    with tf.variable_scope("batch_norm"):
        gamma = tf.get_variable("gamma", [shape[-1]],
                                initializer=tf.random_normal_initializer(1., 0.02))
        beta = tf.get_variable("beta", [shape[-1]],
                               initializer=tf.constant_initializer(0.))
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        assign_mean, assign_var = tf.nn.moments(X, [0, 1, 2])
        ema_assign_op = ema.apply([assign_mean, assign_var])
        ema_mean, ema_var = ema.average(assign_mean), ema.average(assign_var)
        def update_mean_var():
            with tf.control_dependencies([ema_assign_op]):
                return tf.identity(assign_mean), tf.identity(assign_var)
        IS_TRAINING = tf.get_collection("IS_TRAINING") # TODO: this is always empty
        mean, variance = control_flow_ops.cond(IS_TRAINING,
                                               update_mean_var,
                                               lambda: (ema_mean, ema_var))
        return tf.nn.batch_norm_with_global_normalization(
            X, mean, variance, beta, gamma, epsilon,
            scale_after_normalization=scale_after_normalization)
