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


def batch_normalize(tensor_in, epsilon=1e-5, convnet=True, decay=0.9,
                    scale_after_normalization=True):
    """Batch Normalization

    Args:
        tensor_in: input Tensor, 4D shape:
                   [batch, in_height, in_width, in_depth].
        epsilon : A float number to avoid being divided by 0.
        decay: decay rate for exponential moving average.
        convnet: Whether this is for convolutional net use. If this is True,
                 moments will sum across axis [0, 1, 2]. Otherwise, only [0].
        scale_after_normalization: Whether to scale after normalization.
    """
    shape = tensor_in.get_shape().as_list()

    with tf.variable_scope("batch_norm"):
        gamma = tf.get_variable("gamma", [shape[-1]],
                                initializer=tf.random_normal_initializer(1., 0.02))
        beta = tf.get_variable("beta", [shape[-1]],
                               initializer=tf.constant_initializer(0.))
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        if convnet:
            assign_mean, assign_var = tf.nn.moments(tensor_in, [0, 1, 2])
        else:
            assign_mean, assign_var = tf.nn.moments(tensor_in, [0])
        ema_assign_op = ema.apply([assign_mean, assign_var])
        ema_mean, ema_var = ema.average(assign_mean), ema.average(assign_var)
        def update_mean_var():
            """Internal function that updates mean and variance during training"""
            with tf.control_dependencies([ema_assign_op]):
                return tf.identity(assign_mean), tf.identity(assign_var)
        is_training = tf.squeeze(tf.get_collection("IS_TRAINING"))
        mean, variance = tf.python.control_flow_ops.cond(
            is_training, update_mean_var, lambda: (ema_mean, ema_var))
        return tf.nn.batch_norm_with_global_normalization(
            tensor_in, mean, variance, beta, gamma, epsilon,
            scale_after_normalization=scale_after_normalization)
