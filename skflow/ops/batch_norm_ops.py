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

def batch_normalize(X, batch_size=-1, epsilon=1e-5, momentum=0.1):
    """Batch Normalization

    Args:
        X: Input Tensor
        batch_size : Size of the batch, or -1 for size to fit.
        epsilon : A float number to avoid being divided by 0.
        momentum : momentum for the moving average.
    """
    shape = X.get_shape().as_list()

    with tf.variable_scope("batch_norm"):
        ema = tf.train.ExponentialMovingAverage(decay=momentum)
        gamma = tf.get_variable("gamma", [shape[-1]],
                                initializer=tf.random_normal_initializer(1., 0.02))
        beta = tf.get_variable("beta", [shape[-1]],
                                initializer=tf.constant_initializer(0.))
        mean, variance = tf.nn.moments(X, [0, 1, 2])
        return tf.nn.batch_norm_with_global_normalization(
            X, mean, variance, beta, gamma, epsilon,
            scale_after_normalization=True)
