# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# pylint: disable=unused-import,g-bad-import-order
"""Contains the normalization layer classes and their functional aliases.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.training import moving_averages
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import variables

from tensorflow.python.layers import base
from tensorflow.python.layers import utils


class BatchNormalization(base._Layer):  # pylint: disable=protected-access
  """Batch Normalization layer from http://arxiv.org/abs/1502.03167.

  "Batch Normalization: Accelerating Deep Network Training by Reducing
  Internal Covariate Shift"

  Sergey Ioffe, Christian Szegedy

  Arguments:
    axis: Integer, the axis that should be normalized (typically the features
      axis). For instance, after a `Convolution2D` layer with
      `data_format="channels_first"`, set `axis=1` in `BatchNormalization`.
    momentum: Momentum for the moving average.
    epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    moving_mean_initializer: Initializer for the moving mean.
    moving_variance_initializer: Initializer for the moving variance.
    beta_regularizer: Optional regularizer for the beta weight.
    gamma_regularizer: Optional regularizer for the gamma weight.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    name: A string, the name of the layer.
  """

  def __init__(self,
               axis=-1,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer=init_ops.zeros_initializer(),
               gamma_initializer=init_ops.ones_initializer(),
               moving_mean_initializer=init_ops.zeros_initializer(),
               moving_variance_initializer=init_ops.ones_initializer(),
               beta_regularizer=None,
               gamma_regularizer=None,
               trainable=True,
               name=None,
               **kwargs):
    super(BatchNormalization, self).__init__(
        name=name, trainable=trainable, **kwargs)
    self.axis = axis
    self.momentum = momentum
    self.epsilon = epsilon
    self.center = center
    self.scale = scale
    self.beta_initializer = beta_initializer
    self.gamma_initializer = gamma_initializer
    self.moving_mean_initializer = moving_mean_initializer
    self.moving_variance_initializer = moving_variance_initializer
    self.beta_regularizer = beta_regularizer
    self.gamma_regularizer = gamma_regularizer

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if not input_shape.ndims:
      raise ValueError('Input has undefined rank:', input_shape)
    ndim = len(input_shape)
    if self.axis < 0:
      axis = ndim + self.axis
    else:
      axis = self.axis
    if axis < 0 or axis >= ndim:
      raise ValueError('Value of `axis` argument ' + str(self.axis) +
                       ' is out of range for input with rank ' + str(ndim))
    param_dim = input_shape[axis]
    if not param_dim.value:
      raise ValueError('Input has undefined `axis` dimension. Input shape: ',
                       input_shape)

    if self.center:
      self.beta = vs.get_variable('beta',
                                  shape=(param_dim,),
                                  initializer=self.beta_initializer,
                                  regularizer=self.beta_regularizer,
                                  trainable=True)
    else:
      self.beta = None
    if self.scale:
      self.gamma = vs.get_variable('gamma',
                                   shape=(param_dim,),
                                   initializer=self.gamma_initializer,
                                   regularizer=self.gamma_regularizer,
                                   trainable=True)
    else:
      self.gamma = None

    # Disable variable partitioning when creating the moving mean and variance
    partitioner = vs.get_variable_scope().partitioner
    try:
      vs.get_variable_scope().set_partitioner(None)
      self.moving_mean = vs.get_variable(
          'moving_mean',
          shape=(param_dim,),
          initializer=self.moving_mean_initializer,
          trainable=False)
      self.moving_variance = vs.get_variable(
          'moving_variance',
          shape=(param_dim,),
          initializer=self.moving_variance_initializer,
          trainable=False)
    finally:
      vs.get_variable_scope().set_partitioner(partitioner)

  def call(self, inputs, training=False):
    # First, compute the axes along which to reduce the mean / variance,
    # as well as the broadcast shape to be used for all parameters.
    input_shape = inputs.get_shape()
    ndim = len(input_shape)
    reduction_axes = list(range(len(input_shape)))
    del reduction_axes[self.axis]
    broadcast_shape = [1] * len(input_shape)
    broadcast_shape[self.axis] = input_shape[self.axis].value

    # Determines whether broadcasting is needed.
    needs_broadcasting = (sorted(reduction_axes) != range(ndim)[:-1])

    # Determine a boolean value for `training`: could be True, False, or None.
    training_value = utils.constant_value(training)

    if needs_broadcasting:
      # In this case we must explictly broadcast all parameters.
      if self.center:
        broadcast_beta = array_ops.reshape(self.beta, broadcast_shape)
      else:
        broadcast_beta = None
      if self.scale:
        broadcast_gamma = array_ops.reshape(self.gamma, broadcast_shape)
      else:
        broadcast_gamma = None

    if training_value is not False:
      if needs_broadcasting:
        broadcast_mean, broadcast_variance = nn.moments(
            inputs, reduction_axes, keep_dims=True)
        mean = array_ops.reshape(broadcast_mean, [-1])
        variance = array_ops.reshape(broadcast_variance, [-1])
      else:
        mean, variance = nn.moments(inputs, reduction_axes)

      # Prepare updates if necessary.
      if not self.updates:
        mean_update = moving_averages.assign_moving_average(
            self.moving_mean, mean, self.momentum, zero_debias=False)
        variance_update = moving_averages.assign_moving_average(
            self.moving_variance, variance, self.momentum, zero_debias=False)
        # In the future this should be refactored into a self.add_update
        # methods in order to allow for instance-based BN layer sharing
        # across unrelated input streams (e.g. like in Keras).
        self.updates.append(mean_update)
        self.updates.append(variance_update)

    # Normalize batch. We do this inside separate functions for training
    # and inference so as to avoid evaluating both branches.
    def normalize_in_test():
      if needs_broadcasting:
        broadcast_moving_mean = array_ops.reshape(self.moving_mean,
                                                  broadcast_shape)
        broadcast_moving_variance = array_ops.reshape(self.moving_variance,
                                                      broadcast_shape)
        return nn.batch_normalization(inputs,
                                      broadcast_moving_mean,
                                      broadcast_moving_variance,
                                      broadcast_beta,
                                      broadcast_gamma,
                                      self.epsilon)
      else:
        return nn.batch_normalization(inputs,
                                      self.moving_mean,
                                      self.moving_variance,
                                      self.beta if self.center else None,
                                      self.gamma if self.scale else None,
                                      self.epsilon)

    def normalize_in_training():
      if needs_broadcasting:
        return nn.batch_normalization(inputs,
                                      broadcast_mean,
                                      broadcast_variance,
                                      broadcast_beta,
                                      broadcast_gamma,
                                      self.epsilon)
      else:
        return nn.batch_normalization(inputs,
                                      mean,
                                      variance,
                                      self.beta if self.center else None,
                                      self.gamma if self.scale else None,
                                      self.epsilon)

    return utils.smart_cond(training,
                            normalize_in_training,
                            normalize_in_test)


def batch_normalization(inputs,
                        axis=-1,
                        momentum=0.99,
                        epsilon=1e-3,
                        center=True,
                        scale=True,
                        beta_initializer=init_ops.zeros_initializer(),
                        gamma_initializer=init_ops.ones_initializer(),
                        moving_mean_initializer=init_ops.zeros_initializer(),
                        moving_variance_initializer=init_ops.ones_initializer(),
                        beta_regularizer=None,
                        gamma_regularizer=None,
                        training=False,
                        trainable=True,
                        name=None,
                        reuse=None):
  """Functional interface for the batch normalization layer.

  Reference: http://arxiv.org/abs/1502.03167

  "Batch Normalization: Accelerating Deep Network Training by Reducing
  Internal Covariate Shift"

  Sergey Ioffe, Christian Szegedy

  Arguments:
    inputs: Tensor input.
    axis: Integer, the axis that should be normalized (typically the features
      axis). For instance, after a `Convolution2D` layer with
      `data_format="channels_first"`, set `axis=1` in `BatchNormalization`.
    momentum: Momentum for the moving average.
    epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    moving_mean_initializer: Initializer for the moving mean.
    moving_variance_initializer: Initializer for the moving variance.
    beta_regularizer: Optional regularizer for the beta weight.
    gamma_regularizer: Optional regularizer for the gamma weight.
    training: Either a Python boolean, or a TensorFlow boolean scalar tensor
      (e.g. a placeholder). Whether to return the output in training mode
      (normalized with statistics of the current batch) or in inference mode
      (normalized with moving statistics).
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    name: String, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns:
    Output tensor.
  """
  layer = BatchNormalization(
      axis=axis,
      momentum=momentum,
      epsilon=epsilon,
      center=center,
      scale=scale,
      beta_initializer=beta_initializer,
      gamma_initializer=gamma_initializer,
      moving_mean_initializer=moving_mean_initializer,
      moving_variance_initializer=moving_variance_initializer,
      beta_regularizer=beta_regularizer,
      gamma_regularizer=gamma_regularizer,
      trainable=trainable,
      name=name,
      _reuse=reuse,
      _scope=name)
  return layer.apply(inputs, training=training)


# Aliases

BatchNorm = BatchNormalization
batch_norm = batch_normalization
