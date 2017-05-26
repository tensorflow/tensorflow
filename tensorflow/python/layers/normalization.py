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


class BatchNormalization(base.Layer):
  """Batch Normalization layer from http://arxiv.org/abs/1502.03167.

  "Batch Normalization: Accelerating Deep Network Training by Reducing
  Internal Covariate Shift"

  Sergey Ioffe, Christian Szegedy

  Arguments:
    axis: Integer, the axis that should be normalized (typically the features
      axis). For instance, after a `Conv2D` layer with
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
    renorm: Whether to use Batch Renormalization
      (https://arxiv.org/abs/1702.03275). This adds extra variables during
      training. The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
      scalar `Tensors` used to clip the renorm correction. The correction
      `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
      `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
      dmax are set to inf, 0, inf, respectively.
    renorm_momentum: Momentum used to update the moving means and standard
      deviations with renorm. Unlike `momentum`, this affects training
      and should be neither too small (which would add noise) nor too large
      (which would give stale estimates). Note that `momentum` is still applied
      to get the means and variances for inference.
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
               renorm=False,
               renorm_clipping=None,
               renorm_momentum=0.99,
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
    self.renorm = renorm
    if renorm:
      renorm_clipping = renorm_clipping or {}
      keys = ['rmax', 'rmin', 'dmax']
      if set(renorm_clipping) - set(keys):
        raise ValueError('renorm_clipping %s contains keys not in %s' %
                         (renorm_clipping, keys))
      self.renorm_clipping = renorm_clipping
      self.renorm_momentum = renorm_momentum

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
    self.input_spec = base.InputSpec(ndim=ndim,
                                     axes={self.axis: param_dim.value})

    if self.center:
      self.beta = self.add_variable(name='beta',
                                    shape=(param_dim,),
                                    initializer=self.beta_initializer,
                                    regularizer=self.beta_regularizer,
                                    trainable=True)
    else:
      self.beta = None
    if self.scale:
      self.gamma = self.add_variable(name='gamma',
                                     shape=(param_dim,),
                                     initializer=self.gamma_initializer,
                                     regularizer=self.gamma_regularizer,
                                     trainable=True)
    else:
      self.gamma = None

    # Disable variable partitioning when creating the moving mean and variance
    partitioner = self._scope.partitioner
    try:
      self._scope.set_partitioner(None)
      self.moving_mean = self.add_variable(
          name='moving_mean',
          shape=(param_dim,),
          initializer=self.moving_mean_initializer,
          trainable=False)
      self.moving_variance = self.add_variable(
          name='moving_variance',
          shape=(param_dim,),
          initializer=self.moving_variance_initializer,
          trainable=False)
      if self.renorm:
        # Create variables to maintain the moving mean and standard deviation.
        # These are used in training and thus are different from the moving
        # averages above. The renorm variables are colocated with moving_mean
        # and moving_variance.
        # NOTE: below, the outer `with device` block causes the current device
        # stack to be cleared. The nested ones use a `lambda` to set the desired
        # device and ignore any devices that may be set by the custom getter.
        def _renorm_variable(name, shape):
          var = self.add_variable(name=name,
                                  shape=shape,
                                  initializer=init_ops.zeros_initializer(),
                                  trainable=False)
          return var
        with ops.device(None):
          with ops.device(lambda _: self.moving_mean.device):
            self.renorm_mean = _renorm_variable('renorm_mean', (param_dim,))
            self.renorm_mean_weight = _renorm_variable('renorm_mean_weight', ())
          # We initialize renorm_stddev to 0, and maintain the (0-initialized)
          # renorm_stddev_weight. This allows us to (1) mix the average
          # stddev with the minibatch stddev early in training, and (2) compute
          # the unbiased average stddev by dividing renorm_stddev by the weight.
          with ops.device(lambda _: self.moving_variance.device):
            self.renorm_stddev = _renorm_variable('renorm_stddev', (param_dim,))
            self.renorm_stddev_weight = _renorm_variable(
                'renorm_stddev_weight', ())
    finally:
      self._scope.set_partitioner(partitioner)
    self.built = True

  def _renorm_correction_and_moments(self, mean, variance, training):
    """Returns the correction and update values for renorm."""
    stddev = math_ops.sqrt(variance + self.epsilon)
    # Compute the average mean and standard deviation, as if they were
    # initialized with this batch's moments.
    mixed_renorm_mean = (self.renorm_mean +
                         (1. - self.renorm_mean_weight) * mean)
    mixed_renorm_stddev = (self.renorm_stddev +
                           (1. - self.renorm_stddev_weight) * stddev)
    # Compute the corrections for batch renorm.
    r = stddev / mixed_renorm_stddev
    d = (mean - mixed_renorm_mean) / mixed_renorm_stddev
    # Ensure the corrections use pre-update moving averages.
    with ops.control_dependencies([r, d]):
      mean = array_ops.identity(mean)
      stddev = array_ops.identity(stddev)
    rmin, rmax, dmax = [self.renorm_clipping.get(key)
                        for key in ['rmin', 'rmax', 'dmax']]
    if rmin is not None:
      r = math_ops.maximum(r, rmin)
    if rmax is not None:
      r = math_ops.minimum(r, rmax)
    if dmax is not None:
      d = math_ops.maximum(d, -dmax)
      d = math_ops.minimum(d, dmax)
    # When not training, use r=1, d=0, and decay=1 meaning no updates.
    r = _smart_select(training, lambda: r, lambda: array_ops.ones_like(r))
    d = _smart_select(training, lambda: d, lambda: array_ops.zeros_like(d))
    decay = _smart_select(training, lambda: self.renorm_momentum, lambda: 1.)
    def _update_renorm_variable(var, weight, value):
      """Updates a moving average and weight, returns the unbiased value."""
      # Update the variables without zero debiasing. The debiasing will be
      # accomplished by dividing the exponential moving average by the weight.
      # For example, after a single update, the moving average would be
      # (1-decay) * value. and the weight will be 1-decay, with their ratio
      # giving value.
      # Make sure the weight is not updated until before r and d computation.
      value = array_ops.identity(value)
      with ops.control_dependencies([value]):
        weight_value = array_ops.constant(1., dtype=weight.dtype)
      new_var = moving_averages.assign_moving_average(
          var, value, decay, zero_debias=False)
      new_weight = moving_averages.assign_moving_average(
          weight, weight_value, decay, zero_debias=False)
      return new_var / new_weight

    with ops.colocate_with(self.moving_mean):
      new_mean = _update_renorm_variable(self.renorm_mean,
                                         self.renorm_mean_weight,
                                         mean)
    with ops.colocate_with(self.moving_variance):
      new_stddev = _update_renorm_variable(self.renorm_stddev,
                                           self.renorm_stddev_weight,
                                           stddev)
      # Make sqrt(moving_variance + epsilon) = new_stddev.
      new_variance = math_ops.square(new_stddev) - self.epsilon

    return (r, d, new_mean, new_variance)

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
    needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

    scale, offset = self.gamma, self.beta

    # Determine a boolean value for `training`: could be True, False, or None.
    training_value = utils.constant_value(training)
    if training_value is not False:
      # Some of the computations here are not necessary when training==False
      # but not a constant. However, this makes the code simpler.
      mean, variance = nn.moments(inputs, reduction_axes)
      mean = _smart_select(training,
                           lambda: mean,
                           lambda: self.moving_mean)
      variance = _smart_select(training,
                               lambda: variance,
                               lambda: self.moving_variance)

      if self.renorm:
        r, d, new_mean, new_variance = self._renorm_correction_and_moments(
            mean, variance, training)
        # When training, the normalized values (say, x) will be transformed as
        # x * gamma + beta without renorm, and (x * r + d) * gamma + beta
        # = x * (r * gamma) + (d * gamma + beta) with renorm.
        scale = array_ops.stop_gradient(r, name='renorm_r')
        offset = array_ops.stop_gradient(d, name='renorm_d')
        if self.gamma is not None:
          scale *= self.gamma
          offset *= self.gamma
        if self.beta is not None:
          offset += self.beta
      else:
        new_mean, new_variance = mean, variance

      # Update moving averages when training, and prevent updates otherwise.
      decay = _smart_select(training, lambda: self.momentum, lambda: 1.)
      mean_update = moving_averages.assign_moving_average(
          self.moving_mean, new_mean, decay, zero_debias=False)
      variance_update = moving_averages.assign_moving_average(
          self.moving_variance, new_variance, decay, zero_debias=False)

      self.add_update(mean_update, inputs=inputs)
      self.add_update(variance_update, inputs=inputs)

    else:
      mean, variance = self.moving_mean, self.moving_variance

    def _broadcast(v):
      if needs_broadcasting and v is not None:
        # In this case we must explicitly broadcast all parameters.
        return array_ops.reshape(v, broadcast_shape)
      return v

    return nn.batch_normalization(inputs,
                                  _broadcast(mean),
                                  _broadcast(variance),
                                  _broadcast(offset),
                                  _broadcast(scale),
                                  self.epsilon)


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
                        reuse=None,
                        renorm=False,
                        renorm_clipping=None,
                        renorm_momentum=0.99):
  """Functional interface for the batch normalization layer.

  Reference: http://arxiv.org/abs/1502.03167

  "Batch Normalization: Accelerating Deep Network Training by Reducing
  Internal Covariate Shift"

  Sergey Ioffe, Christian Szegedy

  Note: when training, the moving_mean and moving_variance need to be updated.
  By default the update ops are placed in `tf.GraphKeys.UPDATE_OPS`, so they
  need to be added as a dependency to the `train_op`. For example:

  ```python
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss)
  ```

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
      (normalized with moving statistics). **NOTE**: make sure to set this
      parameter correctly, or else your training/inference will not work
      properly.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    name: String, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.
    renorm: Whether to use Batch Renormalization
      (https://arxiv.org/abs/1702.03275). This adds extra variables during
      training. The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
      scalar `Tensors` used to clip the renorm correction. The correction
      `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
      `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
      dmax are set to inf, 0, inf, respectively.
    renorm_momentum: Momentum used to update the moving means and standard
      deviations with renorm. Unlike `momentum`, this affects training
      and should be neither too small (which would add noise) nor too large
      (which would give stale estimates). Note that `momentum` is still applied
      to get the means and variances for inference.

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
      renorm=renorm,
      renorm_clipping=renorm_clipping,
      renorm_momentum=renorm_momentum,
      name=name,
      _reuse=reuse,
      _scope=name)
  return layer.apply(inputs, training=training)


# Aliases

BatchNorm = BatchNormalization
batch_norm = batch_normalization


# Helper function


def _smart_select(pred, fn_then, fn_else):
  """Selects fn_then() or fn_else() based on the value of pred.

  The purpose of this function is the same as `utils.smart_cond`. However, at
  the moment there is a bug (b/36297356) that seems to kick in only when
  `smart_cond` delegates to `tf.cond`, which sometimes results in the training
  hanging when using parameter servers. This function will output the result
  of `fn_then` or `fn_else` if `pred` is known at graph construction time.
  Otherwise, it will use `tf.where` which will result in some redundant work
  (both branches will be computed but only one selected). However, the tensors
  involved will usually be small (means and variances in batchnorm), so the
  cost will be small and will not be incurred at all if `pred` is a constant.

  Args:
    pred: A boolean scalar `Tensor`.
    fn_then: A callable to use when pred==True.
    fn_else: A callable to use when pred==False.

  Returns:
    A `Tensor` whose value is fn_then() or fn_else() based on the value of pred.
  """
  pred_value = utils.constant_value(pred)
  if pred_value:
    return fn_then()
  elif pred_value is False:
    return fn_else()
  t_then = array_ops.expand_dims(fn_then(), 0)
  t_else = array_ops.expand_dims(fn_else(), 0)
  pred = array_ops.reshape(pred, [1])
  result = array_ops.where(pred, t_then, t_else)
  return array_ops.squeeze(result, [0])
