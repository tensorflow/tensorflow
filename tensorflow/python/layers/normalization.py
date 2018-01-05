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

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import moving_averages


class BatchNormalization(base.Layer):
  """Batch Normalization layer from http://arxiv.org/abs/1502.03167.

  "Batch Normalization: Accelerating Deep Network Training by Reducing
  Internal Covariate Shift"

  Sergey Ioffe, Christian Szegedy

  Arguments:
    axis: An `int` or list of `int`, the axis or axes that should be
        normalized, typically the features axis/axes. For instance, after a
        `Conv2D` layer with `data_format="channels_first"`, set `axis=1`. If a
        list of axes is provided, each axis in `axis` will be normalized
        simultaneously. Default is `-1` which takes uses last axis. Note: when
        using multi-axis batch norm, the `beta`, `gamma`, `moving_mean`, and
        `moving_variance` variables are the same rank as the input Tensor, with
        dimension size 1 in all reduced (non-axis) dimensions).
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
    beta_constraint: An optional projection function to be applied to the `beta`
        weight after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    gamma_constraint: An optional projection function to be applied to the
        `gamma` weight after being updated by an `Optimizer`.
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
    fused: if `True`, use a faster, fused implementation if possible.
      If `None`, use the system recommended implementation.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    virtual_batch_size: An `int`. By default, `virtual_batch_size` is `None`,
      which means batch normalization is performed across the whole batch. When
      `virtual_batch_size` is not `None`, instead perform "Ghost Batch
      Normalization", which creates virtual sub-batches which are each
      normalized separately (with shared gamma, beta, and moving statistics).
      Must divide the actual batch size during execution.
    adjustment: A function taking the `Tensor` containing the (dynamic) shape of
      the input tensor and returning a pair (scale, bias) to apply to the
      normalized values (before gamma and beta), only during training. For
      example, if axis==-1,
        `adjustment = lambda shape: (
          tf.random_uniform(shape[-1:], 0.93, 1.07),
          tf.random_uniform(shape[-1:], -0.1, 0.1))`
      will scale the normalized value by up to 7% up or down, then shift the
      result by up to 0.1 (with independent scaling and bias for each feature
      but shared across all examples), and finally apply gamma and/or beta. If
      `None`, no adjustment is applied. Cannot be specified if
      virtual_batch_size is specified.
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
               beta_constraint=None,
               gamma_constraint=None,
               renorm=False,
               renorm_clipping=None,
               renorm_momentum=0.99,
               fused=None,
               trainable=True,
               virtual_batch_size=None,
               adjustment=None,
               name=None,
               **kwargs):
    super(BatchNormalization, self).__init__(
        name=name, trainable=trainable, **kwargs)
    if isinstance(axis, list):
      self.axis = axis[:]
    else:
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
    self.beta_constraint = beta_constraint
    self.gamma_constraint = gamma_constraint
    self.renorm = renorm
    self.virtual_batch_size = virtual_batch_size
    self.adjustment = adjustment
    if fused is None:
      fused = True

    self.fused = fused
    self._bessels_correction_test_only = True

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
    ndims = len(input_shape)

    # Convert axis to list and resolve negatives
    if isinstance(self.axis, int):
      self.axis = [self.axis]

    if not isinstance(self.axis, list):
      raise TypeError('axis must be int or list, type given: %s'
                      % type(self.axis))

    for idx, x in enumerate(self.axis):
      if x < 0:
        self.axis[idx] = ndims + x

    # Validate axes
    for x in self.axis:
      if x < 0 or x >= ndims:
        raise ValueError('Invalid axis: %d' % x)
    if len(self.axis) != len(set(self.axis)):
      raise ValueError('Duplicate axis: %s' % self.axis)

    if self.virtual_batch_size is not None:
      if self.virtual_batch_size <= 0:
        raise ValueError('virtual_batch_size must be a positive integer that '
                         'divides the true batch size of the input Tensor')
      # If using virtual batches, the first dimension must be the batch
      # dimension and cannot be the batch norm axis
      if 0 in self.axis:
        raise ValueError('When using virtual_batch_size, the batch dimension '
                         'must be 0 and thus axis cannot include 0')
      if self.adjustment is not None:
        raise ValueError('When using virtual_batch_size, adjustment cannot '
                         'be specified')

    if self.fused:
      # Currently fused batch norm doesn't support renorm. It also only supports
      # an input tensor of rank 4 and a channel dimension on axis 1 or 3.
      # TODO(yaozhang): if input is not 4D, reshape it to 4D and reshape the
      # output back to its original shape accordingly.
      self.fused = (not self.renorm and
                    ndims == 4 and
                    self.axis in [[1], [3]] and
                    self.virtual_batch_size is None and
                    self.adjustment is None)
      # TODO(chrisying): fused batch norm is currently not supported for
      # multi-axis batch norm and by extension virtual batches. In some cases,
      # it might be possible to use fused batch norm but would require reshaping
      # the Tensor to 4D with the axis in 1 or 3 (preferred 1) which is
      # particularly tricky. A compromise might be to just support the most
      # common use case (turning 5D w/ virtual batch to NCHW)

    if self.fused:
      if self.axis == [1]:
        self._data_format = 'NCHW'
      elif self.axis == [3]:
        self._data_format = 'NHWC'
      else:
        raise ValueError('Unsupported axis, fused batch norm only supports '
                         'axis == [1] or axis == [3]')

    # Raise parameters of fp16 batch norm to fp32
    if self.dtype == dtypes.float16 or self.dtype == dtypes.bfloat16:
      param_dtype = dtypes.float32
    else:
      param_dtype = self.dtype or dtypes.float32

    axis_to_dim = {x: input_shape[x].value for x in self.axis}
    for x in axis_to_dim:
      if axis_to_dim[x] is None:
        raise ValueError('Input has undefined `axis` dimension. Input shape: ',
                         input_shape)
    self.input_spec = base.InputSpec(ndim=ndims, axes=axis_to_dim)

    if len(axis_to_dim) == 1 and self.virtual_batch_size is None:
      # Single axis batch norm (most common/default use-case)
      param_shape = (list(axis_to_dim.values())[0],)
    else:
      # Parameter shape is the original shape but with 1 in all non-axis dims
      param_shape = [axis_to_dim[i] if i in axis_to_dim
                     else 1 for i in range(ndims)]
      if self.virtual_batch_size is not None:
        # When using virtual batches, add an extra dim at index 1
        param_shape.insert(1, 1)
        for idx, x in enumerate(self.axis):
          self.axis[idx] = x + 1      # Account for added dimension

    if self.scale:
      self.gamma = self.add_variable(
          name='gamma',
          shape=param_shape,
          dtype=param_dtype,
          initializer=self.gamma_initializer,
          regularizer=self.gamma_regularizer,
          constraint=self.gamma_constraint,
          trainable=True)
    else:
      self.gamma = None
      if self.fused:
        self._gamma_const = array_ops.constant(
            1.0, dtype=param_dtype, shape=param_shape)

    if self.center:
      self.beta = self.add_variable(
          name='beta',
          shape=param_shape,
          dtype=param_dtype,
          initializer=self.beta_initializer,
          regularizer=self.beta_regularizer,
          constraint=self.beta_constraint,
          trainable=True)
    else:
      self.beta = None
      if self.fused:
        self._beta_const = array_ops.constant(
            0.0, dtype=param_dtype, shape=param_shape)

    # Disable variable partitioning when creating the moving mean and variance
    try:
      if self._scope:
        partitioner = self._scope.partitioner
        self._scope.set_partitioner(None)
      else:
        partitioner = None
      self.moving_mean = self.add_variable(
          name='moving_mean',
          shape=param_shape,
          dtype=param_dtype,
          initializer=self.moving_mean_initializer,
          trainable=False)

      self.moving_variance = self.add_variable(
          name='moving_variance',
          shape=param_shape,
          dtype=param_dtype,
          initializer=self.moving_variance_initializer,
          trainable=False)

      self._one_minus_decay = 1.0 - self.momentum
      if self.renorm:
        # Create variables to maintain the moving mean and standard deviation.
        # These are used in training and thus are different from the moving
        # averages above. The renorm variables are colocated with moving_mean
        # and moving_variance.
        # NOTE: below, the outer `with device` block causes the current device
        # stack to be cleared. The nested ones use a `lambda` to set the desired
        # device and ignore any devices that may be set by the custom getter.
        def _renorm_variable(name, shape):
          var = self.add_variable(
              name=name,
              shape=shape,
              dtype=param_dtype,
              initializer=init_ops.zeros_initializer(),
              trainable=False)
          return var

        with ops.device(None):
          device = ((lambda _: self.moving_mean.device)
                    if context.in_graph_mode() else self.moving_mean.device)
          with ops.device(device):
            self.renorm_mean = _renorm_variable('renorm_mean', param_shape)
            self.renorm_mean_weight = _renorm_variable('renorm_mean_weight', ())
          # We initialize renorm_stddev to 0, and maintain the (0-initialized)
          # renorm_stddev_weight. This allows us to (1) mix the average
          # stddev with the minibatch stddev early in training, and (2) compute
          # the unbiased average stddev by dividing renorm_stddev by the weight.
          device = ((lambda _: self.moving_variance.device)
                    if context.in_graph_mode() else self.moving_variance.device)
          with ops.device(device):
            self.renorm_stddev = _renorm_variable('renorm_stddev', param_shape)
            self.renorm_stddev_weight = _renorm_variable(
                'renorm_stddev_weight', ())
    finally:
      if partitioner:
        self._scope.set_partitioner(partitioner)
    self.built = True

  def _assign_moving_average(self, variable, value, one_minus_decay):
    with ops.name_scope(None, 'AssignMovingAvg',
                        [variable, value, one_minus_decay]) as scope:
      with ops.colocate_with(variable):
        update_delta = math_ops.multiply(
            math_ops.subtract(variable.read_value(), value),
            one_minus_decay)
        if isinstance(variable, resource_variable_ops.ResourceVariable):
          # state_ops.assign_sub does an extra read_variable_op after the
          # assign. We avoid that here.
          return gen_resource_variable_ops.assign_sub_variable_op(
              variable.handle, update_delta, name=scope)
        else:
          return state_ops.assign_sub(variable, update_delta, name=scope)

  def _fused_batch_norm(self, inputs, training):
    """Returns the output of fused batch norm."""
    beta = self.beta if self.center else self._beta_const
    gamma = self.gamma if self.scale else self._gamma_const

    def _fused_batch_norm_training():
      return nn.fused_batch_norm(
          inputs,
          gamma,
          beta,
          epsilon=self.epsilon,
          data_format=self._data_format)

    def _fused_batch_norm_inference():
      return nn.fused_batch_norm(
          inputs,
          gamma,
          beta,
          mean=self.moving_mean,
          variance=self.moving_variance,
          epsilon=self.epsilon,
          is_training=False,
          data_format=self._data_format)

    output, mean, variance = utils.smart_cond(
        training, _fused_batch_norm_training, _fused_batch_norm_inference)
    if not self._bessels_correction_test_only:
      # Remove Bessel's correction to be consistent with non-fused batch norm.
      # Note that the variance computed by fused batch norm is
      # with Bessel's correction.
      sample_size = math_ops.cast(
          array_ops.size(inputs) / array_ops.size(variance), variance.dtype)
      factor = (sample_size - math_ops.cast(1.0, variance.dtype)) / sample_size
      variance *= factor

    training_value = utils.constant_value(training)
    if training_value is None:
      one_minus_decay = utils.smart_cond(training,
                                         lambda: self._one_minus_decay,
                                         lambda: 0.)
    else:
      one_minus_decay = ops.convert_to_tensor(self._one_minus_decay)
    if training_value or training_value is None:
      mean_update = self._assign_moving_average(self.moving_mean, mean,
                                                one_minus_decay)
      variance_update = self._assign_moving_average(self.moving_variance,
                                                    variance, one_minus_decay)
      if context.in_graph_mode():
        # Note that in Eager mode, the updates are already executed when running
        # assign_moving_averages. So we do not need to put them into
        # collections.
        self.add_update(mean_update, inputs=inputs)
        self.add_update(variance_update, inputs=inputs)

    return output

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
    # When not training, use r=1, d=0.
    r = utils.smart_cond(training, lambda: r, lambda: array_ops.ones_like(r))
    d = utils.smart_cond(training, lambda: d, lambda: array_ops.zeros_like(d))

    def _update_renorm_variable(var, weight, value):
      """Updates a moving average and weight, returns the unbiased value."""
      value = array_ops.identity(value)
      def _do_update():
        # Update the variables without zero debiasing. The debiasing will be
        # accomplished by dividing the exponential moving average by the weight.
        # For example, after a single update, the moving average would be
        # (1-decay) * value. and the weight will be 1-decay, with their ratio
        # giving the value.
        # Make sure the weight is not updated until before r and d computation.
        with ops.control_dependencies([value]):
          weight_value = array_ops.constant(1., dtype=weight.dtype)
        new_var = moving_averages.assign_moving_average(
            var, value, self.renorm_momentum, zero_debias=False)
        new_weight = moving_averages.assign_moving_average(
            weight, weight_value, self.renorm_momentum, zero_debias=False)
        return new_var / new_weight
      def _fake_update():
        return array_ops.identity(var)
      return utils.smart_cond(training, _do_update, _fake_update)

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
    if self.virtual_batch_size is not None:
      # Virtual batches (aka ghost batches) can be simulated by reshaping the
      # Tensor and reusing the existing batch norm implementation
      original_shape = [-1] + inputs.shape.as_list()[1:]
      expanded_shape = [self.virtual_batch_size, -1] + original_shape[1:]

      # Will cause errors if virtual_batch_size does not divide the batch size
      inputs = array_ops.reshape(inputs, expanded_shape)

      def undo_virtual_batching(outputs):
        outputs = array_ops.reshape(outputs, original_shape)
        return outputs

    if self.fused:
      outputs = self._fused_batch_norm(inputs, training=training)
      if self.virtual_batch_size is not None:
        # Currently never reaches here since fused_batch_norm does not support
        # virtual batching
        return undo_virtual_batching(outputs)
      return outputs

    # Compute the axes along which to reduce the mean / variance
    input_shape = inputs.get_shape()
    ndims = len(input_shape)
    reduction_axes = [i for i in range(ndims) if i not in self.axis]
    if self.virtual_batch_size is not None:
      del reduction_axes[1]     # Do not reduce along virtual batch dim

    # Broadcasting only necessary for single-axis batch norm where the axis is
    # not the last dimension
    broadcast_shape = [1] * ndims
    broadcast_shape[self.axis[0]] = input_shape[self.axis[0]].value
    def _broadcast(v):
      if (v is not None and
          len(v.get_shape()) != ndims and
          reduction_axes != list(range(ndims - 1))):
        return array_ops.reshape(v, broadcast_shape)
      return v

    scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

    def _compose_transforms(scale, offset, then_scale, then_offset):
      if then_scale is not None:
        scale *= then_scale
        offset *= then_scale
      if then_offset is not None:
        offset += then_offset
      return (scale, offset)

    # Determine a boolean value for `training`: could be True, False, or None.
    training_value = utils.constant_value(training)
    if training_value is not False:
      if self.adjustment:
        adj_scale, adj_bias = self.adjustment(array_ops.shape(inputs))
        # Adjust only during training.
        adj_scale = utils.smart_cond(training,
                                     lambda: adj_scale,
                                     lambda: array_ops.ones_like(adj_scale))
        adj_bias = utils.smart_cond(training,
                                    lambda: adj_bias,
                                    lambda: array_ops.zeros_like(adj_bias))
        scale, offset = _compose_transforms(adj_scale, adj_bias, scale, offset)

      # Some of the computations here are not necessary when training==False
      # but not a constant. However, this makes the code simpler.
      keep_dims = self.virtual_batch_size is not None or len(self.axis) > 1
      mean, variance = nn.moments(inputs, reduction_axes, keep_dims=keep_dims)

      moving_mean = self.moving_mean
      moving_variance = self.moving_variance

      mean = utils.smart_cond(training,
                              lambda: mean,
                              lambda: moving_mean)
      variance = utils.smart_cond(training,
                                  lambda: variance,
                                  lambda: moving_variance)

      if self.renorm:
        r, d, new_mean, new_variance = self._renorm_correction_and_moments(
            mean, variance, training)
        # When training, the normalized values (say, x) will be transformed as
        # x * gamma + beta without renorm, and (x * r + d) * gamma + beta
        # = x * (r * gamma) + (d * gamma + beta) with renorm.
        r = _broadcast(array_ops.stop_gradient(r, name='renorm_r'))
        d = _broadcast(array_ops.stop_gradient(d, name='renorm_d'))
        scale, offset = _compose_transforms(r, d, scale, offset)
      else:
        new_mean, new_variance = mean, variance

      if self.virtual_batch_size is not None:
        # This isn't strictly correct since in ghost batch norm, you are
        # supposed to sequentially update the moving_mean and moving_variance
        # with each sub-batch. However, since the moving statistics are only
        # used during evaluation, it is more efficient to just update in one
        # step and should not make a significant difference in the result.
        new_mean = math_ops.reduce_mean(new_mean,
                                        axis=1, keep_dims=True)
        new_variance = math_ops.reduce_mean(new_variance,
                                            axis=1, keep_dims=True)

      def _do_update(var, value):
        return moving_averages.assign_moving_average(
            var, value, self.momentum, zero_debias=False)

      mean_update = utils.smart_cond(
          training,
          lambda: _do_update(self.moving_mean, new_mean),
          lambda: self.moving_mean)
      variance_update = utils.smart_cond(
          training,
          lambda: _do_update(self.moving_variance, new_variance),
          lambda: self.moving_variance)
      if context.in_graph_mode():
        self.add_update(mean_update, inputs=inputs)
        self.add_update(variance_update, inputs=inputs)

    else:
      mean, variance = self.moving_mean, self.moving_variance

    outputs = nn.batch_normalization(inputs,
                                     _broadcast(mean),
                                     _broadcast(variance),
                                     offset,
                                     scale,
                                     self.epsilon)
    # If some components of the shape got lost due to adjustments, fix that.
    outputs.set_shape(input_shape)

    if self.virtual_batch_size is not None:
      return undo_virtual_batching(outputs)

    return outputs

  def compute_output_shape(self, input_shape):
    return input_shape


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
                        beta_constraint=None,
                        gamma_constraint=None,
                        training=False,
                        trainable=True,
                        name=None,
                        reuse=None,
                        renorm=False,
                        renorm_clipping=None,
                        renorm_momentum=0.99,
                        fused=None,
                        virtual_batch_size=None,
                        adjustment=None):
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
    axis: An `int`, the axis that should be normalized (typically the features
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
    beta_constraint: An optional projection function to be applied to the `beta`
        weight after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    gamma_constraint: An optional projection function to be applied to the
        `gamma` weight after being updated by an `Optimizer`.
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
    fused: if `True`, use a faster, fused implementation if possible.
      If `None`, use the system recommended implementation.
    virtual_batch_size: An `int`. By default, `virtual_batch_size` is `None`,
      which means batch normalization is performed across the whole batch. When
      `virtual_batch_size` is not `None`, instead perform "Ghost Batch
      Normalization", which creates virtual sub-batches which are each
      normalized separately (with shared gamma, beta, and moving statistics).
      Must divide the actual batch size during execution.
    adjustment: A function taking the `Tensor` containing the (dynamic) shape of
      the input tensor and returning a pair (scale, bias) to apply to the
      normalized values (before gamma and beta), only during training. For
      example, if axis==-1,
        `adjustment = lambda shape: (
          tf.random_uniform(shape[-1:], 0.93, 1.07),
          tf.random_uniform(shape[-1:], -0.1, 0.1))`
      will scale the normalized value by up to 7% up or down, then shift the
      result by up to 0.1 (with independent scaling and bias for each feature
      but shared across all examples), and finally apply gamma and/or beta. If
      `None`, no adjustment is applied. Cannot be specified if
      virtual_batch_size is specified.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.
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
      beta_constraint=beta_constraint,
      gamma_constraint=gamma_constraint,
      renorm=renorm,
      renorm_clipping=renorm_clipping,
      renorm_momentum=renorm_momentum,
      fused=fused,
      trainable=trainable,
      virtual_batch_size=virtual_batch_size,
      adjustment=adjustment,
      name=name,
      dtype=inputs.dtype.base_dtype,
      _reuse=reuse,
      _scope=name)
  return layer.apply(inputs, training=training)


# Aliases

BatchNorm = BatchNormalization
batch_norm = batch_normalization
