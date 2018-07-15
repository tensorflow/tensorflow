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
# ==============================================================================
"""Normalization layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import distribute as distribute_lib
from tensorflow.python.util.tf_export import tf_export


@tf_export('keras.layers.BatchNormalization')
class BatchNormalization(Layer):
  """Batch normalization layer (Ioffe and Szegedy, 2014).

  Normalize the activations of the previous layer at each batch,
  i.e. applies a transformation that maintains the mean activation
  close to 0 and the activation standard deviation close to 1.

  Arguments:
    axis: Integer, the axis that should be normalized
        (typically the features axis).
        For instance, after a `Conv2D` layer with
        `data_format="channels_first"`,
        set `axis=1` in `BatchNormalization`.
    momentum: Momentum for the moving average.
    epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: If True, multiply by `gamma`.
        If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    moving_mean_initializer: Initializer for the moving mean.
    moving_variance_initializer: Initializer for the moving variance.
    beta_regularizer: Optional regularizer for the beta weight.
    gamma_regularizer: Optional regularizer for the gamma weight.
    beta_constraint: Optional constraint for the beta weight.
    gamma_constraint: Optional constraint for the gamma weight.
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
    fused: if `None` or `True`, use a faster, fused implementation if possible.
      If `False`, use the system recommended implementation.
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

  Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

  Output shape:
      Same shape as input.

  References:
      - [Batch Normalization: Accelerating Deep Network Training by Reducing
        Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
  """

  def __init__(self,
               axis=-1,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='ones',
               moving_mean_initializer='zeros',
               moving_variance_initializer='ones',
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
    self.beta_initializer = initializers.get(beta_initializer)
    self.gamma_initializer = initializers.get(gamma_initializer)
    self.moving_mean_initializer = initializers.get(moving_mean_initializer)
    self.moving_variance_initializer = initializers.get(
        moving_variance_initializer)
    self.beta_regularizer = regularizers.get(beta_regularizer)
    self.gamma_regularizer = regularizers.get(gamma_regularizer)
    self.beta_constraint = constraints.get(beta_constraint)
    self.gamma_constraint = constraints.get(gamma_constraint)
    self.renorm = renorm
    self.virtual_batch_size = virtual_batch_size
    self.adjustment = adjustment
    if fused is None:
      fused = True
    self.supports_masking = True

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
    self.input_spec = InputSpec(ndim=ndims, axes=axis_to_dim)

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
      self.gamma = self.add_weight(
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
      self.beta = self.add_weight(
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

    try:
      # Disable variable partitioning when creating the moving mean and variance
      if hasattr(self, '_scope') and self._scope:
        partitioner = self._scope.partitioner
        self._scope.set_partitioner(None)
      else:
        partitioner = None
      self.moving_mean = self.add_weight(
          name='moving_mean',
          shape=param_shape,
          dtype=param_dtype,
          initializer=self.moving_mean_initializer,
          synchronization=variable_scope.VariableSynchronization.ON_READ,
          trainable=False,
          aggregation=variable_scope.VariableAggregation.MEAN)

      self.moving_variance = self.add_weight(
          name='moving_variance',
          shape=param_shape,
          dtype=param_dtype,
          initializer=self.moving_variance_initializer,
          synchronization=variable_scope.VariableSynchronization.ON_READ,
          trainable=False,
          aggregation=variable_scope.VariableAggregation.MEAN)

      if self.renorm:
        # Create variables to maintain the moving mean and standard deviation.
        # These are used in training and thus are different from the moving
        # averages above. The renorm variables are colocated with moving_mean
        # and moving_variance.
        # NOTE: below, the outer `with device` block causes the current device
        # stack to be cleared. The nested ones use a `lambda` to set the desired
        # device and ignore any devices that may be set by the custom getter.
        def _renorm_variable(name, shape):
          var = self.add_weight(
              name=name,
              shape=shape,
              dtype=param_dtype,
              initializer=init_ops.zeros_initializer(),
              synchronization=variable_scope.VariableSynchronization.ON_READ,
              trainable=False,
              aggregation=variable_scope.VariableAggregation.MEAN)
          return var

        with distribute_lib.get_distribution_strategy().colocate_vars_with(
            self.moving_mean):
          self.renorm_mean = _renorm_variable('renorm_mean', param_shape)
          self.renorm_mean_weight = _renorm_variable('renorm_mean_weight', ())
        # We initialize renorm_stddev to 0, and maintain the (0-initialized)
        # renorm_stddev_weight. This allows us to (1) mix the average
        # stddev with the minibatch stddev early in training, and (2) compute
        # the unbiased average stddev by dividing renorm_stddev by the weight.
        with distribute_lib.get_distribution_strategy().colocate_vars_with(
            self.moving_variance):
          self.renorm_stddev = _renorm_variable('renorm_stddev', param_shape)
          self.renorm_stddev_weight = _renorm_variable('renorm_stddev_weight',
                                                       ())
    finally:
      if partitioner:
        self._scope.set_partitioner(partitioner)
    self.built = True

  def _assign_moving_average(self, variable, value, momentum):
    with ops.name_scope(None, 'AssignMovingAvg',
                        [variable, value, momentum]) as scope:
      with ops.colocate_with(variable):
        decay = ops.convert_to_tensor(1.0 - momentum, name='decay')
        if decay.dtype != variable.dtype.base_dtype:
          decay = math_ops.cast(decay, variable.dtype.base_dtype)
        update_delta = (variable - math_ops.cast(value, variable.dtype)) * decay
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

    output, mean, variance = tf_utils.smart_cond(
        training, _fused_batch_norm_training, _fused_batch_norm_inference)
    if not self._bessels_correction_test_only:
      # Remove Bessel's correction to be consistent with non-fused batch norm.
      # Note that the variance computed by fused batch norm is
      # with Bessel's correction.
      sample_size = math_ops.cast(
          array_ops.size(inputs) / array_ops.size(variance), variance.dtype)
      factor = (sample_size - math_ops.cast(1.0, variance.dtype)) / sample_size
      variance *= factor

    training_value = tf_utils.constant_value(training)
    if training_value is None:
      momentum = tf_utils.smart_cond(training,
                                     lambda: self.momentum,
                                     lambda: 1.0)
    else:
      momentum = ops.convert_to_tensor(self.momentum)
    if training_value or training_value is None:
      mean_update = self._assign_moving_average(self.moving_mean, mean,
                                                momentum)
      variance_update = self._assign_moving_average(self.moving_variance,
                                                    variance, momentum)
      self.add_update(mean_update, inputs=True)
      self.add_update(variance_update, inputs=True)

    return output

  def _renorm_correction_and_moments(self, mean, variance, training):
    """Returns the correction and update values for renorm."""
    stddev = math_ops.sqrt(variance + self.epsilon)
    # Compute the average mean and standard deviation, as if they were
    # initialized with this batch's moments.
    param_dtype = self.renorm_mean.dtype
    mean = math_ops.cast(mean, param_dtype)
    stddev = math_ops.cast(stddev, param_dtype)

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
      rmin = math_ops.cast(rmin, param_dtype)
      r = math_ops.maximum(r, rmin)
    if rmax is not None:
      rmax = math_ops.cast(rmax, param_dtype)
      r = math_ops.minimum(r, rmax)
    if dmax is not None:
      dmax = math_ops.cast(dmax, param_dtype)
      d = math_ops.maximum(d, -dmax)
      d = math_ops.minimum(d, dmax)
    # When not training, use r=1, d=0.
    r = tf_utils.smart_cond(training, lambda: r, lambda: array_ops.ones_like(r))
    d = tf_utils.smart_cond(training,
                            lambda: d,
                            lambda: array_ops.zeros_like(d))

    def _update_renorm_variable(var, weight, value):
      """Updates a moving average and weight, returns the unbiased value."""
      value = array_ops.identity(value)
      def _do_update():
        """Updates the var and weight, returns their updated ratio."""
        # Update the variables without zero debiasing. The debiasing will be
        # accomplished by dividing the exponential moving average by the weight.
        # For example, after a single update, the moving average would be
        # (1-decay) * value. and the weight will be 1-decay, with their ratio
        # giving the value.
        # Make sure the weight is not updated until before r and d computation.
        with ops.control_dependencies([value]):
          weight_value = array_ops.constant(1., dtype=weight.dtype)
        new_var = self._assign_moving_average(var, value, self.renorm_momentum)
        new_weight = self._assign_moving_average(weight, weight_value,
                                                 self.renorm_momentum)
        # TODO(yuefengz): the updates to var and weighted can not be batched
        # together if we fetch their updated values here. Consider calculating
        # new values and delaying the updates.
        return new_var / new_weight

      def _fake_update():
        return array_ops.identity(var)
      return tf_utils.smart_cond(training, _do_update, _fake_update)

    # TODO(yuefengz): colocate the operations
    new_mean = _update_renorm_variable(self.renorm_mean,
                                       self.renorm_mean_weight, mean)
    new_stddev = _update_renorm_variable(self.renorm_stddev,
                                         self.renorm_stddev_weight, stddev)
    # Make sqrt(moving_variance + epsilon) = new_stddev.
    new_variance = math_ops.square(new_stddev) - self.epsilon

    return (r, d, new_mean, new_variance)

  def call(self, inputs, training=None):
    original_training_value = training
    if training is None:
      training = K.learning_phase()

    in_eager_mode = context.executing_eagerly()
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
        outputs = undo_virtual_batching(outputs)
      if not context.executing_eagerly() and original_training_value is None:
        outputs._uses_learning_phase = True  # pylint: disable=protected-access
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
    training_value = tf_utils.constant_value(training)
    if training_value is not False:
      if self.adjustment:
        adj_scale, adj_bias = self.adjustment(array_ops.shape(inputs))
        # Adjust only during training.
        adj_scale = tf_utils.smart_cond(training,
                                        lambda: adj_scale,
                                        lambda: array_ops.ones_like(adj_scale))
        adj_bias = tf_utils.smart_cond(training,
                                       lambda: adj_bias,
                                       lambda: array_ops.zeros_like(adj_bias))
        scale, offset = _compose_transforms(adj_scale, adj_bias, scale, offset)

      # Some of the computations here are not necessary when training==False
      # but not a constant. However, this makes the code simpler.
      keep_dims = self.virtual_batch_size is not None or len(self.axis) > 1
      mean, variance = nn.moments(inputs, reduction_axes, keep_dims=keep_dims)

      moving_mean = self.moving_mean
      moving_variance = self.moving_variance

      mean = tf_utils.smart_cond(training,
                                 lambda: mean,
                                 lambda: moving_mean)
      variance = tf_utils.smart_cond(training,
                                     lambda: variance,
                                     lambda: moving_variance)

      if self.virtual_batch_size is not None:
        # This isn't strictly correct since in ghost batch norm, you are
        # supposed to sequentially update the moving_mean and moving_variance
        # with each sub-batch. However, since the moving statistics are only
        # used during evaluation, it is more efficient to just update in one
        # step and should not make a significant difference in the result.
        new_mean = math_ops.reduce_mean(mean, axis=1, keepdims=True)
        new_variance = math_ops.reduce_mean(variance, axis=1, keepdims=True)
      else:
        new_mean, new_variance = mean, variance

      if self.renorm:
        r, d, new_mean, new_variance = self._renorm_correction_and_moments(
            new_mean, new_variance, training)
        # When training, the normalized values (say, x) will be transformed as
        # x * gamma + beta without renorm, and (x * r + d) * gamma + beta
        # = x * (r * gamma) + (d * gamma + beta) with renorm.
        r = _broadcast(array_ops.stop_gradient(r, name='renorm_r'))
        d = _broadcast(array_ops.stop_gradient(d, name='renorm_d'))
        scale, offset = _compose_transforms(r, d, scale, offset)

      def _do_update(var, value):
        if in_eager_mode and not self.trainable:
          return

        return self._assign_moving_average(var, value, self.momentum)

      mean_update = tf_utils.smart_cond(
          training,
          lambda: _do_update(self.moving_mean, new_mean),
          lambda: self.moving_mean)
      variance_update = tf_utils.smart_cond(
          training,
          lambda: _do_update(self.moving_variance, new_variance),
          lambda: self.moving_variance)
      if not context.executing_eagerly():
        self.add_update(mean_update, inputs=True)
        self.add_update(variance_update, inputs=True)

    else:
      mean, variance = self.moving_mean, self.moving_variance

    mean = math_ops.cast(mean, inputs.dtype)
    variance = math_ops.cast(variance, inputs.dtype)
    if offset is not None:
      offset = math_ops.cast(offset, inputs.dtype)
    if scale is not None:
      scale = math_ops.cast(scale, inputs.dtype)
    outputs = nn.batch_normalization(inputs,
                                     _broadcast(mean),
                                     _broadcast(variance),
                                     offset,
                                     scale,
                                     self.epsilon)
    # If some components of the shape got lost due to adjustments, fix that.
    outputs.set_shape(input_shape)

    if self.virtual_batch_size is not None:
      outputs = undo_virtual_batching(outputs)
    if not context.executing_eagerly() and original_training_value is None:
      outputs._uses_learning_phase = True  # pylint: disable=protected-access
    return outputs

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'axis': self.axis,
        'momentum': self.momentum,
        'epsilon': self.epsilon,
        'center': self.center,
        'scale': self.scale,
        'beta_initializer': initializers.serialize(self.beta_initializer),
        'gamma_initializer': initializers.serialize(self.gamma_initializer),
        'moving_mean_initializer':
            initializers.serialize(self.moving_mean_initializer),
        'moving_variance_initializer':
            initializers.serialize(self.moving_variance_initializer),
        'beta_regularizer': regularizers.serialize(self.beta_regularizer),
        'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
        'beta_constraint': constraints.serialize(self.beta_constraint),
        'gamma_constraint': constraints.serialize(self.gamma_constraint)
    }
    # Only add TensorFlow-specific parameters if they are set, so as to preserve
    # model compatibility with external Keras.
    if self.renorm:
      config['renorm'] = True
      config['renorm_clipping'] = self.renorm_clipping
      config['renorm_momentum'] = self.renorm_momentum
    if self.virtual_batch_size is not None:
      config['virtual_batch_size'] = self.virtual_batch_size
    # Note: adjustment is not serializable.
    if self.adjustment is not None:
      logging.warning('The `adjustment` function of this `BatchNormalization` '
                      'layer cannot be serialized and has been omitted from '
                      'the layer config. It will not be included when '
                      're-creating the layer from the saved config.')
    base_config = super(BatchNormalization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
