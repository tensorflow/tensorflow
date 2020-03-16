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

from tensorflow.python.compat import compat
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export


class BatchNormalizationBase(Layer):
  r"""Normalize and scale inputs or activations. (Ioffe and Szegedy, 2014).

  Normalize the activations of the previous layer at each batch,
  i.e. applies a transformation that maintains the mean activation
  close to 0 and the activation standard deviation close to 1.

  Batch normalization differs from other layers in several key aspects:

  1) Adding BatchNormalization with `training=True` to a model causes the
  result of one example to depend on the contents of all other examples in a
  minibatch. Be careful when padding batches or masking examples, as these can
  change the minibatch statistics and affect other examples.

  2) Updates to the weights (moving statistics) are based on the forward pass
  of a model rather than the result of gradient computations.

  3) When performing inference using a model containing batch normalization, it
  is generally (though not always) desirable to use accumulated statistics
  rather than mini-batch statistics. This is accomplished by passing
  `training=False` when calling the model, or using `model.predict`.

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
    fused: if `True`, use a faster, fused implementation, or raise a ValueError
      if the fused implementation cannot be used. If `None`, use the faster
      implementation if possible. If False, do not used the fused
      implementation.
    trainable: Boolean, if `True` the variables will be marked as trainable.
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
          tf.random.uniform(shape[-1:], 0.93, 1.07),
          tf.random.uniform(shape[-1:], -0.1, 0.1))`
      will scale the normalized value by up to 7% up or down, then shift the
      result by up to 0.1 (with independent scaling and bias for each feature
      but shared across all examples), and finally apply gamma and/or beta. If
      `None`, no adjustment is applied. Cannot be specified if
      virtual_batch_size is specified.

  Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode.
      - `training=True`: The layer will normalize its inputs using the
        mean and variance of the current batch of inputs.
      - `training=False`: The layer will normalize its inputs using the
        mean and variance of its moving statistics, learned during training.

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as input.

  {{TRAINABLE_ATTRIBUTE_NOTE}}

  Normalization equations:
    Consider the intermediate activations \(x\) of a mini-batch of size
    \\(m\\):

    We can compute the mean and variance of the batch

    \\({\mu_B} = \frac{1}{m} \sum_{i=1}^{m} {x_i}\\)

    \\({\sigma_B^2} = \frac{1}{m} \sum_{i=1}^{m} ({x_i} - {\mu_B})^2\\)

    and then compute a normalized \\(x\\), including a small factor
    \\({\epsilon}\\) for numerical stability.

    \\(\hat{x_i} = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}\\)

    And finally \\(\hat{x}\) is linearly transformed by \({\gamma}\\)
    and \\({\beta}\\), which are learned parameters:

    \\({y_i} = {\gamma * \hat{x_i} + \beta}\\)

  References:
  - [Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
  """

  # By default, the base class uses V2 behavior. The BatchNormalization V1
  # subclass sets this to False to use the V1 behavior.
  _USE_V2_BEHAVIOR = True

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
    super(BatchNormalizationBase, self).__init__(
        name=name, **kwargs)
    if isinstance(axis, (list, tuple)):
      self.axis = axis[:]
    elif isinstance(axis, int):
      self.axis = axis
    else:
      raise TypeError('Expected an int or a list/tuple of ints for the '
                      'argument \'axis\', but received: %r' % axis)
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
    if self._USE_V2_BEHAVIOR:
      if fused:
        self._raise_if_fused_cannot_be_used()
      # We leave fused as None if self._fused_can_be_used()==True, since we
      # still may set it to False in self.build() if the input rank is not 4.
      elif fused is None and not self._fused_can_be_used():
        fused = False
    elif fused is None:
      fused = True
    self.supports_masking = True

    self.fused = fused
    self._bessels_correction_test_only = True
    self._trainable_var = None
    self.trainable = trainable

    if renorm:
      renorm_clipping = renorm_clipping or {}
      keys = ['rmax', 'rmin', 'dmax']
      if set(renorm_clipping) - set(keys):
        raise ValueError('renorm_clipping %s contains keys not in %s' %
                         (renorm_clipping, keys))
      self.renorm_clipping = renorm_clipping
      self.renorm_momentum = renorm_momentum

  def _raise_if_fused_cannot_be_used(self):
    """Raises a ValueError if fused implementation cannot be used.

    In addition to the checks done in this function, the input tensors rank must
    be 4. The input rank check can only be done once the input shape is known.
    """
    # Note the ValueErrors in this function are caught and not reraised in
    # _fused_can_be_used(). No other exception besides ValueError should be
    # raised here.

    # Currently fused batch norm doesn't support renorm. It also only supports a
    # channel dimension on axis 1 or 3, when no virtual batch size or adjustment
    # is used.
    if self.renorm:
      raise ValueError('Passing both fused=True and renorm=True is '
                       'unsupported')
    axis = [self.axis] if isinstance(self.axis, int) else self.axis
    # Axis -3 is equivalent to 1, and axis -1 is equivalent to 3, because the
    # input rank is required to be 4 (which is checked later).
    if len(axis) > 1 or axis[0] not in (-3, -1, 1, 3):
      raise ValueError('Passing fused=True is only supported when axis is 1 '
                       'or 3')
    if self.virtual_batch_size is not None:
      raise ValueError('Passing fused=True is unsupported when '
                       'virtual_batch_size is specified.')
    if self.adjustment is not None:
      raise ValueError('Passing fused=True is unsupported when '
                       'adjustment is specified.')
    # TODO(reedwm): Support fp64 in FusedBatchNorm then remove this check.
    if self._compute_dtype not in ('float16', 'bfloat16', 'float32', None):
      raise ValueError('Passing fused=True is only supported when the compute '
                       'dtype is float16, bfloat16, or float32. Got dtype: %s'
                       % (self._compute_dtype,))

  def _fused_can_be_used(self):
    try:
      self._raise_if_fused_cannot_be_used()
      return True
    except ValueError:
      return False

  @property
  def trainable(self):
    return self._trainable

  @trainable.setter
  def trainable(self, value):
    self._trainable = value
    if self._trainable_var is not None:
      self._trainable_var.update_value(value)

  def _get_trainable_var(self):
    if self._trainable_var is None:
      self._trainable_var = K.freezable_variable(
          self._trainable, name=self.name + '_trainable')
    return self._trainable_var

  @property
  def _param_dtype(self):
    # Raise parameters of fp16 batch norm to fp32
    if self.dtype == dtypes.float16 or self.dtype == dtypes.bfloat16:
      return dtypes.float32
    else:
      return self.dtype or dtypes.float32

  def _support_zero_size_input(self):
    return distribution_strategy_context.has_strategy() and getattr(
        distribution_strategy_context.get_strategy().extended,
        'experimental_enable_get_next_as_optional', False)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if not input_shape.ndims:
      raise ValueError('Input has undefined rank:', input_shape)
    ndims = len(input_shape)

    # Convert axis to list and resolve negatives
    if isinstance(self.axis, int):
      self.axis = [self.axis]

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

    if self.fused in (None, True):
      # TODO(yaozhang): if input is not 4D, reshape it to 4D and reshape the
      # output back to its original shape accordingly.
      if self._USE_V2_BEHAVIOR:
        if self.fused is None:
          self.fused = (ndims == 4)
        elif self.fused and ndims != 4:
          raise ValueError('Batch normalization layers with fused=True only '
                           'support 4D input tensors.')
      else:
        assert self.fused is not None
        self.fused = (ndims == 4 and self._fused_can_be_used())
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

    axis_to_dim = {x: input_shape.dims[x].value for x in self.axis}
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
          dtype=self._param_dtype,
          initializer=self.gamma_initializer,
          regularizer=self.gamma_regularizer,
          constraint=self.gamma_constraint,
          trainable=True,
          experimental_autocast=False)
    else:
      self.gamma = None
      if self.fused:
        self._gamma_const = K.constant(
            1.0, dtype=self._param_dtype, shape=param_shape)

    if self.center:
      self.beta = self.add_weight(
          name='beta',
          shape=param_shape,
          dtype=self._param_dtype,
          initializer=self.beta_initializer,
          regularizer=self.beta_regularizer,
          constraint=self.beta_constraint,
          trainable=True,
          experimental_autocast=False)
    else:
      self.beta = None
      if self.fused:
        self._beta_const = K.constant(
            0.0, dtype=self._param_dtype, shape=param_shape)

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
          dtype=self._param_dtype,
          initializer=self.moving_mean_initializer,
          synchronization=tf_variables.VariableSynchronization.ON_READ,
          trainable=False,
          aggregation=tf_variables.VariableAggregation.MEAN,
          experimental_autocast=False)

      self.moving_variance = self.add_weight(
          name='moving_variance',
          shape=param_shape,
          dtype=self._param_dtype,
          initializer=self.moving_variance_initializer,
          synchronization=tf_variables.VariableSynchronization.ON_READ,
          trainable=False,
          aggregation=tf_variables.VariableAggregation.MEAN,
          experimental_autocast=False)

      if self.renorm:
        # In batch renormalization we track the inference moving stddev instead
        # of the moving variance to more closely align with the paper.
        def moving_stddev_initializer(*args, **kwargs):
          return math_ops.sqrt(
              self.moving_variance_initializer(*args, **kwargs))

        with distribution_strategy_context.get_strategy(
        ).extended.colocate_vars_with(self.moving_variance):
          self.moving_stddev = self.add_weight(
              name='moving_stddev',
              shape=param_shape,
              dtype=self._param_dtype,
              initializer=moving_stddev_initializer,
              synchronization=tf_variables.VariableSynchronization.ON_READ,
              trainable=False,
              aggregation=tf_variables.VariableAggregation.MEAN,
              experimental_autocast=False)

        # Create variables to maintain the moving mean and standard deviation.
        # These are used in training and thus are different from the moving
        # averages above. The renorm variables are colocated with moving_mean
        # and moving_stddev.
        # NOTE: below, the outer `with device` block causes the current device
        # stack to be cleared. The nested ones use a `lambda` to set the desired
        # device and ignore any devices that may be set by the custom getter.
        def _renorm_variable(name,
                             shape,
                             initializer=init_ops.zeros_initializer()):
          """Create a renorm variable."""
          var = self.add_weight(
              name=name,
              shape=shape,
              dtype=self._param_dtype,
              initializer=initializer,
              synchronization=tf_variables.VariableSynchronization.ON_READ,
              trainable=False,
              aggregation=tf_variables.VariableAggregation.MEAN,
              experimental_autocast=False)
          return var

        with distribution_strategy_context.get_strategy(
        ).extended.colocate_vars_with(self.moving_mean):
          self.renorm_mean = _renorm_variable('renorm_mean', param_shape,
                                              self.moving_mean_initializer)
        with distribution_strategy_context.get_strategy(
        ).extended.colocate_vars_with(self.moving_stddev):
          self.renorm_stddev = _renorm_variable('renorm_stddev', param_shape,
                                                moving_stddev_initializer)
    finally:
      if partitioner:
        self._scope.set_partitioner(partitioner)
    self.built = True

  def _assign_moving_average(self, variable, value, momentum, inputs_size):
    with K.name_scope('AssignMovingAvg') as scope:
      with ops.colocate_with(variable):
        decay = ops.convert_to_tensor_v2(1.0 - momentum, name='decay')
        if decay.dtype != variable.dtype.base_dtype:
          decay = math_ops.cast(decay, variable.dtype.base_dtype)
        update_delta = (
            variable - math_ops.cast(value, variable.dtype)) * decay
        if inputs_size is not None:
          update_delta = array_ops.where(inputs_size > 0, update_delta,
                                         K.zeros_like(update_delta))
        return state_ops.assign_sub(variable, update_delta, name=scope)

  def _assign_new_value(self, variable, value):
    with K.name_scope('AssignNewValue') as scope:
      with ops.colocate_with(variable):
        return state_ops.assign(variable, value, name=scope)

  def _fused_batch_norm(self, inputs, training):
    """Returns the output of fused batch norm."""
    beta = self.beta if self.center else self._beta_const
    gamma = self.gamma if self.scale else self._gamma_const

    # TODO(b/129279393): Support zero batch input in non DistributionStrategy
    # code as well.
    if self._support_zero_size_input():
      inputs_size = array_ops.size(inputs)
    else:
      inputs_size = None

    # TODO(rmlarsen): Support using fused avg updates for non-eager execution
    # after fixing graph pattern matching and enabling fused_batch_norm to
    # take exponential_avg_factor as a tensor input.
    use_fused_avg_updates = (
        compat.forward_compatible(2020, 3, 6) and
        ops.executing_eagerly_outside_functions() and
        isinstance(self.momentum, (float, int)))
    if use_fused_avg_updates:
      exponential_avg_factor = 1.0 - self.momentum
    else:
      exponential_avg_factor = None

    def _maybe_add_or_remove_bessels_correction(variance, remove=True):
      r"""Add or remove Bessel's correction."""
      # Removes Bessel's correction if remove == True, adds it otherwise.
      # This is to be consistent with non-fused batch norm. Note that the
      # variance computed by fused batch norm is with Bessel's correction.
      # This is only used in legacy V1 batch norm tests.
      if self._bessels_correction_test_only:
        return variance
      sample_size = math_ops.cast(
          array_ops.size(inputs) / array_ops.size(variance), variance.dtype)
      if remove:
        factor = (sample_size -
                  math_ops.cast(1.0, variance.dtype)) / sample_size
      else:
        factor = sample_size / (
            sample_size - math_ops.cast(1.0, variance.dtype))
      return variance * factor

    def _fused_batch_norm_training():
      return nn.fused_batch_norm(
          inputs,
          gamma,
          beta,
          mean=self.moving_mean,
          variance=_maybe_add_or_remove_bessels_correction(
              self.moving_variance, remove=False),
          epsilon=self.epsilon,
          is_training=True,
          data_format=self._data_format,
          exponential_avg_factor=exponential_avg_factor)

    def _fused_batch_norm_training_empty():
      return inputs, self.moving_mean, self.moving_variance

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

    train_op = _fused_batch_norm_training
    if use_fused_avg_updates and inputs_size is not None:
      train_op = lambda: tf_utils.smart_cond(inputs_size > 0,
                                             _fused_batch_norm_training,
                                             _fused_batch_norm_training_empty)

    output, mean, variance = tf_utils.smart_cond(training, train_op,
                                                 _fused_batch_norm_inference)
    variance = _maybe_add_or_remove_bessels_correction(variance, remove=True)

    training_value = tf_utils.constant_value(training)
    if training_value or training_value is None:
      if not use_fused_avg_updates:
        if training_value is None:
          momentum = tf_utils.smart_cond(training, lambda: self.momentum,
                                         lambda: 1.0)
        else:
          momentum = ops.convert_to_tensor_v2(self.momentum)

      def mean_update():
        """Update self.moving_mean with the most recent data point."""
        if use_fused_avg_updates:
          return self._assign_new_value(self.moving_mean, mean)
        else:
          return self._assign_moving_average(self.moving_mean, mean, momentum,
                                             inputs_size)

      def variance_update():
        """Update self.moving_variance with the most recent data point."""
        if use_fused_avg_updates:
          return self._assign_new_value(self.moving_variance, variance)
        else:
          return self._assign_moving_average(self.moving_variance, variance,
                                             momentum, inputs_size)

      self.add_update(mean_update)
      self.add_update(variance_update)

    return output

  def _renorm_correction_and_moments(self, mean, variance, training,
                                     inputs_size):
    """Returns the correction and update values for renorm."""
    stddev = math_ops.sqrt(variance + self.epsilon)
    # Compute the average mean and standard deviation, as if they were
    # initialized with this batch's moments.
    renorm_mean = self.renorm_mean
    # Avoid divide by zero early on in training.
    renorm_stddev = math_ops.maximum(self.renorm_stddev,
                                     math_ops.sqrt(self.epsilon))
    # Compute the corrections for batch renorm.
    r = stddev / renorm_stddev
    d = (mean - renorm_mean) / renorm_stddev
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
    r = tf_utils.smart_cond(training, lambda: r, lambda: array_ops.ones_like(r))
    d = tf_utils.smart_cond(training,
                            lambda: d,
                            lambda: array_ops.zeros_like(d))

    def _update_renorm_variable(var, value, inputs_size):
      """Updates a moving average and weight, returns the unbiased value."""
      value = array_ops.identity(value)
      def _do_update():
        """Updates the var, returns the updated value."""
        new_var = self._assign_moving_average(var, value, self.renorm_momentum,
                                              inputs_size)
        return new_var

      def _fake_update():
        return array_ops.identity(var)
      return tf_utils.smart_cond(training, _do_update, _fake_update)

    # TODO(yuefengz): colocate the operations
    update_new_mean = _update_renorm_variable(self.renorm_mean, mean,
                                              inputs_size)
    update_new_stddev = _update_renorm_variable(self.renorm_stddev, stddev,
                                                inputs_size)

    # Update the inference mode moving averages with the batch value.
    with ops.control_dependencies([update_new_mean, update_new_stddev]):
      out_mean = array_ops.identity(mean)
      out_variance = array_ops.identity(variance)

    return (r, d, out_mean, out_variance)

  def _calculate_mean_and_var(self, inputs, reduction_axes, keep_dims):
    return nn.moments(inputs, reduction_axes, keep_dims=keep_dims)

  def _moments(self, inputs, reduction_axes, keep_dims):
    mean, variance = self._calculate_mean_and_var(inputs, reduction_axes,
                                                  keep_dims)
    # TODO(b/129279393): Support zero batch input in non DistributionStrategy
    # code as well.
    if self._support_zero_size_input():
      inputs_size = array_ops.size(inputs)
      mean = array_ops.where(inputs_size > 0, mean, K.zeros_like(mean))
      variance = array_ops.where(inputs_size > 0, variance,
                                 K.zeros_like(variance))
    return mean, variance

  def _get_training_value(self, training=None):
    if training is None:
      training = K.learning_phase()
    if self._USE_V2_BEHAVIOR:
      if isinstance(training, int):
        training = bool(training)
      if base_layer_utils.is_in_keras_graph():
        training = math_ops.logical_and(training, self._get_trainable_var())
      elif not self.trainable:
        # When the layer is not trainable, it overrides the value passed from
        # model.
        training = self.trainable
    return training

  def call(self, inputs, training=None):
    training = self._get_training_value(training)

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
      return outputs

    # Compute the axes along which to reduce the mean / variance
    input_shape = inputs.shape
    ndims = len(input_shape)
    reduction_axes = [i for i in range(ndims) if i not in self.axis]
    if self.virtual_batch_size is not None:
      del reduction_axes[1]     # Do not reduce along virtual batch dim

    # Broadcasting only necessary for single-axis batch norm where the axis is
    # not the last dimension
    broadcast_shape = [1] * ndims
    broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value
    def _broadcast(v):
      if (v is not None and len(v.shape) != ndims and
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
    if training_value == False:  # pylint: disable=singleton-comparison,g-explicit-bool-comparison
      mean, variance = self.moving_mean, self.moving_variance
    else:
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
      mean, variance = self._moments(
          math_ops.cast(inputs, self._param_dtype),
          reduction_axes,
          keep_dims=keep_dims)

      moving_mean = self.moving_mean
      moving_variance = self.moving_variance

      mean = tf_utils.smart_cond(training, lambda: mean,
                                 lambda: ops.convert_to_tensor_v2(moving_mean))
      variance = tf_utils.smart_cond(
          training, lambda: variance,
          lambda: ops.convert_to_tensor_v2(moving_variance))

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

      if self._support_zero_size_input():
        inputs_size = array_ops.size(inputs)
      else:
        inputs_size = None
      if self.renorm:
        r, d, new_mean, new_variance = self._renorm_correction_and_moments(
            new_mean, new_variance, training, inputs_size)
        # When training, the normalized values (say, x) will be transformed as
        # x * gamma + beta without renorm, and (x * r + d) * gamma + beta
        # = x * (r * gamma) + (d * gamma + beta) with renorm.
        r = _broadcast(array_ops.stop_gradient(r, name='renorm_r'))
        d = _broadcast(array_ops.stop_gradient(d, name='renorm_d'))
        scale, offset = _compose_transforms(r, d, scale, offset)

      def _do_update(var, value):
        """Compute the updates for mean and variance."""
        return self._assign_moving_average(var, value, self.momentum,
                                           inputs_size)

      def mean_update():
        true_branch = lambda: _do_update(self.moving_mean, new_mean)
        false_branch = lambda: self.moving_mean
        return tf_utils.smart_cond(training, true_branch, false_branch)

      def variance_update():
        """Update the moving variance."""

        def true_branch_renorm():
          # We apply epsilon as part of the moving_stddev to mirror the training
          # code path.
          moving_stddev = _do_update(self.moving_stddev,
                                     math_ops.sqrt(new_variance + self.epsilon))
          return self._assign_new_value(
              self.moving_variance,
              # Apply relu in case floating point rounding causes it to go
              # negative.
              K.relu(moving_stddev * moving_stddev - self.epsilon))

        if self.renorm:
          true_branch = true_branch_renorm
        else:
          true_branch = lambda: _do_update(self.moving_variance, new_variance)

        false_branch = lambda: self.moving_variance
        return tf_utils.smart_cond(training, true_branch, false_branch)

      self.add_update(mean_update)
      self.add_update(variance_update)

    mean = math_ops.cast(mean, inputs.dtype)
    variance = math_ops.cast(variance, inputs.dtype)
    if offset is not None:
      offset = math_ops.cast(offset, inputs.dtype)
    if scale is not None:
      scale = math_ops.cast(scale, inputs.dtype)
    # TODO(reedwm): Maybe do math in float32 if given float16 inputs, if doing
    # math in float16 hurts validation accuracy of popular models like resnet.
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
    base_config = super(BatchNormalizationBase, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def replace_in_base_docstring(replacements):
  string = BatchNormalizationBase.__doc__
  for old, new in replacements:
    assert old in string
    string = string.replace(old, new)
  return string


@keras_export(v1=['keras.layers.BatchNormalization'])  # pylint: disable=missing-docstring
class BatchNormalization(BatchNormalizationBase):

  __doc__ = replace_in_base_docstring(
      [('''
    fused: if `True`, use a faster, fused implementation, or raise a ValueError
      if the fused implementation cannot be used. If `None`, use the faster
      implementation if possible. If False, do not used the fused
      implementation.''',
        '''
    fused: if `None` or `True`, use a faster, fused implementation if possible.
      If `False`, use the system recommended implementation.'''),
       ('{{TRAINABLE_ATTRIBUTE_NOTE}}', '')])

  _USE_V2_BEHAVIOR = False


@keras_export('keras.layers.LayerNormalization')
class LayerNormalization(Layer):
  """Layer normalization layer (Ba et al., 2016).

  Normalize the activations of the previous layer for each given example in a
  batch independently, rather than across a batch like Batch Normalization.
  i.e. applies a transformation that maintains the mean activation within each
  example close to 0 and the activation standard deviation close to 1.

  Arguments:
    axis: Integer or List/Tuple. The axis that should be normalized
      (typically the features axis).
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
    beta_regularizer: Optional regularizer for the beta weight.
    gamma_regularizer: Optional regularizer for the gamma weight.
    beta_constraint: Optional constraint for the beta weight.
    gamma_constraint: Optional constraint for the gamma weight.
    trainable: Boolean, if `True` the variables will be marked as trainable.

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as input.

  References:
    - [Layer Normalization](https://arxiv.org/abs/1607.06450)
  """

  def __init__(self,
               axis=-1,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='ones',
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    super(LayerNormalization, self).__init__(
        name=name, trainable=trainable, **kwargs)
    if isinstance(axis, (list, tuple)):
      self.axis = axis[:]
    elif isinstance(axis, int):
      self.axis = axis
    else:
      raise TypeError('Expected an int or a list/tuple of ints for the '
                      'argument \'axis\', but received: %r' % axis)

    self.epsilon = epsilon
    self.center = center
    self.scale = scale
    self.beta_initializer = initializers.get(beta_initializer)
    self.gamma_initializer = initializers.get(gamma_initializer)
    self.beta_regularizer = regularizers.get(beta_regularizer)
    self.gamma_regularizer = regularizers.get(gamma_regularizer)
    self.beta_constraint = constraints.get(beta_constraint)
    self.gamma_constraint = constraints.get(gamma_constraint)

    self.supports_masking = True

    # Indicates whether a faster fused implementation can be used. This will be
    # set to True or False in build()"
    self._fused = None

  def _fused_can_be_used(self, ndims):
    """Return false if fused implementation cannot be used.

    Check if the axis is contiguous and can be collapsed into the last axis.
    The self.axis is assumed to have no duplicates.
    """
    axis = sorted(self.axis)
    can_use_fused = False

    if axis[-1] == ndims - 1 and axis[-1] - axis[0] == len(axis) - 1:
      can_use_fused = True

    # fused_batch_norm will silently raise epsilon to be at least 1.001e-5, so
    # we cannot used the fused version if epsilon is below that value. Also, the
    # variable dtype must be float32, as fused_batch_norm only supports float32
    # variables.
    if self.epsilon < 1.001e-5 or self.dtype != 'float32':
      can_use_fused = False

    return can_use_fused

  def build(self, input_shape):
    ndims = len(input_shape)
    if ndims is None:
      raise ValueError('Input shape %s has undefined rank.' % input_shape)

    # Convert axis to list and resolve negatives
    if isinstance(self.axis, int):
      self.axis = [self.axis]
    elif isinstance(self.axis, tuple):
      self.axis = list(self.axis)
    for idx, x in enumerate(self.axis):
      if x < 0:
        self.axis[idx] = ndims + x

    # Validate axes
    for x in self.axis:
      if x < 0 or x >= ndims:
        raise ValueError('Invalid axis: %d' % x)
    if len(self.axis) != len(set(self.axis)):
      raise ValueError('Duplicate axis: {}'.format(tuple(self.axis)))

    param_shape = [input_shape[dim] for dim in self.axis]
    if self.scale:
      self.gamma = self.add_weight(
          name='gamma',
          shape=param_shape,
          initializer=self.gamma_initializer,
          regularizer=self.gamma_regularizer,
          constraint=self.gamma_constraint,
          trainable=True,
          experimental_autocast=False)
    else:
      self.gamma = None

    if self.center:
      self.beta = self.add_weight(
          name='beta',
          shape=param_shape,
          initializer=self.beta_initializer,
          regularizer=self.beta_regularizer,
          constraint=self.beta_constraint,
          trainable=True,
          experimental_autocast=False)
    else:
      self.beta = None

    self._fused = self._fused_can_be_used(ndims)

    self.built = True

  def call(self, inputs):
    # Compute the axes along which to reduce the mean / variance
    input_shape = inputs.shape
    ndims = len(input_shape)

    # Broadcasting only necessary for norm where the axis is not just
    # the last dimension
    broadcast_shape = [1] * ndims
    for dim in self.axis:
      broadcast_shape[dim] = input_shape.dims[dim].value
    def _broadcast(v):
      if (v is not None and len(v.shape) != ndims and
          self.axis != [ndims - 1]):
        return array_ops.reshape(v, broadcast_shape)
      return v

    if not self._fused:
      input_dtype = inputs.dtype
      if input_dtype in ('float16', 'bfloat16') and self.dtype == 'float32':
        # If mixed precision is used, cast inputs to float32 so that this is at
        # least as numerically stable as the fused version.
        inputs = math_ops.cast(inputs, 'float32')

      # Calculate the moments on the last axis (layer activations).
      mean, variance = nn.moments(inputs, self.axis, keep_dims=True)

      scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

      # Compute layer normalization using the batch_normalization function.
      outputs = nn.batch_normalization(
          inputs,
          mean,
          variance,
          offset=offset,
          scale=scale,
          variance_epsilon=self.epsilon)
      outputs = math_ops.cast(outputs, input_dtype)
    else:
      # Collapse dims before self.axis, and dims in self.axis
      pre_dim, in_dim = (1, 1)
      axis = sorted(self.axis)
      tensor_shape = array_ops.shape(inputs)
      for dim in range(0, ndims):
        dim_tensor = tensor_shape[dim]
        if dim < axis[0]:
          pre_dim = pre_dim * dim_tensor
        else:
          assert dim in axis
          in_dim = in_dim * dim_tensor

      squeezed_shape = [1, pre_dim, in_dim, 1]
      # This fused operation requires reshaped inputs to be NCHW.
      data_format = 'NCHW'

      inputs = array_ops.reshape(inputs, squeezed_shape)

      def _set_const_tensor(val, dtype, shape):
        return array_ops.fill(shape, constant_op.constant(val, dtype=dtype))

      # self.gamma and self.beta have the wrong shape for fused_batch_norm, so
      # we cannot pass them as the scale and offset parameters. Therefore, we
      # create two constant tensors in correct shapes for fused_batch_norm and
      # later construct a separate calculation on the scale and offset.
      scale = _set_const_tensor(1.0, self.dtype, [pre_dim])
      offset = _set_const_tensor(0.0, self.dtype, [pre_dim])

      # Compute layer normalization using the fused_batch_norm function.
      outputs, _, _ = nn.fused_batch_norm(
          inputs,
          scale=scale,
          offset=offset,
          epsilon=self.epsilon,
          data_format=data_format)

      outputs = array_ops.reshape(outputs, tensor_shape)

      scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

      if scale is not None:
        outputs = outputs * math_ops.cast(scale, outputs.dtype)
      if offset is not None:
        outputs = outputs + math_ops.cast(offset, outputs.dtype)

    # If some components of the shape got lost due to adjustments, fix that.
    outputs.set_shape(input_shape)

    return outputs

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'axis': self.axis,
        'epsilon': self.epsilon,
        'center': self.center,
        'scale': self.scale,
        'beta_initializer': initializers.serialize(self.beta_initializer),
        'gamma_initializer': initializers.serialize(self.gamma_initializer),
        'beta_regularizer': regularizers.serialize(self.beta_regularizer),
        'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
        'beta_constraint': constraints.serialize(self.beta_constraint),
        'gamma_constraint': constraints.serialize(self.gamma_constraint)
    }
    base_config = super(LayerNormalization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
