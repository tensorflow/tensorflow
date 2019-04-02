# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Virtual batch normalization.

This technique was first introduced in `Improved Techniques for Training GANs`
(Salimans et al, https://arxiv.org/abs/1606.03498). Instead of using batch
normalization on a minibatch, it fixes a reference subset of the data to use for
calculating normalization statistics.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope

__all__ = [
    'VBN',
]


def _static_or_dynamic_batch_size(tensor, batch_axis):
  """Returns the static or dynamic batch size."""
  batch_size = array_ops.shape(tensor)[batch_axis]
  static_batch_size = tensor_util.constant_value(batch_size)
  return static_batch_size or batch_size


def _statistics(x, axes):
  """Calculate the mean and mean square of `x`.

  Modified from the implementation of `tf.nn.moments`.

  Args:
    x: A `Tensor`.
    axes: Array of ints.  Axes along which to compute mean and
      variance.

  Returns:
    Two `Tensor` objects: `mean` and `square mean`.
  """
  # The dynamic range of fp16 is too limited to support the collection of
  # sufficient statistics. As a workaround we simply perform the operations
  # on 32-bit floats before converting the mean and variance back to fp16
  y = math_ops.cast(x, dtypes.float32) if x.dtype == dtypes.float16 else x

  # Compute true mean while keeping the dims for proper broadcasting.
  shift = array_ops.stop_gradient(math_ops.reduce_mean(y, axes, keepdims=True))

  shifted_mean = math_ops.reduce_mean(y - shift, axes, keepdims=True)
  mean = shifted_mean + shift
  mean_squared = math_ops.reduce_mean(math_ops.square(y), axes, keepdims=True)

  mean = array_ops.squeeze(mean, axes)
  mean_squared = array_ops.squeeze(mean_squared, axes)
  if x.dtype == dtypes.float16:
    return (math_ops.cast(mean, dtypes.float16),
            math_ops.cast(mean_squared, dtypes.float16))
  else:
    return (mean, mean_squared)


def _validate_init_input_and_get_axis(reference_batch, axis):
  """Validate input and return the used axis value."""
  if reference_batch.shape.ndims is None:
    raise ValueError('`reference_batch` has unknown dimensions.')

  ndims = reference_batch.shape.ndims
  if axis < 0:
    used_axis = ndims + axis
  else:
    used_axis = axis
  if used_axis < 0 or used_axis >= ndims:
    raise ValueError('Value of `axis` argument ' + str(used_axis) +
                     ' is out of range for input with rank ' + str(ndims))
  return used_axis


def _validate_call_input(tensor_list, batch_dim):
  """Verifies that tensor shapes are compatible, except for `batch_dim`."""
  def _get_shape(tensor):
    shape = tensor.shape.as_list()
    del shape[batch_dim]
    return shape
  base_shape = tensor_shape.TensorShape(_get_shape(tensor_list[0]))
  for tensor in tensor_list:
    base_shape.assert_is_compatible_with(_get_shape(tensor))


class VBN(object):
  """A class to perform virtual batch normalization.

  This technique was first introduced in `Improved Techniques for Training GANs`
  (Salimans et al, https://arxiv.org/abs/1606.03498). Instead of using batch
  normalization on a minibatch, it fixes a reference subset of the data to use
  for calculating normalization statistics.

  To do this, we calculate the reference batch mean and mean square, and modify
  those statistics for each example. We use mean square instead of variance,
  since it is linear.

  Note that if `center` or `scale` variables are created, they are shared
  between all calls to this object.

  The `__init__` API is intended to mimic `tf.layers.batch_normalization` as
  closely as possible.
  """

  def __init__(self,
               reference_batch,
               axis=-1,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer=init_ops.zeros_initializer(),
               gamma_initializer=init_ops.ones_initializer(),
               beta_regularizer=None,
               gamma_regularizer=None,
               trainable=True,
               name=None,
               batch_axis=0):
    """Initialize virtual batch normalization object.

    We precompute the 'mean' and 'mean squared' of the reference batch, so that
    `__call__` is efficient. This means that the axis must be supplied when the
    object is created, not when it is called.

    We precompute 'square mean' instead of 'variance', because the square mean
    can be easily adjusted on a per-example basis.

    Args:
      reference_batch: A minibatch tensors. This will form the reference data
        from which the normalization statistics are calculated. See
        https://arxiv.org/abs/1606.03498 for more details.
      axis: Integer, the axis that should be normalized (typically the features
        axis). For instance, after a `Convolution2D` layer with
        `data_format="channels_first"`, set `axis=1` in `BatchNormalization`.
      epsilon: Small float added to variance to avoid dividing by zero.
      center: If True, add offset of `beta` to normalized tensor. If False,
        `beta` is ignored.
      scale: If True, multiply by `gamma`. If False, `gamma` is
        not used. When the next layer is linear (also e.g. `nn.relu`), this can
        be disabled since the scaling can be done by the next layer.
      beta_initializer: Initializer for the beta weight.
      gamma_initializer: Initializer for the gamma weight.
      beta_regularizer: Optional regularizer for the beta weight.
      gamma_regularizer: Optional regularizer for the gamma weight.
      trainable: Boolean, if `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      name: String, the name of the ops.
      batch_axis: The axis of the batch dimension. This dimension is treated
        differently in `virtual batch normalization` vs `batch normalization`.

    Raises:
      ValueError: If `reference_batch` has unknown dimensions at graph
        construction.
      ValueError: If `batch_axis` is the same as `axis`.
    """
    axis = _validate_init_input_and_get_axis(reference_batch, axis)
    self._epsilon = epsilon
    self._beta = 0
    self._gamma = 1
    self._batch_axis = _validate_init_input_and_get_axis(
        reference_batch, batch_axis)

    if axis == self._batch_axis:
      raise ValueError('`axis` and `batch_axis` cannot be the same.')

    with variable_scope.variable_scope(name, 'VBN',
                                       values=[reference_batch]) as self._vs:
      self._reference_batch = reference_batch

      # Calculate important shapes:
      #  1) Reduction axes for the reference batch
      #  2) Broadcast shape, if necessary
      #  3) Reduction axes for the virtual batchnormed batch
      #  4) Shape for optional parameters
      input_shape = self._reference_batch.shape
      ndims = input_shape.ndims
      reduction_axes = list(range(ndims))
      del reduction_axes[axis]

      self._broadcast_shape = [1] * len(input_shape)
      self._broadcast_shape[axis] = input_shape.dims[axis]

      self._example_reduction_axes = list(range(ndims))
      del self._example_reduction_axes[max(axis, self._batch_axis)]
      del self._example_reduction_axes[min(axis, self._batch_axis)]

      params_shape = self._reference_batch.shape[axis]

      # Determines whether broadcasting is needed. This is slightly different
      # than in the `nn.batch_normalization` case, due to `batch_dim`.
      self._needs_broadcasting = (
          sorted(self._example_reduction_axes) != list(range(ndims))[:-2])

      # Calculate the sufficient statistics for the reference batch in a way
      # that can be easily modified by additional examples.
      self._ref_mean, self._ref_mean_squares = _statistics(
          self._reference_batch, reduction_axes)
      self._ref_variance = (self._ref_mean_squares -
                            math_ops.square(self._ref_mean))

      # Virtual batch normalization uses a weighted average between example
      # statistics and the reference batch statistics.
      ref_batch_size = _static_or_dynamic_batch_size(
          self._reference_batch, self._batch_axis)
      self._example_weight = 1. / (
          math_ops.cast(ref_batch_size, dtypes.float32) + 1.)
      self._ref_weight = 1. - self._example_weight

      # Make the variables, if necessary.
      if center:
        self._beta = variable_scope.get_variable(
            name='beta',
            shape=(params_shape,),
            initializer=beta_initializer,
            regularizer=beta_regularizer,
            trainable=trainable)
      if scale:
        self._gamma = variable_scope.get_variable(
            name='gamma',
            shape=(params_shape,),
            initializer=gamma_initializer,
            regularizer=gamma_regularizer,
            trainable=trainable)

  def _virtual_statistics(self, inputs, reduction_axes):
    """Compute the statistics needed for virtual batch normalization."""
    cur_mean, cur_mean_sq = _statistics(inputs, reduction_axes)
    vb_mean = (self._example_weight * cur_mean +
               self._ref_weight * self._ref_mean)
    vb_mean_sq = (self._example_weight * cur_mean_sq +
                  self._ref_weight * self._ref_mean_squares)
    return (vb_mean, vb_mean_sq)

  def _broadcast(self, v, broadcast_shape=None):
    # The exact broadcast shape depends on the current batch, not the reference
    # batch, unless we're calculating the batch normalization of the reference
    # batch.
    b_shape = broadcast_shape or self._broadcast_shape
    if self._needs_broadcasting and v is not None:
      return array_ops.reshape(v, b_shape)
    return v

  def reference_batch_normalization(self):
    """Return the reference batch, but batch normalized."""
    with ops.name_scope(self._vs.name):
      return nn.batch_normalization(self._reference_batch,
                                    self._broadcast(self._ref_mean),
                                    self._broadcast(self._ref_variance),
                                    self._broadcast(self._beta),
                                    self._broadcast(self._gamma),
                                    self._epsilon)

  def __call__(self, inputs):
    """Run virtual batch normalization on inputs.

    Args:
      inputs: Tensor input.

    Returns:
       A virtual batch normalized version of `inputs`.

    Raises:
       ValueError: If `inputs` shape isn't compatible with the reference batch.
    """
    _validate_call_input([inputs, self._reference_batch], self._batch_axis)

    with ops.name_scope(self._vs.name, values=[inputs, self._reference_batch]):
      # Calculate the statistics on the current input on a per-example basis.
      vb_mean, vb_mean_sq = self._virtual_statistics(
          inputs, self._example_reduction_axes)
      vb_variance = vb_mean_sq - math_ops.square(vb_mean)

      # The exact broadcast shape of the input statistic Tensors depends on the
      # current batch, not the reference batch. The parameter broadcast shape
      # is independent of the shape of the input statistic Tensor dimensions.
      b_shape = self._broadcast_shape[:]  # deep copy
      b_shape[self._batch_axis] = _static_or_dynamic_batch_size(
          inputs, self._batch_axis)
      return nn.batch_normalization(
          inputs,
          self._broadcast(vb_mean, b_shape),
          self._broadcast(vb_variance, b_shape),
          self._broadcast(self._beta, self._broadcast_shape),
          self._broadcast(self._gamma, self._broadcast_shape),
          self._epsilon)
