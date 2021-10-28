# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Normalization preprocessing layer."""
# pylint: disable=g-classes-have-attributes

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import variables
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.experimental.preprocessing.Normalization')
class Normalization(base_preprocessing_layer.PreprocessingLayer):
  """Feature-wise normalization of the data.

  This layer will coerce its inputs into a distribution centered around
  0 with standard deviation 1. It accomplishes this by precomputing the mean and
  variance of the data, and calling (input-mean)/sqrt(var) at runtime.

  What happens in `adapt`: Compute mean and variance of the data and store them
    as the layer's weights. `adapt` should be called before `fit`, `evaluate`,
    or `predict`.

  Args:
      axis: Integer or tuple of integers, the axis or axes that should be
        "kept". These axes are not be summed over when calculating the
        normalization statistics. By default the last axis, the `features` axis
        is kept and any `space` or `time` axes are summed. Each element in the
        the axes that are kept is normalized independently. If `axis` is set to
        'None', the layer will perform scalar normalization (dividing the input
        by a single scalar value). The `batch` axis, 0, is always summed over
        (`axis=0` is not allowed).
      mean: The mean value(s) to use during normalization. The passed value(s)
        will be broadcast to the shape of the kept axes above; if the value(s)
        cannot be broadcast, an error will be raised when this layer's build()
        method is called.
      variance: The variance value(s) to use during normalization. The passed
        value(s) will be broadcast to the shape of the kept axes above; if the
        value(s) cannot be broadcast, an error will be raised when this layer's
        build() method is called.

  Examples:

  Calculate the mean and variance by analyzing the dataset in `adapt`.

  >>> adapt_data = np.array([[1.], [2.], [3.], [4.], [5.]], dtype=np.float32)
  >>> input_data = np.array([[1.], [2.], [3.]], np.float32)
  >>> layer = Normalization()
  >>> layer.adapt(adapt_data)
  >>> layer(input_data)
  <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
  array([[-1.4142135 ],
         [-0.70710677],
         [ 0.        ]], dtype=float32)>

  Pass the mean and variance directly.

  >>> input_data = np.array([[1.], [2.], [3.]], np.float32)
  >>> layer = Normalization(mean=3., variance=2.)
  >>> layer(input_data)
  <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
  array([[-1.4142135 ],
         [-0.70710677],
         [ 0.        ]], dtype=float32)>
  """

  def __init__(self, axis=-1, mean=None, variance=None, **kwargs):
    super().__init__(streaming=True, **kwargs)

    # Standardize `axis` to a tuple.
    if axis is None:
      axis = ()
    elif isinstance(axis, int):
      axis = (axis,)
    else:
      axis = tuple(axis)
    if 0 in axis:
      raise ValueError('The argument \'axis\' may not be 0.')
    self.axis = axis

    # Set `mean` and `variance` if passed.
    if isinstance(mean, variables.Variable):
      raise ValueError('Normalization does not support passing a Variable '
                       'for the `mean` init arg.')
    if isinstance(variance, variables.Variable):
      raise ValueError('Normalization does not support passing a Variable '
                       'for the `variance` init arg.')
    if (mean is not None) != (variance is not None):
      raise ValueError(
          'When setting values directly, both `mean` and `variance` '
          'must be set. Got mean: {} and variance: {}'.format(mean, variance))
    self.input_mean = mean
    self.input_variance = variance

  def build(self, input_shape):
    super().build(input_shape)

    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if len(input_shape) == 1:
      input_shape = input_shape + [1]
    ndim = len(input_shape)

    if any(a < 1 - ndim or a >= ndim for a in self.axis):
      raise ValueError('All `axis` values must be in the range '
                       '[1 - ndim, ndim - 1]. Found '
                       'ndim: `{}`, axis: {}'.format(ndim, self.axis))

    # Axes to be kept, replacing negative values with positive equivalents.
    # Sorted to avoid transposing axes.
    self._keep_axis = sorted([d if d >= 0 else d + ndim for d in self.axis])
    # Axes to be reduced.
    self._reduce_axis = [d for d in range(ndim) if d not in self._keep_axis]
    # 1 if an axis should be reduced, 0 otherwise.
    self._reduce_axis_mask = [
        0 if d in self._keep_axis else 1 for d in range(ndim)
    ]
    # Broadcast any reduced axes.
    self._broadcast_shape = [
        input_shape[d] if d in self._keep_axis else 1 for d in range(ndim)
    ]
    mean_and_var_shape = tuple(input_shape[d] for d in self._keep_axis)

    if self.input_mean is None:
      self.adapt_mean = self.add_weight(
          name='mean',
          shape=mean_and_var_shape,
          dtype=self.dtype,
          initializer=init_ops.zeros_initializer,
          trainable=False)
      self.adapt_variance = self.add_weight(
          name='variance',
          shape=mean_and_var_shape,
          dtype=self.dtype,
          initializer=init_ops.ones_initializer,
          trainable=False)
      self.count = self.add_weight(
          name='count',
          shape=(),
          dtype=dtypes.int64,
          initializer=init_ops.zeros_initializer,
          trainable=False)
      self.finalize_state()
    else:
      # In the no adapt case, make constant tensors for mean and variance with
      # proper broadcast shape for use during call.
      mean = self.input_mean * np.ones(mean_and_var_shape)
      variance = self.input_variance * np.ones(mean_and_var_shape)
      mean = array_ops.reshape(mean, self._broadcast_shape)
      variance = array_ops.reshape(variance, self._broadcast_shape)
      self.mean = math_ops.cast(mean, self.compute_dtype)
      self.variance = math_ops.cast(variance, self.compute_dtype)

  def update_state(self, data):
    if self.input_mean is not None:
      raise ValueError(
          'Cannot `adapt` a Normalization layer that is initialized with '
          'static `mean` and `variance`, you passed mean {} and variance {}.'
          .format(self.input_mean, self.input_variance))

    if not self.built:
      raise RuntimeError('`build` must be called before `update_state`.')

    data = self._standardize_inputs(data)
    data = math_ops.cast(data, self.adapt_mean.dtype)
    batch_mean, batch_variance = nn_impl.moments_v2(
        data, axes=self._reduce_axis)
    batch_shape = array_ops.shape(data, out_type=self.count.dtype)
    batch_reduce_shape = array_ops.gather(batch_shape, self._reduce_axis)
    batch_count = math_ops.reduce_prod(batch_reduce_shape)

    total_count = batch_count + self.count
    batch_weight = (
        math_ops.cast(batch_count, dtype=self.dtype) /
        math_ops.cast(total_count, dtype=self.dtype))
    existing_weight = 1. - batch_weight

    total_mean = self.adapt_mean * existing_weight + batch_mean * batch_weight
    # The variance is computed using the lack-of-fit sum of squares
    # formula (see https://en.wikipedia.org/wiki/Lack-of-fit_sum_of_squares).
    total_variance = ((self.adapt_variance +
                       (self.adapt_mean - total_mean)**2) * existing_weight +
                      (batch_variance +
                       (batch_mean - total_mean)**2) * batch_weight)
    self.adapt_mean.assign(total_mean)
    self.adapt_variance.assign(total_variance)
    self.count.assign(total_count)

  def merge_state(self, layers):
    layers = layers + [self]
    for l in layers:
      if l.input_mean is not None:
        raise ValueError(
            'Cannot merge Normalization layer {} that has initialized with '
            '`mean` and `variance`, you passed `mean={}` and `variance={}`.'
            .format(l.name, l.input_mean, l.input_variance))
      if not l.built:
        raise ValueError(
            'Cannot merge Normalization layer {}, it has no state. You need to '
            'call `adapt` on this layer before merging.'.format(l.name))

    layer_counts = [l.count for l in layers]
    layer_means = [l.adapt_mean for l in layers]
    layer_variances = [l.adapt_variance for l in layers]

    total_count = math_ops.reduce_sum(layer_counts)
    layer_weightings = (
        math_ops.cast(layer_counts, self.dtype) /
        math_ops.cast(total_count, self.dtype))
    layer_weightings = array_ops.reshape(
        layer_weightings,
        shape=[len(layers)] + [1] * self.adapt_mean.shape.rank)

    total_mean = math_ops.reduce_sum(layer_means * layer_weightings, axis=0)
    inter_layer_variances = (layer_means - total_mean)**2
    total_variance = math_ops.reduce_sum(
        ((layer_variances + inter_layer_variances) * layer_weightings), axis=0)

    self.adapt_mean.assign(total_mean)
    self.adapt_variance.assign(total_variance)
    self.count.assign(total_count)
    self.finalize_state()

  def reset_state(self):  # pylint: disable=method-hidden
    if self.input_mean is not None or not self.built:
      return

    self.adapt_mean.assign(array_ops.zeros_like(self.adapt_mean))
    self.adapt_variance.assign(array_ops.ones_like(self.adapt_variance))
    self.count.assign(array_ops.zeros_like(self.count))

  def finalize_state(self):
    if self.input_mean is not None or not self.built:
      return

    # In the adapt case, we make constant tensors for mean and variance with
    # proper broadcast shape and dtype each time `finalize_state` is called.
    self.mean = array_ops.reshape(self.adapt_mean, self._broadcast_shape)
    self.mean = math_ops.cast(self.mean, self.compute_dtype)
    self.variance = array_ops.reshape(self.adapt_variance,
                                      self._broadcast_shape)
    self.variance = math_ops.cast(self.variance, self.compute_dtype)

  def call(self, inputs):
    inputs = self._standardize_inputs(inputs)
    # The base layer automatically casts floating-point inputs, but we
    # explicitly cast here to also allow integer inputs to be passed
    inputs = math_ops.cast(inputs, self.compute_dtype)
    return ((inputs - self.mean) /
            math_ops.maximum(math_ops.sqrt(self.variance), backend.epsilon()))

  def compute_output_shape(self, input_shape):
    return input_shape

  def compute_output_signature(self, input_spec):
    return input_spec

  def get_config(self):
    config = super().get_config()
    config.update({
        'axis': self.axis,
        'mean': self._convert_to_list(self.input_mean),
        'variance': self._convert_to_list(self.input_variance),
    })
    return config

  def _standardize_inputs(self, inputs):
    inputs = ops.convert_to_tensor_v2_with_dispatch(inputs)
    if inputs.shape.rank == 0:
      inputs = array_ops.reshape(inputs, [1, 1])
    elif inputs.shape.rank == 1:
      inputs = array_ops.expand_dims(inputs, 1)
    return inputs

  def _convert_to_list(self, inputs):
    if tensor_util.is_tensor(inputs):
      inputs = inputs.numpy()
    if isinstance(inputs, (np.ndarray)):
      inputs = inputs.tolist()
      inputs = list(inputs)
    return inputs
