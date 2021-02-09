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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import variables
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.experimental.preprocessing.Normalization', v1=[])
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
        value(s)cannot be broadcast, an error will be raised when this layer's
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
    super(Normalization, self).__init__(stateful=True, streaming=True, **kwargs)
    base_preprocessing_layer.keras_kpl_gauge.get_cell('Normalization').set(True)

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
    if mean is not None and variance is not None:
      mean = convert_to_ndarray(mean)
      variance = convert_to_ndarray(variance)
    elif mean is not None or variance is not None:
      raise ValueError(
          'When setting values directly, both `mean` and `variance` '
          'must be set. Got mean: {} and variance: {}'.format(mean, variance))
    self.mean_val = mean
    self.variance_val = variance

  def build(self, input_shape):
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
    # Create variables without keeping reduced axes.
    mean_and_var_shape = tuple(input_shape[d] for d in self._keep_axis)

    self.mean = self.add_weight(
        name='mean',
        shape=mean_and_var_shape,
        dtype=self.dtype,
        initializer=init_ops.zeros_initializer,
        trainable=False)
    self.variance = self.add_weight(
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

    super(Normalization, self).build(input_shape)

    if (self.mean_val is not None and self.variance_val is not None):
      mean_val = self.mean_val * np.ones(mean_and_var_shape)
      variance_val = self.variance_val * np.ones(mean_and_var_shape)
      self.mean.assign(mean_val)
      self.variance.assign(variance_val)

    self.built = True

  def update_state(self, data):
    if not self.built:
      raise RuntimeError('`build` must be called before `update_state`.')

    data = self._standardize_inputs(data)
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

    total_mean = self.mean * existing_weight + batch_mean * batch_weight
    # The variance is computed using the lack-of-fit sum of squares
    # formula (see https://en.wikipedia.org/wiki/Lack-of-fit_sum_of_squares).
    total_variance = ((self.variance +
                       (self.mean - total_mean)**2) * existing_weight +
                      (batch_variance +
                       (batch_mean - total_mean)**2) * batch_weight)
    self.mean.assign(total_mean)
    self.variance.assign(total_variance)
    self.count.assign(total_count)

  def merge_state(self, layers):
    layers = layers + [self]
    if any(not l.built for l in layers):
      raise ValueError(
          'All layers to be merged must have been adapted to some inputs '
          'first (otherwise they have no state).')

    layer_counts = [l.count for l in layers]
    layer_means = [l.mean for l in layers]
    layer_variances = [l.variance for l in layers]

    total_count = math_ops.reduce_sum(layer_counts)
    layer_weightings = (
        math_ops.cast(layer_counts, self.dtype) /
        math_ops.cast(total_count, self.dtype))
    layer_weightings = array_ops.reshape(
        layer_weightings, shape=[len(layers)] + [1] * self.mean.shape.rank)

    total_mean = math_ops.reduce_sum(layer_means * layer_weightings, axis=0)
    inter_layer_variances = (layer_means - total_mean)**2
    total_variance = math_ops.reduce_sum(
        ((layer_variances + inter_layer_variances) * layer_weightings), axis=0)

    self.mean.assign(total_mean)
    self.variance.assign(total_variance)
    self.count.assign(total_count)

  def reset_state(self):  # pylint: disable=method-hidden
    if self.built:
      self.mean.assign(array_ops.zeros_like(self.mean))
      self.variance.assign(array_ops.ones_like(self.variance))
      self.count.assign(array_ops.zeros_like(self.count))

  def call(self, inputs):
    inputs = self._standardize_inputs(inputs)
    # We need to reshape the mean and variance data to ensure that Tensorflow
    # broadcasts the data correctly.
    mean = array_ops.reshape(self.mean, self._broadcast_shape)
    variance = array_ops.reshape(self.variance, self._broadcast_shape)
    return ((inputs - mean) /
            math_ops.maximum(math_ops.sqrt(variance), K.epsilon()))

  def compute_output_shape(self, input_shape):
    return input_shape

  def compute_output_signature(self, input_spec):
    return input_spec

  def get_config(self):
    config = super(Normalization, self).get_config()
    config.update({'axis': self.axis})
    return config

  def set_weights(self, weights):
    """Override for set_weights to ensure we can set just mean/var weights."""
    if len(weights) == 2:
      weights.append(np.array(0))
    super(Normalization, self).set_weights(weights)

  def _standardize_inputs(self, inputs):
    inputs = ops.convert_to_tensor_v2_with_dispatch(inputs)
    if inputs.shape.rank == 0:
      inputs = array_ops.reshape(inputs, [1, 1])
    elif inputs.shape.rank == 1:
      inputs = array_ops.expand_dims(inputs, 1)

    if inputs.dtype != self.dtype:
      inputs = math_ops.cast(inputs, self.dtype)
    return inputs


def convert_to_ndarray(values):
  if isinstance(values, np.ndarray):
    return values
  elif isinstance(values, ops.Tensor):
    return K.get_value(values)
  else:
    return np.array(values)
