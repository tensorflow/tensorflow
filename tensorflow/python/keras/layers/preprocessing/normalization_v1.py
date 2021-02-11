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
"""Tensorflow V1 version of the Normalization preprocessing layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.keras.engine.base_preprocessing_layer_v1 import CombinerPreprocessingLayer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import keras_export

_COUNT_NAME = 'count'
_MEAN_NAME = 'mean'
_VARIANCE_NAME = 'variance'


def convert_to_ndarray(values):
  if isinstance(values, np.ndarray):
    return values
  elif isinstance(values, ops.Tensor):
    return K.get_value(values)
  else:
    return np.array(values)


@keras_export(v1=['keras.layers.experimental.preprocessing.Normalization'])
class Normalization(CombinerPreprocessingLayer):
  """Feature-wise normalization of the data.

  This layer will coerce its inputs into a distribution centered around
  0 with standard deviation 1. It accomplishes this by precomputing the mean and
  variance of the data, and calling (input-mean)/sqrt(var) at runtime.

  What happens in `adapt`: Compute mean and variance of the data and store them
    as the layer's weights. `adapt` should be called before `fit`, `evaluate`,
    or `predict`.

  Attributes:
      axis: Integer or tuple of integers, the axis or axes that should be
        "kept". These axes are not be summed over when calculating the
        normalization statistics. By default the last axis, the `features` axis
        is kept and any `space` or `time` axes are summed. Each element in the
        the axes that are kept is normalized independently. If `axis` is set to
        'None', the layer will perform scalar normalization (dividing the input
        by a single scalar value). The `batch` axis, 0, is always summed over
        (`axis=0` is not allowed).
  """

  def __init__(self, axis=-1, **kwargs):
    # Standardize `axis` to a tuple.
    if axis is None:
      axis = ()
    elif isinstance(axis, int):
      axis = (axis,)
    else:
      axis = tuple(axis)

    mean = kwargs.pop('mean', None)
    variance = kwargs.pop('variance', None)

    super(Normalization, self).__init__(
        combiner=_NormalizingCombiner(axis), **kwargs)
    base_preprocessing_layer.keras_kpl_gauge.get_cell('Normalization').set(True)

    if 0 in axis:
      raise ValueError('The argument \'axis\' may not be 0.')

    self.axis = axis

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

    # Sort `self.axis` to avoid transposing `mean_and_var_shape`.
    # Negative axes are not sortable until you know the number of dimensions.
    original_axis = self.axis
    self.axis = tuple(
        sorted(self.axis, key=lambda a: a if a >= 0 else ndim + a))

    if any(a < 1 - ndim for a in self.axis) or any(
        a >= ndim for a in self.axis):
      raise ValueError('All `axis` values must be in '
                       'the range [1-ndim, ndim-1].\n'
                       'Got:\n'
                       '    ndim: {}\n'
                       '    axis: {}'.format(ndim, original_axis))

    self._broadcast_shape = [1 for _ in range(len(input_shape))]
    mean_and_var_shape = []
    for i in self.axis:
      mean_and_var_shape.append(input_shape[i])
      self._broadcast_shape[i] = input_shape[i]

    # count is not used in this class's call() method, but is used to re-create
    # the accumulator during multiple calls to 'adapt'.
    # TODO(omalleyt): should mean and variance be set to self.dtype?
    self.mean = self._add_state_variable(
        name=_MEAN_NAME,
        shape=mean_and_var_shape,
        dtype=K.floatx(),
        initializer=init_ops.zeros_initializer)
    self.variance = self._add_state_variable(
        name=_VARIANCE_NAME,
        shape=mean_and_var_shape,
        dtype=K.floatx(),
        initializer=init_ops.ones_initializer)
    self.count = self._add_state_variable(
        name=_COUNT_NAME,
        shape=(),
        dtype=dtypes.int64,
        initializer=init_ops.zeros_initializer)

    super(Normalization, self).build(input_shape)

    if (self.mean_val is not None and self.variance_val is not None):
      mean_val = self.mean_val * np.ones(mean_and_var_shape)
      variance_val = self.variance_val * np.ones(mean_and_var_shape)
      self.set_weights([mean_val, variance_val])

  def call(self, inputs):
    inputs = ops.convert_to_tensor_v2_with_dispatch(inputs)
    if inputs.shape.rank == 1:
      inputs = array_ops.expand_dims(inputs, 1)
    # If the inputs are not floats, cast them to floats. This avoids issues
    # with int-float multiplication and division below.
    if inputs.dtype != K.floatx():
      inputs = math_ops.cast(inputs, K.floatx())
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
    config = {'axis': self.axis}
    base_config = super(Normalization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def set_weights(self, weights):
    """Override for set_weights to ensure we can set just mean/var weights."""
    if len(weights) == 2:
      weights.append(np.array(0))
    super(Normalization, self).set_weights(weights)


class _NormalizingCombiner(base_preprocessing_layer.Combiner):
  """Combiner for the Normalization preprocessing layer.

  This class encapsulates the computations for finding the mean and variance
  of a set of data in a stable and numerically correct way. Its associated
  accumulator is a namedtuple('count', 'mean', 'variance').

  Attributes:
    axis: The axis to compute mean and var over.
  """
  COUNT_IDX = 0
  MEAN_IDX = 1
  VAR_IDX = 2

  def __init__(self, axis):
    self.axis = axis

  def compute(self, values, accumulator=None):
    """Compute a step in this computation, returning a new accumulator."""
    values = np.array(values)
    if values.ndim == 1:
      values = np.expand_dims(values, 1)

    # `np.delete` ignores negative indexes, so use a mask to delete items.
    axis_mask = np.ones([values.ndim], dtype=bool)
    axis_mask[np.array(self.axis, dtype=np.int32)] = False

    # This is the shape of all reduced axes (not specified in 'axis').

    reduction_counts = np.array(values.shape)[axis_mask]
    # We get the number of elements that will be reduced by multiplying all
    # values of 'shape' corresponding to the reduced axes.
    count = np.prod(reduction_counts, dtype=np.int64)

    # We want to reduce across dimensions except those specified in 'axis'
    # when using np.mean or np.variance; create the tuple of axes to reduce
    # over here.
    reduction_axes = tuple(np.arange(values.ndim)[axis_mask])

    mean = np.mean(values, axis=reduction_axes, dtype=np.float64)
    variance = np.var(values, axis=reduction_axes, dtype=np.float64)

    # Create an accumulator with our new data and either return it or combine
    # it with the passed accumulator.
    if accumulator is None:
      return self._create_accumulator(count, mean, variance)
    else:
      return self.add_data_to_accumulator(count, mean, variance, accumulator)

  def add_data_to_accumulator(self, count, mean, variance, accumulator):
    """Add new data to the totals in an accumulator."""
    # Combine accumulators and return the result.
    combined_count = count + accumulator[self.COUNT_IDX]

    # To combine accumulator means, we weight each accumulator's mean by the
    # number of elements that were accumulated, and then divide by the
    # total number of elements.
    combined_mean = (mean * count + accumulator[self.MEAN_IDX] *
                     accumulator[self.COUNT_IDX]) / combined_count

    # The variance is computed using the lack-of-fit sum of squares
    # formula (see https://en.wikipedia.org/wiki/Lack-of-fit_sum_of_squares).
    accumulator_var_contribution = accumulator[self.COUNT_IDX] * (
        accumulator[self.VAR_IDX] +
        np.square(accumulator[self.MEAN_IDX] - combined_mean))
    data_var_contribution = count * (variance + np.square(mean - combined_mean))
    combined_variance = (accumulator_var_contribution +
                         data_var_contribution) / combined_count

    accumulator[self.COUNT_IDX] = combined_count
    accumulator[self.MEAN_IDX] = np.nan_to_num(combined_mean)
    accumulator[self.VAR_IDX] = np.nan_to_num(combined_variance)
    return accumulator

  def merge(self, accumulators):
    """Merge several accumulators to a single accumulator."""
    # Combine accumulators and return the result.
    combined_count = np.sum(
        [accumulator[self.COUNT_IDX] for accumulator in accumulators])

    # To combine accumulator means, we weight each accumulator's mean by the
    # number of elements that were accumulated, and then divide by the
    # total number of elements.
    combined_mean = np.add.reduce([
        accumulator[self.MEAN_IDX] * accumulator[self.COUNT_IDX]
        for accumulator in accumulators
    ]) / combined_count

    # The variance is computed using the lack-of-fit sum of squares
    # formula (see https://en.wikipedia.org/wiki/Lack-of-fit_sum_of_squares).
    def variance_contribution(accumulator):
      return accumulator[self.COUNT_IDX] * (
          accumulator[self.VAR_IDX] +
          np.square(accumulator[self.MEAN_IDX] - combined_mean))

    combined_variance = np.add.reduce([
        variance_contribution(accumulator) for accumulator in accumulators
    ]) / combined_count

    return self._create_accumulator(combined_count, combined_mean,
                                    combined_variance)

  def extract(self, accumulator):
    """Convert an accumulator into a dict of output values."""
    return {
        _COUNT_NAME: accumulator[self.COUNT_IDX],
        _MEAN_NAME: accumulator[self.MEAN_IDX],
        _VARIANCE_NAME: accumulator[self.VAR_IDX]
    }

  def restore(self, output):
    """Create an accumulator based on 'output'."""
    # There is no special internal state here, so we just return the relevant
    # internal value.
    count = output[_COUNT_NAME]
    mean = output[_MEAN_NAME]
    var = output[_VARIANCE_NAME]
    if (count == 0 and (mean.any() != 0.0 or var.any() != 0.0)):
      raise RuntimeError(
          'The mean and/or variance of a Normalization preprocessing layer '
          "were set without also setting 'count'. If 'count' is not also set, "
          " or was set to 0, 'adapt' cannot be called unless the 'reset_state'"
          'arg is True.')
    return self._create_accumulator(output[_COUNT_NAME], output[_MEAN_NAME],
                                    output[_VARIANCE_NAME])

  def serialize(self, accumulator):
    """Serialize an accumulator for a remote call."""
    output_dict = {
        _COUNT_NAME: accumulator[self.COUNT_IDX].tolist(),
        _MEAN_NAME: accumulator[self.MEAN_IDX].tolist(),
        _VARIANCE_NAME: accumulator[self.VAR_IDX].tolist()
    }
    return compat.as_bytes(json.dumps(output_dict))

  def deserialize(self, encoded_accumulator):
    """Deserialize an accumulator received from 'serialize()'."""
    value_dict = json.loads(compat.as_text(encoded_accumulator))
    return self._create_accumulator(
        np.array(value_dict[_COUNT_NAME]), np.array(value_dict[_MEAN_NAME]),
        np.array(value_dict[_VARIANCE_NAME]))

  def _create_accumulator(self, count, mean, variance):
    """Convert any 'nan' values in the given accumulator to numeric values."""
    return [count, mean, variance]
