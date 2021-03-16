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
"""Keras discretization preprocessing layer."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.keras.engine.base_preprocessing_layer import Combiner
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_boosted_trees_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.parallel_for import control_flow_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import keras_export


_BINS_NAME = "bins"


def summarize(values, epsilon):
  """Reduce a 1D sequence of values to a summary.

  This algorithm is based on numpy.quantiles but modified to allow for
  intermediate steps between multiple data sets. It first finds the target
  number of bins as the reciprocal of epsilon and then takes the individual
  values spaced at appropriate intervals to arrive at that target.
  The final step is to return the corresponding counts between those values
  If the target num_bins is larger than the size of values, the whole array is
  returned (with weights of 1).

  Args:
      values: 1-D `np.ndarray` to be summarized.
      epsilon: A `'float32'` that determines the approxmiate desired precision.

  Returns:
      A 2-D `np.ndarray` that is a summary of the inputs. First column is the
      interpolated partition values, the second is the weights (counts).
  """

  num_bins = 1.0 / epsilon
  value_shape = values.shape
  n = np.prod([[(1 if dim is None else dim) for dim in value_shape]])
  if num_bins >= n:
    return np.hstack((np.expand_dims(np.sort(values), 1), np.ones((n, 1))))
  step_size = int(n / num_bins)
  partition_indices = np.arange(step_size, n, step_size, np.int64)

  part = np.partition(values, partition_indices)[partition_indices]

  return np.hstack((np.expand_dims(part, 1),
                    step_size * np.ones((np.prod(part.shape), 1))))


def compress(summary, epsilon):
  """Compress a summary to within `epsilon` accuracy.

  The compression step is needed to keep the summary sizes small after merging,
  and also used to return the final target boundaries. It finds the new bins
  based on interpolating cumulative weight percentages from the large summary.
  Taking the difference of the cumulative weights from the previous bin's
  cumulative weight will give the new weight for that bin.

  Args:
      summary: 2-D `np.ndarray` summary to be compressed.
      epsilon: A `'float32'` that determines the approxmiate desired precision.

  Returns:
      A 2-D `np.ndarray` that is a compressed summary. First column is the
      interpolated partition values, the second is the weights (counts).
  """
  if np.prod(summary[:, 0].shape) * epsilon < 1:
    return summary

  percents = epsilon + np.arange(0.0, 1.0, epsilon)
  cum_weights = summary[:, 1].cumsum()
  cum_weight_percents = cum_weights / cum_weights[-1]
  new_bins = np.interp(percents, cum_weight_percents, summary[:, 0])
  cum_weights = np.interp(percents, cum_weight_percents, cum_weights)
  new_weights = cum_weights - np.concatenate((np.array([0]), cum_weights[:-1]))

  return np.hstack((np.expand_dims(new_bins, 1),
                    np.expand_dims(new_weights, 1)))


def merge_summaries(prev_summary, next_summary, epsilon):
  """Weighted merge sort of summaries.

  Given two summaries of distinct data, this function merges (and compresses)
  them to stay within `epsilon` error tolerance.

  Args:
      prev_summary: 2-D `np.ndarray` summary to be merged with `next_summary`.
      next_summary: 2-D `np.ndarray` summary to be merged with `prev_summary`.
      epsilon: A `'float32'` that determines the approxmiate desired precision.

  Returns:
      A 2-D `np.ndarray` that is a merged summary. First column is the
      interpolated partition values, the second is the weights (counts).
  """
  merged = np.concatenate((prev_summary, next_summary))
  merged = merged[merged[:, 0].argsort()]
  if np.prod(merged.shape) * epsilon < 1:
    return merged
  return compress(merged, epsilon)


def get_bucket_boundaries(summary, num_bins):
  return compress(summary, 1.0 / num_bins)[:-1, 0]


@keras_export("keras.layers.experimental.preprocessing.Discretization")
class Discretization(base_preprocessing_layer.CombinerPreprocessingLayer):
  """Buckets data into discrete ranges.

  This layer will place each element of its input data into one of several
  contiguous ranges and output an integer index indicating which range each
  element was placed in.

  Input shape:
    Any `tf.Tensor` or `tf.RaggedTensor` of dimension 2 or higher.

  Output shape:
    Same as input shape.

  Attributes:
    bin_boundaries: A list of bin boundaries. The leftmost and rightmost bins
      will always extend to `-inf` and `inf`, so `bin_boundaries=[0., 1., 2.]`
      generates bins `(-inf, 0.)`, `[0., 1.)`, `[1., 2.)`, and `[2., +inf)`. If
      this option is set, `adapt` should not be called.
    num_bins: The integer number of bins to compute. If this option is set,
      `adapt` should be called to learn the bin boundaries.
    epsilon: Error tolerance, typically a small fraction close to zero (e.g.
      0.01). Higher values of epsilon increase the quantile approximation, and
      hence result in more unequal buckets, but could improve performance
      and resource consumption.

  Examples:

  Bucketize float values based on provided buckets.
  >>> input = np.array([[-1.5, 1.0, 3.4, .5], [0.0, 3.0, 1.3, 0.0]])
  >>> layer = tf.keras.layers.experimental.preprocessing.Discretization(
  ...          bin_boundaries=[0., 1., 2.])
  >>> layer(input)
  <tf.Tensor: shape=(2, 4), dtype=int32, numpy=
  array([[0, 1, 3, 1],
         [0, 3, 2, 0]], dtype=int32)>

  Bucketize float values based on a number of buckets to compute.
  >>> input = np.array([[-1.5, 1.0, 3.4, .5], [0.0, 3.0, 1.3, 0.0]])
  >>> layer = tf.keras.layers.experimental.preprocessing.Discretization(
  ...          num_bins=4, epsilon=0.01)
  >>> layer.adapt(input)
  >>> layer(input)
  <tf.Tensor: shape=(2, 4), dtype=int32, numpy=
  array([[0, 2, 3, 1],
         [0, 3, 2, 0]], dtype=int32)>
  """

  def __init__(self,
               bin_boundaries=None,
               num_bins=None,
               epsilon=0.01,
               **kwargs):
    # bins is a deprecated arg for setting bin_boundaries or num_bins that still
    # has some usage.
    if "bins" in kwargs:
      logging.warning(
          "bins is deprecated, please use bin_boundaries or num_bins instead.")
      if isinstance(kwargs["bins"], int) and num_bins is None:
        num_bins = kwargs["bins"]
      elif bin_boundaries is None:
        bin_boundaries = kwargs["bins"]
      del kwargs["bins"]
    super(Discretization, self).__init__(
        combiner=Discretization.DiscretizingCombiner(
            epsilon, num_bins if num_bins is not None else 1),
        **kwargs)
    base_preprocessing_layer.keras_kpl_gauge.get_cell(
        "Discretization").set(True)
    if num_bins is not None and num_bins < 0:
      raise ValueError("`num_bins` must be must be greater than or equal to 0. "
                       "You passed `num_bins={}`".format(num_bins))
    if num_bins is not None and bin_boundaries is not None:
      raise ValueError("Both `num_bins` and `bin_boundaries` should not be "
                       "set. You passed `num_bins={}` and "
                       "`bin_boundaries={}`".format(num_bins, bin_boundaries))
    self.bin_boundaries = bin_boundaries
    self.num_bins = num_bins
    self.epsilon = epsilon

  def build(self, input_shape):
    if self.bin_boundaries is not None:
      initial_bins = np.append(self.bin_boundaries, [np.Inf])
    else:
      initial_bins = np.zeros(self.num_bins)
    self.bins = self._add_state_variable(
        name=_BINS_NAME,
        shape=(initial_bins.size,),
        dtype=dtypes.float32,
        initializer=init_ops.constant_initializer(initial_bins))
    super(Discretization, self).build(input_shape)

  def get_config(self):
    config = {
        "bin_boundaries": self.bin_boundaries,
        "num_bins": self.num_bins,
        "epsilon": self.epsilon,
    }
    base_config = super(Discretization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def compute_output_shape(self, input_shape):
    return input_shape

  def compute_output_signature(self, input_spec):
    output_shape = self.compute_output_shape(input_spec.shape.as_list())
    output_dtype = dtypes.int64
    if isinstance(input_spec, sparse_tensor.SparseTensorSpec):
      return sparse_tensor.SparseTensorSpec(
          shape=output_shape, dtype=output_dtype)
    return tensor_spec.TensorSpec(shape=output_shape, dtype=output_dtype)

  def call(self, inputs):
    def _bucketize_op(bins):
      bins = [math_ops.cast(bins, dtypes.float32)]
      return lambda inputs: gen_boosted_trees_ops.BoostedTreesBucketize(  # pylint: disable=g-long-lambda
          float_values=[math_ops.cast(inputs, dtypes.float32)],
          bucket_boundaries=bins)[0]

    if tf_utils.is_ragged(inputs):
      integer_buckets = ragged_functional_ops.map_flat_values(
          _bucketize_op(array_ops.squeeze(self.bins)),
          inputs)
      # Ragged map_flat_values doesn't touch the non-values tensors in the
      # ragged composite tensor. If this op is the only op a Keras model,
      # this can cause errors in Graph mode, so wrap the tensor in an identity.
      return array_ops.identity(integer_buckets)
    elif isinstance(inputs, sparse_tensor.SparseTensor):
      integer_buckets = gen_boosted_trees_ops.BoostedTreesBucketize(
          float_values=[math_ops.cast(inputs.values, dtypes.float32)],
          bucket_boundaries=[math_ops.cast(array_ops.squeeze(self.bins),
                                           dtypes.float32)])[0]
      return sparse_tensor.SparseTensor(
          indices=array_ops.identity(inputs.indices),
          values=integer_buckets,
          dense_shape=array_ops.identity(inputs.dense_shape))
    else:
      input_shape = inputs.get_shape()
      if any(dim is None for dim in input_shape.as_list()[1:]):
        raise NotImplementedError(
            "Discretization Layer requires known non-batch shape,"
            "found {}".format(input_shape))

      reshaped = array_ops.reshape(
          inputs,
          [-1, gen_math_ops.Prod(input=input_shape.as_list()[1:], axis=0)])

      return array_ops.reshape(
          control_flow_ops.vectorized_map(
              _bucketize_op(array_ops.squeeze(self.bins)), reshaped),
          array_ops.constant([-1] + input_shape.as_list()[1:]))

  class DiscretizingCombiner(Combiner):
    """Combiner for the Discretization preprocessing layer.

    This class encapsulates the computations for finding the quantile boundaries
    of a set of data in a stable and numerically correct way. Its associated
    accumulator is a namedtuple('summaries'), representing summarizations of
    the data used to generate boundaries.

    Attributes:
      epsilon: Error tolerance.
      num_bins: The desired number of buckets.
    """

    def __init__(self, epsilon, num_bins,):
      self.epsilon = epsilon
      self.num_bins = num_bins

      # TODO(mwunder): Implement elementwise per-column discretization.

    def compute(self, values, accumulator=None):
      """Compute a step in this computation, returning a new accumulator."""

      if isinstance(values, sparse_tensor.SparseTensor):
        values = values.values
      if tf_utils.is_ragged(values):
        values = values.flat_values
      flattened_input = np.reshape(values, newshape=(-1, 1))

      summaries = [summarize(v, self.epsilon) for v in flattened_input.T]

      if accumulator is None:
        return self._create_accumulator(summaries)
      else:
        return self._create_accumulator(
            [merge_summaries(prev_summ, summ, self.epsilon)
             for prev_summ, summ in zip(accumulator.summaries, summaries)])

    def merge(self, accumulators):
      """Merge several accumulators to a single accumulator."""
      # Combine accumulators and return the result.

      merged = accumulators[0].summaries
      for accumulator in accumulators[1:]:
        merged = [merge_summaries(prev, summary, self.epsilon)
                  for prev, summary in zip(merged, accumulator.summaries)]

      return self._create_accumulator(merged)

    def extract(self, accumulator):
      """Convert an accumulator into a dict of output values."""

      boundaries = [np.append(get_bucket_boundaries(summary, self.num_bins),
                              [np.Inf])
                    for summary in accumulator.summaries]
      return {
          _BINS_NAME: np.squeeze(np.vstack(boundaries))
      }

    def restore(self, output):
      """Create an accumulator based on 'output'."""
      raise NotImplementedError(
          "Discretization does not restore or support streaming updates.")

    def serialize(self, accumulator):
      """Serialize an accumulator for a remote call."""
      output_dict = {
          _BINS_NAME: [summary.tolist() for summary in accumulator.summaries]
      }
      return compat.as_bytes(json.dumps(output_dict))

    def deserialize(self, encoded_accumulator):
      """Deserialize an accumulator received from 'serialize()'."""
      value_dict = json.loads(compat.as_text(encoded_accumulator))
      return self._create_accumulator(np.array(value_dict[_BINS_NAME]))

    def _create_accumulator(self, summaries):
      """Represent the accumulator as one or more summaries of the dataset."""
      return collections.namedtuple("Accumulator", ["summaries"])(summaries)
