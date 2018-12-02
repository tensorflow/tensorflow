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
"""Utility functions for adding pruning related ops to the graph.
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope

_NBINS = 256


def weight_mask_variable(var, scope):
  """Create a mask for the weights.

  This function adds a variable 'mask' to the graph.

  Args:
    var: the weight variable that needs to be masked
    scope: The variable scope of the variable var

  Returns:
    the mask variable of the same size and shape as var, initialized to all 1s.
  """
  with variable_scope.variable_scope(scope):
    mask = variable_scope.get_variable(
        'mask',
        var.get_shape(),
        initializer=init_ops.ones_initializer(),
        trainable=False,
        dtype=var.dtype)
  return mask


def weight_threshold_variable(var, scope):
  """Create a scalar threshold for the weights.

  This function adds a variable
  'threshold' to the graph.

  Args:
    var: The weight variable that needs to be masked
    scope: The variable scope of the variable var

  Returns:
    A scalar threshold variable initialized to 0.
  """
  with variable_scope.variable_scope(scope):
    threshold = variable_scope.get_variable(
        'threshold', [],
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        dtype=var.dtype)
    return threshold


def kronecker_product(mat1, mat2):
  """Computes the Kronecker product of two matrices mat1 and mat2.

  Args:
    mat1: A matrix of size m x n
    mat2: A matrix of size p x q
  Returns:
    Kronecker product of matrices mat1 and mat2 of size mp x nq
  """

  m1, n1 = mat1.get_shape().as_list()
  mat1_rsh = array_ops.reshape(mat1, [m1, 1, n1, 1])
  m2, n2 = mat2.get_shape().as_list()
  mat2_rsh = array_ops.reshape(mat2, [1, m2, 1, n2])
  return array_ops.reshape(mat1_rsh * mat2_rsh, [m1 * m2, n1 * n2])


def expand_tensor(tensor, block_dims):
  """Expands a 2D tensor by replicating the tensor values.

  This is equivalent to the kronecker product of the tensor and a matrix of
  ones of size block_dims.

  Example:

  tensor = [[1,2]
            [3,4]]
  block_dims = [2,2]

  result = [[1 1 2 2]
            [1 1 2 2]
            [3 3 4 4]
            [3 3 4 4]]

  Args:
    tensor: A 2D tensor that needs to be expanded.
    block_dims: List of integers specifying the expansion factor.

  Returns:
    The expanded tensor

  Raises:
    ValueError: if tensor is not rank-2 or block_dims is does not have 2
    elements.
  """
  if tensor.get_shape().ndims != 2:
    raise ValueError('Input tensor must be rank 2')

  if len(block_dims) != 2:
    raise ValueError('block_dims must have 2 elements')

  block_height, block_width = block_dims

  def _tile_rows(tensor, multiple):
    """Create a new tensor by tiling the tensor along rows."""
    return array_ops.tile(tensor, [multiple, 1])

  def _generate_indices(num_rows, block_dim):
    indices = np.zeros(shape=[num_rows * block_dim, 1], dtype=np.int32)
    for k in range(block_dim):
      for r in range(num_rows):
        indices[k * num_rows + r] = r * block_dim + k
    return indices

  def _replicate_rows(tensor, multiple):
    tensor_shape = tensor.shape.as_list()
    expanded_shape = [tensor_shape[0] * multiple, tensor_shape[1]]
    indices = constant_op.constant(_generate_indices(tensor_shape[0], multiple))
    return array_ops.scatter_nd(indices, _tile_rows(tensor, multiple),
                                expanded_shape)

  expanded_tensor = tensor

  # Expand rows by factor block_height.
  if block_height > 1:
    expanded_tensor = _replicate_rows(tensor, block_height)

  # Transpose and expand by factor block_width. Transpose the result.
  if block_width > 1:
    expanded_tensor = array_ops.transpose(
        _replicate_rows(array_ops.transpose(expanded_tensor), block_width))

  return expanded_tensor


def _histogram(values, value_range, nbins=100, dtype=dtypes.int32, name=None):
  """Return histogram of values.

  Given the tensor `values`, this operation returns a rank 1 histogram counting
  the number of entries in `values` that fell into every bin.  The bins are
  equal width and determined by the arguments `value_range` and `nbins`.

  Args:
    values:  Numeric `Tensor`.
    value_range:  Shape [2] `Tensor` of same `dtype` as `values`.
      values <= value_range[0] will be mapped to hist[0],
      values >= value_range[1] will be mapped to hist[-1].
    nbins:  Scalar `int32 Tensor`.  Number of histogram bins.
    dtype:  dtype for returned histogram.
    name:  A name for this operation (defaults to 'histogram').

  Returns:
    A 1-D `Tensor` holding histogram of values.

  """
  with ops.name_scope(name, 'histogram', [values, value_range, nbins]) as scope:
    values = ops.convert_to_tensor(values, name='values')
    values = array_ops.reshape(values, [-1])
    nbins_float = np.float32(nbins)

    # Map tensor values that fall within value_range to [0, 1].
    scaled_values = math_ops.truediv(
        values - value_range[0],
        value_range[1] - value_range[0],
        name='scaled_values')

    # map tensor values within the open interval value_range to {0,.., nbins-1},
    # values outside the open interval will be zero or less, or nbins or more.
    indices = math_ops.floor(nbins_float * scaled_values, name='indices')

    # Clip edge cases (e.g. value = value_range[1]) or "outliers."
    indices = math_ops.cast(
        clip_ops.clip_by_value(indices, 0, nbins_float - 1), dtypes.int32)

    return math_ops.unsorted_segment_sum(
        array_ops.ones_like(indices, dtype=dtype), indices, nbins, name=scope)


def compute_cdf_from_histogram(values, value_range, **kwargs):
  """Returns the normalized cumulative distribution of the given values tensor.

  Computes the histogram and uses tf.cumsum to arrive at cdf

  Args:
    values:  Numeric `Tensor`.
    value_range:  Shape [2] `Tensor` of same `dtype` as `values`.
    **kwargs: keyword arguments: nbins, name

  Returns:
    A 1-D `Tensor` holding normalized cdf of values.

  """
  nbins = kwargs.get('nbins', _NBINS)
  name = kwargs.get('name', None)
  with ops.name_scope(name, 'cdf', [values, value_range, nbins]):
    histogram = _histogram(
        values, value_range, dtype=dtypes.float32, nbins=nbins)
    cdf = math_ops.cumsum(histogram)
    return math_ops.div(cdf, math_ops.reduce_max(cdf))


def compute_cdf(values, value_range, **kwargs):
  """Returns the normalized cumulative distribution of the given values tensor.

  Uses tf.while_loop to directly compute the cdf of the values.

  Args:
    values:  Numeric `Tensor`.
    value_range:  Shape [2] `Tensor` of same `dtype` as `values`
    **kwargs: keyword arguments: nbins, name

  Returns:
    A 1-D `Tensor` holding normalized cdf of values.

  """
  nbins = kwargs.get('nbins', _NBINS)
  name = kwargs.get('name', None)
  with ops.name_scope(name, 'cdf', [values, value_range, nbins]):
    values = ops.convert_to_tensor(values, name='values')
    nbins_float = np.float32(nbins)

    # Map tensor values that fall within value_range to [0, 1].
    scaled_values = math_ops.truediv(
        values - value_range[0],
        value_range[1] - value_range[0],
        name='scaled_values')

    # map tensor values within the open interval value_range to {0,.., nbins-1},
    # values outside the open interval will be zero or less, or nbins or more.
    indices = math_ops.floor(nbins_float * scaled_values, name='indices')

    # Clip edge cases (e.g. value = value_range[1]) or "outliers."
    indices = math_ops.cast(
        clip_ops.clip_by_value(indices, 0, nbins_float - 1), dtypes.int32)

    cdf = array_ops.zeros(nbins)
    i = constant_op.constant(0)

    def loop_cond(loop_count, _):
      return math_ops.less(loop_count, nbins)

    def loop_body(loop_count, cdf):
      temp = math_ops.reduce_sum(
          math_ops.cast(
              math_ops.less_equal(indices, loop_count), dtypes.float32))
      cdf = math_ops.add(
          cdf,
          array_ops.one_hot(
              loop_count, depth=nbins, on_value=temp, off_value=0.0))
      return [loop_count + 1, cdf]

    _, cdf = control_flow_ops.while_loop(
        loop_cond, loop_body, [i, cdf], maximum_iterations=nbins)

    return math_ops.div(cdf, math_ops.reduce_max(cdf))


def factorized_pool(input_tensor,
                    window_shape,
                    pooling_type,
                    strides,
                    padding,
                    name=None):
  """Performs m x n pooling through a combination of 1xm and 1xn pooling.

  Args:
    input_tensor: Input tensor. Must be rank 2
    window_shape: Pooling window shape
    pooling_type: Either 'MAX' or 'AVG'
    strides: The stride of the pooling window
    padding: 'SAME' or 'VALID'.
    name: Name of the op

  Returns:
    A rank 2 tensor containing the pooled output

  Raises:
    ValueError: if the input tensor is not rank 2
  """
  if input_tensor.get_shape().ndims != 2:
    raise ValueError('factorized_pool() accepts tensors of rank 2 only')

  [height, width] = input_tensor.get_shape()
  with ops.name_scope(name, 'factorized_pool'):
    input_tensor_aligned = array_ops.reshape(
        input_tensor, [1, 1, height, width],
        name=input_tensor.op.name + '_aligned')

    height_pooling = nn_ops.pool(
        input_tensor_aligned,
        window_shape=[1, window_shape[0]],
        pooling_type=pooling_type,
        strides=[1, strides[0]],
        padding=padding)
    swap_height_width = array_ops.transpose(height_pooling, perm=[0, 1, 3, 2])

    width_pooling = nn_ops.pool(
        swap_height_width,
        window_shape=[1, window_shape[1]],
        pooling_type=pooling_type,
        strides=[1, strides[1]],
        padding=padding)

  return array_ops.squeeze(
      array_ops.transpose(width_pooling, perm=[0, 1, 3, 2]), axis=[0, 1])


def determine_partitioned_axis(partitioned_variable):
  partitioned_axis = 0
  concatenated_variable_shape = partitioned_variable.get_shape()
  for partition in partitioned_variable:
    partition_shape = partition.get_shape()
    maybe_partitioned_axis = np.less(partition_shape,
                                     concatenated_variable_shape)
    # Sanity check: make sure number of partitioned axis == 1
    if np.count_nonzero(maybe_partitioned_axis) != 1:
      raise ValueError('Number of partitioned axes %s not equal to 1' %
                       np.count_nonzero(maybe_partitioned_axis))
    partitioned_axis = np.where(maybe_partitioned_axis)[0][0]
  return partitioned_axis


def variable_assign(var, new_value):
  return state_ops.assign(var, new_value, name=var.op.name + '_assign')


def partitioned_variable_assign(partitioned_var, new_value):
  """Assign op for partitioned variables.

  Args:
    partitioned_var: A partitioned tensorflow variable
    new_value: Value to be assigned to the variable var

  Returns:
    A tensorflow op that groups the assign ops for each of the variable slices
  """
  # Determine which axis was used to partition the variable. Currently
  # tensorflow allows partitioning variable only along 1 axis.
  axis = 0 if len(partitioned_var) == 1 else determine_partitioned_axis(
      partitioned_var)

  partition_sizes = np.array(
      [partition.get_shape()[axis] for partition in partitioned_var])
  new_partitioned_values = array_ops.split(
      new_value,
      ops.convert_to_tensor(partition_sizes, dtype=dtypes.int32),
      axis=axis)
  op_list = []
  for partition in partitioned_var:
    op_list.append(
        variable_assign(partition, new_partitioned_values[len(op_list)]))
  return control_flow_ops.group(
      *op_list, name=partitioned_var.name + '_group_assign')
