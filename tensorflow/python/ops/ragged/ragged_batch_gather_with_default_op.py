# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Array operations for RaggedTensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_dispatch  # pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_operators  # pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_shape
from tensorflow.python.ops.ragged import ragged_where_op


#===============================================================================
# ragged.batch_gather_with_default
#===============================================================================
def batch_gather_with_default(params,
                              indices,
                              default_value='',
                              name=None):
  """Same as `batch_gather` but inserts `default_value` for invalid indices.

  This operation is similar to `batch_gather` except that it will substitute
  the value for invalid indices with `default_value` as the contents.
  See `batch_gather` for more details.


  Args:
    params: A potentially ragged tensor with shape `[B1...BN, P1...PM]` (`N>=0`,
      `M>0`).
    indices: A potentially ragged tensor with shape `[B1...BN, I]` (`N>=0`).
    default_value: A value to be inserted in places where `indices` are out of
      bounds. Must be the same dtype as params and either a scalar or rank 1.
    name: A name for the operation (optional).

  Returns:
    A potentially ragged tensor with shape `[B1...BN, I, P2...PM]`.
    `result.ragged_rank = max(indices.ragged_rank, params.ragged_rank)`.

  #### Example:

  >>> params = tf.ragged.constant([['a', 'b', 'c'], ['d'], [], ['e']])
  >>> indices = tf.ragged.constant([[1, 2, -1], [], [], [0, 10]])
  >>> batch_gather_with_default(params, indices, 'FOO')
  <tf.RaggedTensor [[b'b', b'c', b'FOO'], [], [], [b'e', b'FOO']]>

  """
  with ops.name_scope(name, 'RaggedBatchGatherWithDefault'):
    params = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        params, name='params',
    )
    indices = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        indices, name='indices',
    )
    default_value = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        default_value, name='default_value',
    )
    row_splits_dtype, (params, indices, default_value) = (
        ragged_tensor.match_row_splits_dtypes(params, indices, default_value,
                                              return_dtype=True))
    # TODO(hterry): lift this restriction and support default_values of
    #               of rank > 1
    if default_value.shape.ndims not in (0, 1):
      raise ValueError('"default_value" must be a scalar or vector')
    upper_bounds = None
    if indices.shape.ndims is None:
      raise ValueError('Indices must have a known rank.')
    if params.shape.ndims is None:
      raise ValueError('Params must have a known rank.')

    num_batch_dimensions = indices.shape.ndims - 1
    pad = None
    # The logic for this works as follows:
    # - create a padded params, where:
    #    padded_params[b1...bn, 0] = default_value
    #    padded_params[b1...bn, i] = params[b1...bn, i-1] (i>0)
    # - create an `upper_bounds` Tensor that contains the number of elements
    #   in each innermost rank. Broadcast `upper_bounds` to be the same shape
    #   as `indices`.
    # - check to see which index in `indices` are out of bounds and substitute
    #   it with the index containing `default_value` (the first).
    # - call batch_gather with the indices adjusted.
    with ops.control_dependencies([
        check_ops.assert_greater_equal(array_ops.rank(params),
                                       array_ops.rank(indices))]):
      if ragged_tensor.is_ragged(params):
        row_lengths = ragged_array_ops.expand_dims(
            params.row_lengths(axis=num_batch_dimensions),
            axis=-1)
        upper_bounds = math_ops.cast(row_lengths, indices.dtype)

        pad_shape = _get_pad_shape(params, indices, row_splits_dtype)

        pad = ragged_tensor_shape.broadcast_to(
            default_value, pad_shape)
      else:
        params_shape = array_ops.shape(params)
        pad_shape = array_ops.concat([
            params_shape[:num_batch_dimensions],
            [1],
            params_shape[num_batch_dimensions + 1:params.shape.ndims]
        ], 0)
        upper_bounds = params_shape[num_batch_dimensions]
        pad = array_ops.broadcast_to(default_value, pad_shape)

      # Add `default_value` as the first value in the innermost (ragged) rank.
      pad = math_ops.cast(pad, params.dtype)
      padded_params = array_ops.concat(
          [pad, params], axis=num_batch_dimensions)

      # Adjust the indices by substituting out-of-bound indices to the
      # default-value index (which is the first element)
      shifted_indices = indices + 1
      is_out_of_bounds = (indices < 0) | (indices > upper_bounds)
      adjusted_indices = ragged_where_op.where(
          is_out_of_bounds,
          x=array_ops.zeros_like(indices), y=shifted_indices,
      )
      return array_ops.batch_gather(
          params=padded_params, indices=adjusted_indices, name=name)


def _get_pad_shape(params, indices, row_splits_dtype):
  """Gets the RaggedTensorDynamicShape for the pad tensor."""
  num_batch_dimensions = indices.shape.ndims - 1
  params_shape = ragged_tensor_shape.RaggedTensorDynamicShape.from_tensor(
      params, dim_size_dtype=row_splits_dtype)

  # We want to create a pad tensor that can be concatenated with the params.
  if params.shape.ndims == indices.shape.ndims:
    # When params and indices are the same rank, the shape of the pad tensor is
    # almost identical to params, except the last dimension which has size = 1.
    if params_shape.num_inner_dimensions == 0:
      pad_dims = params_shape.partitioned_dim_sizes[:-1] + (
          array_ops.ones_like(params_shape.partitioned_dim_sizes[-1]),)
      return ragged_tensor_shape.RaggedTensorDynamicShape(
          pad_dims, [])
    else:
      return ragged_tensor_shape.RaggedTensorDynamicShape(
          params_shape.partitioned_dim_sizes,
          array_ops.concat([params_shape.inner_dim_sizes[:-1], [1]], axis=0))
  else:
    # When the rank of indices < params, the pad has the same dimension as
    # params up to the 'num_batch_dimensions' rank. Every dimension after that
    # has size 1.
    pad_dims = None
    if num_batch_dimensions == 0:
      pad_dims = (constant_op.constant(1, dtype=row_splits_dtype),) + (
          constant_op.constant([1], dtype=row_splits_dtype),) * (
              params_shape.num_partitioned_dimensions -
              num_batch_dimensions - 1)
    else:
      batch_dimensions = params_shape.partitioned_dim_sizes[
          :num_batch_dimensions]
      gather_dimension = params_shape.partitioned_dim_sizes[
          num_batch_dimensions]
      pad_dims = batch_dimensions + (
          array_ops.ones_like(gather_dimension),) * (
              params_shape.num_partitioned_dimensions - num_batch_dimensions)

    return ragged_tensor_shape.RaggedTensorDynamicShape(
        pad_dims, params_shape.inner_dim_sizes)
