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
"""Gather operations for RaggedTensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_ragged_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor


#===============================================================================
# ragged_gather
#===============================================================================
# TODO(edloper): Add an `axis` argument
def gather(params, indices, validate_indices=None, axis=0, batch_dims=0,
           name=None):
  """Gathers ragged slices from `params` axis `0` according to `indices`.

  Returns `RaggedTensor` output, such that:

  ```python
  output.shape = indices.shape + params.shape[1:]
  output.ragged_rank = indices.shape.ndims + params.ragged_rank
  output[i...j, d0...dn] = params[indices[i...j], d0...dn]
  ```

  `params` may be ragged.  `indices` may be ragged.
  `indices` must have dtype `int32` or `int64`. If any index is out of bounds,
  then an error is returned.

  Examples:

  >>> params = tf.constant(['a', 'b', 'c', 'd', 'e'])
  >>> indices = tf.constant([3, 1, 2, 1, 0])
  >>> ragged_params = tf.ragged.constant([['a', 'b', 'c'], ['d'], [], ['e']])
  >>> ragged_indices = tf.ragged.constant([[3, 1, 2], [1], [], [0]])

  >>> tf.gather(params, ragged_indices)
  <tf.RaggedTensor [[b'd', b'b', b'c'], [b'b'], [], [b'a']]>

  >>> tf.gather(ragged_params, indices)
  <tf.RaggedTensor [[b'e'], [b'd'], [], [b'd'], [b'a', b'b', b'c']]>

  >>> tf.gather(ragged_params, ragged_indices)
  <tf.RaggedTensor [[[b'e'], [b'd'], []], [[b'd']], [], [[b'a', b'b', b'c']]]>

  Args:
    params: The potentially ragged tensor from which to gather values. Must be
      at least rank 1.
    indices: The potentially ragged tensor indicating which values to gather.
      Must have dtype `int32` or `int64`.  Values must be in the range `[0,
      params.shape[0]]`.
    validate_indices: Ignored.
    axis: Must be zero.
    batch_dims: Must be zero.
    name: A name for the operation (optional).

  Returns:
    A `RaggedTensor`, where `output.dtype=params.dtype` and
    `output.shape=indices.shape + params.shape[1:]` and
    `output.ragged_rank=indices.shape.ndims + params.ragged_rank`.

  Raises:
    ValueError: If indices.shape.ndims is not known statically.
  """
  del validate_indices
  if not isinstance(axis, int) or axis != 0:
    raise ValueError('axis != 0 is not supported for ragged gather yet.')
  if not isinstance(batch_dims, int) or batch_dims != 0:
    raise ValueError('batch_dims != 0 is not supported for ragged gather yet.')
  with ops.name_scope(name, 'RaggedGather', [params, indices]):
    params = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        params, name='params')
    indices = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        indices, name='indices')
    params, indices = ragged_tensor.match_row_splits_dtypes(params, indices)

    if ragged_tensor.is_ragged(indices):
      return indices.with_values(gather(params, indices.values))

    if not ragged_tensor.is_ragged(params):
      return array_ops.gather(params, indices)

    indices = ops.convert_to_tensor(indices)
    if indices.shape.ndims is None:
      raise ValueError('indices.shape.ndims must be known statically')

    result = gen_ragged_array_ops.ragged_gather(
        indices=indices,
        params_dense_values=params.flat_values,
        params_nested_splits=params.nested_row_splits,
        OUTPUT_RAGGED_RANK=indices.shape.ndims + len(params.nested_row_splits) -
        1)

    # Compose the RaggedTensor from splits & values.
    return ragged_tensor.RaggedTensor.from_nested_row_splits(
        result.output_dense_values, result.output_nested_splits, validate=False)


#===============================================================================
# ragged.gather_nd
#===============================================================================
def gather_nd(params, indices, batch_dims=0, name=None):
  """Gather slices from `params` using `n`-dimensional indices.

  This operation is similar to `gather`, but it uses the innermost dimension
  of `indices` to define a slice into `params`.  In particular, if:

  * `indices` has shape `[A1...AN, I]`
  * `params` has shape `[B1...BM]`

  Then:

  * `result` has shape `[A1...AN, B_{I+1}...BM]`.
  * `result[a1...aN] = params[indices[a1...aN, :]]`

  Args:
    params: A potentially ragged tensor with shape `[A1...AN, I]`.
    indices: A potentially ragged tensor with shape `[B1...BM]`.
    batch_dims: Must be zero.
    name: A name for the operation (optional).

  Returns:
    A potentially ragged tensor with shape `[A1...AN, B_{I+1}...BM]`.

  #### Examples:

  >>> params = tf.ragged.constant(
  ...     [ [ ['000', '001'], ['010'              ]          ],
  ...       [ ['100'       ], ['110', '111', '112'], ['120'] ],
  ...       [ [            ], ['210'              ]          ] ])

  >>> # Gather 2D slices from a 3D tensor
  >>> tf.gather_nd(params, [[2], [0]])
  <tf.RaggedTensor [[[], [b'210']], [[b'000', b'001'], [b'010']]]>

  >>> # Gather 1D slices from a 3D tensor
  >>> tf.gather_nd(params, [[2, 1], [0, 0]])
  <tf.RaggedTensor [[b'210'], [b'000', b'001']]>

  >>> # Gather scalars from a 3D tensor
  >>> tf.gather_nd(params, [[0, 0, 1], [1, 1, 2]]).numpy()
  array([b'001', b'112'], dtype=object)
  """
  if not isinstance(batch_dims, int) or batch_dims != 0:
    raise ValueError('batch_dims != 0 is not supported for ragged gather yet.')
  if not (ragged_tensor.is_ragged(params) or ragged_tensor.is_ragged(indices)):
    return array_ops.gather_nd(params, indices, name)

  with ops.name_scope(name, 'RaggedGatherNd', [params, indices]):

    params = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        params, name='params')
    indices = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        indices, name='indices')
    params, indices = ragged_tensor.match_row_splits_dtypes(params, indices)
    indices_shape = indices.shape
    indices_ndims = indices_shape.ndims
    if indices_ndims is None:
      raise ValueError('indices.rank be statically known.')
    if indices_ndims == 0:
      raise ValueError('indices.rank must be at least 1.')
    if (ragged_tensor.is_ragged(indices) and
        indices_ndims == indices.ragged_rank + 1):
      raise ValueError('The innermost dimension of indices may not be ragged')

    # `index_size` is the "n" in "gather_nd" -- i.e., the number of dimensions
    # that each index slices into.
    index_size = tensor_shape.dimension_value(indices_shape[-1])
    if index_size is None:
      raise ValueError('indices.shape[-1] must be statically known.')

    # If `indices` has more than 2 dimensions, then recurse.  If `indices` is
    # dense, then we convert it to ragged before recursing, and then convert
    # the result back to `dense` if appropriate.
    if indices_ndims > 2:
      indices_is_dense = not ragged_tensor.is_ragged(indices)
      if indices_is_dense:
        indices = ragged_tensor.RaggedTensor.from_tensor(
            indices, ragged_rank=indices_ndims - 2,
            row_splits_dtype=params.row_splits.dtype)
      result = indices.with_flat_values(gather_nd(params, indices.flat_values))
      if (indices_is_dense and ragged_tensor.is_ragged(result) and
          result.ragged_rank == indices_ndims - 2):
        result = ragged_tensor.RaggedTensor.to_tensor(result)
      return result

    # indices_ndims <= 2, and the innermost dimension of indices may not be
    # ragged, so `indices` must not be ragged.
    assert not ragged_tensor.is_ragged(indices)
    assert ragged_tensor.is_ragged(params)

    # Handle corner case: An empty index tuple selects the entire `params`
    # value.  So if `index_size` is zero, then tile `params`.
    if index_size == 0:
      params_ndims = params.ragged_rank + array_ops.rank(params.flat_values)
      for dim in range(indices_ndims - 1):
        params = ragged_array_ops.expand_dims(params, axis=0)
      multiples = array_ops.concat([
          array_ops.shape(indices)[:-1],
          array_ops.ones([params_ndims], dtypes.int32)
      ],
                                   axis=0)
      return ragged_array_ops.tile(params, multiples)

    # When index_size=1, we can just flatten the index tuples and use gather.
    elif index_size == 1:
      flattened_index_tuples = array_ops.reshape(indices, [-1])
      return gather(params, flattened_index_tuples)

    # Otherwise, params is a RaggedTensor, and indices is a 1D or 2D Tensor.
    # Flatten both the index tuples and the params, such that the flattened
    # index tuples point to the correct values in the flattened params; and
    # then use ragged.gather on the flattened index tuples & params.
    else:
      indices = math_ops.cast(indices, params.row_splits.dtype)

      # Flatten the outermost 2 dimensions of the index tuples & params.
      flattened_index_tuples = array_ops.gather(params.row_splits,
                                                indices[..., 0])
      flattened_index_tuples += indices[..., 1]
      flattened_params = params.values

      # Flatten any remaining dimensions.
      for dim in range(2, index_size):
        if not ragged_tensor.is_ragged(flattened_params):
          flattened_index_tuples = array_ops.expand_dims(
              flattened_index_tuples, axis=1)
          flattened_index_tuples = array_ops.concat(
              [flattened_index_tuples, indices[..., dim:]], axis=1)
          return array_ops.gather_nd(flattened_params, flattened_index_tuples)

        flattened_index_tuples = array_ops.gather(
            flattened_params.row_starts(), flattened_index_tuples)
        flattened_index_tuples += indices[..., dim]
        flattened_params = flattened_params.values

      # Gather using the flattened index tuples and params.
      return gather(flattened_params, flattened_index_tuples)


#===============================================================================
# Gradient for the RaggedGather kernel
#===============================================================================
@ops.RegisterGradient('RaggedGather')
def _ragged_gather_grad(op, *grads):
  """Gradient for RaggedGather op."""
  param_nested_splits = op.inputs[:-2]
  param_inner_values = op.inputs[-2]
  indices = op.inputs[-1]
  grad_inner_values = grads[-1]

  # For each row in `params`, find the range of values in `params.inner_values`
  # that is covered by that row.  In particular, the values in row `i` are
  # `param_inner_values[combined_splits[i]:combined_splits[i+1]`.
  combined_splits = param_nested_splits[0]
  for row_splits in param_nested_splits[1:]:
    combined_splits = array_ops.gather(row_splits, combined_splits)

  # The outer dimensions of `indices` correspond 1:1 with the outer dimensions
  # of `ragged_grad` that are encoded by `grad_nested_splits`.  Thus, the
  # flattened `indices` correspond 1:1 with `grad_inner_values`.
  flat_indices = array_ops.reshape(indices, [-1])

  # Build an IndexedSlices where the values are taken from `flat_grad`.
  grad_indices = ragged_math_ops.range(
      array_ops.gather(combined_splits, flat_indices),
      array_ops.gather(combined_splits[1:], flat_indices)).values

  param_inner_values_grad = indexed_slices.IndexedSlices(
      values=grad_inner_values, indices=grad_indices,
      dense_shape=array_ops.shape(param_inner_values))
  return [None for _ in param_nested_splits] + [param_inner_values_grad, None]
