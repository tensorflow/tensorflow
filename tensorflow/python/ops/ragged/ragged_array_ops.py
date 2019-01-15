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

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_ragged_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_conversion_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.ops.ragged import segment_id_ops


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

  ```python
  >>> params = tf.constant(['a', 'b', 'c', 'd', 'e'])
  >>> indices = tf.constant([3, 1, 2, 1, 0])
  >>> ragged_params = tf.ragged.constant([['a', 'b', 'c'], ['d'], [], ['e']])
  >>> ragged_indices = tf.ragged.constant([[3, 1, 2], [1], [], [0]])

  >>> print ragged.gather(params, ragged_indices)
  [['d', 'b', 'c'], ['b'], [], ['a']]

  >>> print ragged.gather(ragged_params, indices)
  [['e'], ['d'], [], ['d'], ['a', 'b', 'c']]

  >>> print ragged.gather(ragged_params, ragged_indices)
  [[['e'], ['d'], []], [['d']], [], [['a', 'b', 'c']]]
  ```

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
        result.output_dense_values, result.output_nested_splits)


#===============================================================================
# ragged.batch_gather
#===============================================================================
def batch_gather(params, indices, name=None):
  """Gathers slices from `params` according to `indices` with batch dims.

  This operation is similar to `gather`, but it assumes that the leading `N`
  dimensions of `indices` and `params` are batch dimensions, and performs a
  gather within each batch.  In particular, when using this operation with `N`
  batch dimensions `B1...BN`:

  * `indices` has shape `[B1...BN, I]`
  * `params` has shape `[B1...BN, P1...PM]`.
  * `result` has shape `[B1...BN, I, P2...PM]`.
  * `result[b1...bN, i, p2...pM] =
    params[b1...bN, indices[b1...bN, i], p2...pM]`

  Args:
    params: A potentially ragged tensor with shape `[B1...BN, P1...PM]` (`N>=0`,
      `M>0`).
    indices: A potentially ragged tensor with shape `[B1...BN, I]` (`N>=0`).
    name: A name for the operation (optional).

  Returns:
    A potentially ragged tensor with shape `[B1...BN, I, P2...PM]`.
    `result.ragged_rank = max(indices.ragged_rank, params.ragged_rank)`.

  #### Example:
    ```python
    >>> params = tf.ragged.constant([['a', 'b', 'c'], ['d'], [], ['e']])
    >>> indices = tf.ragged.constant([[1, 2, 0], [], [], [0, 0]])
    >>> ragged.batch_gather(params, indices)
    [['b', 'c', 'a'], [], [], ['e', 'e']]
    ```
  """
  if not (ragged_tensor.is_ragged(params) or ragged_tensor.is_ragged(indices)):
    return array_ops.batch_gather(params, indices, name)

  with ops.name_scope(name, 'RaggedBatchGather', [params, indices]):
    params = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        params, name='params')
    indices = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        indices, name='indices')
    indices_ndims = indices.shape.ndims
    if indices_ndims is None:
      raise ValueError(
          'batch_gather does not allow indices with unknown shape.')
    if indices_ndims == 0:
      raise ValueError('indices.rank must be at least 1.')

    if ragged_tensor.is_ragged(indices):
      # If the outermost ragged dimension is a batch dimension, recurse.
      if indices_ndims > 2:
        if not ragged_tensor.is_ragged(params):
          raise ValueError('batch shape from indices does '
                           'not match params shape')
        checks = [check_ops.assert_equal(params.row_splits, indices.row_splits)]
        with ops.control_dependencies(checks):
          return ragged_tensor.RaggedTensor.from_row_splits(
              batch_gather(params.values, indices.values), indices.row_splits)

      # Otherwise, indices is a 2D ragged tensor with 1 ragged dimension.
      else:
        # Ensure that `params` is ragged and has at least 2 dimensions.
        if not ragged_tensor.is_ragged(params):
          if params.shape.ndims is not None and params.shape.ndims < 2:
            raise ValueError('batch shape from indices does '
                             'not match params shape')
          params = ragged_conversion_ops.from_tensor(params, ragged_rank=1)

        # Adjust indices from within-batch to global (in params.values), and
        # then use ragged.gather to gather them.
        num_indices = indices.row_lengths()
        params_starts = params.row_starts()
        adjustments = ragged_util.repeat(params_starts, num_indices, axis=0)
        adjusted_index_values = math_ops.to_int64(indices.values) + adjustments
        return ragged_tensor.RaggedTensor.from_row_splits(
            gather(params.values, adjusted_index_values), indices.row_splits)

    else:  # params is a RaggedTensor and indices is a Tensor.
      if indices_ndims == 1:
        return gather(params, indices)
      elif indices_ndims == 2:
        # Adjust indices from batch-local to global (in params.values)
        adjustments = array_ops.expand_dims(params.row_starts(), 1)
        adjusted_indices = math_ops.to_int64(indices) + adjustments
        return gather(params.values, adjusted_indices)
      else:
        raise ValueError('batch shape from indices does not match params shape')


#===============================================================================
# ragged.gather_nd
#===============================================================================
def gather_nd(params, indices, name=None):
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
    name: A name for the operation (optional).

  Returns:
    A potentially ragged tensor with shape `[A1...AN, B_{I+1}...BM]`.

  #### Examples:
    ```python
    >>> params = tf.ragged.constant_value(
    ...     [ [ ['000', '001'], ['010'              ]          ],
    ...       [ ['100'       ], ['110', '111', '112'], ['120'] ],
    ...       [ [            ], ['210'              ]          ] ])

    >>> # Gather 2D slices from a 3D tensor
    >>> ragged.gather_nd(params, [[2], [0]])
    [ [ [            ], ['210'] ]
      [ ['000', '001'], ['010'] ] ]

    >>> # Gather 1D slices from a 3D tensor
    >>> ragged.gather_nd(params, [[2, 1], [0, 0]])
    [['210'], ['000', '001']]

    >>> # Gather scalars from a 3D tensor
    >>> ragged.gather_nd(params, [[0, 0, 1], [1, 1, 2]])
    ['001', '112']
    ```
  """
  if not (ragged_tensor.is_ragged(params) or ragged_tensor.is_ragged(indices)):
    return array_ops.gather_nd(params, indices, name)

  with ops.name_scope(name, 'RaggedGatherNd', [params, indices]):

    params = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        params, name='params')
    indices = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        indices, name='indices')
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
        indices = ragged_conversion_ops.from_tensor(
            indices, ragged_rank=indices_ndims - 2)
      result = indices.with_flat_values(gather_nd(params, indices.flat_values))
      if (indices_is_dense and ragged_tensor.is_ragged(result) and
          result.ragged_rank == indices_ndims - 2):
        result = ragged_conversion_ops.to_tensor(result)
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
        params = expand_dims(params, axis=0)
      multiples = array_ops.concat([
          array_ops.shape(indices)[:-1],
          array_ops.ones([params_ndims], dtypes.int32)
      ],
                                   axis=0)
      return tile(params, multiples)

    # When index_size=1, we can just flatten the index tuples and use gather.
    elif index_size == 1:
      flattened_index_tuples = array_ops.reshape(indices, [-1])
      return gather(params, flattened_index_tuples)

    # Otherwise, params is a RaggedTensor, and indices is a 1D or 2D Tensor.
    # Flatten both the index tuples and the params, such that the flattened
    # index tuples point to the correct values in the flattened params; and
    # then use ragged.gather on the flattened index tuples & params.
    else:
      indices = math_ops.to_int64(indices)

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
# Masking
#===============================================================================
def boolean_mask(data, mask, keepdims=False, name=None):
  """Applies a boolean mask to `data`.

  Returns a potentially ragged tensor that is formed by retaining the elements
  in `data` where the corresponding value in `mask` is `True`.

  If `keepdims` is true then outer dimensions (corresponding to the `mask`
  dimensions) are preserved, and:

  * `output[a1...aA, i, b1...bB] = data[a1...aA, j, b1...bB]`

     Where `j` is the `i`th `True` entry of `mask[a1...aA]`.

  If `keepdims` is false, then the outer dimensions are collapsed (similar to
  the behavior of `tf.boolean_mask`), and:

  * `output[i, b1...bB] = data[a1...aA, b1...bB]`

     Where `(a1...aA)` is the `i`th `True` entry of `mask`
     (in row-major order).

  Args:
    data: A potentially ragged tensor.
    mask: A potentially ragged boolean tensor.  `mask`'s shape must be a prefix
      of `data`'s shape.  `rank(mask)` must be known statically.
    keepdims: Whether to preserve the outer dimensions (`keepdims=True`) or
      flatten them (`keepdims=False`).
    name: A name prefix for the returned tensor (optional).

  Returns:
    A potentially ragged tensor that is formed by retaining the elements in
    `data` where the corresponding value in `mask` is `True`.

    If `keepdims` is false:

    * `rank(output) = rank(data) - rank(mask) + 1`.
    * `output.ragged_rank = max(data.ragged_rank - rank(mask) + 1, 0)`.

    If `keepdims` is true:

    * `rank(output) = rank(data)`.
    * `output.ragged_rank = max(data.ragged_rank, rank(mask) - 1)`.

  Raises:
    ValueError: if `rank(mask)` is not known statically; or if `mask.shape` is
      not a prefix of `data.shape`.

  #### Examples:
    ```python
    >>> # Aliases for True & False so data and mask line up.
    >>> T, F = (True, False)

    >>> tf.ragged.boolean_mask(  # Mask a 2D Tensor.  Flatten outer dims.
    ...     data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ...     mask=[[T, F, T], [F, F, F], [T, F, F]],
    ...     keepdims=False).tolist()
    [1, 3, 7]

    >>> tf.ragged.boolean_mask(  # Mask a 2D Tensor.  Preserve outer dims.
    ...     data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ...     mask=[[T, F, T], [F, F, F], [T, F, F]],
    ...     keepdims=True).tolist()
    [[1, 3], [], [7]]

    >>> tf.ragged.boolean_mask(  # Mask a 2D RaggedTensor.  Flatten outer dims.
    ...     tf.ragged.constant([[1, 2, 3], [4], [5, 6]]),
    ...     tf.ragged.constant([[F, F, T], [F], [T, T]]),
    ...     keepdims=False).tolist()
    [3, 5, 6]

    >>> tf.ragged.boolean_mask(  # Mask a 2D RaggedTensor.  Preserve outer dims.
    ...     tf.ragged.constant([[1, 2, 3], [4], [5, 6]]),
    ...     tf.ragged.constant([[F, F, T], [F], [T, T]]),
    ...     keepdims=True).tolist()
    [[3], [], [5, 6]]

    >>> tf.ragged.boolean_mask(  # Mask rows of a 2D RaggedTensor.
    ...     tf.ragged.constant([[1, 2, 3], [4], [5, 6]]),
    ...     tf.ragged.constant([True, False, True]),
    ...     keepdims=True).tolist()
    [[1, 2, 3], [5, 6]]
    ```
  """
  with ops.name_scope(name, 'RaggedMask', [data, mask]):
    # Convert inputs to tensors.
    data = ragged_tensor.convert_to_tensor_or_ragged_tensor(data, name='data')
    mask = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        mask, dtypes.bool, name='mask')

    # Get static rank of mask.
    if mask.shape.ndims is None:
      raise ValueError('mask.shape.ndims must be kown statically.')
    elif mask.shape.ndims == 0:
      raise ValueError('mask cannot be scalar.')

    # If mask is ragged, then recurse with a non-ragged mask.
    if ragged_tensor.is_ragged(mask):
      if not ragged_tensor.is_ragged(data):
        data = ragged_conversion_ops.from_tensor(
            data, ragged_rank=mask.ragged_rank)
      # Check that mask.nested_row_splits is a prefix of
      # data.nested_row_splits.
      splits_list = [
          mask.nested_row_splits, data.nested_row_splits[:mask.ragged_rank]
      ]
      with ops.control_dependencies(
          ragged_util.assert_splits_match(splits_list)):
        # Strip off ragged `splits` until `mask` is non-ragged.  Keep the splits
        # that we strip off in `splits`, so we can add them back on after
        # we recursively mask the non-ragged data.
        splits = []
        while ragged_tensor.is_ragged(mask):
          if mask.shape.ndims > 2:
            splits.append(mask.row_splits)
          else:
            # Count the number of True mask values in each row to find the
            # lengths of the filtered rows; then convert to splits.
            int_mask = ragged_functional_ops.map_flat_values(
                math_ops.cast, mask, dtype=dtypes.int64)
            masked_row_lengths = ragged_math_ops.reduce_sum(int_mask, axis=1)
            splits.append(ragged_util.lengths_to_splits(masked_row_lengths))
          mask = mask.values
          data = data.values

        # Recursively apply the nested non-ragged mask to the nested data.
        masked_values = boolean_mask(data, mask, keepdims)

        # Add the ragged `splits` back to the result.
        if keepdims:
          masked_values = ragged_tensor.RaggedTensor.from_nested_row_splits(
              masked_values, splits)

        return masked_values

    # If mask is non-ragged and has rank 1, and data is ragged, then build a
    # ragged tensor with the indicated rows.
    elif ragged_tensor.is_ragged(data) and mask.shape.ndims == 1:
      # Get the masked splits: first get the length of each row, then filter
      # out the rows that we are deleting, and convert that filtered set of
      # masks back to a splits tensor.
      lengths = data.row_lengths()
      masked_lengths = array_ops.boolean_mask(lengths, mask)
      masked_splits = ragged_util.lengths_to_splits(masked_lengths)

      # Get the masked values: first get row ids corresponding to each
      # value, then use tf.gather to build a boolean mask that's false for
      # values that come from rows that we are deleting, and use that mask to
      # construct the masked values tensor.
      segment_ids = segment_id_ops.row_splits_to_segment_ids(data.row_splits)
      segment_mask = array_ops.gather(mask, segment_ids)
      masked_values = boolean_mask(data.values, segment_mask, keepdims=False)

      return ragged_tensor.RaggedTensor.from_row_splits(masked_values,
                                                        masked_splits)

    # If mask is non-ragged and has rank>1, then convert it to be ragged,
    # with a ragged rank matching data.
    if ragged_tensor.is_ragged(data):
      mask = ragged_conversion_ops.from_tensor(
          mask, ragged_rank=min(data.ragged_rank, mask.shape.ndims - 1))
      return boolean_mask(data, mask, keepdims)

    # Otherwise, data and mask are both `Tensor`s.
    else:
      # Apply `boolean_mask` to get the masked values.
      masked_values = array_ops.boolean_mask(data, mask)

      if mask.shape.ndims >= 2 and keepdims:
        # Add the innermost ragged dimension.  For each innermost cell, get the
        # number of values it contains.  Then flatten that to get a list of
        # cell lengths, and convert it to splits.  Finally, combine the splits
        # and values to get the innermost ragged tensor.
        masked_lengths = math_ops.count_nonzero(mask, axis=-1)
        flattened_masked_lengths = array_ops.reshape(masked_lengths, [-1])
        masked_values = ragged_tensor.RaggedTensor.from_row_lengths(
            masked_values, flattened_masked_lengths)

        # Wrap remaining ragged dimensions.
        if mask.shape.ndims > 2 and keepdims:
          mask_shape = array_ops.shape(mask, out_type=dtypes.int64)
          split_size = math_ops.cumprod(mask_shape) + 1
          for dim in range(mask.shape.ndims - 3, -1, -1):
            elt_size = mask_shape[dim + 1]
            masked_splits = math_ops.range(split_size[dim]) * elt_size
            masked_values = ragged_tensor.RaggedTensor.from_row_splits(
                masked_values, masked_splits)

      return masked_values


#===============================================================================
# Concatenation and Stacking
#===============================================================================
def concat(values, axis, name=None):
  """Concatenates potentially ragged tensors along one dimension.

  Given a list of tensors with the same rank `K` (`K >= axis`), returns a
  rank-`K` `RaggedTensor` `result` such that `result[i0...iaxis]` is the
  concatenation of `[rt[i0...iaxis] for rt in values]`.

  Args:
    values: A list of potentially ragged tensors.  May not be empty. All
      `values` must have the same rank and the same dtype; but unlike
      `tf.concat`, they can have arbitrary shapes.
    axis: A python integer, indicating the dimension along which to concatenate.
      (Note: Unlike `tf.concat`, the `axis` parameter must be statically known.)
        Negative values are supported only if the rank of at least one
        `values` value is statically known.
    name: A name prefix for the returned tensor (optional).

  Returns:
    A `RaggedTensor` with rank `K`.
    `result.ragged_rank=max(axis, max(rt.ragged_rank for rt in values]))`.

  Raises:
    ValueError: If `values` is empty, if `axis` is out of bounds or if
      the input tensors have different ranks.

  #### Example:
    ```python
    >>> t1 = tf.ragged.constant([[1, 2], [3, 4, 5]])
    >>> t2 = tf.ragged.constant([[6], [7, 8, 9]])
    >>> ragged.concat([t1, t2], axis=0)
    [[1, 2], [3, 4, 5], [6], [7, 8, 9]]
    >>> ragged.concat([t1, t2], axis=1)
    [[1, 2, 6], [3, 4, 5, 7, 8, 9]]
    ```
  """
  if not isinstance(values, (list, tuple)):
    values = [values]
  with ops.name_scope(name, 'RaggedConcat', values):
    return _ragged_stack_concat_helper(values, axis, stack_values=False)


def stack(values, axis=0, name=None):
  """Stacks potentially ragged tensors along one dimension.

  Given a list of tensors with the same rank `K` (`K >= axis`), returns a
  rank-`K+1` `RaggedTensor` `result` such that `result[i0...iaxis]` is the
  list `[rt[i0...iaxis] for rt in values]`.

  Args:
    values: A list of potentially ragged tensors.  May not be empty. All
      `values` must have the same rank and the same dtype; but unlike
      `tf.concat`, they can have arbitrary shapes.
    axis: A python integer, indicating the dimension along which to stack.
      (Note: Unlike `tf.stack`, the `axis` parameter must be statically known.)
        Negative values are supported only if the rank of at least one
        `values` value is statically known.
    name: A name prefix for the returned tensor (optional).

  Returns:
    A `RaggedTensor` with rank `K+1`.
    `result.ragged_rank=max(axis, max(rt.ragged_rank for rt in values]))`.

  Raises:
    ValueError: If `values` is empty, if `axis` is out of bounds or if
      the input tensors have different ranks.

  #### Example:
    ```python
    >>> t1 = tf.ragged.constant([[1, 2], [3, 4, 5]])
    >>> t2 = tf.ragged.constant([[6], [7, 8, 9]])
    >>> ragged.stack([t1, t2], axis=0)
    [[[1, 2], [3, 4, 5]], [[6], [7, 9, 0]]]
    >>> ragged.stack([t1, t2], axis=1)
    [[[1, 2], [6]], [[3, 4, 5], [7, 8, 9]]]
    ```
  """
  if not isinstance(values, (list, tuple)):
    values = [values]
  with ops.name_scope(name, 'RaggedConcat', values):
    return _ragged_stack_concat_helper(values, axis, stack_values=True)


def _ragged_stack_concat_helper(rt_inputs, axis, stack_values):
  """Helper function to concatenate or stack ragged tensors.

  Args:
    rt_inputs: A list of RaggedTensors or Tensors to combine.
    axis: The axis along which to concatenate or stack.
    stack_values: A boolean -- if true, then stack values; otherwise,
      concatenate them.

  Returns:
    A RaggedTensor.
  Raises:
    ValueError: If rt_inputs is empty, or if axis is out of range.
  """
  # Validate parameters.
  if not rt_inputs:
    raise ValueError('rt_inputs may not be empty.')

  # Convert input tensors.
  rt_inputs = [
      ragged_tensor.convert_to_tensor_or_ragged_tensor(
          rt_input, name='rt_input') for rt_input in rt_inputs
  ]

  # Special case: if there's only one input, then return it as-is.
  if len(rt_inputs) == 1:
    if stack_values:
      return expand_dims(rt_inputs[0], axis=axis)
    else:
      return rt_inputs[0]

  # Check the rank (number of dimensions) of the input tensors.
  ndims = None
  for rt in rt_inputs:
    if ndims is None:
      ndims = rt.shape.ndims
    else:
      rt.shape.assert_has_rank(ndims)

  out_ndims = ndims if (ndims is None or not stack_values) else ndims + 1
  axis = ragged_util.get_positive_axis(axis, out_ndims)

  # If all the inputs are Tensors, and we're combining the final dimension,
  # then we can delegate to the tf.stack/tf.concat operation, and return a
  # Tensor.
  if all(not ragged_tensor.is_ragged(rt) for rt in rt_inputs):
    if ndims is not None and (axis == out_ndims - 1 or axis == ndims - 1):
      if stack_values:
        return array_ops.stack(rt_inputs, axis)
      else:
        return array_ops.concat(rt_inputs, axis)

  # Convert any Tensor inputs to RaggedTensors.  This makes it
  # possible to concatenate Tensors and RaggedTensors together.
  for i in range(len(rt_inputs)):
    if not ragged_tensor.is_ragged(rt_inputs[i]):
      rt_inputs[i] = ragged_conversion_ops.from_tensor(
          rt_inputs[i], ragged_rank=1)

  # Convert the input tensors to all have the same ragged_rank.
  ragged_rank = max(max(rt.ragged_rank for rt in rt_inputs), 1)
  rt_inputs = [_increase_ragged_rank_to(rt, ragged_rank) for rt in rt_inputs]

  if axis == 0:
    return _ragged_stack_concat_axis_0(rt_inputs, stack_values)
  elif axis == 1:
    return _ragged_stack_concat_axis_1(rt_inputs, stack_values)
  else:  # axis > 1: recurse.
    values = [rt.values for rt in rt_inputs]
    splits = [[rt_input.row_splits] for rt_input in rt_inputs]
    with ops.control_dependencies(ragged_util.assert_splits_match(splits)):
      return ragged_tensor.RaggedTensor.from_row_splits(
          _ragged_stack_concat_helper(values, axis - 1, stack_values),
          splits[0][0])


def _ragged_stack_concat_axis_0(rt_inputs, stack_values):
  """Helper function to concatenate or stack ragged tensors along axis 0.

  Args:
    rt_inputs: A list of RaggedTensors, all with the same rank and ragged_rank.
    stack_values: Boolean.  If true, then stack values; otherwise, concatenate
      them.

  Returns:
    A RaggedTensor.
  """
  # Concatenate the inner values together.
  flat_values = [rt.flat_values for rt in rt_inputs]
  concatenated_flat_values = array_ops.concat(flat_values, axis=0)

  # Concatenate the splits together for each ragged dimension (adjusting
  # split offsets as necessary).
  nested_splits = [rt.nested_row_splits for rt in rt_inputs]
  ragged_rank = rt_inputs[0].ragged_rank
  concatenated_nested_splits = [
      _concat_ragged_splits([ns[dim]
                             for ns in nested_splits])
      for dim in range(ragged_rank)
  ]

  # If we are performing a stack operation, then add another splits.
  if stack_values:
    stack_lengths = array_ops.stack([_nrows(rt) for rt in rt_inputs])
    stack_splits = ragged_util.lengths_to_splits(stack_lengths)
    concatenated_nested_splits.insert(0, stack_splits)

  return ragged_tensor.RaggedTensor.from_nested_row_splits(
      concatenated_flat_values, concatenated_nested_splits)


def _ragged_stack_concat_axis_1(rt_inputs, stack_values):
  """Helper function to concatenate or stack ragged tensors along axis 1.

  Args:
    rt_inputs: A list of RaggedTensors, all with the same rank and ragged_rank.
    stack_values: Boolean.  If true, then stack values; otherwise, concatenate
      them.

  Returns:
    A RaggedTensor.
  """
  num_inputs = len(rt_inputs)

  rt_nrows = _nrows(rt_inputs[0])
  nrows_msg = 'Input tensors have incompatible shapes.'
  nrows_checks = [
      check_ops.assert_equal(_nrows(rt), rt_nrows, message=nrows_msg)
      for rt in rt_inputs[1:]
  ]

  with ops.control_dependencies(nrows_checks):
    # Concatentate the inputs together to put them in a single ragged tensor.
    concatenated_rt = _ragged_stack_concat_axis_0(rt_inputs, stack_values=False)

    # Use ragged.gather to permute the rows of concatenated_rt.  In particular,
    #   permuted_rt = [rt_inputs[0][0], ..., rt_inputs[N][0],
    #                  rt_inputs[0][1], ..., rt_inputs[N][1],
    #                      ...,
    #                  rt_inputs[0][M], ..., rt_input[N][M]]
    # where `N=num_inputs-1` and `M=rt_nrows-1`.
    row_indices = math_ops.range(rt_nrows * num_inputs)
    row_index_matrix = array_ops.reshape(row_indices, [num_inputs, -1])
    transposed_row_index_matrix = array_ops.transpose(row_index_matrix)
    row_permutation = array_ops.reshape(transposed_row_index_matrix, [-1])
    permuted_rt = gather(concatenated_rt, row_permutation)

    if stack_values:
      # Add a new splits tensor to group together the values.
      stack_splits = math_ops.range(0, rt_nrows * num_inputs + 1, num_inputs)
      _copy_row_shape(rt_inputs, stack_splits)
      return ragged_tensor.RaggedTensor.from_row_splits(permuted_rt,
                                                        stack_splits)
    else:
      # Merge together adjacent rows by dropping the row-split indices that
      # separate them.
      concat_splits = permuted_rt.row_splits[::num_inputs]
      _copy_row_shape(rt_inputs, concat_splits)
      return ragged_tensor.RaggedTensor.from_row_splits(permuted_rt.values,
                                                        concat_splits)


def _copy_row_shape(rt_inputs, splits):
  """Sets splits.shape to [rt[shape[0]+1] for each rt in rt_inputs."""
  for rt in rt_inputs:
    if rt.shape[0] is not None:
      splits.set_shape(tensor_shape.TensorShape(rt.shape[0] + 1))


#===============================================================================
# Tiling
#===============================================================================
def tile(input, multiples, name=None):  # pylint: disable=redefined-builtin
  """Constructs a `RaggedTensor` by tiling a given `RaggedTensor`.

  The values of `input` are replicated `multiples[i]` times along the
  `i`th dimension (for each dimension `i`).  For every dimension `axis` in
  `input`, the length of each output element in that dimension is the
  length of corresponding input element multiplied by `multiples[axis]`.

  Args:
    input: A `RaggedTensor`.
    multiples: A 1-D integer `Tensor`.  Length must be the same as the number of
      dimensions in `input`.
    name: A name for the operation (optional).

  Returns:
    A `RaggedTensor` with the same type, rank, and ragged_rank as `input`.

  #### Example:
    ```python
    >>> rt = tf.ragged.constant([[1, 2], [3]])
    >>> ragged.tile(rt, [3, 2])
    [[1, 2, 1, 2], [3, 3], [1, 2, 1, 2], [3, 3], [1, 2, 1, 2], [3, 3]]
    ```
  """
  with ops.name_scope(name, 'RaggedTile', [input, multiples]):
    input = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        input, name='input')
    multiples = ragged_util.convert_to_int_tensor(
        multiples, name='multiples', dtype=dtypes.int64)
    multiples.shape.assert_has_rank(1)
    if not ragged_tensor.is_ragged(input):
      return array_ops.tile(input, multiples, name)

    # If the constant value of `multiples` is available, then we can use it
    # to skip tiling dimensions where `multiples=1`.
    const_multiples = tensor_util.constant_value(multiples)

    return ragged_tensor.RaggedTensor.from_nested_row_splits(
        _tile_ragged_values(input, multiples, const_multiples),
        _tile_ragged_splits(input, multiples, const_multiples))


def _tile_ragged_values(rt_input, multiples, const_multiples=None):
  """Builds flat_values tensor for a tiled `RaggedTensor`.

  Returns a tensor that repeats the values in
  `rt_input.flat_values` in the
  appropriate pattern to construct a `RaggedTensor` that tiles `rt_input` as
  specified by `multiples`.

  Args:
    rt_input: The `RaggedTensor` whose values should be repeated.
    multiples: A 1-D integer `tensor`, indicating how many times each dimension
      should be repeated.
    const_multiples: Optional constant value for multiples.  Used to skip tiling
      dimensions where `multiples=1`.

  Returns:
    A `Tensor` with the same type and rank as `rt_input.flat_values`.

  #### Example:
    ```python
    >>> rt = tf.ragged.constant([[1, 2], [3]])
    >>> _tile_ragged_values(rt, [3, 2])
    [1, 2, 1, 2, 3, 3, 1, 2, 1, 2, 3, 3, 1, 2, 1, 2, 3, 3]
    ```
  """
  ragged_rank = rt_input.ragged_rank
  nested_splits = rt_input.nested_row_splits

  # Pointers to the values in `rt_input.flat_values`.
  inner_value_ids = math_ops.range(nested_splits[-1][-1])

  # For each ragged dimension (working from the innermost to outermost),
  # expand `inner_value_ids` as necessary to tile that dimension.
  prev_splits = None
  for axis in range(ragged_rank, 0, -1):
    # Ragged splits for this dimension.
    splits = nested_splits[axis - 1]

    # Adjust splits so they point into `inner_value_ids` (instead of just
    # pointing into the next dimension's values).
    if prev_splits is not None:  # Not the first pass through the loop.
      splits = array_ops.gather(prev_splits * multiples[axis + 1], splits)

    # Repeat each element in this ragged dimension `multiples[axis]` times.
    if const_multiples is None or const_multiples[axis] != 1:
      inner_value_ids = ragged_util.repeat_ranges(inner_value_ids, splits,
                                                  multiples[axis])

    prev_splits = splits

  # Gather the tiled inner values.
  ragged_tiled_values = array_ops.gather(rt_input.flat_values, inner_value_ids)

  # Tile the flat_values for the uniform dimensions (i.e., for `axis=0` plus
  # `axis=range(ragged_rank, rank)`).
  inner_repeats = array_ops.concat([multiples[:1], multiples[ragged_rank + 1:]],
                                   axis=0)
  return array_ops.tile(ragged_tiled_values, inner_repeats)


def _tile_ragged_splits(rt_input, multiples, const_multiples=None):
  """Builds nested_split tensors for a tiled `RaggedTensor`.

  Returns a list of split tensors that can be used to construct the
  `RaggedTensor` that tiles `rt_input` as specified by `multiples`.

  Args:
    rt_input: The `RaggedTensor` that is being tiled.
    multiples: A 1-D integer `tensor`, indicating how many times each dimension
      should be repeated.
    const_multiples: Optional constant value for multiples.  Used to skip tiling
      dimensions where `multiples=1`.

  Returns:
    A list of 1-D `int64` `Tensor`s (one for each ragged dimension in
    `rt_input`).

  #### Example:
    ```python
    >>> rt = tf.ragged.constant([[1, 2], [3]])
    >>> _tile_ragged_splits(rt, [3, 2])
    [0, 4, 6, 10, 12, 16, 18]
    ```
  """
  ragged_rank = rt_input.ragged_rank
  nested_splits = rt_input.nested_row_splits

  # projected_splits[src_axis, dst_axis] contains the split points that divide
  # the rows from src_axis in the list of dst_axis values.  E.g.,
  # projected_splits[i, i] = nested_splits[i], and
  # projected_splits[i, i+1] = gather(nested_splits[i+1], nested_splits[i]).
  projected_splits = [{i: nested_splits[i]} for i in range(ragged_rank)]
  for src_axis in range(ragged_rank):
    for dst_axis in range(src_axis + 1, ragged_rank - 1):
      projected_splits[src_axis][dst_axis] = array_ops.gather(
          nested_splits[dst_axis],
          projected_splits[src_axis][dst_axis - 1])

  # For each ragged dimension: nested_splits[axis] -> result_splits[axis].
  result_splits = []
  for axis in range(ragged_rank):
    # Get the length of each row for the input tensor for this dimension.
    input_lengths = nested_splits[axis][1:] - nested_splits[axis][:-1]

    # Multiply those lengths by the `multiples` of dimension axis+1, since
    # each value will be repeated that number of times.
    output_lengths = input_lengths * multiples[axis + 1]

    # Repeat ranges of the row lengths as necessary for them to be tiled in
    # each ragged dimension `d < axis`.  (Start with dimension d=axis-1, and
    # work our way up to dimension d=0.)
    repeats = 1
    for d in range(axis - 1, -1, -1):
      if const_multiples is None or const_multiples[d + 1] != 1:
        splits = projected_splits[d][axis - 1] * repeats
        output_lengths = ragged_util.repeat_ranges(output_lengths, splits,
                                                   multiples[d + 1])
      repeats *= multiples[d + 1]

    # Tile splits for the outermost (uniform) dimension.
    output_lengths = array_ops.tile(output_lengths, multiples[:1])

    # Convert to splits.
    result_splits.append(ragged_util.lengths_to_splits(output_lengths))

  return result_splits


#===============================================================================
# Reshaping
#===============================================================================


def expand_dims(input, axis, name=None):  # pylint: disable=redefined-builtin
  """Inserts a dimension with shape 1 into a potentially ragged tensor's shape.

  Given a potentially ragged tenor `input`, this operation inserts a
  dimension with size 1 at the dimension `axis` of `input`'s shape.

  * If `input` is a `Tensor`, then this is equivalent to
    `tf.expand_dims`.
  * If `input` is ragged, and `axis=0`, then the new dimension will be
    uniform; but the previously outermost dimension will become ragged.
  * If `input` is ragged, and `0 < axis < input.ragged_rank`, then the
    new dimension will be ragged.
  * If `input` is ragged, and axis >= input.ragged_rank`, then the new
    dimension will be uniform.

  The following table gives some examples showing how `ragged.expand_dims`
  impacts the shapes of different input tensors.  Ragged dimensions are
  indicated by enclosing them in parentheses.

  input.shape             | axis | result.shape
  ----------------------- | ---- | -----------------------------
  `[D1, D2]`              |  `0` | `[1, D1, D2]`
  `[D1, D2]`              |  `1` | `[D1, 1, D2]`
  `[D1, D2]`              |  `2` | `[D1, D2, 1]`
  `[D1, (D2), (D3), D4]`  |  `0` | `[1, (D1), (D2), (D3), D4]`
  `[D1, (D2), (D3), D4]`  |  `1` | `[D1, (1), (D2), (D3), D4]`
  `[D1, (D2), (D3), D4]`  |  `2` | `[D1, (D2), (1), (D3), D4]`
  `[D1, (D2), (D3), D4]`  |  `3` | `[D1, (D2), (D3), 1, D4]`
  `[D1, (D2), (D3), D4]`  |  `4` | `[D1, (D2), (D3), D4, 1]`

  Args:
    input: The potentially tensor that should be expanded with a new
      dimension.
    axis: An integer constant indicating where the new dimension should be
      inserted.
    name: A name for the operation (optional).

  Returns:
    A tensor with the same values as `input`, with an added dimension of
    size 1 at `axis`.

  #### Examples:
    ```python
    >>> rt = tf.ragged.constant([[1, 2], [3]])
    >>> print rt.shape
    TensorShape([2, None])

    >>> expanded = ragged.expand_dims(rt, axis=0)
    >>> print(expanded.shape, expanded)
    TensorShape([1, None, None]) [[[1, 2], [3]]]

    >>> expanded = ragged.expand_dims(rt, axis=1)
    >>> print(expanded.shape, expanded)
    TensorShape([2, None, None]) [[[1, 2]], [[3]]]

    >>> expanded = ragged.expand_dims(rt, axis=2)
    >>> print(expanded.shape, expanded)
    TensorShape([2, None, 1]) [[[1], [2]], [[3]]]
    ```
  """
  with ops.name_scope(name, 'RaggedExpandDims', [input]):
    input = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        input, name='input')

    if not ragged_tensor.is_ragged(input):
      return array_ops.expand_dims(input, axis)

    ndims = None if input.shape.ndims is None else input.shape.ndims + 1
    axis = ragged_util.get_positive_axis(axis, ndims)
    if axis == 0:
      values = input
      splits = array_ops.stack([0, input.nrows()])
    elif axis == 1:
      values = input
      splits = math_ops.range(input.nrows() + 1)
    else:
      values = expand_dims(input.values, axis - 1)
      splits = input.row_splits

    return ragged_tensor.RaggedTensor.from_row_splits(values, splits)


#===============================================================================
# ragged.where
#===============================================================================
def where(condition, x=None, y=None, name=None):
  """Return the elements, either from `x` or `y`, depending on the `condition`.

  : If both `x` and `y` are `None`:
    Returns the coordinates of true elements of `condition`. The coordinates
    are returned in a 2-D tensor with shape
    `[num_true_values, dim_size(condition)]`, where `result[i]` is the
    coordinates of the `i`th true value (in row-major order).

  : If both `x` and `y` are non-`None`:
    Returns a tensor formed by selecting values from `x` where condition is
    true, and from `y` when condition is false.  In particular:

    : If `condition`, `x`, and `y` all have the same shape:

      * `result[i1...iN] = x[i1...iN]` if `condition[i1...iN]` is true.
      * `result[i1...iN] = y[i1...iN]` if `condition[i1...iN]` is false.

    : Otherwise:

      * `condition` must be a vector.
      * `x` and `y` must have the same number of dimensions.
      * The outermost dimensions of `condition`, `x`, and `y` must all have the
        same size.
      * `result[i] = x[i]` if `condition[i]` is true.
      * `result[i] = y[i]` if `condition[i]` is false.

  Args:
    condition: A potentially ragged tensor of type `bool`
    x: A potentially ragged tensor (optional).
    y: A potentially ragged tensor (optional).  Must be specified if `x` is
      specified.  Must have the same rank and type as `x`.
    name: A name of the operation (optional)

  Returns:
    : If both `x` and `y` are `None`:
      A `Tensor` with shape `(num_true, dim_size(condition))`.
    : Otherwise:
      A potentially ragged tensor with the same type, rank, and outermost
      dimension size as `x` and `y`.
      `result.ragged_rank = max(x.ragged_rank, y.ragged_rank)`.

  Raises:
    ValueError: When exactly one of `x` or `y` is non-`None`; or when
      `condition`, `x`, and `y` have incompatible shapes.

  #### Examples:
    ```python
    >>> # Coordinates where condition is true.
    >>> condition = tf.ragged.constant_value(
    ...     [[True, False, True], [False, True]])
    >>> ragged.where(condition)
    [[0, 0], [0, 2], [1, 1]]

    >>> # Elementwise selection between x and y, based on condition.
    >>> condition = tf.ragged.constant_value(
    ...     [[True, False, True], [False, True]])
    >>> x = tf.ragged.constant_value([['A', 'B', 'C'], ['D', 'E']])
    >>> y = tf.ragged.constant_value([['a', 'b', 'c'], ['d', 'e']])
    >>> ragged.where(condition, x, y)
    [['A', 'b', 'C'], ['d', 'E']]

    >>> # Row selection between x and y, based on condition.
    >>> condition = [True, False]
    >>> x = tf.ragged.constant_value([['A', 'B', 'C'], ['D', 'E']])
    >>> y = tf.ragged.constant_value([['a', 'b', 'c'], ['d', 'e']])
    >>> ragged.where(condition, x, y)
    [['A', 'B', 'C'], ['d', 'e']]
    ```
  """
  if (x is None) != (y is None):
    raise ValueError('x and y must be either both None or both non-None')
  with ops.name_scope('RaggedWhere', name, [condition, x, y]):
    condition = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        condition, name='condition')
    if x is None:
      return _coordinate_where(condition)
    else:
      x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x, name='x')
      y = ragged_tensor.convert_to_tensor_or_ragged_tensor(y, name='y')
      return _elementwise_where(condition, x, y)


def _elementwise_where(condition, x, y):
  """Ragged version of tf.where(condition, x, y)."""
  condition_is_ragged = isinstance(condition, ragged_tensor.RaggedTensor)
  x_is_ragged = isinstance(x, ragged_tensor.RaggedTensor)
  y_is_ragged = isinstance(y, ragged_tensor.RaggedTensor)

  if not (condition_is_ragged or x_is_ragged or y_is_ragged):
    return array_ops.where(condition, x, y)

  elif condition_is_ragged and x_is_ragged and y_is_ragged:
    return ragged_functional_ops.map_flat_values(array_ops.where, condition, x,
                                                 y)
  elif not condition_is_ragged:
    # Concatenate x and y, and then use `gather` to assemble the selected rows.
    condition.shape.assert_has_rank(1)
    x_nrows = _nrows(x)
    x_and_y = concat([x, y], axis=0)
    indices = array_ops.where(condition, math_ops.range(x_nrows),
                              x_nrows + math_ops.range(_nrows(y)))
    return gather(x_and_y, indices)

  else:
    raise ValueError('Input shapes do not match.')


def _coordinate_where(condition):
  """Ragged version of tf.where(condition)."""
  if not isinstance(condition, ragged_tensor.RaggedTensor):
    return array_ops.where(condition)

  # The coordinate for each `true` value in condition.values.
  selected_coords = _coordinate_where(condition.values)

  # Convert the first index in each coordinate to a row index and column index.
  first_index = selected_coords[:, 0]
  selected_rows = array_ops.gather(condition.value_rowids(), first_index)
  selected_row_starts = array_ops.gather(condition.row_splits, selected_rows)
  selected_cols = first_index - selected_row_starts

  # Assemble the row & column index with the indices for inner dimensions.
  return array_ops.concat([
      array_ops.expand_dims(selected_rows, 1),
      array_ops.expand_dims(selected_cols, 1), selected_coords[:, 1:]
  ],
                          axis=1)


#===============================================================================
# RaggedTensor Size
#===============================================================================


def size(input, out_type=dtypes.int32, name=None):  # pylint: disable=redefined-builtin
  """Returns the size of a potentially ragged tensor.

  The size of a ragged tensor is the size of its inner values.

  Args:
    input: A potentially ragged `Tensor`.
    out_type: The numeric output type for the operation.
    name: A name for the operation (optional).

  Returns:
    A Tensor of type `out_type`.

  #### Example:
    ```python
    >>> tf.size(tf.ragged.constant([[1, 2], [3]]))
    3
    ```
  """
  if ragged_tensor.is_ragged(input):
    return array_ops.size(input.flat_values, out_type=out_type, name=name)
  else:
    return array_ops.size(input, out_type=out_type, name=name)


#===============================================================================
# Internal Helper Functions
#===============================================================================


def _increase_ragged_rank_to(rt_input, ragged_rank):
  """Adds ragged dimensions to `rt_input` so it has the desired ragged rank."""
  if ragged_rank > 0:
    if not ragged_tensor.is_ragged(rt_input):
      rt_input = ragged_conversion_ops.from_tensor(rt_input)
    if rt_input.ragged_rank < ragged_rank:
      rt_input = rt_input.with_values(
          _increase_ragged_rank_to(rt_input.values, ragged_rank - 1))
  return rt_input


def _concat_ragged_splits(splits_list):
  """Concatenates a list of RaggedTensor splits to form a single splits."""
  pieces = [splits_list[0]]
  splits_offset = splits_list[0][-1]
  for splits in splits_list[1:]:
    pieces.append(splits[1:] + splits_offset)
    splits_offset += splits[-1]
  return array_ops.concat(pieces, axis=0)


def _nrows(rt_input, out_type=dtypes.int64, name=None):
  if isinstance(rt_input, ragged_tensor.RaggedTensor):
    return rt_input.nrows(out_type=out_type, name=name)
  else:
    with ops.name_scope(name, 'RaggedNRows', [rt_input]):
      return array_ops.shape(rt_input, out_type=out_type)[0]


#===============================================================================
# ragged.rank
#===============================================================================
def rank(input, name=None):  # pylint: disable=redefined-builtin
  """Returns the rank of a RaggedTensor.

  Returns a 0-D `int32` `Tensor` representing the rank of `input`.

  For example:

  ```python
  # shape of tensor 't' is [2, None, None]
  t = tf.ragged.constant([[[1], [2, 2]], [[3, 3, 3], [4, 4, 4, 4]]])
  tf.rank(t)  # 3
  ```

  Args:
    input: A `RaggedTensor`
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  with ops.name_scope(name, 'RaggedRank', [input]) as name:
    if not ragged_tensor.is_ragged(input):
      return array_ops.rank(input, name)

    return input.ragged_rank + array_ops.rank(input.flat_values)
