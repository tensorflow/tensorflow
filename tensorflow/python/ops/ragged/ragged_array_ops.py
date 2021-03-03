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
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_ragged_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

#===============================================================================
# Masking
#===============================================================================


@tf_export('ragged.boolean_mask')
@dispatch.add_dispatch_support
def boolean_mask(data, mask, name=None):
  """Applies a boolean mask to `data` without flattening the mask dimensions.

  Returns a potentially ragged tensor that is formed by retaining the elements
  in `data` where the corresponding value in `mask` is `True`.

  * `output[a1...aA, i, b1...bB] = data[a1...aA, j, b1...bB]`

     Where `j` is the `i`th `True` entry of `mask[a1...aA]`.

  Note that `output` preserves the mask dimensions `a1...aA`; this differs
  from `tf.boolean_mask`, which flattens those dimensions.

  Args:
    data: A potentially ragged tensor.
    mask: A potentially ragged boolean tensor.  `mask`'s shape must be a prefix
      of `data`'s shape.  `rank(mask)` must be known statically.
    name: A name prefix for the returned tensor (optional).

  Returns:
    A potentially ragged tensor that is formed by retaining the elements in
    `data` where the corresponding value in `mask` is `True`.

    * `rank(output) = rank(data)`.
    * `output.ragged_rank = max(data.ragged_rank, rank(mask) - 1)`.

  Raises:
    ValueError: if `rank(mask)` is not known statically; or if `mask.shape` is
      not a prefix of `data.shape`.

  #### Examples:

  >>> # Aliases for True & False so data and mask line up.
  >>> T, F = (True, False)

  >>> tf.ragged.boolean_mask(  # Mask a 2D Tensor.
  ...     data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
  ...     mask=[[T, F, T], [F, F, F], [T, F, F]]).to_list()
  [[1, 3], [], [7]]

  >>> tf.ragged.boolean_mask(  # Mask a 2D RaggedTensor.
  ...     tf.ragged.constant([[1, 2, 3], [4], [5, 6]]),
  ...     tf.ragged.constant([[F, F, T], [F], [T, T]])).to_list()
  [[3], [], [5, 6]]

  >>> tf.ragged.boolean_mask(  # Mask rows of a 2D RaggedTensor.
  ...     tf.ragged.constant([[1, 2, 3], [4], [5, 6]]),
  ...     tf.ragged.constant([True, False, True])).to_list()
  [[1, 2, 3], [5, 6]]
  """
  with ops.name_scope(name, 'RaggedMask', [data, mask]):
    # Convert inputs to tensors.
    data = ragged_tensor.convert_to_tensor_or_ragged_tensor(data, name='data')
    mask = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        mask, dtypes.bool, name='mask')
    row_splits_dtype, (data, mask) = ragged_tensor.match_row_splits_dtypes(
        data, mask, return_dtype=True)

    # Get static rank of mask.
    if mask.shape.ndims is None:
      raise ValueError('mask.shape.ndims must be known statically.')
    elif mask.shape.ndims == 0:
      raise ValueError('mask cannot be scalar.')

    # If mask is ragged, then recurse with a non-ragged mask.
    if ragged_tensor.is_ragged(mask):
      if not ragged_tensor.is_ragged(data):
        data = ragged_tensor.RaggedTensor.from_tensor(
            data,
            ragged_rank=mask.ragged_rank,
            row_splits_dtype=mask.row_splits.dtype)
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
                math_ops.cast, mask, dtype=row_splits_dtype)
            masked_row_lengths = ragged_math_ops.reduce_sum(int_mask, axis=1)
            splits.append(ragged_util.lengths_to_splits(masked_row_lengths))
          mask = mask.values
          data = data.values

        # Recursively apply the nested non-ragged mask to the nested data.
        masked_values = boolean_mask(data, mask)

        # Add the ragged `splits` back to the result.
        masked_values = ragged_tensor.RaggedTensor.from_nested_row_splits(
            masked_values, splits, validate=False)

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
      masked_values = boolean_mask(data.values, segment_mask)

      return ragged_tensor.RaggedTensor.from_row_splits(
          masked_values, masked_splits, validate=False)

    # If mask is non-ragged and has rank>1, then convert it to be ragged,
    # with a ragged rank matching data.
    if ragged_tensor.is_ragged(data):
      mask = ragged_tensor.RaggedTensor.from_tensor(
          mask,
          ragged_rank=min(data.ragged_rank, mask.shape.ndims - 1),
          row_splits_dtype=data.row_splits.dtype)
      return boolean_mask(data, mask)

    # Otherwise, data and mask are both `Tensor`s.
    else:
      # Apply `boolean_mask` to get the masked values.
      masked_values = array_ops.boolean_mask(data, mask)

      if mask.shape.ndims >= 2:
        # Add the innermost ragged dimension.  For each innermost cell, get the
        # number of values it contains.  Then flatten that to get a list of
        # cell lengths, and convert it to splits.  Finally, combine the splits
        # and values to get the innermost ragged tensor.
        masked_lengths = math_ops.count_nonzero(
            mask, axis=-1, dtype=row_splits_dtype)
        flattened_masked_lengths = array_ops.reshape(masked_lengths, [-1])
        masked_values = ragged_tensor.RaggedTensor.from_row_lengths(
            masked_values, flattened_masked_lengths, validate=False)

        # Wrap remaining ragged dimensions.
        if mask.shape.ndims > 2:
          mask_shape = array_ops.shape(mask, out_type=row_splits_dtype)
          split_size = math_ops.cumprod(mask_shape) + 1
          for dim in range(mask.shape.ndims - 3, -1, -1):
            elt_size = mask_shape[dim + 1]
            masked_splits = math_ops.range(split_size[dim]) * elt_size
            masked_values = ragged_tensor.RaggedTensor.from_row_splits(
                masked_values, masked_splits, validate=False)

      return masked_values


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

  >>> rt = tf.ragged.constant([[1, 2], [3]])
  >>> tf.tile(rt, [3, 2]).to_list()
  [[1, 2, 1, 2], [3, 3], [1, 2, 1, 2], [3, 3], [1, 2, 1, 2], [3, 3]]
  """
  with ops.name_scope(name, 'RaggedTile', [input, multiples]):
    input = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        input, name='input')
    if not ragged_tensor.is_ragged(input):
      return array_ops.tile(input, multiples, name)
    multiples = ragged_util.convert_to_int_tensor(
        multiples, name='multiples', dtype=input.row_splits.dtype)
    multiples.shape.assert_has_rank(1)

    # If the constant value of `multiples` is available, then we can use it
    # to skip tiling dimensions where `multiples=1`.
    const_multiples = tensor_util.constant_value(multiples)

    return ragged_tensor.RaggedTensor.from_nested_row_splits(
        _tile_ragged_values(input, multiples, const_multiples),
        _tile_ragged_splits(input, multiples, const_multiples),
        validate=False)


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

  >>> rt = tf.ragged.constant([[1, 2], [3]])
  >>> _tile_ragged_values(rt, tf.constant([3, 2])).numpy()
  array([1, 2, 1, 2, 3, 3, 1, 2, 1, 2, 3, 3, 1, 2, 1, 2, 3, 3], dtype=int32)
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
    A list of 1-D integer `Tensor`s (one for each ragged dimension in
    `rt_input`).

  #### Example:

  >>> rt = tf.ragged.constant([[1, 2], [3]])
  >>> _tile_ragged_splits(rt, [3, 2])
  [<tf.Tensor: shape=(7,), dtype=int64,
  numpy=array([ 0,  4,  6, 10, 12, 16, 18])>]
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
          nested_splits[dst_axis], projected_splits[src_axis][dst_axis - 1])

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

  The following table gives some examples showing how `ragged.expand_dims`
  impacts the shapes of different input tensors.  Ragged dimensions are
  indicated by enclosing them in parentheses.

  input.shape             | axis | result.shape
  ----------------------- | ---- | -----------------------------
  `[D1, D2]`              |  `0` | `[1, D1, D2]`
  `[D1, D2]`              |  `1` | `[D1, 1, D2]`
  `[D1, D2]`              |  `2` | `[D1, D2, 1]`
  `[D1, (D2), (D3), D4]`  |  `0` | `[1, D1, (D2), (D3), D4]`
  `[D1, (D2), (D3), D4]`  |  `1` | `[D1, 1, (D2), (D3), D4]`
  `[D1, (D2), (D3), D4]`  |  `2` | `[D1, (D2), 1, (D3), D4]`
  `[D1, (D2), (D3), D4]`  |  `3` | `[D1, (D2), (D3), 1, D4]`
  `[D1, (D2), (D3), D4]`  |  `4` | `[D1, (D2), (D3), D4, 1]`

  Args:
    input: The potentially tensor that should be expanded with a new dimension.
    axis: An integer constant indicating where the new dimension should be
      inserted.
    name: A name for the operation (optional).

  Returns:
    A tensor with the same values as `input`, with an added dimension of
    size 1 at `axis`.

  #### Examples:

  >>> rt = tf.ragged.constant([[1, 2], [3]])
  >>> print(rt.shape)
  (2, None)

  >>> expanded = tf.expand_dims(rt, axis=0)
  >>> print(expanded.shape, expanded)
  (1, 2, None) <tf.RaggedTensor [[[1, 2], [3]]]>

  >>> expanded = tf.expand_dims(rt, axis=1)
  >>> print(expanded.shape, expanded)
  (2, 1, None) <tf.RaggedTensor [[[1, 2]], [[3]]]>

  >>> expanded = tf.expand_dims(rt, axis=2)
  >>> print(expanded.shape, expanded)
  (2, None, 1) <tf.RaggedTensor [[[1], [2]], [[3]]]>
  """
  with ops.name_scope(name, 'RaggedExpandDims', [input]):
    input = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        input, name='input')

    if not ragged_tensor.is_ragged(input):
      return array_ops.expand_dims(input, axis)

    ndims = None if input.shape.ndims is None else input.shape.ndims + 1
    axis = array_ops.get_positive_axis(axis, ndims, ndims_name='rank(input)')

    if axis == 0:
      return ragged_tensor.RaggedTensor.from_uniform_row_length(
          input, uniform_row_length=input.nrows(), nrows=1, validate=False)
    elif axis == 1:
      return ragged_tensor.RaggedTensor.from_uniform_row_length(
          input, uniform_row_length=1, nrows=input.nrows(), validate=False)
    else:
      if ragged_tensor.is_ragged(input.values):
        return input.with_values(expand_dims(input.values, axis - 1))
      else:
        return input.with_values(array_ops.expand_dims(input.values, axis - 1))


#===============================================================================
# RaggedTensor Size
#===============================================================================


def size(input, out_type=dtypes.int32, name=None):  # pylint: disable=redefined-builtin
  """Returns the size of a potentially ragged tensor.

  The size of a ragged tensor is the size of its inner values.

  #### Example:

  >>> tf.size(tf.ragged.constant([[1, 2], [3]])).numpy()
  3

  Args:
    input: A potentially ragged `Tensor`.
    out_type: The numeric output type for the operation.
    name: A name for the operation (optional).

  Returns:
    A Tensor of type `out_type`.
  """
  if ragged_tensor.is_ragged(input):
    return array_ops.size(input.flat_values, out_type=out_type, name=name)
  else:
    return array_ops.size(input, out_type=out_type, name=name)


#===============================================================================
# ragged.rank
#===============================================================================
def rank(input, name=None):  # pylint: disable=redefined-builtin
  """Returns the rank of a RaggedTensor.

  Returns a 0-D `int32` `Tensor` representing the rank of `input`.

  #### Example:

  >>> # shape of tensor 't' is [2, None, None]
  >>> t = tf.ragged.constant([[[1], [2, 2]], [[3, 3, 3], [4, 4, 4, 4]]])
  >>> tf.rank(t).numpy()
  3

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


#===============================================================================
# ragged.one_hot
#===============================================================================
def ragged_one_hot(indices,
                   depth,
                   on_value=None,
                   off_value=None,
                   axis=None,
                   dtype=None,
                   name=None):
  """Applies tf.one_hot along the values of a RaggedTensor."""
  # Get the adjusted axis value for the call to array_ops.one_hot.
  # Note: the only negative `axis` value supported by array_ops.one_hot is -1.
  if isinstance(axis, int) and axis >= 0:
    if axis <= indices.ragged_rank:
      raise ValueError('axis (%d) must be greater than indices.ragged_rank '
                       '(%d).' % (axis, indices.ragged_rank))
    axis -= indices.ragged_rank

  with ops.name_scope(name, 'RaggedOneHot',
                      [indices, depth, on_value, off_value, axis]):
    indices = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        indices, name='indices')
    return indices.with_flat_values(
        array_ops.one_hot(indices.flat_values, depth, on_value, off_value, axis,
                          dtype, name))


#===============================================================================
# ragged.stack_dynamic_partitions
#===============================================================================
@tf_export('ragged.stack_dynamic_partitions')
@dispatch.add_dispatch_support
def stack_dynamic_partitions(data, partitions, num_partitions, name=None):
  """Stacks dynamic partitions of a Tensor or RaggedTensor.

  Returns a RaggedTensor `output` with `num_partitions` rows, where the row
  `output[i]` is formed by stacking all slices `data[j1...jN]` such that
  `partitions[j1...jN] = i`.  Slices of `data` are stacked in row-major
  order.

  If `num_partitions` is an `int` (not a `Tensor`), then this is equivalent to
  `tf.ragged.stack(tf.dynamic_partition(data, partitions, num_partitions))`.

  #### Example:

  >>> data           = ['a', 'b', 'c', 'd', 'e']
  >>> partitions     = [  3,   0,   2,   2,   3]
  >>> num_partitions = 5
  >>> tf.ragged.stack_dynamic_partitions(data, partitions, num_partitions)
  <tf.RaggedTensor [[b'b'], [], [b'c', b'd'], [b'a', b'e'], []]>

  Args:
    data: A `Tensor` or `RaggedTensor` containing the values to stack.
    partitions: An `int32` or `int64` `Tensor` or `RaggedTensor` specifying the
      partition that each slice of `data` should be added to. `partitions.shape`
      must be a prefix of `data.shape`.  Values must be greater than or equal to
      zero, and less than `num_partitions`. `partitions` is not required to be
      sorted.
    num_partitions: An `int32` or `int64` scalar specifying the number of
      partitions to output.  This determines the number of rows in `output`.
    name: A name prefix for the returned tensor (optional).

  Returns:
    A `RaggedTensor` containing the stacked partitions.  The returned tensor
    has the same dtype as `data`, and its shape is
    `[num_partitions, (D)] + data.shape[partitions.rank:]`, where `(D)` is a
    ragged dimension whose length is the number of data slices stacked for
    each `partition`.
  """
  with ops.name_scope(name, 'SegmentStack', [data, partitions, num_partitions]):
    # Convert inputs to tensors.
    data = ragged_tensor.convert_to_tensor_or_ragged_tensor(data, name='data')
    row_splits_dtype = (
        data.row_splits.dtype
        if isinstance(data, ragged_tensor.RaggedTensor) else None)
    partitions = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        partitions, name='partitions', preferred_dtype=row_splits_dtype)
    num_partitions = ops.convert_to_tensor(
        num_partitions, name='num_partitions', preferred_dtype=partitions.dtype)
    if row_splits_dtype is not None:
      partitions = math_ops.cast(partitions, row_splits_dtype)
    num_partitions = math_ops.cast(num_partitions, partitions.dtype)

    # Sanity-checks for shapes.
    partitions_rank = partitions.shape.ndims
    if partitions_rank is None:
      raise ValueError('partitions must have known rank.')
    num_partitions.shape.assert_has_rank(0)
    partitions.shape.assert_is_compatible_with(data.shape[:partitions_rank])

    if partitions_rank == 0:
      # If partitions is a scalar, then just create a RaggedTensor containing
      # that single the complete `data` value in the specified row.
      return ragged_tensor.RaggedTensor.from_value_rowids(
          values=array_ops.stack([data]),
          value_rowids=array_ops.stack([partitions]),
          nrows=num_partitions,
          validate=False)

    elif partitions_rank == 1:
      # If partitions is a vector (the typical case): we can just use data and
      # partitions as the `values` and `value_rowids` for `from_value_rowids`,
      # as long as we sort them first.
      permutation = sort_ops.argsort(partitions, stable=True)
      value_rowids = array_ops.gather(partitions, permutation)
      values = array_ops.gather(data, permutation)
      check = check_ops.assert_less(
          value_rowids[-1:],
          num_partitions,
          message='partitions must be less than num_partitions')
      with ops.control_dependencies([check]):
        return ragged_tensor.RaggedTensor.from_value_rowids(
            values, value_rowids, nrows=num_partitions, validate=False)

    else:
      # Handle higher-dimensional partitions via recursion.
      if not isinstance(data, ragged_tensor.RaggedTensor):
        data = ragged_tensor.RaggedTensor.from_tensor(
            data, row_splits_dtype=partitions.dtype, ragged_rank=1)
      if not isinstance(partitions, ragged_tensor.RaggedTensor):
        partitions = ragged_tensor.RaggedTensor.from_tensor(
            partitions,
            row_splits_dtype=partitions.dtype,
            ragged_rank=max(data.ragged_rank, partitions_rank - 1))
      check = check_ops.assert_equal(
          data.row_splits,
          partitions.row_splits,
          message='data and partitions have incompatible ragged shapes')
      with ops.control_dependencies([check]):
        return stack_dynamic_partitions(data.values, partitions.values,
                                        num_partitions)


#===============================================================================
# Reverse
#===============================================================================
def reverse(tensor, axis, name=None):
  """Reverses a RaggedTensor along the specified axes.

  #### Example:

  >>> data = tf.ragged.constant([
  ...   [[1, 2], [3, 4]], [[5, 6]], [[7, 8], [9, 10], [11, 12]]])
  >>> tf.reverse(data, axis=[0, 2])
  <tf.RaggedTensor [[[8, 7], [10, 9], [12, 11]], [[6, 5]], [[2, 1], [4, 3]]]>

  Args:
    tensor: A 'RaggedTensor' to reverse.
    axis: A list or tuple of 'int' or a constant 1D 'tf.Tensor'. The indices of
      the axes to reverse.
    name: A name prefix for the returned tensor (optional).

  Returns:
    A 'RaggedTensor'.
  """
  type_error_msg = ('`axis` must be a list of int or a constant tensor'
                    'when reversing axes in a ragged tensor')

  with ops.name_scope(name, 'Reverse', [tensor, axis]):
    if isinstance(axis, ops.Tensor):
      axis = tensor_util.constant_value(axis)
      if axis is None:
        raise TypeError(type_error_msg)
    elif not (isinstance(axis, (list, tuple)) and
              all(isinstance(dim, int) for dim in axis)):
      raise TypeError(type_error_msg)

    tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        tensor, name='tensor')

    # Allow usage of negative values to specify innermost axes.
    axis = [
        array_ops.get_positive_axis(dim, tensor.shape.rank, 'axis[%d]' % i,
                                    'rank(tensor)')
        for i, dim in enumerate(axis)
    ]

    # We only need to slice up to the max axis. If the axis list
    # is empty, it should be 0.
    slices = [slice(None)] * (max(axis) + 1 if axis else 0)

    for dim in axis:
      slices[dim] = slice(None, None, -1)

    return tensor[tuple(slices)]


#===============================================================================
# Cross
#===============================================================================


@tf_export('ragged.cross')
@dispatch.add_dispatch_support
def cross(inputs, name=None):
  """Generates feature cross from a list of tensors.

  The input tensors must have `rank=2`, and must all have the same number of
  rows.  The result is a `RaggedTensor` with the same number of rows as the
  inputs, where `result[row]` contains a list of all combinations of values
  formed by taking a single value from each input's corresponding row
  (`inputs[i][row]`).  Values are combined by joining their strings with '_X_'.
  E.g.:

  >>> tf.ragged.cross([tf.ragged.constant([['a'], ['b', 'c']]),
  ...                  tf.ragged.constant([['d'], ['e']]),
  ...                  tf.ragged.constant([['f'], ['g']])])
  <tf.RaggedTensor [[b'a_X_d_X_f'], [b'b_X_e_X_g', b'c_X_e_X_g']]>

  Args:
    inputs: A list of `RaggedTensor` or `Tensor` or `SparseTensor`.
    name: Optional name for the op.

  Returns:
    A 2D `RaggedTensor` of type `string`.
  """
  return _cross_internal(inputs=inputs, hashed_output=False, name=name)


@tf_export('ragged.cross_hashed')
@dispatch.add_dispatch_support
def cross_hashed(inputs, num_buckets=0, hash_key=None, name=None):
  """Generates hashed feature cross from a list of tensors.

  The input tensors must have `rank=2`, and must all have the same number of
  rows.  The result is a `RaggedTensor` with the same number of rows as the
  inputs, where `result[row]` contains a list of all combinations of values
  formed by taking a single value from each input's corresponding row
  (`inputs[i][row]`).  Values are combined by hashing together their
  fingerprints. E.g.:

  >>> tf.ragged.cross_hashed([tf.ragged.constant([['a'], ['b', 'c']]),
  ...                         tf.ragged.constant([['d'], ['e']]),
  ...                         tf.ragged.constant([['f'], ['g']])],
  ...                        num_buckets=100)
  <tf.RaggedTensor [[78], [66, 74]]>

  Args:
    inputs: A list of `RaggedTensor` or `Tensor` or `SparseTensor`.
    num_buckets: A non-negative `int` that used to bucket the hashed values. If
      `num_buckets != 0`, then `output = hashed_value % num_buckets`.
    hash_key: Integer hash_key that will be used by the `FingerprintCat64`
      function. If not given, a default key is used.
    name: Optional name for the op.

  Returns:
    A 2D `RaggedTensor` of type `int64`.
  """
  return _cross_internal(
      inputs=inputs,
      hashed_output=True,
      num_buckets=num_buckets,
      hash_key=hash_key,
      name=name)


_DEFAULT_CROSS_HASH_KEY = 0xDECAFCAFFE


def _cross_internal(inputs,
                    hashed_output=False,
                    num_buckets=0,
                    hash_key=None,
                    name=None):
  """Generates feature cross from a list of ragged and dense tensors."""
  if not isinstance(inputs, (tuple, list)):
    raise TypeError('Inputs must be a list')

  if hash_key is None:
    hash_key = _DEFAULT_CROSS_HASH_KEY

  ragged_inputs = []
  sparse_inputs = []
  dense_inputs = []
  input_order = []
  with ops.name_scope(name, 'RaggedCross', inputs):
    for i, t in enumerate(inputs):
      if sparse_tensor.is_sparse(t):
        t = sparse_tensor.SparseTensor.from_value(t)
      else:
        t = ragged_tensor.convert_to_tensor_or_ragged_tensor(t)
      if t.dtype.is_integer:
        t = math_ops.cast(t, dtypes.int64)
      elif t.dtype != dtypes.string:
        raise ValueError('Unexpected dtype for inputs[%d]: %s' % (i, t.dtype))
      if isinstance(t, ragged_tensor.RaggedTensor):
        if t.ragged_rank != 1:
          raise ValueError('tf.ragged.cross only supports inputs with rank=2')
        ragged_inputs.append(t)
        input_order.append('R')
      elif isinstance(t, sparse_tensor.SparseTensor):
        sparse_inputs.append(t)
        input_order.append('S')
      else:
        dense_inputs.append(t)
        input_order.append('D')

    out_values_type = dtypes.int64 if hashed_output else dtypes.string
    if ragged_inputs and all(
        t.row_splits.dtype == dtypes.int32 for t in ragged_inputs):
      out_row_splits_type = dtypes.int32
    else:
      out_row_splits_type = dtypes.int64

    # Convert hash_key from uint64 -> int64, since we need to pass it via
    # an int64 attr.
    if hash_key > 2**63:
      hash_key -= 2**64

    values_out, splits_out = gen_ragged_array_ops.ragged_cross(
        ragged_values=[rt.values for rt in ragged_inputs],
        ragged_row_splits=[rt.row_splits for rt in ragged_inputs],
        sparse_indices=[st.indices for st in sparse_inputs],
        sparse_values=[st.values for st in sparse_inputs],
        sparse_shape=[st.dense_shape for st in sparse_inputs],
        dense_inputs=dense_inputs,
        input_order=''.join(input_order),
        hashed_output=hashed_output,
        num_buckets=num_buckets,
        hash_key=hash_key,
        out_values_type=out_values_type.as_datatype_enum,
        out_row_splits_type=out_row_splits_type.as_datatype_enum,
        name=name)

    return ragged_tensor.RaggedTensor.from_row_splits(
        values_out, splits_out, validate=False)
