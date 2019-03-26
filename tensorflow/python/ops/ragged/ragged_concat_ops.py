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
"""Concat and stack operations for RaggedTensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_conversion_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util


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
      return ragged_array_ops.expand_dims(rt_inputs[0], axis=axis)
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
    stack_lengths = array_ops.stack([rt.nrows() for rt in rt_inputs])
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

  rt_nrows = rt_inputs[0].nrows()
  nrows_msg = 'Input tensors have incompatible shapes.'
  nrows_checks = [
      check_ops.assert_equal(rt.nrows(), rt_nrows, message=nrows_msg)
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
    permuted_rt = ragged_gather_ops.gather(concatenated_rt, row_permutation)

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
