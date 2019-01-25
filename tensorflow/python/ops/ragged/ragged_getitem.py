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
"""Python-style indexing and slicing for RaggedTensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor


def ragged_tensor_getitem(self, key):
  """Returns the specified piece of this RaggedTensor.

  Supports multidimensional indexing and slicing, with one restriction:
  indexing into a ragged inner dimension is not allowed.  This case is
  problematic because the indicated value may exist in some rows but not
  others.  In such cases, it's not obvious whether we should (1) report an
  IndexError; (2) use a default value; or (3) skip that value and return a
  tensor with fewer rows than we started with.  Following the guiding
  principles of Python ("In the face of ambiguity, refuse the temptation to
  guess"), we simply disallow this operation.

  Any dimensions added by `array_ops.newaxis` will be ragged if the following
  dimension is ragged.

  Args:
    self: The RaggedTensor to slice.
    key: Indicates which piece of the RaggedTensor to return, using standard
      Python semantics (e.g., negative values index from the end).  `key`
      may have any of the following types:

      * `int` constant
      * Scalar integer `Tensor`
      * `slice` containing integer constants and/or scalar integer
        `Tensor`s
      * `Ellipsis`
      * `tf.newaxis`
      * `tuple` containing any of the above (for multidimentional indexing)

  Returns:
    A `Tensor` or `RaggedTensor` object.  Values that include at least one
    ragged dimension are returned as `RaggedTensor`.  Values that include no
    ragged dimensions are returned as `Tensor`.  See above for examples of
    expressions that return `Tensor`s vs `RaggedTensor`s.

  Raises:
    ValueError: If `key` is out of bounds.
    ValueError: If `key` is not supported.
    TypeError: If the indices in `key` have an unsupported type.

  Examples:

    ```python
    >>> # A 2-D ragged tensor with 1 ragged dimension.
    >>> rt = ragged.constant([['a', 'b', 'c'], ['d', 'e'], ['f'], ['g']])
    >>> rt[0].eval().tolist()       # First row (1-D `Tensor`)
    ['a', 'b', 'c']
    >>> rt[:3].eval().tolist()      # First three rows (2-D RaggedTensor)
    [['a', 'b', 'c'], ['d', 'e'], '[f'], [g']]
    >>> rt[3, 0].eval().tolist()    # 1st element of 4th row (scalar)
    'g'

    >>> # A 3-D ragged tensor with 2 ragged dimensions.
    >>> rt = ragged.constant([[[1, 2, 3], [4]],
    ...                    [[5], [], [6]],
    ...                    [[7]],
    ...                    [[8, 9], [10]]])
    >>> rt[1].eval().tolist()       # Second row (2-D RaggedTensor)
    [[5], [], [6]]
    >>> rt[3, 0].eval().tolist()    # First element of fourth row (1-D Tensor)
    [8, 9]
    >>> rt[:, 1:3].eval().tolist()  # Items 1-3 of each row (3-D RaggedTensor)
    [[[4]], [[], [6]], [], [[10]]]
    >>> rt[:, -1:].eval().tolist()  # Last item of each row (3-D RaggedTensor)
    [[[4]], [[6]], [[7]], [[10]]]
    ```
  """
  scope_tensors = [self] + list(_tensors_in_key_list(key))
  if isinstance(key, (list, tuple)):
    key = list(key)
  else:
    key = [key]
  with ops.name_scope(None, "RaggedGetItem", scope_tensors):
    return _ragged_getitem(self, key)


def _ragged_getitem(rt_input, key_list):
  """Helper for indexing and slicing ragged tensors with __getitem__().

  Extracts the specified piece of the `rt_input`.  See
  `RaggedTensor.__getitem__` for examples and restrictions.

  Args:
    rt_input: The `RaggedTensor` from which a piece should be returned.
    key_list: The list of keys specifying which piece to return. Each key
      corresponds with a separate dimension.

  Returns:
    The indicated piece of rt_input.

  Raises:
    ValueError: If `key_list` is not supported.
    TypeError: If any keys in `key_list` have an unsupported type.
  """
  if not key_list:
    return rt_input
  row_key = key_list[0]
  inner_keys = key_list[1:]

  if row_key is Ellipsis:
    expanded_key_list = _expand_ellipsis(key_list, rt_input.shape.ndims)
    return _ragged_getitem(rt_input, expanded_key_list)

  # Adding a new axis: Get rt_input[inner_keys], and wrap it in a RaggedTensor
  # that puts all values in a single row.
  if row_key is array_ops.newaxis:
    inner_rt = _ragged_getitem(rt_input, inner_keys)
    nsplits = array_ops.shape(inner_rt.row_splits, out_type=dtypes.int64)[0]
    return ragged_tensor.RaggedTensor.from_row_splits(
        inner_rt, array_ops.stack([0, nsplits - 1]))

  # Slicing a range of rows: first slice the outer dimension, and then
  # call `_ragged_getitem_inner_dimensions` to handle the inner keys.
  if isinstance(row_key, slice):
    sliced_rt_input = _slice_ragged_row_dimension(rt_input, row_key)
    return _ragged_getitem_inner_dimensions(sliced_rt_input, inner_keys)

  # Indexing a single row: slice values to get the indicated row, and then
  # use a recursive call to __getitem__ to handle the inner keys.
  else:
    starts = rt_input.row_splits[:-1]
    limits = rt_input.row_splits[1:]
    row = rt_input.values[starts[row_key]:limits[row_key]]
    return row.__getitem__(inner_keys)


def _slice_ragged_row_dimension(rt_input, row_key):
  """Slice the outer dimension of `rt_input` according to the given `slice`.

  Args:
    rt_input: The `RaggedTensor` to slice.
    row_key: The `slice` object that should be used to slice `rt_input`.

  Returns:
    A `RaggedTensor` containing the indicated slice of `rt_input`.
  """
  if row_key.start is None and row_key.stop is None and row_key.step is None:
    return rt_input

  # Use row_key to slice the starts & limits.
  new_starts = rt_input.row_splits[:-1][row_key]
  new_limits = rt_input.row_splits[1:][row_key]
  zero_pad = array_ops.zeros([1], dtypes.int64)

  # If there's no slice step, then we can just select a single continuous
  # span of `ragged.values(rt_input)`.
  if row_key.step is None or row_key.step == 1:
    # Construct the new splits.  If new_starts and new_limits are empty,
    # then this reduces to [0].  Otherwise, this reduces to:
    #   concat([[new_starts[0]], new_limits])
    new_splits = array_ops.concat(
        [zero_pad[array_ops.size(new_starts):], new_starts[:1], new_limits],
        axis=0)
    values_start = new_splits[0]
    values_limit = new_splits[-1]
    return ragged_tensor.RaggedTensor.from_row_splits(
        rt_input.values[values_start:values_limit], new_splits - values_start)

  # If there is a slice step (aka a strided slice), then use ragged_gather to
  # collect the necessary elements of `ragged.values(rt_input)`.
  else:
    return _build_ragged_tensor_from_value_ranges(new_starts, new_limits, 1,
                                                  rt_input.values)


def _ragged_getitem_inner_dimensions(rt_input, key_list):
  """Retrieve inner dimensions, keeping outermost dimension unchanged.

  Args:
    rt_input: The `RaggedTensor` or `Tensor` from which a piece should be
      extracted.
    key_list: The __getitem__ keys for slicing the inner dimensions.

  Returns:
    A `RaggedTensor`.

  Raises:
    ValueError: If key_list is not supported.
  """
  if not key_list:
    return rt_input

  if isinstance(rt_input, ops.Tensor):
    return rt_input.__getitem__([slice(None, None, None)] + key_list)

  column_key = key_list[0]
  if column_key is Ellipsis:
    expanded_key_list = _expand_ellipsis(key_list, rt_input.values.shape.ndims)
    return _ragged_getitem_inner_dimensions(rt_input, expanded_key_list)

  # Adding a new axis to a ragged inner dimension: recursively get the inner
  # dimensions of rt_input with key_list[1:], and then wrap the result in a
  # RaggedTensor that puts each value in its own row.
  if column_key is array_ops.newaxis:
    inner_rt = _ragged_getitem_inner_dimensions(rt_input, key_list[1:])
    nsplits = array_ops.shape(inner_rt.row_splits, out_type=dtypes.int64)[0]
    return ragged_tensor.RaggedTensor.from_row_splits(inner_rt,
                                                      math_ops.range(nsplits))

  # Slicing a range of columns in a ragged inner dimension.  We use a
  # recursive call to process the values, and then assemble a RaggedTensor
  # with those values.
  if isinstance(column_key, slice):
    if (column_key.start is None and column_key.stop is None and
        column_key.step is None):
      # Trivial slice: recursively process all values, & splits is unchanged.
      return rt_input.with_values(
          _ragged_getitem_inner_dimensions(rt_input.values, key_list[1:]))
    else:
      # Nontrivial slice: use ragged_gather to extract the indicated slice as
      # a new RaggedTensor (inner_rt), and then recursively process its values.
      # The splits can be taken from inner_rt.row_splits().
      inner_rt_starts = rt_input.row_splits[:-1]
      inner_rt_limits = rt_input.row_splits[1:]
      if column_key.start is not None and column_key.start != 0:
        inner_rt_starts = _add_offset_to_ranges(
            column_key.start, rt_input.row_splits[:-1], rt_input.row_splits[1:])
      if column_key.stop is not None and column_key.stop != 0:
        inner_rt_limits = _add_offset_to_ranges(
            column_key.stop, rt_input.row_splits[:-1], rt_input.row_splits[1:])
      inner_rt = _build_ragged_tensor_from_value_ranges(
          inner_rt_starts, inner_rt_limits, column_key.step, rt_input.values)
      return inner_rt.with_values(
          _ragged_getitem_inner_dimensions(inner_rt.values, key_list[1:]))

  # Indexing a single column in a ragged inner dimension: raise an Exception.
  # See RaggedTensor.__getitem__.__doc__ for an explanation of why indexing
  # into a ragged inner dimension is problematic.
  else:
    raise ValueError("Cannot index into an inner ragged dimension.")


def _expand_ellipsis(key_list, num_remaining_dims):
  """Expands the ellipsis at the start of `key_list`.

  Assumes that the first element of `key_list` is Ellipsis.  This will either
  remove the Ellipsis (if it corresponds to zero indices) or prepend a new
  `slice(None, None, None)` (if it corresponds to more than zero indices).

  Args:
    key_list: The arguments to `__getitem__()`.
    num_remaining_dims: The number of dimensions remaining.

  Returns:
    A copy of `key_list` with he ellipsis expanded.
  Raises:
    ValueError: If ragged_rank.shape.ndims is None
    IndexError: If there are too many elements in `key_list`.
  """
  if num_remaining_dims is None:
    raise ValueError("Ellipsis not supported for unknown shape RaggedTensors")
  num_indices = sum(1 for idx in key_list if idx is not array_ops.newaxis)
  if num_indices > num_remaining_dims + 1:
    raise IndexError("Too many indices for RaggedTensor")
  elif num_indices == num_remaining_dims + 1:
    return key_list[1:]
  else:
    return [slice(None, None, None)] + key_list


def _tensors_in_key_list(key_list):
  """Generates all Tensors in the given slice spec."""
  if isinstance(key_list, ops.Tensor):
    yield key_list
  if isinstance(key_list, (list, tuple)):
    for v in key_list:
      for tensor in _tensors_in_key_list(v):
        yield tensor
  if isinstance(key_list, slice):
    for tensor in _tensors_in_key_list(key_list.start):
      yield tensor
    for tensor in _tensors_in_key_list(key_list.stop):
      yield tensor
    for tensor in _tensors_in_key_list(key_list.step):
      yield tensor


def _build_ragged_tensor_from_value_ranges(starts, limits, step, values):
  """Returns a `RaggedTensor` containing the specified sequences of values.

  Returns a RaggedTensor `output` where:

  ```python
  output.shape[0] = starts.shape[0]
  output[i] = values[starts[i]:limits[i]:step]
  ```

  Requires that `starts.shape == limits.shape` and
  `0 <= starts[i] <= limits[i] <= values.shape[0]`.

  Args:
    starts: 1D integer Tensor specifying the start indices for the sequences of
      values to include.
    limits: 1D integer Tensor specifying the limit indices for the sequences of
      values to include.
    step: Integer value specifying the step size for strided slices.
    values: The set of values to select from.

  Returns:
    A `RaggedTensor`.

  Raises:
    ValueError: Until the prerequisite ops are checked in.
  """
  # Use `ragged_range` to get the index of each value we should include.
  if step is None:
    step = 1
  step = ops.convert_to_tensor(step, name="step")
  if step.dtype.is_integer:
    step = math_ops.cast(step, dtypes.int64)
  else:
    raise TypeError("slice strides must be integers or None")
  value_indices = ragged_math_ops.range(starts, limits, step)

  # Use `ragged_gather` or `array_ops.gather` to collect the values.
  if isinstance(values, ragged_tensor.RaggedTensor):
    gathered_values = ragged_gather_ops.gather(
        params=values, indices=value_indices.values)
  else:
    gathered_values = array_ops.gather(
        params=values, indices=value_indices.values)

  # Assemble the RaggedTensor from splits & values.
  return value_indices.with_values(gathered_values)


def _add_offset_to_ranges(offset, starts, limits):
  """Adds an indexing offset to each of the specified ranges.

  If offset>=0, then return output[i]=min(starts[i]+offset, limits[i])
  If offset<0, then return output[i]=max(limits[i]+offset, starts[i])

  Args:
    offset: The offset to add.  None, or an int, or a scalar Tensor.
    starts: 1-D int64 tensor containing start indices.
    limits: 1-D int64 tensor containing limit indices.

  Returns:
    A 1-D int64 tensor.
  """

  def map_positive_offset(offset):
    return math_ops.minimum(starts + offset, limits)

  def map_negative_offset(offset):
    return math_ops.maximum(limits + offset, starts)

  if isinstance(offset, ops.Tensor):
    offset = math_ops.cast(offset, dtypes.int64)
    return control_flow_ops.cond(offset >= 0,
                                 lambda: map_positive_offset(offset),
                                 lambda: map_negative_offset(offset))
  elif isinstance(offset, int):
    return (map_positive_offset(offset)
            if offset > 0 else map_negative_offset(offset))

  else:
    raise TypeError("slice offsets must be integers or None")
