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
"""Private convenience functions for RaggedTensors.

None of these methods are exposed in the main "ragged" package.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import math_ops


def convert_to_int_tensor(tensor, name, dtype=dtypes.int32):
  """Converts the given value to an integer Tensor."""
  tensor = ops.convert_to_tensor(tensor, name=name, preferred_dtype=dtype)
  if tensor.dtype.is_integer:
    tensor = math_ops.cast(tensor, dtype)
  else:
    raise TypeError(
        "%s must be an integer tensor; dtype=%s" % (name, tensor.dtype))
  return tensor


def get_positive_axis(axis, ndims):
  """Validate an `axis` parameter, and normalize it to be positive.

  If `ndims` is known (i.e., not `None`), then check that `axis` is in the
  range `-ndims <= axis < ndims`, and return `axis` (if `axis >= 0`) or
  `axis + ndims` (otherwise).
  If `ndims` is not known, and `axis` is positive, then return it as-is.
  If `ndims` is not known, and `axis` is negative, then report an error.

  Args:
    axis: An integer constant
    ndims: An integer constant, or `None`

  Returns:
    The normalized `axis` value.

  Raises:
    ValueError: If `axis` is out-of-bounds, or if `axis` is negative and
      `ndims is None`.
  """
  if not isinstance(axis, int):
    raise TypeError("axis must be an int; got %s" % type(axis).__name__)
  if ndims is not None:
    if 0 <= axis < ndims:
      return axis
    elif -ndims <= axis < 0:
      return axis + ndims
    else:
      raise ValueError(
          "axis=%s out of bounds: expected %s<=axis<%s" % (axis, -ndims, ndims))
  elif axis < 0:
    raise ValueError("axis may only be negative if ndims is statically known.")
  return axis


def assert_splits_match(nested_splits_lists):
  """Checks that the given splits lists are identical.

  Performs static tests to ensure that the given splits lists are identical,
  and returns a list of control dependency op tensors that check that they are
  fully identical.

  Args:
    nested_splits_lists: A list of nested_splits_lists, where each split_list is
      a list of `splits` tensors from a `RaggedTensor`, ordered from outermost
      ragged dimension to innermost ragged dimension.

  Returns:
    A list of control dependency op tensors.
  Raises:
    ValueError: If the splits are not identical.
  """
  error_msg = "Inputs must have identical ragged splits"
  for splits_list in nested_splits_lists:
    if len(splits_list) != len(nested_splits_lists[0]):
      raise ValueError(error_msg)
  return [
      check_ops.assert_equal(s1, s2, message=error_msg)
      for splits_list in nested_splits_lists[1:]
      for (s1, s2) in zip(nested_splits_lists[0], splits_list)
  ]


# This op is intended to exactly match the semantics of numpy.repeat, with
# one exception: numpy.repeat has special (and somewhat non-intuitive) behavior
# when axis is not specified.  Rather than implement that special behavior, we
# simply make `axis` be a required argument.
#
# External (OSS) `tf.repeat` feature request:
# https://github.com/tensorflow/tensorflow/issues/8246
def repeat(data, repeats, axis, name=None):
  """Repeats elements of `data`.

  Args:
    data: An `N`-dimensional tensor.
    repeats: A 1-D integer tensor specifying how many times each element in
      `axis` should be repeated.  `len(repeats)` must equal `data.shape[axis]`.
      Supports broadcasting from a scalar value.
    axis: `int`.  The axis along which to repeat values.  Must be less than
      `max(N, 1)`.
    name: A name for the operation.

  Returns:
    A tensor with `max(N, 1)` dimensions.  Has the same shape as `data`,
    except that dimension `axis` has size `sum(repeats)`.

  #### Examples:
    ```python
    >>> repeat(['a', 'b', 'c'], repeats=[3, 0, 2], axis=0)
    ['a', 'a', 'a', 'c', 'c']
    >>> repeat([[1, 2], [3, 4]], repeats=[2, 3], axis=0)
    [[1, 2], [1, 2], [3, 4], [3, 4], [3, 4]]
    >>> repeat([[1, 2], [3, 4]], repeats=[2, 3], axis=1)
    [[1, 1, 2, 2, 2], [3, 3, 4, 4, 4]]
    ```
  """
  if not isinstance(axis, int):
    raise TypeError("axis must be an int; got %s" % type(axis).__name__)

  with ops.name_scope(name, "Repeat", [data, repeats]):
    data = ops.convert_to_tensor(data, name="data")
    repeats = convert_to_int_tensor(repeats, name="repeats")
    repeats.shape.with_rank_at_most(1)

    # If `data` is a scalar, then upgrade it to a vector.
    data = _with_nonzero_rank(data)
    data_shape = array_ops.shape(data)

    # If `axis` is negative, then convert it to a positive value.
    axis = get_positive_axis(axis, data.shape.ndims)

    # Check data Tensor shapes.
    if repeats.shape.ndims == 1:
      data.shape.dims[axis].assert_is_compatible_with(repeats.shape[0])

    # If we know that `repeats` is a scalar, then we can just tile & reshape.
    if repeats.shape.ndims == 0:
      expanded = array_ops.expand_dims(data, axis + 1)
      tiled = tile_one_dimension(expanded, axis + 1, repeats)
      result_shape = array_ops.concat(
          [data_shape[:axis], [-1], data_shape[axis + 1:]], axis=0)
      return array_ops.reshape(tiled, result_shape)

    # Broadcast the `repeats` tensor so rank(repeats) == axis + 1.
    if repeats.shape.ndims != axis + 1:
      repeats_shape = array_ops.shape(repeats)
      repeats_ndims = array_ops.rank(repeats)
      broadcast_shape = array_ops.concat(
          [data_shape[:axis + 1 - repeats_ndims], repeats_shape], axis=0)
      repeats = array_ops.broadcast_to(repeats, broadcast_shape)
      repeats.set_shape([None] * (axis + 1))

    # Create a "sequence mask" based on `repeats`, where slices across `axis`
    # contain one `True` value for each repetition.  E.g., if
    # `repeats = [3, 1, 2]`, then `mask = [[1, 1, 1], [1, 0, 0], [1, 1, 0]]`.
    max_repeat = math_ops.maximum(0, math_ops.reduce_max(repeats))
    mask = array_ops.sequence_mask(repeats, max_repeat)

    # Add a new dimension around each value that needs to be repeated, and
    # then tile that new dimension to match the maximum number of repetitions.
    expanded = array_ops.expand_dims(data, axis + 1)
    tiled = tile_one_dimension(expanded, axis + 1, max_repeat)

    # Use `boolean_mask` to discard the extra repeated values.  This also
    # flattens all dimensions up through `axis`.
    masked = array_ops.boolean_mask(tiled, mask)

    # Reshape the output tensor to add the outer dimensions back.
    if axis == 0:
      result = masked
    else:
      result_shape = array_ops.concat(
          [data_shape[:axis], [-1], data_shape[axis + 1:]], axis=0)
      result = array_ops.reshape(masked, result_shape)

    # Preserve shape information.
    if data.shape.ndims is not None:
      new_axis_size = 0 if repeats.shape[0] == 0 else None
      result.set_shape(data.shape[:axis].concatenate(
          [new_axis_size]).concatenate(data.shape[axis + 1:]))

    return result


def tile_one_dimension(data, axis, multiple):
  """Tiles a single dimension of a tensor."""
  # Assumes axis is a nonnegative int.
  if data.shape.ndims is not None:
    multiples = [1] * data.shape.ndims
    multiples[axis] = multiple
  else:
    ones = array_ops.ones(array_ops.rank(data), dtypes.int32)
    multiples = array_ops.concat([ones[:axis], [multiple], ones[axis + 1:]],
                                 axis=0)
  return array_ops.tile(data, multiples)


def _with_nonzero_rank(data):
  """If `data` is scalar, then add a dimension; otherwise return as-is."""
  if data.shape.ndims is not None:
    if data.shape.ndims == 0:
      return array_ops.stack([data])
    else:
      return data
  else:
    data_shape = array_ops.shape(data)
    data_ndims = array_ops.rank(data)
    return array_ops.reshape(
        data,
        array_ops.concat([[1], data_shape], axis=0)[-data_ndims:])


def lengths_to_splits(lengths):
  """Returns splits corresponding to the given lengths."""
  return array_ops.concat([[0], math_ops.cumsum(lengths)], axis=-1)


def repeat_ranges(params, splits, repeats):
  """Repeats each range of `params` (as specified by `splits`) `repeats` times.

  Let the `i`th range of `params` be defined as
  `params[splits[i]:splits[i + 1]]`.  Then this function returns a tensor
  containing range 0 repeated `repeats[0]` times, followed by range 1 repeated
  `repeats[1]`, ..., followed by the last range repeated `repeats[-1]` times.

  Args:
    params: The `Tensor` whose values should be repeated.
    splits: A splits tensor indicating the ranges of `params` that should be
      repeated.
    repeats: The number of times each range should be repeated.  Supports
      broadcasting from a scalar value.

  Returns:
    A `Tensor` with the same rank and type as `params`.

  #### Example:
    ```python
    >>> repeat_ranges(['a', 'b', 'c'], [0, 2, 3], 3)
    ['a', 'b', 'a', 'b', 'a', 'b', 'c', 'c', 'c']
    ```
  """
  # Divide `splits` into starts and limits, and repeat them `repeats` times.
  if repeats.shape.ndims != 0:
    repeated_starts = repeat(splits[:-1], repeats, axis=0)
    repeated_limits = repeat(splits[1:], repeats, axis=0)
  else:
    # Optimization: we can just call repeat once, and then slice the result.
    repeated_splits = repeat(splits, repeats, axis=0)
    n_splits = array_ops.shape(repeated_splits, out_type=dtypes.int64)[0]
    repeated_starts = repeated_splits[:n_splits - repeats]
    repeated_limits = repeated_splits[repeats:]

  # Get indices for each range from starts to limits, and use those to gather
  # the values in the desired repetition pattern.
  one = array_ops.ones((), repeated_starts.dtype)
  offsets = gen_ragged_math_ops.ragged_range(
      repeated_starts, repeated_limits, one)
  return array_ops.gather(params, offsets.rt_dense_values)
