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

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import math_ops



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


# Note: imported here to avoid circular dependency of array_ops.
get_positive_axis = array_ops.get_positive_axis
convert_to_int_tensor = array_ops.convert_to_int_tensor
repeat = array_ops.repeat_with_axis


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
    n_splits = array_ops.shape(repeated_splits, out_type=repeats.dtype)[0]
    repeated_starts = repeated_splits[:n_splits - repeats]
    repeated_limits = repeated_splits[repeats:]

  # Get indices for each range from starts to limits, and use those to gather
  # the values in the desired repetition pattern.
  one = array_ops.ones((), repeated_starts.dtype)
  offsets = gen_ragged_math_ops.ragged_range(
      repeated_starts, repeated_limits, one)
  return array_ops.gather(params, offsets.rt_dense_values)
