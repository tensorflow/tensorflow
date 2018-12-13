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
"""Support for ragged tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.util.tf_export import tf_export


@tf_export("ragged.map_flat_values")
def map_flat_values(op, *args, **kwargs):
  """Applies `op` to the inner values of one or more RaggedTensors.

  Replaces any `RaggedTensor` in `args` or `kwargs` with its `flat_values`
  tensor, and then calls `op`.  Returns a `RaggedTensor` that is constructed
  from the input `RaggedTensor`s' `splits` and the value returned by
  the `op`.

  If the input arguments contain multiple `RaggedTensor`s, then they must have
  identical `splits`.

  Examples:

  ```python
  >>> rt = ragged.constant([[1, 2, 3], [], [4, 5], [6]])
  >>> ragged.map_flat_values(tf.ones_like, rt).eval().tolist()
  [[1, 1, 1], [], [1, 1], [1]]
  >>> ragged.map_flat_values(tf.multiply, rt, rt).eval().tolist()
  [[1, 4, 9], [], [16, 25], [36]]
  >>> ragged.map_flat_values(tf.add, rt, 5).eval().tolist()
  [[6, 7, 8], [], [9, 10], [11]]
  ```

  Args:
    op: The operation that should be applied to the RaggedTensor `flat_values`.
      `op` is typically an element-wise operation (such as math_ops.add), but
      any operation that preserves the size of the outermost dimension can be
      used.  I.e., `shape[0]` of the value returned by `op` must match
      `shape[0]` of the `RaggedTensor`s' `flat_values` tensors.
    *args: Arguments for `op`.
    **kwargs: Keyword arguments for `op`.

  Returns:
    A `RaggedTensor` whose `ragged_rank` matches the `ragged_rank` of all
    input `RaggedTensor`s.
  Raises:
    ValueError: If args contains no `RaggedTensors`, or if the `nested_splits`
      of the input `RaggedTensor`s are not identical.
  """
  # Replace RaggedTensors with their values; and collect the splits tensors
  # from each RaggedTensor.
  nested_splits_lists = []
  inner_args = _replace_ragged_with_flat_values(args, nested_splits_lists)
  inner_kwargs = _replace_ragged_with_flat_values(kwargs, nested_splits_lists)
  if not nested_splits_lists:
    return op(*args, **kwargs)

  with ops.control_dependencies(
      ragged_util.assert_splits_match(nested_splits_lists)):
    # Delegate to op, and then compose the result from the transformed values
    # and the splits.
    return ragged_tensor.RaggedTensor.from_nested_row_splits(
        op(*inner_args, **inner_kwargs), nested_splits_lists[0])


def _replace_ragged_with_flat_values(value, nested_splits_lists):
  """Replace RaggedTensors with their flat_values, and record their splits.

  Returns a copy of `value`, with any nested `RaggedTensor`s replaced by their
  `flat_values` tensor.  Looks inside lists, tuples, and dicts.

  Appends each `RaggedTensor`'s `nested_splits` to `nested_splits_lists`.

  Args:
    value: The value that should be transformed by replacing `RaggedTensors`.
    nested_splits_lists: An output parameter used to record the `nested_splits`
      for any `RaggedTensors` that were replaced.

  Returns:
    A copy of `value` with nested `RaggedTensors` replaced by their `values`.
  """
  # Base case
  if ragged_tensor.is_ragged(value):
    value = ragged_tensor.convert_to_tensor_or_ragged_tensor(value)
    nested_splits_lists.append(value.nested_row_splits)
    return value.flat_values

  # Recursion cases
  def recurse(v):
    return _replace_ragged_with_flat_values(v, nested_splits_lists)

  if isinstance(value, list):
    return [recurse(v) for v in value]
  elif isinstance(value, tuple):
    return tuple(recurse(v) for v in value)
  elif isinstance(value, dict):
    return dict((k, recurse(v)) for (k, v) in value.items())
  else:
    return value
