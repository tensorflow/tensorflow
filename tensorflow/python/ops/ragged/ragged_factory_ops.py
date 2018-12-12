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
"""Operations for constructing RaggedTensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.util.tf_export import tf_export


#===============================================================================
# Op to construct a constant RaggedTensor from a nested Python list.
#===============================================================================
@tf_export("ragged.constant")
def constant(pylist, dtype=None, ragged_rank=None, inner_shape=None, name=None):
  """Constructs a constant RaggedTensor from a nested Python list.

  Example:

  ```python
  >>> ragged.constant([[1, 2], [3], [4, 5, 6]]).eval()
  RaggedTensorValue(values=[1, 2, 3, 4, 5, 6], splits=[0, 2, 3, 6])
  ```

  All scalar values in `pylist` must have the same nesting depth `K`, and the
  returned `RaggedTensor` will have rank `K`.  If `pylist` contains no scalar
  values, then `K` is one greater than the maximum depth of empty lists in
  `pylist`.  All scalar values in `pylist` must be compatible with `dtype`.

  Args:
    pylist: A nested `list` or `tuple`.  Any nested element that is not a `list`
      or `tuple` must be a scalar value compatible with `dtype`.
    dtype: The type of elements for the returned `RaggedTensor`.  If not
      specified, then a default is chosen based on the scalar values in
      `pylist`.
    ragged_rank: An integer specifying the ragged rank of the returned
      `RaggedTensor`.  Must be nonnegative and less than `K`. Defaults to
      `max(0, K - 1)` if `inner_shape` is not specified.  Defaults to `max(0, K
      - 1 - len(inner_shape))` if `inner_shape` is specified.
    inner_shape: A tuple of integers specifying the shape for individual inner
      values in the returned `RaggedTensor`.  Defaults to `()` if `ragged_rank`
      is not specified.  If `ragged_rank` is specified, then a default is chosen
      based on the contents of `pylist`.
    name: A name prefix for the returned tensor (optional).

  Returns:
    A potentially ragged tensor with rank `K` and the specified `ragged_rank`,
    containing the values from `pylist`.

  Raises:
    ValueError: If the scalar values in `pylist` have inconsistent nesting
      depth; or if ragged_rank or inner_shape are incompatible with `pylist`.
  """
  with ops.name_scope(name, "RaggedConstant"):
    return _constant_value(ragged_tensor.RaggedTensor.from_row_splits,
                           constant_op.constant, pylist, dtype, ragged_rank,
                           inner_shape)


@tf_export(v1=["ragged.constant_value"])
def constant_value(pylist, dtype=None, ragged_rank=None, inner_shape=None):
  """Constructs a RaggedTensorValue from a nested Python list.

  > Warning: This function returns a `RaggedTensorValue`, not a `RaggedTensor`.
  > If you wish to construct a constant `RaggedTensor`, use
  > [`ragged.constant(...)`](constant.md) instead.

  Example:

  ```python
  >>> ragged.constant_value([[1, 2], [3], [4, 5, 6]])
  RaggedTensorValue(values=[1, 2, 3, 4, 5, 6], splits=[0, 2, 3, 6])
  ```

  All scalar values in `pylist` must have the same nesting depth `K`, and the
  returned `RaggedTensorValue` will have rank `K`.  If `pylist` contains no
  scalar values, then `K` is one greater than the maximum depth of empty lists
  in `pylist`.  All scalar values in `pylist` must be compatible with `dtype`.

  Args:
    pylist: A nested `list` or `tuple`.  Any nested element that is not a `list`
      or `tuple` must be a scalar value compatible with `dtype`.
    dtype: `numpy.dtype`.  The type of elements for the returned `RaggedTensor`.
      If not specified, then a default is chosen based on the scalar values in
      `pylist`.
    ragged_rank: An integer specifying the ragged rank of the returned
      `RaggedTensorValue`.  Must be nonnegative and less than `K`. Defaults to
      `max(0, K - 1)` if `inner_shape` is not specified.  Defaults to `max(0, K
      - 1 - len(inner_shape))` if `inner_shape` is specified.
    inner_shape: A tuple of integers specifying the shape for individual inner
      values in the returned `RaggedTensorValue`.  Defaults to `()` if
      `ragged_rank` is not specified.  If `ragged_rank` is specified, then a
      default is chosen based on the contents of `pylist`.

  Returns:
    A `RaggedTensorValue` or `numpy.array` with rank `K` and the specified
    `ragged_rank`, containing the values from `pylist`.

  Raises:
    ValueError: If the scalar values in `pylist` have inconsistent nesting
      depth; or if ragged_rank or inner_shape are incompatible with `pylist`.
  """

  def _ragged_factory(values, row_splits):
    row_splits = np.array(row_splits, dtype=np.int64)
    return ragged_tensor_value.RaggedTensorValue(values, row_splits)

  def _inner_factory(pylist, dtype, shape, name=None):  # pylint: disable=unused-argument
    return np.reshape(np.array(pylist, dtype=dtype), shape)

  return _constant_value(_ragged_factory, _inner_factory, pylist, dtype,
                         ragged_rank, inner_shape)


def _constant_value(ragged_factory, inner_factory, pylist, dtype, ragged_rank,
                    inner_shape):
  """Constructs a constant RaggedTensor or RaggedTensorValue.

  Args:
    ragged_factory: A factory function with the signature:
      `ragged_factory(values, row_splits)`
    inner_factory: A factory function with the signature: `inner_factory(pylist,
      dtype, shape, name)`
    pylist: A nested `list` or `tuple`.
    dtype: Data type for returned value.
    ragged_rank: Ragged rank for returned value.
    inner_shape: Inner value shape for returned value.

  Returns:
    A value returned by `ragged_factory` or `inner_factory`.

  Raises:
    ValueError: If the scalar values in `pylist` have inconsistent nesting
      depth; or if ragged_rank or inner_shape are incompatible with `pylist`.
  """
  if ragged_tensor.is_ragged(pylist):
    raise TypeError("pylist may not be a RaggedTensor or RaggedTensorValue.")

  if not isinstance(pylist, (list, tuple)):
    # Scalar value
    if ragged_rank is not None and ragged_rank != 0:
      raise ValueError("Invalid pylist=%r: incompatible with ragged_rank=%d" %
                       (pylist, ragged_rank))
    if inner_shape is not None and inner_shape:
      raise ValueError(
          "Invalid pylist=%r: incompatible with dim(inner_shape)=%d" %
          (pylist, len(inner_shape)))
    return inner_factory(pylist, dtype, ())

  if ragged_rank is not None and ragged_rank < 0:
    raise ValueError(
        "Invalid ragged_rank=%r: must be nonnegative" % ragged_rank)

  # Find the depth of scalar values in `pylist`.
  scalar_depth, max_depth = _find_scalar_and_max_depth(pylist)
  if scalar_depth is not None:
    if max_depth > scalar_depth:
      raise ValueError("Invalid pylist=%r: empty list nesting is greater "
                       "than scalar value nesting" % pylist)

  # If both inner_shape and ragged_rank were specified, then check that
  # they are compatible with pylist.
  if inner_shape is not None and ragged_rank is not None:
    expected_depth = ragged_rank + len(inner_shape) + 1
    if ((scalar_depth is not None and expected_depth != scalar_depth) or
        (scalar_depth is None and expected_depth < max_depth)):
      raise ValueError(
          "Invalid pylist=%r: incompatible with ragged_rank=%d "
          "and dim(inner_shape)=%d" % (pylist, ragged_rank, len(inner_shape)))

  # Check if the result is a `Tensor`.
  if (ragged_rank == 0 or
      (ragged_rank is None and
       ((max_depth < 2) or
        (inner_shape is not None and max_depth - len(inner_shape) < 2)))):
    return inner_factory(pylist, dtype, inner_shape)

  # Compute default value for inner_shape.
  if inner_shape is None:
    if ragged_rank is None:
      inner_shape = ()
    else:
      inner_shape = _default_inner_shape_for_pylist(pylist, ragged_rank)

  # Compute default value for ragged_rank.
  if ragged_rank is None:
    if scalar_depth is None:
      ragged_rank = max(1, max_depth - 1)
    else:
      ragged_rank = max(1, scalar_depth - 1 - len(inner_shape))

  # Build the splits for each ragged rank, and concatenate the inner values
  # into a single list.
  nested_splits = []
  values = pylist
  for dim in range(ragged_rank):
    nested_splits.append([0])
    concatenated_values = []
    for row in values:
      nested_splits[dim].append(nested_splits[dim][-1] + len(row))
      concatenated_values.extend(row)
    values = concatenated_values

  values = inner_factory(
      values, dtype=dtype, shape=(len(values),) + inner_shape, name="values")
  for row_splits in reversed(nested_splits):
    values = ragged_factory(values, row_splits)
  return values


def _find_scalar_and_max_depth(pylist):
  """Finds nesting depth of scalar values in pylist.

  Args:
    pylist: A nested python `list` or `tuple`.

  Returns:
    A tuple `(scalar_depth, max_depth)`.  `scalar_depth` is the nesting
    depth of scalar values in `pylist`, or `None` if `pylist` contains no
    scalars.  `max_depth` is the maximum depth of `pylist` (including
    empty lists).

  Raises:
    ValueError: If pylist has inconsistent nesting depths for scalars.
  """
  if isinstance(pylist, (list, tuple)):
    scalar_depth = None
    max_depth = 1
    for child in pylist:
      child_scalar_depth, child_max_depth = _find_scalar_and_max_depth(child)
      if child_scalar_depth is not None:
        if scalar_depth is not None and scalar_depth != child_scalar_depth + 1:
          raise ValueError("all scalar values must have the same nesting depth")
        scalar_depth = child_scalar_depth + 1
      max_depth = max(max_depth, child_max_depth + 1)
    return (scalar_depth, max_depth)
  else:
    return (0, 0)


def _default_inner_shape_for_pylist(pylist, ragged_rank):
  """Computes a default inner shape for the given python list."""

  def get_inner_shape(item):
    """Returns the inner shape for a python list `item`."""
    if not isinstance(item, (list, tuple)):
      return ()
    elif item:
      return (len(item),) + get_inner_shape(item[0])
    else:
      return (0,)

  def check_inner_shape(item, shape):
    """Checks that `item` has a consistent shape matching `shape`."""
    is_nested = isinstance(item, (list, tuple))
    if is_nested != bool(shape):
      raise ValueError("inner values have inconsistent shape")
    if is_nested:
      if shape[0] != len(item):
        raise ValueError("inner values have inconsistent shape")
      for child in item:
        check_inner_shape(child, shape[1:])

  # Collapse the ragged layers to get the list of inner values.
  flat_values = pylist
  for dim in range(ragged_rank):
    if not all(isinstance(v, (list, tuple)) for v in flat_values):
      raise ValueError("pylist has scalar values depth %d, but ragged_rank=%d "
                       "requires scalar value depth greater than %d" %
                       (dim + 1, ragged_rank, ragged_rank))
    flat_values = sum((list(v) for v in flat_values), [])

  # Compute the inner shape looking only at the leftmost elements; and then
  # use check_inner_shape to verify that other elements have the same shape.
  inner_shape = get_inner_shape(flat_values)
  check_inner_shape(flat_values, inner_shape)
  return inner_shape[1:]
