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
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value


#===============================================================================
# Op to construct a constant RaggedTensor from a nested Python list.
#===============================================================================
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
      `max(0, K - 1)` if `inner_shape` is not specified.  Defaults to
      `max(0, K - 1 - len(inner_shape))` if `inner_shape` is specified.
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
  with ops.name_scope(name, 'RaggedConstant'):
    return _constant_value(from_row_splits, constant_op.constant, pylist, dtype,
                           ragged_rank, inner_shape)


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
    raise TypeError('pylist may not be a RaggedTensor or RaggedTensorValue.')

  if not isinstance(pylist, (list, tuple)):
    # Scalar value
    if ragged_rank is not None and ragged_rank != 0:
      raise ValueError('Invalid pylist=%r: incompatible with ragged_rank=%d' %
                       (pylist, ragged_rank))
    if inner_shape is not None and inner_shape:
      raise ValueError(
          'Invalid pylist=%r: incompatible with dim(inner_shape)=%d' %
          (pylist, len(inner_shape)))
    return inner_factory(pylist, dtype, ())

  if ragged_rank is not None and ragged_rank < 0:
    raise ValueError(
        'Invalid ragged_rank=%r: must be nonnegative' % ragged_rank)

  # Find the depth of scalar values in `pylist`.
  scalar_depth, max_depth = _find_scalar_and_max_depth(pylist)
  if scalar_depth is not None:
    if max_depth > scalar_depth:
      raise ValueError('Invalid pylist=%r: empty list nesting is greater '
                       'than scalar value nesting' % pylist)

  # If both inner_shape and ragged_rank were specified, then check that
  # they are compatible with pylist.
  if inner_shape is not None and ragged_rank is not None:
    expected_depth = ragged_rank + len(inner_shape) + 1
    if ((scalar_depth is not None and expected_depth != scalar_depth) or
        (scalar_depth is None and expected_depth < max_depth)):
      raise ValueError(
          'Invalid pylist=%r: incompatible with ragged_rank=%d '
          'and dim(inner_shape)=%d' % (pylist, ragged_rank, len(inner_shape)))

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
      values, dtype=dtype, shape=(len(values),) + inner_shape, name='values')
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
          raise ValueError('all scalar values must have the same nesting depth')
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
      raise ValueError('inner values have inconsistent shape')
    if is_nested:
      if shape[0] != len(item):
        raise ValueError('inner values have inconsistent shape')
      for child in item:
        check_inner_shape(child, shape[1:])

  # Collapse the ragged layers to get the list of inner values.
  inner_values = pylist
  for dim in range(ragged_rank):
    if not all(isinstance(v, (list, tuple)) for v in inner_values):
      raise ValueError('pylist has scalar values depth %d, but ragged_rank=%d '
                       'requires scalar value depth greater than %d' %
                       (dim + 1, ragged_rank, ragged_rank))
    inner_values = sum((list(v) for v in inner_values), [])

  # Compute the inner shape looking only at the leftmost elements; and then
  # use check_inner_shape to verify that other elements have the same shape.
  inner_shape = get_inner_shape(inner_values)
  check_inner_shape(inner_values, inner_shape)
  return inner_shape[1:]


#===============================================================================
# Convert value -> tensor
#===============================================================================
def convert_to_tensor_or_ragged_tensor(value,
                                       dtype=None,
                                       preferred_dtype=None,
                                       name=None):
  """Converts value to a `RaggedTensor` or `Tensor`.

  * If `value` is a `RaggedTensor`, then return it as-is.
  * If `value` is a `RaggedTensorValue`, return a corresponding constant
    `RaggedTensor`.
  * Otherwise, use `convert_to_tensor` to convert `value` to a `Tensor`.

  Args:
    value: A `RaggedTensor`, a `RaggedTensorValue`, or an object whose type has
      a registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor.  If missing the type
      is inferred from the type of `value`.
    preferred_dtype: Optional element type for the returned tensor, used when
      dtype is None.  This argument has no effect if `value` is already a
      tensor, or when conversion is not possible.
    name: Optional name to use if a new `Tensor` is created.

  Returns:
    A `Tensor` or `RaggedTensor`.
  """
  if isinstance(value, ragged_tensor.RaggedTensor):
    if dtype and not dtype.is_compatible_with(value.dtype):
      raise ValueError('Tensor conversion requested dtype %s for '
                       'RaggedTensor with dtype %s: %r' %
                       (dtype.name, value.dtype.name, value))
    return value
  elif isinstance(value, ragged_tensor_value.RaggedTensorValue):
    with ops.name_scope(name, 'ConvertToTensorOrRaggedTensor', []):
      inner_values = ops.convert_to_tensor(
          value=value.inner_values,
          dtype=dtype,
          preferred_dtype=preferred_dtype,
          name='inner_values')
      return from_nested_row_splits(inner_values, value.nested_row_splits)
  else:
    return ops.convert_to_tensor(
        value=value, dtype=dtype, preferred_dtype=preferred_dtype, name=name)


#===============================================================================
# Ops to construct RaggedTensor from row-partitioned values.
#===============================================================================


def from_value_rowids(values, value_rowids, nrows=None, name=None):
  """Creates a `RaggedTensor` with rows partitioned by `value_rowids`.

  The returned `RaggedTensor` corresponds with the python list defined by:

  ```python
  result = [[values[i] for i in range(len(values)) if value_rowids[i] == row]
            for row in range(nrows)]
  ```

  Warning: currently, this needs to cast value_rowids to int64 before
  converting, since `tf.bincount` only supports `int32`.

  Args:
    values: A potentially ragged tensor with shape `[nvals, ...]`.
    value_rowids: A 1-D int64 tensor with shape `[nvals]`, which corresponds
      one-to-one with `values`, and specifies each value's row index.  Must be
      nonnegative, and must be sorted in ascending order.
    nrows: An int64 scalar specifying the number of rows.  This should be
      specified if the `RaggedTensor` may containing empty training rows.  Must
      be greater than `value_rowids[-1]` (or zero if `value_rowids` is empty).
      Defaults to `value_rowids[-1]` (or zero if `value_rowids` is empty).
    name: A name prefix for the RaggedTensor (optional).

  Returns:
    A `RaggedTensor`.  `result.rank = values.rank + 1`.
    `result.ragged_rank = values.ragged_rank + 1`.

  Raises:
    ValueError: If `nrows` is incompatible with `value_rowids`.

  #### Example:
    ```python
    >>> rt = ragged.from_value_rowids(
    ...     values=[3, 1, 4, 1, 5, 9, 2, 6],
    ...     value_rowids=[0, 0, 0, 0, 2, 2, 2, 3],
    ...     nrows=5)
    >>> rt.eval().tolist()
    [[3, 1, 4, 1], [], [5, 9, 2], [6], []]
    ```
  """
  with ops.name_scope(name, 'RaggedFromValueRowIds',
                      [values, value_rowids, nrows]):
    values = convert_to_tensor_or_ragged_tensor(values, name='values')
    value_rowids = ops.convert_to_tensor(
        value_rowids, dtypes.int64, name='value_rowids')
    if nrows is None:
      const_rowids = tensor_util.constant_value(value_rowids)
      if const_rowids is None:
        nrows = array_ops.concat([value_rowids[-1:], [-1]], axis=0)[0] + 1
        const_nrows = None
      else:
        const_nrows = const_rowids[-1] + 1 if const_rowids.size > 0 else 0
        nrows = ops.convert_to_tensor(const_nrows, dtypes.int64, name='nrows')
    else:
      nrows = ops.convert_to_tensor(nrows, dtypes.int64, 'nrows')
      const_nrows = tensor_util.constant_value(nrows)
      if const_nrows is not None:
        if const_nrows < 0:
          raise ValueError('Expected nrows >= 0; got %d' % const_nrows)
        const_rowids = tensor_util.constant_value(value_rowids)
        if const_rowids is not None and const_rowids.size > 0:
          if not const_nrows >= const_rowids[-1] + 1:
            raise ValueError(
                'Expected nrows >= value_rowids[-1] + 1; got nrows=%d, '
                'value_rowids[-1]=%d' % (const_nrows, const_rowids[-1]))

    value_rowids.shape.assert_has_rank(1)
    nrows.shape.assert_has_rank(0)
    values.shape[:1].assert_is_compatible_with(value_rowids.shape)

    # Convert value_rowids & nrows to row_splits.
    # Note: we don't use segment_ids_to_row_splits() here because we want
    # to save the intermediate value `row_lengths`, so we can cache it.
    # TODO(b/116708836) Upgrade bincount to accept int64 so we can skip the cast
    # (Remove the warning in the docstring when we do.)
    value_rowids_int32 = math_ops.cast(value_rowids, dtypes.int32)
    nrows_int32 = math_ops.cast(nrows, dtypes.int32)
    row_lengths = math_ops.bincount(
        value_rowids_int32,
        minlength=nrows_int32,
        maxlength=nrows_int32,
        dtype=dtypes.int64)
    row_splits = array_ops.concat([[0], math_ops.cumsum(row_lengths)], axis=0)
    if const_nrows is not None:
      row_lengths.set_shape([const_nrows])
      row_splits.set_shape([const_nrows + 1])

    return ragged_tensor.RaggedTensor(
        values,
        row_splits,
        cached_row_lengths=row_lengths,
        cached_value_rowids=value_rowids,
        cached_nrows=nrows,
        internal=True)


def from_row_splits(values, row_splits, name=None):
  """Creates a `RaggedTensor` with rows partitioned by `row_splits`.

  The returned `RaggedTensor` corresponds with the python list defined by:

  ```python
  result = [values[row_splits[i]:row_splits[i + 1]]
            for i in range(len(row_splits) - 1)]
  ```

  Args:
    values: A potentially ragged tensor with shape `[nvals, ...]`.
    row_splits: A 1-D int64 tensor with shape `[nrows+1]`.  Must not be empty,
      and must be sorted in ascending order.  `row_splits[0]` must be zero and
      `row_splits[-1]` must be `nvals`.
    name: A name prefix for the RaggedTensor (optional).

  Returns:
    A `RaggedTensor`.  `result.rank = values.rank + 1`.
    `result.ragged_rank = values.ragged_rank + 1`.

  Raises:
    ValueError: If `row_splits` is an empty list.

  #### Example:
    ```python
    >>> rt = ragged.from_row_splits(
    ...     values=[3, 1, 4, 1, 5, 9, 2, 6],
    ...     row_splits=[0, 4, 4, 7, 8, 8])
    >>> rt.eval().tolist()
    [[3, 1, 4, 1], [], [5, 9, 2], [6], []]
    ```
  """
  if isinstance(row_splits, (list, tuple)) and not row_splits:
    raise ValueError('row_splits tensor may not be empty.')
  with ops.name_scope(name, 'RaggedFromRowSplits', [values, row_splits]):
    values = convert_to_tensor_or_ragged_tensor(values, name='values')
    row_splits = ops.convert_to_tensor(row_splits, dtypes.int64, 'row_splits')
    row_splits.shape.assert_has_rank(1)
    return ragged_tensor.RaggedTensor(
        values=values, row_splits=row_splits, internal=True)


def from_row_lengths(values, row_lengths, name=None):
  """Creates a `RaggedTensor` with rows partitioned by `row_lengths`.

  The returned `RaggedTensor` corresponds with the python list defined by:

  ```python
  result = [[values.pop(0) for i in range(length)]
            for length in row_lengths]
  ```

  Args:
    values: A potentially ragged tensor with shape `[nvals, ...]`.
    row_lengths: A 1-D int64 tensor with shape `[nrows]`.  Must be nonnegative.
      `sum(row_lengths)` must be `nvals`.
    name: A name prefix for the RaggedTensor (optional).

  Returns:
    A `RaggedTensor`.  `result.rank = values.rank + 1`.
    `result.ragged_rank = values.ragged_rank + 1`.

  #### Example:
    ```python
    >>> rt = ragged.from_row_lengths(
    ...     values=[3, 1, 4, 1, 5, 9, 2, 6],
    ...     row_lengths=[4, 0, 3, 1, 0])
    >>> rt.eval().tolist()
    [[3, 1, 4, 1], [], [5, 9, 2], [6], []]
    ```
  """
  with ops.name_scope(name, 'RaggedFromRowLengths', [values, row_lengths]):
    values = convert_to_tensor_or_ragged_tensor(values, name='values')
    row_lengths = ops.convert_to_tensor(row_lengths, dtypes.int64,
                                        'row_lengths')
    row_lengths.shape.assert_has_rank(1)
    row_limits = math_ops.cumsum(row_lengths)
    row_splits = array_ops.concat([[0], row_limits], axis=0)
    return ragged_tensor.RaggedTensor(
        values=values,
        row_splits=row_splits,
        cached_row_lengths=row_lengths,
        internal=True)


def from_row_starts(values, row_starts, name=None):
  """Creates a `RaggedTensor` with rows partitioned by `row_starts`.

  Equivalent to: `from_row_splits(values, concat([row_starts, nvals]))`.

  Args:
    values: A potentially ragged tensor with shape `[nvals, ...]`.
    row_starts: A 1-D int64 tensor with shape `[nrows]`.  Must be nonnegative
      and sorted in ascending order.  If `nrows>0`, then `row_starts[0]` must be
      zero.
    name: A name prefix for the RaggedTensor (optional).

  Returns:
    A `RaggedTensor`.  `result.rank = values.rank + 1`.
    `result.ragged_rank = values.ragged_rank + 1`.

  #### Example:
    ```python
    >>> rt = ragged.from_row_starts(
    ...     values=[3, 1, 4, 1, 5, 9, 2, 6],
    ...     row_starts=[0, 4, 4, 7, 8])
    >>> rt.eval().tolist()
    [[3, 1, 4, 1], [], [5, 9, 2], [6], []]
    ```
  """
  with ops.name_scope(name, 'RaggedFromRowStarts', [values, row_starts]):
    values = convert_to_tensor_or_ragged_tensor(values, name='values')
    row_starts = ops.convert_to_tensor(row_starts, dtypes.int64, 'row_starts')
    row_starts.shape.assert_has_rank(1)
    nvals = array_ops.shape(values, out_type=dtypes.int64)[:1]
    row_splits = array_ops.concat([row_starts, nvals], axis=0)
    return ragged_tensor.RaggedTensor(
        values=values, row_splits=row_splits, internal=True)


def from_row_limits(values, row_limits, name=None):
  """Creates a `RaggedTensor` with rows partitioned by `row_limits`.

  Equivalent to: `from_row_splits(values, concat([0, row_limits]))`.

  Args:
    values: A potentially ragged tensor with shape `[nvals, ...]`.
    row_limits: A 1-D int64 tensor with shape `[nrows]`.  Must be sorted in
      ascending order.  If `nrows>0`, then `row_limits[-1]` must be `nvals`.
    name: A name prefix for the RaggedTensor (optional).

  Returns:
    A `RaggedTensor`.  `result.rank = values.rank + 1`.
    `result.ragged_rank = values.ragged_rank + 1`.

  #### Example:
    ```python
    >>> rt = ragged.from_row_limits(
    ...     values=[3, 1, 4, 1, 5, 9, 2, 6],
    ...     row_limits=[4, 4, 7, 8, 8])
    >>> rt.eval().tolist()
    [[3, 1, 4, 1], [], [5, 9, 2], [6], []]
    ```
  """
  with ops.name_scope(name, 'RaggedFromRowLimits', [values, row_limits]):
    values = convert_to_tensor_or_ragged_tensor(values, name='values')
    row_limits = ops.convert_to_tensor(row_limits, dtypes.int64, 'row_limits')
    row_limits.shape.assert_has_rank(1)
    zero = array_ops.zeros([1], dtypes.int64)
    row_splits = array_ops.concat([zero, row_limits], axis=0)
    return ragged_tensor.RaggedTensor(
        values=values, row_splits=row_splits, internal=True)


def from_nested_value_rowids(inner_values,
                             nested_value_rowids,
                             nested_nrows=None,
                             name=None):
  """Creates a `RaggedTensor` from a nested list of `value_rowids` tensors.

  Equivalent to:

  ```python
  result = inner_values
  for (value_rowids, nrows) in reversed(zip(nested_value_rowids, nested_nrows)):
    result = from_value_rowids(result, value_rowids, nrows)
  ```

  Args:
    inner_values: A potentially ragged tensor.
    nested_value_rowids: A list of 1-D int64 tensors.  The `i`th tensor is used
      as the `value_rowids` for the `i`th ragged dimension.
    nested_nrows: A list of int64 scalars.  The `i`th scalar is used as the
      `nrows` for the `i`th ragged dimension.
    name: A name prefix for the RaggedTensor (optional).

  Returns:
    A `RaggedTensor` (or `inner_values` if `nested_value_rowids` is empty).

  Raises:
    ValueError: If `len(nested_values_rowids) != len(nested_nrows)`.
  """
  if isinstance(nested_value_rowids, ops.Tensor):
    raise TypeError('nested_value_rowids must be a list of Tensors')
  if nested_nrows is None:
    nested_nrows = [None] * len(nested_value_rowids)
  else:
    if isinstance(nested_nrows, ops.Tensor):
      raise TypeError('nested_nrows must be a list of Tensors')
    if len(nested_nrows) != len(nested_value_rowids):
      raise ValueError('nested_nrows must have the same length as '
                       'nested_value_rowids')

  with ops.name_scope(
      name, 'RaggedFromNestedValueRowIds',
      [inner_values] + list(nested_value_rowids) + list(nested_nrows)):
    result = inner_values
    for value_rowids, nrows in reversed(
        list(zip(nested_value_rowids, nested_nrows))):
      result = from_value_rowids(result, value_rowids, nrows)
    return result


def from_nested_row_splits(inner_values, nested_row_splits, name=None):
  """Creates a `RaggedTensor` from a nested list of `row_splits` tensors.

  Equivalent to:

  ```python
  result = inner_values
  for row_splits in reversed(nested_row_splits):
    result = from_row_splits(result, row_splits)
  ```

  Args:
    inner_values: A potentially ragged tensor.
    nested_row_splits: A list of 1-D int64 tensors.  The `i`th tensor is used as
      the `row_splits` for the `i`th ragged dimension.
    name: A name prefix for the RaggedTensor (optional).

  Returns:
    A `RaggedTensor` (or `inner_values` if `nested_row_splits` is empty).
  """
  if isinstance(nested_row_splits, ops.Tensor):
    raise TypeError('nested_row_splits must be a list of Tensors')
  with ops.name_scope(name, 'RaggedFromNestedRowSplits',
                      [inner_values] + list(nested_row_splits)):
    result = inner_values
    for splits in reversed(nested_row_splits):
      result = from_row_splits(result, splits)
    return result
