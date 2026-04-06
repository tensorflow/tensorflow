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

from typing import Union

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.util import dispatch
from tensorflow.python.util.numpy_compat import np_reshape
from tensorflow.python.util.tf_export import tf_export


# ===============================================================================
# Op to construct a constant RaggedTensor from a nested Python list.
# ===============================================================================
@tf_export("ragged.constant")
@dispatch.add_dispatch_support
def constant(
    pylist,
    dtype=None,
    ragged_rank=None,
    inner_shape=None,
    name=None,
    row_splits_dtype=dtypes.int64,
) -> Union[ragged_tensor.RaggedTensor, ops._EagerTensorBase, ops.Operation]:
  """Constructs a constant RaggedTensor from a nested Python list.

  Example:

  >>> tf.ragged.constant([[1, 2], [3], [4, 5, 6]])
  <tf.RaggedTensor [[1, 2], [3], [4, 5, 6]]>

  All scalar values in `pylist` must have the same nesting depth `K`, and the
  returned `RaggedTensor` will have rank `K`.  If `pylist` contains no scalar
  values, then `K` is one greater than the maximum depth of empty lists in
  `pylist`.  All scalar values in `pylist` must be compatible with `dtype`.

  Args:
    pylist: A nested `list`, `tuple` or `np.ndarray`.  Any nested element that
      is not a `list`, `tuple` or `np.ndarray` must be a scalar value compatible
      with `dtype`.
    dtype: The type of elements for the returned `RaggedTensor`.  If not
      specified, then a default is chosen based on the scalar values in
      `pylist`. If there are no scalar values in `pylist`, then the default is
      `tf.float32`.
    ragged_rank: An integer specifying the ragged rank of the returned
      `RaggedTensor`.  Must be nonnegative and less than `K`. Defaults to
      `max(0, K - 1)` if `inner_shape` is not specified.  Defaults to `max(0, K
      - 1 - len(inner_shape))` if `inner_shape` is specified.
    inner_shape: A tuple of integers specifying the shape for individual inner
      values in the returned `RaggedTensor`.  Defaults to `()` if `ragged_rank`
      is not specified.  If `ragged_rank` is specified, then a default is chosen
      based on the contents of `pylist`.
    name: A name prefix for the returned tensor (optional).
    row_splits_dtype: data type for the constructed `RaggedTensor`'s row_splits.
      One of `tf.int32` or `tf.int64`.

  Returns:
    A potentially ragged tensor with rank `K` and the specified `ragged_rank`,
    containing the values from `pylist`.

  Raises:
    ValueError: If the scalar values in `pylist` have inconsistent nesting
      depth; or if ragged_rank or inner_shape are incompatible with `pylist`.
  """
  def ragged_factory(values, row_splits):
    row_splits = constant_op.constant(row_splits, dtype=row_splits_dtype)
    return ragged_tensor.RaggedTensor.from_row_splits(values, row_splits,
                                                      validate=False)

  with ops.name_scope(name, "RaggedConstant"):
    return _constant_value(ragged_factory, constant_op.constant, pylist, dtype,
                           ragged_rank, inner_shape)


@tf_export(v1=["ragged.constant_value"])
@dispatch.add_dispatch_support
def constant_value(
    pylist,
    dtype=None,
    ragged_rank=None,
    inner_shape=None,
    row_splits_dtype="int64",
) -> Union[ragged_tensor_value.RaggedTensorValue, np.ndarray]:
  """Constructs a RaggedTensorValue from a nested Python list.

  Warning: This function returns a `RaggedTensorValue`, not a `RaggedTensor`.
  If you wish to construct a constant `RaggedTensor`, use
  [`ragged.constant(...)`](constant.md) instead.

  Example:

  >>> tf.compat.v1.ragged.constant_value([[1, 2], [3], [4, 5, 6]])
  tf.RaggedTensorValue(values=array([1, 2, 3, 4, 5, 6]),
                       row_splits=array([0, 2, 3, 6]))

  All scalar values in `pylist` must have the same nesting depth `K`, and the
  returned `RaggedTensorValue` will have rank `K`.  If `pylist` contains no
  scalar values, then `K` is one greater than the maximum depth of empty lists
  in `pylist`.  All scalar values in `pylist` must be compatible with `dtype`.

  Args:
    pylist: A nested `list`, `tuple` or `np.ndarray`.  Any nested element that
      is not a `list` or `tuple` must be a scalar value compatible with `dtype`.
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
    row_splits_dtype: data type for the constructed `RaggedTensorValue`'s
      row_splits.  One of `numpy.int32` or `numpy.int64`.

  Returns:
    A `tf.RaggedTensorValue` or `numpy.array` with rank `K` and the specified
    `ragged_rank`, containing the values from `pylist`.

  Raises:
    ValueError: If the scalar values in `pylist` have inconsistent nesting
      depth; or if ragged_rank or inner_shape are incompatible with `pylist`.
  """
  if dtype is not None and isinstance(dtype, dtypes.DType):
    dtype = dtype.as_numpy_dtype
  row_splits_dtype = dtypes.as_dtype(row_splits_dtype).as_numpy_dtype
  def _ragged_factory(values, row_splits):
    row_splits = np.array(row_splits, dtype=row_splits_dtype)
    return ragged_tensor_value.RaggedTensorValue(values, row_splits)

  def _inner_factory(pylist, dtype, shape, name=None):  # pylint: disable=unused-argument
    if dtype is object or dtype is None:
      return np_reshape(np.array(pylist, dtype=dtype), shape)
    else:
      return np_reshape(np.array(pylist).astype(dtype), shape)

  return _constant_value(
      _ragged_factory, _inner_factory, pylist, dtype, ragged_rank, inner_shape
  )


def _get_uniform_dims(pylist):
  if not isinstance(pylist, (list, tuple)) or not pylist:
    return 0
  first_len = len(pylist[0]) if isinstance(pylist[0], (list, tuple)) else -1
  if first_len == -1:
    return 0
  for item in pylist:
    if not isinstance(item, (list, tuple)) or len(item) != first_len:
      return 0
  return 1 + _get_uniform_dims(pylist[0])

def _constant_value(
    ragged_factory, inner_factory, pylist, dtype, ragged_rank, inner_shape
):
  if ragged_tensor.is_ragged(pylist):
    raise TypeError("pylist may not be a RaggedTensor or RaggedTensorValue.")

  if not isinstance(pylist, (list, tuple)) and np.ndim(pylist) == 0:
    if ragged_rank is not None and ragged_rank != 0:
      raise ValueError(
          f"Invalid pylist={pylist}: incompatible with ragged_rank={ragged_rank}")
    if inner_shape is not None and inner_shape:
      raise ValueError(
          f"Invalid pylist={pylist}: incompatible with dim(inner_shape)={len(inner_shape)}")
    return inner_factory(pylist, dtype, ())

  if ragged_rank is not None and ragged_rank < 0:
    raise ValueError(f"Invalid ragged_rank={ragged_rank}: must be nonnegative")

  scalar_depth, max_depth = _find_scalar_and_max_depth(pylist)

  if inner_shape is None:
    inner_shape = () if ragged_rank is None else _default_inner_shape_for_pylist(pylist, ragged_rank)

  uniform_dims = _get_uniform_dims(pylist)

  if ragged_rank is None:
    if scalar_depth is None:
      ragged_rank = max(1, max_depth - 1 - uniform_dims)
    else:
      ragged_rank = max(1, scalar_depth - 1 - len(inner_shape) - uniform_dims)
  else:
    uniform_dims = max(0, max_depth - 1 - ragged_rank)

  # Record uniform lengths before flattening
  uniform_lengths = []
  temp = pylist
  for _ in range(uniform_dims):
    uniform_lengths.append(len(temp[0]))
    flattened = []
    for row in temp:
      if isinstance(row, (list, tuple)):
        flattened.extend(row)
      else:
        flattened.append(row)
    temp = flattened

  # Flatten uniform dimensions for processing
  values = pylist
  for _ in range(uniform_dims):
    flattened = []
    for row in values:
      if isinstance(row, (list, tuple)):
        flattened.extend(row)
      else:
        flattened.append(row)
    values = flattened

  # Build splits for ragged dimensions
  nested_splits = []
  for _ in range(ragged_rank):
    splits = [0]
    new_values = []
    for row in values:
      splits.append(splits[-1] + len(row))
      new_values.extend(row)
    nested_splits.append(splits)
    values = new_values

  values = inner_factory(
      values, dtype=dtype, shape=(len(values),) + inner_shape, name="values")

  for row_splits in reversed(nested_splits):
    values = ragged_factory(values, row_splits)

  # Restore uniform outer dimensions
  for length in reversed(uniform_lengths):
    values = ragged_tensor.RaggedTensor.from_uniform_row_length(
        values, uniform_row_length=length)

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
  # Check if pylist is not scalar. np.ndim builds an array, so we
  # short-circuit lists and tuples.
  if isinstance(pylist, (list, tuple)) or np.ndim(pylist) != 0:
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
  return (0, 0)


def _default_inner_shape_for_pylist(pylist, ragged_rank):
  """Computes a default inner shape for the given python list."""

  def get_inner_shape(item):
    """Returns the inner shape for a python list `item`."""
    if not isinstance(item, (list, tuple)) and np.ndim(item) == 0:
      return ()
    # Note that we need this check here in case `item` is not a Python list but
    # fakes as being one (pylist). For a scenario of this, see test added in
    # https://github.com/tensorflow/tensorflow/pull/48945
    elif len(item) > 0:  # pylint: disable=g-explicit-length-test
      return (len(item),) + get_inner_shape(item[0])
    return (0,)

  def check_inner_shape(item, shape):
    """Checks that `item` has a consistent shape matching `shape`."""
    is_nested = isinstance(item, (list, tuple)) or np.ndim(item) != 0
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
    if not all(
        isinstance(v, (list, tuple)) or np.ndim(v) != 0 for v in flat_values):
      raise ValueError("pylist has scalar values depth %d, but ragged_rank=%d "
                       "requires scalar value depth greater than %d" %
                       (dim + 1, ragged_rank, ragged_rank))
    flat_values = sum((list(v) for v in flat_values), [])

  # Compute the inner shape looking only at the leftmost elements; and then
  # use check_inner_shape to verify that other elements have the same shape.
  inner_shape = get_inner_shape(flat_values)
  check_inner_shape(flat_values, inner_shape)
  return inner_shape[1:]


@tf_export(v1=["ragged.placeholder"])
@dispatch.add_dispatch_support
def placeholder(dtype, ragged_rank, value_shape=None, name=None):
  """Creates a placeholder for a `tf.RaggedTensor` that will always be fed.

  **Important**: This ragged tensor will produce an error if evaluated.
  Its value must be fed using the `feed_dict` optional argument to
  `Session.run()`, `Tensor.eval()`, or `Operation.run()`.


  Args:
    dtype: The data type for the `RaggedTensor`.
    ragged_rank: The ragged rank for the `RaggedTensor`
    value_shape: The shape for individual flat values in the `RaggedTensor`.
    name: A name for the operation (optional).

  Returns:
    A `RaggedTensor` that may be used as a handle for feeding a value, but
    not evaluated directly.

  Raises:
    RuntimeError: if eager execution is enabled

  @compatibility(TF2)
  This API is not compatible with eager execution and `tf.function`. To migrate
  to TF2, rewrite the code to be compatible with eager execution. Check the
  [migration
  guide](https://www.tensorflow.org/guide/migrate#1_replace_v1sessionrun_calls)
  on replacing `Session.run` calls. In TF2, you can just pass tensors directly
  into ops and layers. If you want to explicitly set up your inputs, also see
  [Keras functional API](https://www.tensorflow.org/guide/keras/functional) on
  how to use `tf.keras.Input` to replace `tf.compat.v1.ragged.placeholder`.
  `tf.function` arguments also do the job of `tf.compat.v1.ragged.placeholder`.
  For more details please read [Better
  performance with tf.function](https://www.tensorflow.org/guide/function).
  @end_compatibility
  """
  if ragged_rank == 0:
    return array_ops.placeholder(dtype, value_shape, name)

  with ops.name_scope(name, "RaggedPlaceholder", []):
    flat_shape = tensor_shape.TensorShape([None]).concatenate(value_shape)
    result = array_ops.placeholder(dtype, flat_shape, "flat_values")
    for i in reversed(range(ragged_rank)):
      row_splits = array_ops.placeholder(dtypes.int64, [None],
                                         "row_splits_%d" % i)
      result = ragged_tensor.RaggedTensor.from_row_splits(result, row_splits,
                                                          validate=False)
    return result