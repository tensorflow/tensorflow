# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Python layer for set_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python.framework import tensor_util

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import resource_loader


_set_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile("_set_ops.so"))
assert _set_ops, "Could not load _set_ops.so."


_VALID_DTYPES = set([
    dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64,
    dtypes.uint8, dtypes.uint16, dtypes.string])


@ops.RegisterShape("SetSize")
def _size_shape(unused_op):
  """Shape function for SetSize op."""
  return [tensor_shape.unknown_shape()]


def set_size(a, validate_indices=True):
  """Compute number of unique elements along last dimension of `a`.

  Args:
    a: `SparseTensor`, with indices sorted in row-major order.
    validate_indices: Whether to validate the order and range of sparse indices
       in `a`.

  Returns:
    For `a` ranked `n`, this is a `Tensor` with rank `n-1`, and the same 1st
    `n-1` dimensions as `a`. Each value is the number of unique elements in
    the corresponding `[0...n-1]` dimension of `a`.

  Raises:
    TypeError: If `a` is an invalid types.
  """
  a = tensor_util.convert_to_tensor_or_sparse_tensor(a, name="a")
  if not isinstance(a, ops.SparseTensor):
    raise TypeError("Expected `SparseTensor`, got %s." % a)
  if a.values.dtype.base_dtype not in _VALID_DTYPES:
    raise TypeError("Invalid dtype %s." % a.values.dtype)
  # pylint: disable=protected-access
  return _set_ops.set_size(a.indices, a.values, a.shape, validate_indices)

ops.NoGradient("SetSize")


@ops.RegisterShape("DenseToDenseSetOperation")
def _dense_to_dense_shape(op):
  """Shapes for `SparseTensor` result given 2 dense inputs.

  Args:
    op: Operation with 2 dense `Tensor` inputs.

  Returns:
    Tuple of three shapes corresponding to the indices, values, and shape
    `Tensor` components of the result `SparseTensor`.

  Raises:
    ValueError: if either input `Tensor` has rank < 2, or ranks do not match, or
    first n-1 dims of input shapes are not compatible.
  """
  # The following should stay in sync with `ComputeDenseToDense` shape
  # assertions in kernels/set_kernels.cc.
  input0_shape = op.inputs[0].get_shape()
  input0_rank = input0_shape.ndims
  if (input0_rank is not None) and (input0_rank < 2):
    raise ValueError("Input 0, expected rank >= 2, got shape %s." %
                     input0_shape)
  # Dimension n contains the set values to be compared, so ranks and the first
  # n-1 dimensions of inputs and output must match.
  input1_shape = op.inputs[1].get_shape()
  input1_rank = input1_shape.ndims
  if (input0_rank is not None) and (input1_rank is not None) and (
      input0_rank != input1_rank):
    raise ValueError(
        "Ranks do not match: input 0 with shape %s, input 1 with shape %s." %
        (input0_shape, input1_shape))
  output_rank = input1_rank if input0_rank is None else input0_rank
  output_dim0 = input1_shape[1] if input0_shape[0] is None else input0_shape[0]
  input0_dims = input0_shape.dims
  if input0_dims is None:
    group0_shape = tensor_shape.unknown_shape()
  else:
    group0_shape = tensor_shape.TensorShape(input0_dims[:-1])
  input1_dims = input1_shape.dims
  if input1_dims is None:
    group1_shape = tensor_shape.unknown_shape()
  else:
    group1_shape = tensor_shape.TensorShape(input1_dims[:-1])
  group0_shape.assert_is_compatible_with(group1_shape)

  indices_shape = tensor_shape.TensorShape((output_dim0, output_rank))
  values_shape = tensor_shape.unknown_shape(1)
  shape_shape = tensor_shape.TensorShape((output_rank,))
  return (indices_shape, values_shape, shape_shape)


@ops.RegisterShape("DenseToSparseSetOperation")
def _dense_to_sparse_shape(op):
  """Shapes for `SparseTensor` result given 1 dense input and 1 sparse input.

  Args:
    op: Operation with 1 dense `Tensor` and 1 `SparseTensor` input.

  Returns:
    Tuple of three shapes corresponding to the indices, values, and shape
    `Tensor` components of the result `SparseTensor`.

  Raises:
    ValueError: if either input `Tensor` has rank < 2.
  """
  # The following should stay in sync with `ComputeDenseToSparse` shape
  # assertions in kernels/set_kernels.cc.
  input_shape = op.inputs[0].get_shape()
  input_rank = input_shape.ndims
  if (input_rank is not None) and (input_rank < 2):
    raise ValueError("Expected rank >= 2, got %s." % input_shape)
  # Assert valid dimensions for the 3 `Tensor` components of `SparseTensor`.
  ops.SparseTensor(op.inputs[1], op.inputs[2], op.inputs[3])

  indices_shape = tensor_shape.TensorShape((input_shape[0], input_rank))
  values_shape = tensor_shape.unknown_shape(1)
  shape_shape = tensor_shape.TensorShape((input_rank,))
  return (indices_shape, values_shape, shape_shape)


@ops.RegisterShape("SparseToSparseSetOperation")
def _sparse_to_sparse_shape(op):
  """Shapes for `SparseTensor` result given 2 sparse inputs.

  Args:
    op: Operation with 2 `SparseTensor` inputs.

  Returns:
    Tuple of three shapes corresponding to the indices, values, and shape
    `Tensor` components of the result `SparseTensor`.
  """
  # The following should stay in sync with `ComputeSparseToSparse` shape
  # assertions in kernels/set_kernels.cc.
  # Assert valid dimensions for the 3 `Tensor` components of `SparseTensor`.
  ops.SparseTensor(op.inputs[0], op.inputs[1], op.inputs[2])
  ops.SparseTensor(op.inputs[3], op.inputs[4], op.inputs[5])

  indices_shape = tensor_shape.unknown_shape(2)
  values_shape = tensor_shape.unknown_shape(1)
  shape_shape = tensor_shape.unknown_shape(1)
  return (indices_shape, values_shape, shape_shape)


ops.NoGradient("DenseToDenseSetOperation")
ops.NoGradient("DenseToSparseSetOperation")
ops.NoGradient("SparseToSparseSetOperation")


def _set_operation(a, b, set_operation, validate_indices=True):
  """Compute set operation of elements in last dimension of `a` and `b`.

  All but the last dimension of `a` and `b` must match.

  Args:
    a: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices
        must be sorted in row-major order.
    b: `Tensor` or `SparseTensor` of the same type as `a`. Must be
        `SparseTensor` if `a` is `SparseTensor`. If sparse, indices must be
        sorted in row-major order.
    set_operation: String indicating set operaiton. See
        SetOperationOp::SetOperationFromContext for valid values.
    validate_indices: Whether to validate the order and range of sparse indices
       in `a` and `b`.

  Returns:
    A `SparseTensor` with the same rank as `a` and `b`, and all but the last
    dimension the same. Elements along the last dimension contain the results
    of the set operation.

  Raises:
    TypeError: If inputs are invalid types.
    ValueError: If `a` is sparse and `b` is dense.
  """
  a = tensor_util.convert_to_tensor_or_sparse_tensor(a, name="a")
  if a.dtype.base_dtype not in _VALID_DTYPES:
    raise TypeError("'a' invalid dtype %s." % a.dtype)
  b = tensor_util.convert_to_tensor_or_sparse_tensor(b, name="b")
  if b.dtype.base_dtype != a.dtype.base_dtype:
    raise TypeError("Types don't match, %s vs %s." % (a.dtype, b.dtype))
  # pylint: disable=protected-access
  if isinstance(a, ops.SparseTensor):
    if isinstance(b, ops.SparseTensor):
      indices, values, shape = _set_ops.sparse_to_sparse_set_operation(
          a.indices, a.values, a.shape, b.indices, b.values, b.shape,
          set_operation, validate_indices)
    else:
      raise ValueError("Sparse,Dense is not supported, but Dense,Sparse is. "
                       "Please flip the order of your inputs.")
  elif isinstance(b, ops.SparseTensor):
    indices, values, shape = _set_ops.dense_to_sparse_set_operation(
        a, b.indices, b.values, b.shape, set_operation, validate_indices)
  else:
    indices, values, shape = _set_ops.dense_to_dense_set_operation(
        a, b, set_operation, validate_indices)
  # pylint: enable=protected-access
  return ops.SparseTensor(indices, values, shape)


def set_intersection(a, b, validate_indices=True):
  """Compute set intersection of elements in last dimension of `a` and `b`.

  All but the last dimension of `a` and `b` must match.

  Args:
    a: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices
        must be sorted in row-major order.
    b: `Tensor` or `SparseTensor` of the same type as `a`. Must be
        `SparseTensor` if `a` is `SparseTensor`. If sparse, indices must be
        sorted in row-major order.
    validate_indices: Whether to validate the order and range of sparse indices
       in `a` and `b`.

  Returns:
    A `SparseTensor` with the same rank as `a` and `b`, and all but the last
    dimension the same. Elements along the last dimension contain the
    intersections.
  """
  return _set_operation(a, b, "intersection", validate_indices)


def set_difference(a, b, aminusb=True, validate_indices=True):
  """Compute set difference of elements in last dimension of `a` and `b`.

  All but the last dimension of `a` and `b` must match.

  Args:
    a: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices
        must be sorted in row-major order.
    b: `Tensor` or `SparseTensor` of the same type as `a`. Must be
        `SparseTensor` if `a` is `SparseTensor`. If sparse, indices must be
        sorted in row-major order.
    aminusb: Whether to subtract `b` from `a`, vs vice versa.
    validate_indices: Whether to validate the order and range of sparse indices
       in `a` and `b`.

  Returns:
    A `SparseTensor` with the same rank as `a` and `b`, and all but the last
    dimension the same. Elements along the last dimension contain the
    differences.
  """
  return _set_operation(a, b, "a-b" if aminusb else "b-a", validate_indices)


def set_union(a, b, validate_indices=True):
  """Compute set union of elements in last dimension of `a` and `b`.

  All but the last dimension of `a` and `b` must match.

  Args:
    a: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices
        must be sorted in row-major order.
    b: `Tensor` or `SparseTensor` of the same type as `a`. Must be
        `SparseTensor` if `a` is `SparseTensor`. If sparse, indices must be
        sorted in row-major order.
    validate_indices: Whether to validate the order and range of sparse indices
       in `a` and `b`.

  Returns:
    A `SparseTensor` with the same rank as `a` and `b`, and all but the last
    dimension the same. Elements along the last dimension contain the
    unions.
  """
  return _set_operation(a, b, "union", validate_indices)
