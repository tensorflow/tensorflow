# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""CSR Sparse Matrix Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

import six

# pylint: disable=g-direct-tensorflow-import, wildcard-import
from tensorflow.python.eager import context
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.linalg.sparse import gen_sparse_csr_matrix_ops as sm_ops
from tensorflow.python.ops.linalg.sparse.gen_sparse_csr_matrix_ops import *


__all__ = [
    "SparseMatrix",
    "CSRSparseMatrix",
    "matmul",
    "dense_shape_and_type",
]
# pylint: disable=invalid-name
__all__ += [_x for _x in dir(sm_ops) if not _x.startswith("_")]


class DenseShapeAndType(
    collections.namedtuple("DenseShapeAndType", ("shape", "dtype"))):
  pass


def _get_handle_data(tensor):
  return resource_variable_ops.get_eager_safe_handle_data(tensor)


def _create_handle_data_proto(shape_proto, dtype_enum):
  """Create handle data based on shape and dtype protos."""
  variant_shape_and_type_data = \
    cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData()
  variant_shape_and_type_data.is_set = True
  # NOTE(ebrevdo): shape_and_type lacks append() in some versions of protobuf.
  variant_shape_and_type_data.shape_and_type.extend([
      cpp_shape_inference_pb2.CppShapeInferenceResult.HandleShapeAndType(
          shape=shape_proto, dtype=dtype_enum)
  ])
  return variant_shape_and_type_data


def _make_handle_data(tensor):
  """Create handle data based on tensor shape and dtype."""
  return _create_handle_data_proto(tensor.shape.as_proto(),
                                   tensor.dtype.as_datatype_enum)


def get_shape_and_type(matrix):
  """Return matrix's shape and type if available."""
  handle_data = getattr(matrix, "_handle_data", None)
  if handle_data is None:
    return None
  if len(handle_data.shape_and_type) != 1:
    raise ValueError(
        "shape_and_type array in _handle_data must have length one, but saw: %d"
        % len(handle_data.shape_and_type))
  return handle_data.shape_and_type[0]


def dense_shape_and_type(matrix):
  """Get dense shape and dtype of the tf.Tensor containing the matrix.

  Args:
    matrix: A `tf.Tensor` of type `tf.variant` storing a sparse matrix.

  Returns:
    An instance of `ShapeAndType` with properties `shape` (a `tf.TensorShape`)
    and `dtype` (a `tf.DType`).

  Raises:
    TypeError: if `matrix` is not a tensor or its dtype is not variant.
    ValueError: if `matrix` lacks static handle data containing the dense
      shape and dtype.
  """
  if not isinstance(matrix, ops.Tensor):
    raise TypeError("matrix should be a tensor, but saw: %s" % (matrix,))
  if matrix.dtype != dtypes.variant:
    raise TypeError(
        "expected matrix to be type tf.variant, but saw: %s" % (matrix.dtype,))
  handle_data = _get_handle_data(matrix)
  if not handle_data or not handle_data.is_set:
    raise ValueError("matrix has missing handle data: %s" % (matrix,))
  if len(handle_data.shape_and_type) != 1:
    raise ValueError("len(matrix.handle_data.shape_and_type) != 1: '%s'" %
                     (handle_data.shape_and_type,))
  return DenseShapeAndType(
      tensor_shape.TensorShape(handle_data.shape_and_type[0].shape),
      dtypes.DType(handle_data.shape_and_type[0].dtype))


def matmul_shape_inference(a, b, c, transpose_a, transpose_b, adjoint_a,
                           adjoint_b):
  """Helper function for matmul to set the result matrix's handle data."""
  c_handle = getattr(c, "_handle_data", None)
  a_shape_and_type = get_shape_and_type(a)
  b_shape_and_type = get_shape_and_type(b)
  if (c_handle is None and a_shape_and_type is not None and
      b_shape_and_type is not None):

    transpose_a = transpose_a or adjoint_a
    transpose_b = transpose_b or adjoint_b

    a_shape = a_shape_and_type.shape
    b_shape = b_shape_and_type.shape
    rank = len(a_shape.dim)

    # Creates the output shape.
    c_rows = a_shape.dim[rank - (1 if transpose_a else 2)].size
    c_cols = b_shape.dim[rank - (2 if transpose_b else 1)].size
    c_shape = tensor_shape.TensorShape(a_shape)
    c_shape = tensor_shape.TensorShape(c_shape[:rank - 2] + [c_rows, c_cols])
    c_handle = _create_handle_data_proto(c_shape.as_proto(),
                                         a_shape_and_type.dtype)
  return c_handle


def matmul(a,
           b,
           transpose_a=False,
           transpose_b=False,
           adjoint_a=False,
           adjoint_b=False,
           name=None):
  """Perform a sparse matrix matmul between `a` and `b`.

  Performs a contraction between `a` and `b` along the two innermost dimensions.
  If both `a` and `b` are instances of `SparseMatrix`, returns a new instance
  of `SparseMatrix` (same type as `a`).  If one is not an instance of
  `SparseMatrix`, returns a dense `Tensor`:

  ```
  c = opA(a) . opB(b)
  ```
  where `opA` (resp. `opB`) is the transpose or hermitian transpose depending
  on the values of `transpose_a` (resp. `transpose_b`) and `adjoint_a`
  (resp. `adjoint_b`).

  Args:
    a: `Tensor` or `SparseMatrix`, having rank `2` or `3`.
    b: `Tensor` or `SparseMatrix`, having rank `2` or `3`.
    transpose_a: Python `bool`.
    transpose_b: Python `bool`.
    adjoint_a: Python `bool`.
    adjoint_b: Python `bool`.
    name: Optional name to use when creating ops.

  Returns:
    A `SparseMatrix` if both `a` and `b` are instances of `SparseMatrix`,
    otherwise a dense `Tensor`.
  """
  if not isinstance(a, SparseMatrix) and not isinstance(b, SparseMatrix):
    return math_ops.matmul(
        a,
        b,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        adjoint_a=adjoint_a,
        adjoint_b=adjoint_b,
        name=name)

  # pylint: disable=protected-access
  a_matrix = a._matrix if isinstance(a, SparseMatrix) else a
  b_matrix = b._matrix if isinstance(b, SparseMatrix) else b
  with ops.name_scope(name, "SparseMatrixMatMul", [a_matrix, b_matrix]):
    if isinstance(a, SparseMatrix) and isinstance(b, SparseMatrix):
      if not (isinstance(a, type(b)) or isinstance(b, type(a))):
        raise TypeError("SparseMatrix types don't inherit from each other: "
                        "%s and %s" % (type(a), type(b)))
      c = sm_ops.sparse_matrix_sparse_mat_mul(
          a_matrix,
          b_matrix,
          transpose_a=transpose_a,
          transpose_b=transpose_b,
          adjoint_a=adjoint_a,
          adjoint_b=adjoint_b,
          type=a.dtype)

      # In eager mode, shape inference functions are not called, and the output
      # shape is not set. We have to infer the output shape here.
      # TODO(penporn): Set this from the C++ kernel instead.
      c_handle = matmul_shape_inference(a_matrix, b_matrix, c, transpose_a,
                                        transpose_b, adjoint_a, adjoint_b)
      return a._from_matrix(c, handle_data=c_handle)

    elif isinstance(a, SparseMatrix):
      return sm_ops.sparse_matrix_mat_mul(
          a_matrix,
          b,
          transpose_a=transpose_a,
          transpose_b=transpose_b,
          adjoint_a=adjoint_a,
          adjoint_b=adjoint_b)
    else:
      # opA(A) . opB(B) = t(nopB(B) . nopA(A))
      if not adjoint_a and not adjoint_b:
        return sm_ops.sparse_matrix_mat_mul(
            b_matrix,
            a,
            transpose_a=not transpose_b,
            transpose_b=not transpose_a,
            transpose_output=True)
      elif not transpose_a and not transpose_b:
        return sm_ops.sparse_matrix_mat_mul(
            b_matrix,
            a,
            adjoint_a=not adjoint_b,
            adjoint_b=not adjoint_a,
            transpose_output=True,
            conjugate_output=True)
      else:
        return sm_ops.sparse_matrix_mat_mul(
            b_matrix,
            math_ops.conj(a),
            transpose_output=True,
            conjugate_output=adjoint_b)


class SparseMatrix(six.with_metaclass(abc.ABCMeta)):
  """Abstract class for sparse matrix types."""

  @abc.abstractmethod
  def __init__(self):
    self._eager_mode = context.executing_eagerly()

  @abc.abstractproperty
  def _matrix(self):
    pass

  @abc.abstractmethod
  def _from_matrix(self, matrix, handle_data=None):
    pass

  @abc.abstractmethod
  def to_dense(self):
    pass

  @abc.abstractmethod
  def to_sparse_tensor(self):
    pass

  @property
  def graph(self):
    return self._matrix.graph

  @property
  def shape(self):
    return dense_shape_and_type(self._matrix).shape

  @property
  def dtype(self):
    return dense_shape_and_type(self._matrix).dtype

  @property
  def eager_handle_data(self):
    """Return the matrix's handle data iff in eager mode."""
    return _get_handle_data(self._matrix) if self._eager_mode else None

  def conj(self):
    return self._from_matrix(
        math_ops.conj(self._matrix), self.eager_handle_data)

  def hermitian_transpose(self):
    """Return the hermitian transpose of the matrix."""
    return self._from_matrix(
        sm_ops.sparse_matrix_transpose(
            self._matrix, conjugate=True, type=self.dtype),
        self.eager_handle_data)

  def nnz(self):
    """Number of stored values, including explicit zeros."""
    return sm_ops.sparse_matrix_nnz(self._matrix)

  nonzero = nnz

  def sorted_indices(self):
    # TODO(ebrevdo): A more efficient implementation?
    return self.to_sparse_tensor().indices

  def transpose(self):
    return self._from_matrix(
        sm_ops.sparse_matrix_transpose(self._matrix, type=self.dtype),
        self.eager_handle_data)


class CSRSparseMatrix(SparseMatrix):
  """(Optionally batched) CSR Sparse Matrix."""

  def __init__(self, value, indices=None, name=None):
    """Construct a CSRSparseMatrix from a dense matrix or SparseTensor.

    Args:
      value: A dense `2D` or `3D` Tensor or `SparseTensor`.
      indices: The nonzero indices of `value`
        (if `value` is not a `SparseTensor`).
      name: Optional op name.

    Raises:
      ValueError: if `value` is a `SparseTensor` and `indices` is not `None`.
    """
    super(CSRSparseMatrix, self).__init__()
    if isinstance(value, sparse_tensor.SparseTensor):
      if indices is not None:
        raise ValueError("indices must be None if value is a SparseTensor.")
      self._dtype = value.dtype
      self._csr_matrix = sm_ops.sparse_tensor_to_csr_sparse_matrix(
          indices=value.indices,
          values=value.values,
          dense_shape=value.dense_shape)
    else:
      value = ops.convert_to_tensor(value)
      self._dtype = value.dtype
      if indices is not None:
        indices = ops.convert_to_tensor(indices, dtype=dtypes.int64)
      else:
        indices = array_ops.stop_gradient(array_ops.where(value))
      self._csr_matrix = sm_ops.dense_to_csr_sparse_matrix(value, indices)

    # Eager mode doesn't call shape inference functions, so we have to set the
    # shape and dtype handle data directly.
    if self._eager_mode:
      # pylint: disable=protected-access
      self._csr_matrix._handle_data = _make_handle_data(value)
      # pylint: enable=protected-access

  @property
  def _matrix(self):
    return self._csr_matrix

  def _from_matrix(self, matrix, handle_data=None):
    assert isinstance(matrix, ops.Tensor) and matrix.dtype == dtypes.variant
    ret = type(self).__new__(type(self))
    # pylint: disable=protected-access
    ret._dtype = self._dtype
    if self._eager_mode:
      if matrix._handle_data is None:
        matrix._handle_data = handle_data
      assert matrix._handle_data is not None
    ret._csr_matrix = matrix
    # pylint: enable=protected-access
    return ret

  def to_dense(self):
    return sm_ops.csr_sparse_matrix_to_dense(self._matrix, type=self.dtype)

  def to_sparse_tensor(self):
    r = sm_ops.csr_sparse_matrix_to_sparse_tensor(self._matrix, type=self.dtype)
    return sparse_tensor.SparseTensor(
        indices=r.indices, values=r.values, dense_shape=r.dense_shape)
