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
"""CSR sparse matrix tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import sparse

from tensorflow.core.framework import tensor_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

CPU = "/device:CPU:0"
GPU = "/device:GPU:0"


def dense_to_csr_sparse_matrix(dense):
  dense_t = ops.convert_to_tensor(dense)
  locs = array_ops.stop_gradient(array_ops.where(math_ops.abs(dense_t) > 0))
  return sparse_csr_matrix_ops.dense_to_csr_sparse_matrix(dense_t, locs)


def _swap(a, i, j):
  a[i], a[j] = a[j], a[i]


def twist_matrix(matrix, permutation_indices):
  """Permute the rows and columns of a 2D or (batched) 3D Tensor."""
  # Shuffle the rows and columns with the same permutation.
  if matrix.shape.ndims == 2:
    # Invert the permutation since `tf.gather` and `tf.gather_nd` need the
    # mapping from each index `i` to the index that maps to `i`.
    permutation_indices_inv = array_ops.invert_permutation(permutation_indices)
    matrix = array_ops.gather(matrix, permutation_indices_inv, axis=0)
    matrix = array_ops.gather(matrix, permutation_indices_inv, axis=1)
  elif matrix.shape.ndims == 3:
    permutation_indices_inv = map_fn.map_fn(array_ops.invert_permutation,
                                            permutation_indices)
    # For 3D Tensors, it's easy to shuffle the rows but not the columns. We
    # permute the rows, transpose, permute the rows again, and transpose back.
    batch_size = matrix.shape[0]
    batch_indices = array_ops.broadcast_to(
        math_ops.range(batch_size)[:, None], permutation_indices.shape)
    for _ in range(2):
      matrix = array_ops.gather_nd(
          matrix,
          array_ops.stack([batch_indices, permutation_indices_inv], axis=-1))
      # Transpose the matrix, or equivalently, swap dimensions 1 and 2.
      matrix = array_ops.transpose(matrix, perm=[0, 2, 1])
  else:
    raise ValueError("Input matrix must have rank 2 or 3. Got: {}".format(
        matrix.shape.ndims))

  return matrix


class CSRSparseMatrixOpsTest(test.TestCase):

  @classmethod
  def setUpClass(cls):  # pylint: disable=g-missing-super-call
    cls._gpu_available = test_util.is_gpu_available()

  # TODO(ebrevdo): This will work once we find a way to get rendezvous
  # working for CSRSparseMatrix and can remove the HostMemory
  # annotations for the other ops.
  @test_util.run_in_graph_and_eager_modes
  def DISABLEDtestFromProto(self):
    if not self._gpu_available:
      return

    a_indices = np.array([[0, 0], [2, 3]])
    a_values = np.asarray([1.0, 5.0], dtype=np.float32)
    a_dense_shape = np.asarray([5, 6], dtype=np.int64)
    a_sparse_mat = sparse.coo_matrix(
        (a_values, (a_indices[:, 0], a_indices[:, 1])), shape=a_dense_shape)
    a_csr_mat = a_sparse_mat.tocsr()
    a_col_inds = a_csr_mat.indices
    a_row_ptrs = a_csr_mat.indptr

    # Format of SparseMatrix:
    #  type_name == "tensorflow::CSRSparseMatrix"
    #  metadata == b (validated)
    #  tensors == [dense_shape, row_ptrs, col_indices, values]
    dense_shape_proto = tensor_util.make_tensor_proto(a_dense_shape)
    row_ptrs_proto = tensor_util.make_tensor_proto(a_row_ptrs)
    col_inds_proto = tensor_util.make_tensor_proto(a_col_inds)
    values_proto = tensor_util.make_tensor_proto(a_values)
    variant_tensor_data = tensor_pb2.VariantTensorDataProto(
        type_name="tensorflow::CSRSparseMatrix",
        metadata=np.asarray(True).tobytes(),
        tensors=[
            dense_shape_proto, row_ptrs_proto, col_inds_proto, values_proto
        ])
    tensor_proto = tensor_pb2.TensorProto(
        dtype=dtypes.variant.as_datatype_enum,
        tensor_shape=tensor_shape.TensorShape([]).as_proto())
    tensor_proto.variant_val.extend([variant_tensor_data])
    a_sm = constant_op.constant(tensor_proto)
    a_rt = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
        a_sm, type=dtypes.float32)
    self.evaluate(a_rt)

  @test_util.run_in_graph_and_eager_modes
  def testSparseTensorConversion(self):
    a_indices = np.array([[0, 0], [2, 3], [2, 4], [3, 0]])
    a_values = [1.0, 5.0, -1.0, -2.0]
    a_dense_shape = [5, 6]
    a_sparse_mat = sparse.coo_matrix(
        (a_values, (a_indices[:, 0], a_indices[:, 1])), shape=a_dense_shape)
    a_csr_mat = a_sparse_mat.tocsr()

    # Convert 2D SparseTensor to CSR Matrix
    a_st = sparse_tensor.SparseTensor(a_indices, a_values, a_dense_shape)
    a_st = math_ops.cast(a_st, dtypes.float32)
    a_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
        a_st.indices, a_st.values, a_st.dense_shape)

    # Get row indices and columns for batch 0.
    a_sm_row_ptrs, a_sm_col_inds, a_sm_values = (
        sparse_csr_matrix_ops.csr_sparse_matrix_components(
            a_sm, 0, type=a_st.dtype))

    a_sm_row_ptrs_values, a_sm_col_inds_values, a_sm_values_values = (
        self.evaluate((a_sm_row_ptrs, a_sm_col_inds, a_sm_values)))

    self.assertAllEqual(a_csr_mat.indices, a_sm_col_inds_values)
    self.assertAllEqual(a_csr_mat.indptr, a_sm_row_ptrs_values)
    self.assertAllClose(a_values, a_sm_values_values)

    # Convert CSR Matrix to 2D SparseTensor
    a_st_rt = sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(
        a_sm, type=a_st.dtype)
    a_st_rt_value = self.evaluate(a_st_rt)

    self.assertAllEqual(a_indices, a_st_rt_value.indices)
    self.assertAllClose(a_values, a_st_rt_value.values)
    self.assertAllEqual(a_dense_shape, a_st_rt_value.dense_shape)

  # TODO(b/139491352): Add handle_data propagation to array_ops.identity.
  @test_util.run_deprecated_v1
  def testCSRSparseMatrixResourceVariable(self):
    if not self._gpu_available:
      return

    sparsify = lambda m: m * (m > 0)
    dense_shape = [53, 65, 127]
    a_mats = sparsify(np.random.randn(*dense_shape)).astype(np.float32)

    a_sm = dense_to_csr_sparse_matrix(a_mats)
    with ops.device("/gpu:0"):
      v = variable_scope.get_variable("sm", initializer=a_sm, use_resource=True)
      v_id = array_ops.identity(v)
      self.assertEqual(
          sparse_csr_matrix_ops.dense_shape_and_type(v_id).shape, a_mats.shape)
      a_rt = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
          v, type=dtypes.float32)
    v_reassign = state_ops.assign(v, v_id).op
    with self.assertRaisesOpError("uninitialized"):
      self.evaluate(a_rt)
    self.evaluate(v.initializer)
    a_rt_value = self.evaluate(a_rt)
    self.assertAllClose(a_mats, a_rt_value)
    self.evaluate(v_reassign)
    a_rt_reassigned_value = self.evaluate(a_rt)
    self.assertAllClose(a_mats, a_rt_reassigned_value)

  @test_util.run_in_graph_and_eager_modes
  def testBatchSparseTensorConversion(self):
    a_indices = np.array([[0, 0, 0], [0, 2, 3], [2, 0, 1]])
    a_values = [1.0, 5.0, 6.0]
    a_dense_shape = [3, 5, 6]
    a_sparse_mats = [
        sparse.coo_matrix(([1.0, 5.0], ([0, 2], [0, 3])),
                          shape=a_dense_shape[1:]),
        sparse.coo_matrix(([], ([], [])), shape=a_dense_shape[1:]),
        sparse.coo_matrix(([6.0], ([0], [1])), shape=a_dense_shape[1:])
    ]
    a_csr_mats = [m.tocsr() for m in a_sparse_mats]

    # Convert 3D SparseTensor to CSR Matrix
    a_st = sparse_tensor.SparseTensor(a_indices, a_values, a_dense_shape)
    a_st = math_ops.cast(a_st, dtypes.float32)
    a_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
        a_st.indices, a_st.values, a_st.dense_shape)

    # Get row indices and columns for batches.
    a_sm_components = [
        sparse_csr_matrix_ops.csr_sparse_matrix_components(
            a_sm, i, type=a_st.dtype) for i in range(3)
    ]

    a_sm_values = self.evaluate(a_sm_components)

    for i, (a_sm_val, a_csr_mat) in enumerate(zip(a_sm_values, a_csr_mats)):
      tf_logging.info("Comparing batch %d" % i)
      self.assertAllEqual(a_csr_mat.indptr, a_sm_val.row_ptrs)
      self.assertAllEqual(a_csr_mat.indices, a_sm_val.col_inds)
      self.assertAllClose(a_csr_mat.data, a_sm_val.values)

    # Convert CSR batched Matrix to 3D SparseTensor
    a_st_rt = sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(
        a_sm, type=a_st.dtype)
    a_st_rt_value = self.evaluate(a_st_rt)

    self.assertAllEqual(a_indices, a_st_rt_value.indices)
    self.assertAllClose(a_values, a_st_rt_value.values)
    self.assertAllEqual(a_dense_shape, a_st_rt_value.dense_shape)

  @test_util.run_in_graph_and_eager_modes
  def testLargeBatchSparseTensorConversion(self):
    # Test two sets of conversions to check behavior of the ops in a
    # concurrent environment (parallel executions of the ST -> SM ops).

    sparsify = lambda m: m * (m > 0)
    dense_shape = [53, 65, 127]

    mats = [
        sparsify(np.random.randn(*dense_shape)).astype(np.float32)
        for _ in range(2)
    ]
    csr_mats = [list(map(sparse.csr_matrix, mat)) for mat in mats]
    mats_t = [ops.convert_to_tensor(mat) for mat in mats]
    mats_locs = [array_ops.where(mat_t > 0) for mat_t in mats_t]
    sparse_tensors = list()
    for mat_t, mat_loc in zip(mats_t, mats_locs):
      sparse_tensors.append(
          sparse_tensor.SparseTensor(mat_loc,
                                     array_ops.gather_nd(mat_t,
                                                         mat_loc), dense_shape))
    sparse_matrices = [
        sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
            st.indices, st.values, st.dense_shape) for st in sparse_tensors
    ]
    sm_nnz = [
        sparse_csr_matrix_ops.sparse_matrix_nnz(sm) for sm in sparse_matrices
    ]

    # Get row indices and columns for batches.
    sm_components = list()
    for sm in sparse_matrices:
      sm_components.append([
          sparse_csr_matrix_ops.csr_sparse_matrix_components(
              sm, i, type=dtypes.float32) for i in range(dense_shape[0])
      ])

    sm_nnz_values, sm_values = self.evaluate((sm_nnz, sm_components))

    for i, (sm_values_i, csr_mats_i) in enumerate(zip(sm_values, csr_mats)):
      for b, (sm_val, csr_mat) in enumerate(zip(sm_values_i, csr_mats_i)):
        tf_logging.info("Comparing matrix %d batch %d" % (i, b))
        self.assertEqual(csr_mat.nnz, sm_nnz_values[i][b])
        self.assertAllEqual(csr_mat.indptr, sm_val.row_ptrs)
        self.assertAllEqual(csr_mat.indices, sm_val.col_inds)
        self.assertAllClose(csr_mat.data, sm_val.values)

    # Convert CSR batched Matrix to 3D SparseTensor
    st_rt = [
        sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(
            sm, type=dtypes.float32) for sm in sparse_matrices
    ]

    st_values, st_rt_values = self.evaluate((sparse_tensors, st_rt))

    for (st_value, st_rt_value) in zip(st_values, st_rt_values):
      self.assertAllEqual(st_value.indices, st_rt_value.indices)
      self.assertAllClose(st_value.values, st_rt_value.values)
      self.assertAllEqual(dense_shape, st_rt_value.dense_shape)

  @test_util.run_in_graph_and_eager_modes
  def testDenseConversion(self):
    a_indices = np.array([[0, 0], [2, 3], [2, 4], [3, 0]])
    a_values = np.array([1.0, 5.0, -1.0, -2.0]).astype(np.float32)
    a_dense_shape = [5, 6]
    a_sparse_mat = sparse.coo_matrix(
        (a_values, (a_indices[:, 0], a_indices[:, 1])), shape=a_dense_shape)
    a_csr_mat = a_sparse_mat.tocsr()
    a_dense = a_sparse_mat.todense()

    # Convert 2D SparseTensor to CSR Matrix
    a_sm = dense_to_csr_sparse_matrix(a_dense)

    # Get row indices and columns for batch 0.
    a_sm_row_ptrs, a_sm_col_inds, a_sm_values = (
        sparse_csr_matrix_ops.csr_sparse_matrix_components(
            a_sm, 0, type=dtypes.float32))

    a_sm_row_ptrs_values, a_sm_col_inds_values, a_sm_values_values = (
        self.evaluate((a_sm_row_ptrs, a_sm_col_inds, a_sm_values)))

    self.assertAllEqual(a_csr_mat.indices, a_sm_col_inds_values)
    self.assertAllEqual(a_csr_mat.indptr, a_sm_row_ptrs_values)
    self.assertAllClose(a_values, a_sm_values_values)

    # Convert CSR Matrix to 2D dense matrix
    a_rt = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
        a_sm, dtypes.float32)
    a_rt_value = self.evaluate(a_rt)

    self.assertAllEqual(a_dense, a_rt_value)

  @test_util.run_in_graph_and_eager_modes
  def testBatchDenseConversion(self):
    a_dense_shape = [4, 5, 6]
    a_sparse_mats = [
        sparse.coo_matrix(([1.0, 5.0], ([0, 2], [0, 3])),
                          shape=a_dense_shape[1:]),
        sparse.coo_matrix(([], ([], [])), shape=a_dense_shape[1:]),
        sparse.coo_matrix(([6.0], ([0], [1])), shape=a_dense_shape[1:]),
        sparse.coo_matrix(([], ([], [])), shape=a_dense_shape[1:]),
    ]
    a_csr_mats = [m.tocsr() for m in a_sparse_mats]
    a_dense = np.asarray([m.todense() for m in a_sparse_mats], dtype=np.float32)

    # Convert 3D SparseTensor to CSR Matrix
    a_sm = dense_to_csr_sparse_matrix(a_dense)

    # Get row indices and columns for batches.
    a_sm_components = [
        sparse_csr_matrix_ops.csr_sparse_matrix_components(
            a_sm, i, type=dtypes.float32) for i in range(3)
    ]

    a_sm_values = self.evaluate(a_sm_components)

    for i, (a_sm_val, a_csr_mat) in enumerate(zip(a_sm_values, a_csr_mats)):
      tf_logging.info("Comparing batch %d" % i)
      self.assertAllEqual(a_csr_mat.indptr, a_sm_val.row_ptrs)
      self.assertAllEqual(a_csr_mat.indices, a_sm_val.col_inds)
      self.assertAllClose(a_csr_mat.data, a_sm_val.values)

    # Convert CSR batched Matrix to 3D SparseTensor
    a_rt = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
        a_sm, type=dtypes.float32)
    a_rt_value = self.evaluate(a_rt)

    self.assertAllEqual(a_dense, a_rt_value)

  @test_util.run_in_graph_and_eager_modes
  def testLargeBatchDenseConversion(self):
    # Test two sets of conversions to check behavior of the ops in a
    # concurrent environment (parallel executions of the ST -> SM
    # ops).

    sparsify = lambda m: m * (m > 0)
    dense_shape = [53, 65, 127]

    mats = [
        sparsify(np.random.randn(*dense_shape)).astype(np.float32)
        for _ in range(2)
    ]
    csr_mats = [[sparse.csr_matrix(m) for m in mat] for mat in mats]
    mats_t = [ops.convert_to_tensor(mat) for mat in mats]
    mats_locs = [array_ops.where(mat_t > 0) for mat_t in mats_t]
    sparse_matrices = [
        sparse_csr_matrix_ops.dense_to_csr_sparse_matrix(mat, mat_loc)
        for (mat, mat_loc) in zip(mats_t, mats_locs)
    ]
    sm_nnz = [
        sparse_csr_matrix_ops.sparse_matrix_nnz(sm) for sm in sparse_matrices
    ]

    # Get row indices and columns for batches.
    sm_components = []
    for sm in sparse_matrices:
      sm_components.append([
          sparse_csr_matrix_ops.csr_sparse_matrix_components(
              sm, i, type=dtypes.float32) for i in range(dense_shape[0])
      ])

    sm_nnz_values, sm_values = self.evaluate((sm_nnz, sm_components))

    for i, (sm_values_i, csr_mats_i) in enumerate(zip(sm_values, csr_mats)):
      for b, (sm_val, csr_mat) in enumerate(zip(sm_values_i, csr_mats_i)):
        tf_logging.info("Comparing matrix %d batch %d" % (i, b))
        self.assertEqual(csr_mat.nnz, sm_nnz_values[i][b])
        self.assertAllEqual(csr_mat.indptr, sm_val.row_ptrs)
        self.assertAllEqual(csr_mat.indices, sm_val.col_inds)
        self.assertAllClose(csr_mat.data, sm_val.values)

    # Convert CSR batched Matrix to 3D dense tensor
    sm_rt = [
        sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
            sm, type=dtypes.float32) for sm in sparse_matrices
    ]

    sm_rt_values = self.evaluate(sm_rt)

    for (mat, sm_rt_value) in zip(mats, sm_rt_values):
      self.assertAllEqual(mat, sm_rt_value)

  @test_util.run_in_graph_and_eager_modes
  def testSparseMatrixAdd(self):
    if not self._gpu_available:
      return

    a_indices = np.array([[0, 0], [2, 3]])
    a_values = np.array([1.0, 5.0]).astype(np.float32)
    a_dense_shape = [5, 6]
    a_sparse_mat = sparse.coo_matrix(
        (a_values, (a_indices[:, 0], a_indices[:, 1])), shape=a_dense_shape)
    a_dense = a_sparse_mat.todense()

    b_indices = np.array([[1, 0], [1, 4], [2, 3], [4, 1]])
    b_values = np.array([1.0, 0.5, -5.0, 2.0]).astype(np.float32)
    b_dense_shape = [5, 6]
    b_sparse_mat = sparse.coo_matrix(
        (b_values, (b_indices[:, 0], b_indices[:, 1])), shape=b_dense_shape)
    b_dense = b_sparse_mat.todense()

    for (alpha, beta) in [(1.0, 1.0), (1.0, -1.0), (0.25, 0.5)]:
      a_sum_b_sparse_mat = alpha * a_sparse_mat + beta * b_sparse_mat

      # Convert 2D SparseTensor to CSR Matrix
      a_sm = dense_to_csr_sparse_matrix(a_dense)
      b_sm = dense_to_csr_sparse_matrix(b_dense)
      alpha = np.float32(alpha)
      beta = np.float32(beta)
      c_sm = sparse_csr_matrix_ops.sparse_matrix_add(
          a_sm, b_sm, alpha=alpha, beta=beta)
      c_dense = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
          c_sm, dtypes.float32)
      c_dense_value = self.evaluate(c_dense)

      self.assertAllClose(a_sum_b_sparse_mat.todense(), c_dense_value)

  @test_util.run_in_graph_and_eager_modes
  def testLargeBatchSparseMatrixAdd(self):
    if not self._gpu_available:
      return

    sparsify = lambda m: m * (m > 0)
    dense_shape = [53, 65, 127]
    a_mats = sparsify(np.random.randn(*dense_shape)).astype(np.float32)
    b_mats = sparsify(np.random.randn(*dense_shape)).astype(np.float32)
    for (alpha, beta) in [(1.0, 1.0), (1.0, -1.0), (0.25, 0.5)]:
      tf_logging.info("testLargeBatchSparseMatrixAdd, comparing "
                      "alpha, beta (%d, %d)" % (alpha, beta))
      a_sm = dense_to_csr_sparse_matrix(a_mats)
      b_sm = dense_to_csr_sparse_matrix(b_mats)
      alpha = np.float32(alpha)
      beta = np.float32(beta)
      c_sm = sparse_csr_matrix_ops.sparse_matrix_add(
          a_sm, b_sm, alpha=alpha, beta=beta)
      c_dense = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
          c_sm, dtypes.float32)
      c_dense_value = self.evaluate(c_dense)

      self.assertAllClose(c_dense_value, alpha * a_mats + beta * b_mats)

  @test_util.run_in_graph_and_eager_modes
  def testSparseMatrixMatMul(self):
    for shapes in [[(5, 6), (6, 1)], [(5, 6), (6, 2)]]:
      a_indices = np.array([[0, 0], [2, 3]])
      a_values = np.array([1.0, 5.0]).astype(np.float32)
      a_dense_shape = shapes[0]
      a_sparse_mat = sparse.coo_matrix(
          (a_values, (a_indices[:, 0], a_indices[:, 1])), shape=a_dense_shape)
      a_dense = a_sparse_mat.todense()

      # Will multiply sparse a (shape=shapes[0]) by dense b (shape=shapes[1]).
      b = np.random.randn(*shapes[1]).astype(np.float32)

      a_sm = dense_to_csr_sparse_matrix(a_dense)
      c = sparse_csr_matrix_ops.sparse_matrix_mat_mul(a=a_sm, b=b)
      c_value = self.evaluate(c)

      expected_c_value = a_sparse_mat.dot(b)
      self.assertAllClose(expected_c_value, c_value)

  @test_util.run_in_graph_and_eager_modes
  def testSparseMatrixMatMulConjugateOutput(self):
    for shapes in [[(5, 6), (6, 1)], [(5, 6), (6, 2)]]:
      a_indices = np.array([[0, 0], [2, 3]])
      a_values = np.array([1.0 + 1.j, 5.0 - 2.j]).astype(np.complex64)
      a_dense_shape = shapes[0]
      a_sparse_mat = sparse.coo_matrix(
          (a_values, (a_indices[:, 0], a_indices[:, 1])), shape=a_dense_shape)
      a_dense = a_sparse_mat.todense()

      # Will multiply sparse a (shape=shapes[0]) by dense b (shape=shapes[1]).
      b = np.random.randn(*shapes[1]).astype(np.complex64)

      a_sm = dense_to_csr_sparse_matrix(a_dense)
      c = sparse_csr_matrix_ops.sparse_matrix_mat_mul(
          a=a_sm, b=b, conjugate_output=True)
      c_value = self.evaluate(c)

      expected_c_value = self.evaluate(
          math_ops.conj(test_util.matmul_without_tf32(a_dense, b)))
      self.assertAllClose(expected_c_value, c_value)

  @test_util.run_in_graph_and_eager_modes
  def testLargeBatchSparseMatrixMatMul(self):
    dtypes_to_test = [np.float32, np.complex64]
    sparsify = lambda m: m * (m > 0)
    for dtype in dtypes_to_test:
      for (transpose_a, transpose_b) in ((False, False), (False, True),
                                         (True, False), (True, True)):
        for (adjoint_a, adjoint_b) in ((False, False), (False, True),
                                       (True, False), (True, True)):
          if (transpose_a and adjoint_a) or (transpose_b and adjoint_b):
            continue
          for shapes in [[[53, 127, 65], [53, 65, 1]],
                         [[53, 127, 1], [53, 1, 65]],
                         [[53, 127, 65], [53, 65, 127]]]:
            a_dense_shape = shapes[0]
            b_dense_shape = shapes[1]
            if transpose_a or adjoint_a:
              _swap(a_dense_shape, -2, -1)
            if transpose_b or adjoint_b:
              _swap(b_dense_shape, -2, -1)
            a_mats = sparsify(
                (np.random.randn(*a_dense_shape) +
                 1.j * np.random.randn(*a_dense_shape))).astype(dtype)
            b_mats = (np.random.randn(*b_dense_shape) +
                      1.j * np.random.randn(*b_dense_shape)).astype(dtype)
            tf_logging.info(
                "testLargeBatchSparseMatrixMatMul transpose_a %s transpose_b "
                "%s adjoint_a %s adjoint_b %s" %
                (transpose_a, transpose_b, adjoint_a, adjoint_b))
            a_sm = dense_to_csr_sparse_matrix(a_mats)
            c_t = sparse_csr_matrix_ops.sparse_matrix_mat_mul(
                a_sm,
                b_mats,
                transpose_output=False,
                conjugate_output=False,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                adjoint_a=adjoint_a,
                adjoint_b=adjoint_b)
            c_dense_t = test_util.matmul_without_tf32(
                a_mats,
                b_mats,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                adjoint_a=adjoint_a,
                adjoint_b=adjoint_b)
            self.assertAllEqual(c_dense_t.shape, c_t.shape)
            c_t_value, c_dense_t_value = self.evaluate((c_t, c_dense_t))

            self.assertAllClose(
                c_t_value, c_dense_t_value, rtol=1e-6, atol=2e-5)

  @test_util.run_in_graph_and_eager_modes
  def testLargeBatchSparseMatrixMatMulTransposed(self):
    dtypes_to_test = [np.float32, np.complex64]

    sparsify = lambda m: m * (m > 0)
    for dtype in dtypes_to_test:
      for (transpose_a, transpose_b) in ((False, False), (False, True),
                                         (True, False), (True, True)):
        for (adjoint_a, adjoint_b) in ((False, False), (False, True),
                                       (True, False), (True, True)):
          if (transpose_a and adjoint_a) or (transpose_b and adjoint_b):
            continue
          for shapes in [[[53, 127, 65], [53, 65, 1]],
                         [[53, 127, 1], [53, 1, 65]],
                         [[53, 127, 65], [53, 65, 127]]]:
            a_dense_shape = shapes[0]
            b_dense_shape = shapes[1]
            if transpose_a or adjoint_a:
              _swap(a_dense_shape, -2, -1)
            if transpose_b or adjoint_b:
              _swap(b_dense_shape, -2, -1)
            a_mats = sparsify(
                (np.random.randn(*a_dense_shape) +
                 1.j * np.random.randn(*a_dense_shape))).astype(dtype)
            b_mats = (np.random.randn(*b_dense_shape) +
                      1.j * np.random.randn(*b_dense_shape)).astype(dtype)
            tf_logging.info(
                "testLargeBatchSparseMatrixMatMul transpose_a %s transpose_b "
                "%s adjoint_a %s adjoint_b %s" %
                (transpose_a, transpose_b, adjoint_a, adjoint_b))
            a_sm = dense_to_csr_sparse_matrix(a_mats)
            c_t = sparse_csr_matrix_ops.sparse_matrix_mat_mul(
                a_sm,
                b_mats,
                transpose_output=True,
                conjugate_output=False,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                adjoint_a=adjoint_a,
                adjoint_b=adjoint_b)

            # Example: t(adj(a) . b) = t(b) . conj(a)
            c_dense_t = test_util.matmul_without_tf32(
                math_ops.conj(b_mats) if adjoint_b else b_mats,
                math_ops.conj(a_mats) if adjoint_a else a_mats,
                transpose_a=not (transpose_b or adjoint_b),
                transpose_b=not (transpose_a or adjoint_a),
                adjoint_a=False,
                adjoint_b=False)
            self.assertAllEqual(c_t.shape, c_dense_t.shape)
            c_t_value, c_dense_t_value = self.evaluate((c_t, c_dense_t))
            self.assertAllClose(
                c_t_value, c_dense_t_value, rtol=1e-6, atol=2e-5)

  @test_util.run_in_graph_and_eager_modes
  def testLargeBatchSparseMatrixMatMulConjugate(self):
    sparsify = lambda m: m * (m > 0)
    a_dense_shape = [53, 65, 127]
    b_dense_shape = [53, 127, 67]
    a_mats = sparsify(
        (np.random.randn(*a_dense_shape) +
         1.j * np.random.randn(*a_dense_shape))).astype(np.complex64)
    b_mats = (np.random.randn(*b_dense_shape) +
              1.j * np.random.randn(*b_dense_shape)).astype(np.complex64)
    a_sm = dense_to_csr_sparse_matrix(a_mats)
    c_t = sparse_csr_matrix_ops.sparse_matrix_mat_mul(
        a_sm, b_mats, conjugate_output=True)

    c_dense_t = math_ops.conj(test_util.matmul_without_tf32(a_mats, b_mats))
    self.assertAllEqual(c_t.shape, c_dense_t.shape)
    c_t_value, c_dense_t_value = self.evaluate((c_t, c_dense_t))

    self.assertAllClose(c_t_value, c_dense_t_value, atol=1e-5, rtol=1e-5)

  @test_util.run_in_graph_and_eager_modes
  def testSparseMatrixSparseMatMul(self):
    a_indices = np.array([[0, 0], [2, 3]])
    a_values = np.array([1.0, 5.0]).astype(np.float32)
    a_dense_shape = [5, 6]
    a_sparse_mat = sparse.coo_matrix(
        (a_values, (a_indices[:, 0], a_indices[:, 1])), shape=a_dense_shape)
    a_dense = a_sparse_mat.todense()

    b_indices = np.array([[0, 0], [3, 0], [3, 1]])
    b_values = np.array([2.0, 7.0, 8.0]).astype(np.float32)
    b_dense_shape = [6, 7]
    b_sparse_mat = sparse.coo_matrix(
        (b_values, (b_indices[:, 0], b_indices[:, 1])), shape=b_dense_shape)
    b_dense = b_sparse_mat.todense()

    a_sm = dense_to_csr_sparse_matrix(a_dense)
    b_sm = dense_to_csr_sparse_matrix(b_dense)
    c_sm = sparse_csr_matrix_ops.sparse_matrix_sparse_mat_mul(
        a=a_sm, b=b_sm, type=dtypes.float32)

    c_sm_dense = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
        c_sm, dtypes.float32)
    c_sm_dense_value = self.evaluate(c_sm_dense)

    expected_c_value = a_sparse_mat.dot(b_sparse_mat).todense()
    self.assertAllClose(expected_c_value, c_sm_dense_value)

  @test_util.run_in_graph_and_eager_modes
  def testSparseMatrixSparseMatMul_NumericZerosNotPruned(self):
    # Tests that numeric zeros appearing from the sparse-sparse matrix
    # multiplication are not pruned from the sparse structural
    a_indices = np.array([[0, 0], [0, 2]])
    a_values = np.array([2.0, -1.0]).astype(np.float32)
    a_dense_shape = [2, 3]
    a_sparse_mat = sparse.coo_matrix(
        (a_values, (a_indices[:, 0], a_indices[:, 1])), shape=a_dense_shape)
    a_dense = a_sparse_mat.todense()

    b_indices = np.array([[0, 1], [2, 1]])
    b_values = np.array([3.0, 6.0]).astype(np.float32)
    b_dense_shape = [3, 2]
    b_sparse_mat = sparse.coo_matrix(
        (b_values, (b_indices[:, 0], b_indices[:, 1])), shape=b_dense_shape)
    b_dense = b_sparse_mat.todense()

    # Convert to CSRSparseMatrix while removing numeric zeros from the
    # structural representation.
    a_sm = dense_to_csr_sparse_matrix(a_dense)
    b_sm = dense_to_csr_sparse_matrix(b_dense)

    # Compute the matmul.
    c_sm = sparse_csr_matrix_ops.sparse_matrix_sparse_mat_mul(
        a=a_sm, b=b_sm, type=dtypes.float32)
    c_nnz = sparse_csr_matrix_ops.sparse_matrix_nnz(c_sm)
    c_nnz_value = self.evaluate(c_nnz)

    # Expect that there is a single numeric zero at index (0, 1) if zeros are
    # not pruned, since 2.0 * 3.0 + (-1.0) * 6.0 = 0.0.
    self.assertAllClose(1, c_nnz_value)

  @test_util.run_in_graph_and_eager_modes
  def testLargeBatchSparseMatrixSparseMatMul(self):
    sparsify = lambda m: m * (m > 0)

    for (transpose_a, transpose_b) in ((False, False), (False, True),
                                       (True, False), (True, True)):
      for (adjoint_a, adjoint_b) in ((False, False), (False, True),
                                     (True, False), (True, True)):
        if (transpose_a and adjoint_a) or (transpose_b and adjoint_b):
          continue

        a_dense_shape = ([53, 127, 65]
                         if transpose_a or adjoint_a else [53, 65, 127])
        b_dense_shape = ([53, 67, 127]
                         if transpose_b or adjoint_b else [53, 127, 67])

        a_mats = sparsify(np.random.randn(*a_dense_shape)).astype(np.float32)
        b_mats = sparsify(np.random.randn(*b_dense_shape).astype(np.float32))

        a_sm = dense_to_csr_sparse_matrix(a_mats)
        b_sm = dense_to_csr_sparse_matrix(b_mats)
        c_sm = sparse_csr_matrix_ops.sparse_matrix_sparse_mat_mul(
            a_sm,
            b_sm,
            type=dtypes.float32,
            transpose_a=transpose_a,
            adjoint_a=adjoint_a,
            transpose_b=transpose_b,
            adjoint_b=adjoint_b)
        c_sm_dense = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
            c_sm, dtypes.float32)
        c_dense_t = test_util.matmul_without_tf32(
            a_mats,
            b_mats,
            transpose_a=transpose_a,
            adjoint_a=adjoint_a,
            transpose_b=transpose_b,
            adjoint_b=adjoint_b)
        c_dense_t_value, c_sm_dense_value = self.evaluate(
            (c_dense_t, c_sm_dense))

        self.assertAllClose(c_sm_dense_value, c_dense_t_value)

  @test_util.run_in_graph_and_eager_modes
  def testLargeBatchRegisteredAddN(self):
    if not self._gpu_available:
      return

    sparsify = lambda m: m * (m > 0)
    dense_shape = [53, 65, 127]
    matrices = [
        sparsify(np.random.randn(*dense_shape)).astype(np.float32)
        for _ in range(16)
    ]
    sparse_matrices = [dense_to_csr_sparse_matrix(mat) for mat in matrices]
    sparse_matrices_sum = math_ops.add_n(sparse_matrices)
    sparse_matrices_sum_dense = \
        sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
            sparse_matrices_sum, dtypes.float32)
    sparse_matrices_sum_dense_value = self.evaluate(sparse_matrices_sum_dense)

    # Ensure that the dense (numpy) sum across all batches matches the result
    # of add_n converted back to dense.
    expected_sum = np.sum(matrices, axis=0)
    self.assertAllClose(expected_sum, sparse_matrices_sum_dense_value)

  @test_util.run_in_graph_and_eager_modes
  def testCSRZeros(self):
    if not self._gpu_available:
      return
    a_dense_shape = [65, 127]
    b_dense_shape = [53, 127, 67]
    data_types = [
        dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128
    ]
    for dtype in data_types:
      # Check both rank-2 and rank-3 tensors.
      a_sm = sparse_csr_matrix_ops.sparse_matrix_zeros(
          a_dense_shape, type=dtype)
      b_sm = sparse_csr_matrix_ops.sparse_matrix_zeros(
          b_dense_shape, type=dtype)
      a_rt = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(a_sm, type=dtype)
      b_rt = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(b_sm, type=dtype)
      a_rt_value, b_rt_value = self.evaluate((a_rt, b_rt))

      self.assertAllEqual(a_rt_value, np.zeros(a_dense_shape))
      self.assertAllEqual(b_rt_value, np.zeros(b_dense_shape))

  @test_util.run_in_graph_and_eager_modes
  def testLargeBatchZerosLike(self):
    if not self._gpu_available:
      return

    batch_size = 53
    rows = 128
    cols = 67
    dense_shape = [batch_size, rows, cols]
    data_types = [
        dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128
    ]
    for dtype in data_types:
      sparse_matrices = sparse_csr_matrix_ops.sparse_matrix_zeros(
          dense_shape, type=dtype)
      zeros_like_sparse_matrices = array_ops.zeros_like(sparse_matrices)
      zeros_like_components = [
          sparse_csr_matrix_ops.csr_sparse_matrix_components(
              zeros_like_sparse_matrices, i, type=dtype)
          for i in range(batch_size)
      ]
      zeros_like_components_values = self.evaluate(zeros_like_components)
      for component in zeros_like_components_values:
        self.assertAllEqual(component.row_ptrs, np.zeros(rows + 1, np.int32))
        self.assertAllEqual(component.col_inds, np.empty([0], np.int32))
        self.assertAllEqual(component.values, np.empty([0],
                                                       dtype.as_numpy_dtype))

  @test_util.run_in_graph_and_eager_modes
  def testTranspose(self):
    sparsify = lambda m: m * (m > 0)
    dense_shape = [127, 65]
    data_types = [
        dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128
    ]
    for dtype in data_types:
      mats = sparsify(
          (np.random.randn(*dense_shape) +
           1.j * np.random.randn(*dense_shape))).astype(dtype.as_numpy_dtype)
      for conjugate in False, True:
        expected = np.transpose(mats)
        if conjugate:
          expected = np.conj(expected)
        matrices = math_ops.cast(mats, dtype)
        sparse_matrices = dense_to_csr_sparse_matrix(matrices)
        transpose_sparse_matrices = \
            sparse_csr_matrix_ops.sparse_matrix_transpose(
                sparse_matrices, conjugate=conjugate, type=dtype)
        dense_transposed = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
            transpose_sparse_matrices, dtype)
        dense_transposed_values = self.evaluate(dense_transposed)
        self.assertAllClose(expected, dense_transposed_values)

  @test_util.run_in_graph_and_eager_modes
  def testLargeBatchTranspose(self):
    sparsify = lambda m: m * (m > 0)
    dense_shape = [53, 65, 127]
    data_types = [
        dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128
    ]
    for dtype in data_types:
      mats = sparsify(
          (np.random.randn(*dense_shape) +
           1.j * np.random.randn(*dense_shape))).astype(dtype.as_numpy_dtype)
      expected = np.transpose(mats, (0, 2, 1))
      for conjugate in False, True:
        if conjugate:
          expected = np.conj(expected)
        matrices = math_ops.cast(mats, dtype)
        sparse_matrices = dense_to_csr_sparse_matrix(matrices)
        transpose_sparse_matrices = \
            sparse_csr_matrix_ops.sparse_matrix_transpose(
                sparse_matrices, conjugate=conjugate, type=dtype)
        dense_transposed = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
            transpose_sparse_matrices, dtype)
        dense_transposed_values = self.evaluate(dense_transposed)
        self.assertAllClose(expected, dense_transposed_values)

  @test_util.run_in_graph_and_eager_modes
  def testSoftmax(self):
    if not self._gpu_available:
      return

    sparsify = lambda m: m * (m > 0)
    dense_shape = [127, 65]
    logits = sparsify(np.random.randn(*dense_shape))
    logits_with_ninf = np.copy(logits)
    logits_with_ninf[logits == 0] = -np.inf
    data_types = [dtypes.float32, dtypes.float64]
    for dtype in data_types:
      logits_t = math_ops.cast(logits, dtype)
      logits_t_with_ninf = math_ops.cast(logits_with_ninf, dtype)
      expected = nn_ops.softmax(logits_t_with_ninf)
      sparse_logits_t = dense_to_csr_sparse_matrix(logits_t)
      softmax_sparse_logits_t = sparse_csr_matrix_ops.sparse_matrix_softmax(
          sparse_logits_t, type=dtype)
      dense_softmax = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
          softmax_sparse_logits_t, dtype)
      dense_softmax_values, expected_values = self.evaluate(
          (dense_softmax, expected))
      self.assertAllClose(expected_values, dense_softmax_values)

  @test_util.run_in_graph_and_eager_modes
  def testLargeBatchSoftmax(self):
    if not self._gpu_available:
      return

    sparsify = lambda m: m * (m > 0)
    dense_shape = [53, 65, 127]
    logits = sparsify(np.random.randn(*dense_shape))
    logits_with_ninf = np.copy(logits)
    logits_with_ninf[logits == 0] = -np.inf
    data_types = [dtypes.float32, dtypes.float64]
    for dtype in data_types:
      logits_t = math_ops.cast(logits, dtype)
      logits_t_with_ninf = math_ops.cast(logits_with_ninf, dtype)
      expected = nn_ops.softmax(logits_t_with_ninf)
      sparse_logits_t = dense_to_csr_sparse_matrix(logits_t)
      softmax_sparse_logits_t = sparse_csr_matrix_ops.sparse_matrix_softmax(
          sparse_logits_t, type=dtype)
      dense_softmax = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
          softmax_sparse_logits_t, dtype)
      dense_softmax_values, expected_values = self.evaluate(
          (dense_softmax, expected))
      self.assertAllClose(expected_values, dense_softmax_values)

  @test_util.run_in_graph_and_eager_modes
  def testLargeBatchSoftmaxEmpty(self):
    if not self._gpu_available:
      return

    dense_shape = [53, 65, 127]
    sparse_logits_t = sparse_csr_matrix_ops.sparse_matrix_zeros(
        dense_shape, type=dtypes.float32)
    softmax_sparse_logits_t = sparse_csr_matrix_ops.sparse_matrix_softmax(
        sparse_logits_t, type=dtypes.float32)
    dense_softmax = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
        softmax_sparse_logits_t, dtypes.float32)
    dense_softmax_values = self.evaluate(dense_softmax)
    self.assertAllEqual(
        np.zeros_like(dense_softmax_values), dense_softmax_values)

  @test_util.run_in_graph_and_eager_modes
  def testSoftmaxGrad(self):
    if not self._gpu_available:
      return

    sparsify = lambda m: m * (m > 0)
    dense_shape = [127, 65]
    softmax = sparsify(np.random.randn(*dense_shape))
    grad_softmax = sparsify(np.random.randn(*dense_shape))
    expected = (
        (grad_softmax - np.sum(grad_softmax * softmax, -1, keepdims=True)) *
        softmax)
    data_types = [dtypes.float32, dtypes.float64]
    for dtype in data_types:
      softmax_t = math_ops.cast(softmax, dtype)
      grad_softmax_t = math_ops.cast(grad_softmax, dtype)
      softmax_sparse = dense_to_csr_sparse_matrix(softmax_t)
      grad_softmax_sparse = dense_to_csr_sparse_matrix(grad_softmax_t)
      gradients_sparse = sparse_csr_matrix_ops.sparse_matrix_softmax_grad(
          softmax_sparse, grad_softmax_sparse, dtype)
      dense_gradients = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
          gradients_sparse, dtype)
      dense_gradients_values = self.evaluate((dense_gradients))
      self.assertAllClose(expected, dense_gradients_values)

  @test_util.run_in_graph_and_eager_modes
  def testLargeBatchSoftmaxGrad(self):
    if not self._gpu_available:
      return

    sparsify = lambda m: m * (m > 0)
    dense_shape = [53, 65, 127]
    softmax = sparsify(np.random.randn(*dense_shape))
    grad_softmax = sparsify(np.random.randn(*dense_shape))
    expected = (
        (grad_softmax - np.sum(grad_softmax * softmax, -1, keepdims=True)) *
        softmax)
    data_types = [dtypes.float32, dtypes.float64]
    for dtype in data_types:
      softmax_t = math_ops.cast(softmax, dtype)
      grad_softmax_t = math_ops.cast(grad_softmax, dtype)
      softmax_sparse = dense_to_csr_sparse_matrix(softmax_t)
      grad_softmax_sparse = dense_to_csr_sparse_matrix(grad_softmax_t)
      gradients_sparse = sparse_csr_matrix_ops.sparse_matrix_softmax_grad(
          softmax_sparse, grad_softmax_sparse, dtype)
      dense_gradients = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
          gradients_sparse, dtype)
      dense_gradients_values = self.evaluate((dense_gradients))
      self.assertAllClose(expected, dense_gradients_values)

  @test_util.run_in_graph_and_eager_modes
  def testLargeBatchSoftmaxGradEmpty(self):
    if not self._gpu_available:
      return

    sparsify = lambda m: m * (m > 0)
    dense_shape = [53, 65, 127]
    not_empty = sparsify(np.random.randn(*dense_shape)).astype(np.float32)
    sparse_empty = sparse_csr_matrix_ops.sparse_matrix_zeros(
        dense_shape, type=dtypes.float32)
    sparse_not_empty = dense_to_csr_sparse_matrix(not_empty)
    gradients_empty_softmax = sparse_csr_matrix_ops.sparse_matrix_softmax_grad(
        sparse_empty, sparse_not_empty, dtypes.float32)
    gradients_empty_grad_softmax = (
        sparse_csr_matrix_ops.sparse_matrix_softmax_grad(
            sparse_not_empty, sparse_empty, dtypes.float32))
    gradients_empty_both = sparse_csr_matrix_ops.sparse_matrix_softmax_grad(
        sparse_empty, sparse_empty, dtypes.float32)
    ges = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
        gradients_empty_softmax, dtypes.float32)
    gegs = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
        gradients_empty_grad_softmax, dtypes.float32)
    geb = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
        gradients_empty_both, dtypes.float32)
    ges_v, gegs_v, geb_v = self.evaluate((ges, gegs, geb))
    for v in (ges_v, gegs_v, geb_v):
      self.assertAllEqual(np.zeros(dense_shape), v)

  @test_util.run_in_graph_and_eager_modes
  def testLargeBatchConj(self):
    if not self._gpu_available:
      return

    sparsify = lambda m: m * (np.real(m) > 0)
    dense_shape = [53, 65, 127]
    matrices = (
        sparsify(np.random.randn(*dense_shape)) +
        1j * np.random.randn(*dense_shape))
    data_types = [
        dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128
    ]
    for dtype in data_types:
      matrices_t = matrices.astype(dtype.as_numpy_dtype)
      expected = np.conj(matrices_t)
      sparse_matrices = dense_to_csr_sparse_matrix(matrices_t)
      conj_sparse_matrices = math_ops.conj(sparse_matrices)
      dense_conj_matrices = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
          conj_sparse_matrices, dtype)
      conj_values = self.evaluate(dense_conj_matrices)
      self.assertAllClose(expected, conj_values)

  @test_util.run_in_graph_and_eager_modes
  def testLargeBatchSparseMatrixMulScalar(self):
    if not self._gpu_available:
      return

    sparsify = lambda m: m * (m > 0)
    a_dense_shape = [53, 65, 127]
    a_mats = sparsify(np.random.randn(*a_dense_shape)).astype(np.float32)
    b = np.float32(3.5)
    expected = a_mats * b
    a_sm = dense_to_csr_sparse_matrix(a_mats)
    c_t = sparse_csr_matrix_ops.sparse_matrix_mul(a_sm, b)
    c_dense_t = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
        c_t, dtypes.float32)
    c_dense_t_value = self.evaluate(c_dense_t)

    self.assertAllClose(expected, c_dense_t_value)

  @test_util.run_in_graph_and_eager_modes
  def testLargeBatchSparseMatrixMulVec(self):
    if not self._gpu_available:
      return

    sparsify = lambda m: m * (m > 0)
    a_dense_shape = [53, 65, 127]
    a_mats = sparsify(np.random.randn(*a_dense_shape)).astype(np.float32)
    b = np.random.randn(53, 1, 1).astype(np.float32)
    expected = a_mats * b
    a_sm = dense_to_csr_sparse_matrix(a_mats)
    c_t = sparse_csr_matrix_ops.sparse_matrix_mul(a_sm, b)
    c_dense_t = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
        c_t, dtypes.float32)
    c_dense_t_value = self.evaluate(c_dense_t)

    self.assertAllClose(expected, c_dense_t_value)

  @test_util.run_in_graph_and_eager_modes
  def testSparseCholesky(self):
    dense_matrix = np.array([
        [2, 0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 0],
        [1, 1, 7, 0, 0, 0],
        [0, 0, 0, 4, 0, 0],
        [0, 0, 1, 0, 5, 0],
        [0, 0, 2, 0, 1, 6],
    ]).astype(np.complex128)

    data_types = [
        dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128
    ]
    for dtype in data_types:
      with test_util.force_cpu():
        if dtype.is_complex:
          dense_matrix += 0.5j * np.tril(dense_matrix, -1)

        sparse_matrix = dense_to_csr_sparse_matrix(
            math_ops.cast(dense_matrix, dtype))
        # Obtain the Sparse Cholesky factor using AMD Ordering for reducing
        # fill-in.
        ordering_amd = sparse_csr_matrix_ops.sparse_matrix_ordering_amd(
            sparse_matrix)
        cholesky_sparse_matrices = (
            sparse_csr_matrix_ops.sparse_matrix_sparse_cholesky(
                sparse_matrix, ordering_amd, type=dtype))
        dense_cholesky = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
            cholesky_sparse_matrices, dtype)
        # Compute L * Lh where L is the Sparse Cholesky factor.
        verification = test_util.matmul_without_tf32(
            dense_cholesky, array_ops.transpose(dense_cholesky, conjugate=True))
        verification = twist_matrix(verification, ordering_amd)
        # Assert that input matrix A satisfies A = L * Lh.
        verification_values = self.evaluate(verification)
        full_dense_matrix = (
            dense_matrix +
            np.conjugate(np.transpose(np.tril(dense_matrix, -1))))
        self.assertAllClose(full_dense_matrix, verification_values)

  @test_util.run_in_graph_and_eager_modes
  def testBatchSparseCholesky(self):
    dense_mat = np.array([
        # A diagonal matrix.
        [
            [1, 0, 0, 0],  #
            [0, 2, 0, 0],  #
            [0, 0, 3, 0],  #
            [0, 0, 0, 4],
        ],  #
        # A tridiagonal hermitian matrix.
        [
            [5 + 0j, 1 + 0j, 0 + 0j, 0 + 0j],  #
            [1 + 0j, 4 + 0j, 1 + 2j, 0 + 0j],  #
            [0 + 0j, 1 - 2j, 9 + 0j, 3 - 3j],  #
            [0 + 0j, 0 + 0j, 3 + 3j, 7 + 0j],
        ],  #
        # A diagonal matrix with a corner element; for which
        # OrderingAMD returns a non-identity permutation.
        [
            [1, 0, 0, 1.],  #
            [0, 2, 0, 0.],  #
            [0, 0, 3, 0.],  #
            [1, 0, 0, 4.],
        ]  #
    ]).astype(np.complex128)

    data_types = [
        dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128
    ]
    for dtype in data_types:
      sparse_matrix = dense_to_csr_sparse_matrix(
          math_ops.cast(dense_mat, dtype))
      ordering_amd = sparse_csr_matrix_ops.sparse_matrix_ordering_amd(
          sparse_matrix)

      cholesky_sparse_matrix = (
          sparse_csr_matrix_ops.sparse_matrix_sparse_cholesky(
              sparse_matrix, ordering_amd, type=dtype))
      dense_cholesky = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
          cholesky_sparse_matrix, dtype)

      # Compute L * Lh.
      verification = test_util.matmul_without_tf32(
          dense_cholesky,
          array_ops.transpose(dense_cholesky, perm=[0, 2, 1], conjugate=True))
      verification = twist_matrix(verification, ordering_amd)

      verification_values = self.evaluate(verification)
      self.assertAllClose(
          dense_mat.astype(dtype.as_numpy_dtype), verification_values)

  @test_util.run_in_graph_and_eager_modes
  def testLargeBatchSparseCholesky(self):
    sparsity = 0.1
    sparsify = lambda m: m * (m > 1 - sparsity)

    batch_size = 53
    num_rows = 147
    dense_shape = [batch_size, num_rows, num_rows]

    dense_matrix = sparsify(np.random.uniform(size=dense_shape)).astype(
        np.float32)

    # Create a "random" SPD matrix, by choosing each entry of A between
    # 0 and 1 at the specified density, and computing 0.5(A + At) + n*I.
    # This ensures diagonal dominance which implies positive-definiteness.
    dense_matrix = (
        0.5 *
        (dense_matrix + array_ops.transpose(dense_matrix, perm=[0, 2, 1])) +
        num_rows * linalg_ops.eye(dense_shape[-1], batch_shape=[batch_size]))
    # Compute the fill-in reducing permutation and use it to perform
    # the Sparse Cholesky factorization.
    sparse_matrix = dense_to_csr_sparse_matrix(dense_matrix)
    ordering_amd = sparse_csr_matrix_ops.sparse_matrix_ordering_amd(
        sparse_matrix)

    cholesky_sparse_matrix = \
        sparse_csr_matrix_ops.sparse_matrix_sparse_cholesky(
            sparse_matrix, ordering_amd, type=dtypes.float32)
    dense_cholesky = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
        cholesky_sparse_matrix, dtypes.float32)

    # Compute L * Lh.
    verification = test_util.matmul_without_tf32(
        dense_cholesky, array_ops.transpose(dense_cholesky, perm=[0, 2, 1]))
    verification = twist_matrix(verification, ordering_amd)
    verification_values = self.evaluate(verification)
    self.assertAllClose(dense_matrix, verification_values, atol=1e-5, rtol=1e-5)

  @test_util.run_in_graph_and_eager_modes
  def testSparseCholesky_InvalidMatrix(self):
    # Verify that non-SPD matrices result in an Invalid Argument error.
    invalid_matrices = [
        # zero matrix.
        np.array([
            [0., 0., 0., 0.],  #
            [0., 0., 0., 0.],  #
            [0., 0., 0., 0.],  #
            [0., 0., 0., 0.]  #
        ]),
        # zero diagonal entry.
        np.array([
            [9., 0., 5., 0.],  #
            [0., 0., 0., 1.],  #
            [5., 0., 8., 0.],  #
            [0., 1., 0., 7.]  #
        ]),
        # not positive definite.
        np.array([
            [2., -2., 0., 0.],  #
            [-2., 2., 0., 0.],  #
            [0., 0., 3., -3.],  #
            [0., 0., -3., 3.]  #
        ]),
    ]

    with test_util.force_cpu():
      for invalid_matrix in invalid_matrices:
        with self.assertRaises(errors.InvalidArgumentError):
          sparse_matrix = dense_to_csr_sparse_matrix(
              invalid_matrix.astype(np.float32))
          # Compute the fill-in reducing permutation and use it to perform
          # the Sparse Cholesky factorization.
          ordering_amd = sparse_csr_matrix_ops.sparse_matrix_ordering_amd(
              sparse_matrix)
          cholesky_sparse_matrices = (
              sparse_csr_matrix_ops.sparse_matrix_sparse_cholesky(
                  sparse_matrix, ordering_amd, type=dtypes.float32))
          # Convert the Cholesky factor to a dense matrix to be evaluated.
          dense_cholesky = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
              cholesky_sparse_matrices, type=dtypes.float32)
          self.evaluate(dense_cholesky)

  @test_util.run_in_graph_and_eager_modes
  def testOrderingAMD(self):
    num_rows = 6
    # An SPD matrix where AMD ordering can reduce fill-in for Cholesky factor.
    dense_matrix = np.array([
        [7, 0, 0, 0, 0, 0],
        [1, 4, 0, 0, 0, 0],
        [1, 1, 3, 0, 0, 0],
        [0, 0, 0, 4, 0, 0],
        [2, 0, 0, 0, 5, 0],
        [1, 2, 2, 0, 0, 6],
    ]).astype(np.float32)

    with test_util.force_cpu():
      sparse_matrix = dense_to_csr_sparse_matrix(dense_matrix)

      # Obtain the Sparse Cholesky factor with the identity permutation as the
      # fill-in reducing ordering.
      cholesky_without_ordering = (
          sparse_csr_matrix_ops.sparse_matrix_sparse_cholesky(
              sparse_matrix, math_ops.range(num_rows), type=dtypes.float32))
      cholesky_without_ordering_nnz = sparse_csr_matrix_ops.sparse_matrix_nnz(
          cholesky_without_ordering)

      # Obtain the Sparse Cholesky factor using AMD Ordering for reducing
      # fill-in.
      ordering_amd = sparse_csr_matrix_ops.sparse_matrix_ordering_amd(
          sparse_matrix)
      cholesky_with_amd = sparse_csr_matrix_ops.sparse_matrix_sparse_cholesky(
          sparse_matrix, ordering_amd, type=dtypes.float32)
      cholesky_with_amd_nnz = sparse_csr_matrix_ops.sparse_matrix_nnz(
          cholesky_with_amd)

      (ordering_amd_value, cholesky_with_amd_nnz_value,
       cholesky_without_ordering_nnz_value) = self.evaluate(
           [ordering_amd, cholesky_with_amd_nnz, cholesky_without_ordering_nnz])

      # AMD ordering should return a valid permutation.
      self.assertAllClose(np.arange(num_rows), np.sort(ordering_amd_value))
      # Check that cholesky with AMD ordering has a strictly lower nonzero count
      # for this matrix.
      self.assertLess(cholesky_with_amd_nnz_value,
                      cholesky_without_ordering_nnz_value)


class CSRSparseMatrixOpsBenchmark(test.Benchmark):

  def benchmark_sparse_matrix_mat_mul_gpu(self):
    if not test_util.is_gpu_available():
      return

    sparsify = lambda m: array_ops.where(m > 2, m, array_ops.zeros_like(m))

    # XW, X dense and W sparse
    # X is shaped [{1, 8, 16}, 2000]
    # W is shaped [2000, 4000]

    for batch_size in [1, 8, 16]:
      x_dense_shape = [batch_size, 2000]
      w_dense_shape = [2000, 4000]

      with ops.Graph().as_default(), ops.device("/gpu:0"):
        x_mats = random_ops.random_normal(x_dense_shape, dtype=dtypes.float32)
        w_mats = sparsify(
            random_ops.random_normal(w_dense_shape, dtype=dtypes.float32))
        nnz = array_ops.shape(array_ops.where(w_mats))[0]
        ratio = math_ops.cast(nnz, dtypes.float32) / np.prod(w_dense_shape)
        w_sm = dense_to_csr_sparse_matrix(w_mats)
        with ops.name_scope("w_sm_var"):
          w_sm_var = variable_scope.get_variable(
              "sm", initializer=w_sm, use_resource=True)
          w_sm_var_v = w_sm_var.read_value()
        with ops.name_scope("w_var"):
          w_var = variable_scope.get_variable(
              "sm_dense", initializer=w_mats, use_resource=True)
          w_var_v = w_var.read_value()
        with ops.name_scope("b"):
          x = variable_scope.get_variable(
              "b", initializer=x_mats, use_resource=True)
          x_v = x.read_value()
        # X*W = (W'*X')'
        xw_sparse = sparse_csr_matrix_ops.sparse_matrix_mat_mul(
            w_sm_var_v,
            x_v,
            transpose_a=True,
            transpose_b=True,
            transpose_output=True)
        xw_dense = math_ops.matmul(x_v, w_var_v)

        with session.Session() as sess:
          self.evaluate(
              [w_var.initializer, w_sm_var.initializer, x.initializer])
          nnz_value, ratio_value = self.evaluate((nnz, ratio))
          name_template = (
              "sparse_matrix_mat_mul_gpu_%s_W_2000x4000_batch_size_%d")
          self.run_op_benchmark(
              sess,
              xw_sparse.op,
              name=name_template % ("sparse", batch_size),
              extras={
                  "percentage_nonzero": ratio_value,
                  "num_nonzero": nnz_value
              },
              min_iters=50)
          self.run_op_benchmark(
              sess,
              xw_dense.op,
              name=name_template % ("dense", batch_size),
              extras={
                  "percentage_nonzero": ratio_value,
                  "num_nonzero": nnz_value
              },
              min_iters=50)

  def benchmark_sparse_matrix_mat_vec_mul(self):
    # num_rows, device, transpose.
    cases = [
        [2000, CPU, False],
        [8000, CPU, False],
        [12000, CPU, False],
        [2000, CPU, True],
        [8000, CPU, True],
        [12000, CPU, True],
    ]
    seed = 42

    for num_rows, device, transpose in cases:
      if device == GPU and not test_util.is_gpu_available():
        continue
      for num_threads in [1, 2, 4, 6, 8, 10]:
        device_str = "cpu" if device == CPU else "gpu"
        w_dense_shape = [num_rows, num_rows]
        x_dense_shape = [num_rows, 1]

        with ops.Graph().as_default(), ops.device(device):
          random_seed.set_random_seed(seed)
          x = random_ops.random_normal(x_dense_shape, dtype=dtypes.float32)
          w_np = sparse.rand(
              w_dense_shape[0],
              w_dense_shape[1],
              density=0.01,
              dtype=np.float32,
              random_state=np.random.RandomState(seed))
          w_st = sparse_tensor.SparseTensor(
              zip(w_np.row, w_np.col), w_np.data, w_np.shape)
          w_st = sparse_ops.sparse_reorder(w_st)

          nnz = array_ops.shape(w_st.values)[0]
          ratio = math_ops.cast(nnz, dtypes.float32) / np.prod(w_np.shape)

          w_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
              w_st.indices, w_st.values, w_st.dense_shape)
          xw_sparse_matrix = sparse_csr_matrix_ops.sparse_matrix_mat_mul(
              w_sm,
              x,
              transpose_a=transpose,
              transpose_b=False,
              transpose_output=False)
          xw_sparse_tensor = sparse_ops.sparse_tensor_dense_matmul(
              w_st, x, adjoint_a=transpose, adjoint_b=False)

          with session.Session(
              config=config_pb2.ConfigProto(
                  intra_op_parallelism_threads=num_threads)) as sess:
            nnz_value, ratio_value = sess.run((nnz, ratio))
            name_template = ("mat_vec_mul_%s_%s_W_%d_transpose_%s_threads_%d")
            self.run_op_benchmark(
                sess,
                xw_sparse_matrix.op,
                name=name_template %
                (device_str, "sparse_matrix", num_rows, transpose, num_threads),
                extras={
                    "percentage_nonzero": ratio_value,
                    "num_nonzero": nnz_value,
                },
                min_iters=10)
            self.run_op_benchmark(
                sess,
                xw_sparse_tensor.op,
                name=name_template %
                (device_str, "sparse_tensor", num_rows, transpose, num_threads),
                extras={
                    "percentage_nonzero": ratio_value,
                    "num_nonzero": nnz_value,
                },
                min_iters=10)

  def benchmark_sparse_matrix_sparse_matmul(self):
    density = 0.05
    # pylint: disable=g-long-lambda
    sparsify = lambda m: array_ops.where(m > 1. - density, m,
                                         array_ops.zeros_like(m))
    # pylint: enable=g-long-lambda

    for batch_size in [1, 16]:
      for num_threads in [1, 4, 12]:
        dense_shape = [batch_size, 250, 250]

        for device in [CPU, GPU]:
          if device == GPU and not test_util.is_gpu_available():
            continue

          with ops.Graph().as_default(), ops.device(device):
            x_mats = sparsify(
                random_ops.random_uniform(dense_shape, dtype=dtypes.float32))
            y_mats = sparsify(
                random_ops.random_uniform(dense_shape, dtype=dtypes.float32))

            nnz = array_ops.shape(array_ops.where(x_mats))[0] + array_ops.shape(
                array_ops.where(y_mats))[0]
            ratio = math_ops.cast(nnz,
                                  dtypes.float32) / (2 * np.prod(dense_shape))

            x_sm = dense_to_csr_sparse_matrix(x_mats)
            y_sm = dense_to_csr_sparse_matrix(y_mats)

            xy_sparse = sparse_csr_matrix_ops.sparse_matrix_sparse_mat_mul(
                x_sm, y_sm, type=dtypes.float32)

            with session.Session(
                config=config_pb2.ConfigProto(
                    intra_op_parallelism_threads=num_threads)) as sess:
              nnz_value, ratio_value = self.evaluate((nnz, ratio))
              name_template = (
                  "sparse_matrix_sparse_matmul_%s_N_%d_batch_size_%d_threads_%d"
              )
              device_str = "cpu" if device == CPU else "gpu"
              self.run_op_benchmark(
                  sess,
                  xy_sparse.op,
                  name=name_template %
                  (device_str, dense_shape[-1], batch_size, num_threads),
                  extras={
                      "percentage_nonzero": ratio_value,
                      "num_nonzero": nnz_value
                  },
                  min_iters=50)

  def benchmark_sparse_dense_conversion(self):
    sparsity = 0.05

    for batch_size in [1, 16]:
      for num_threads in [1, 4, 12]:
        dense_shape = [batch_size, 750, 750]

        for device in [CPU, GPU]:
          if device == GPU and not test_util.is_gpu_available():
            continue

          with ops.Graph().as_default(), ops.device(device):
            mats = random_ops.random_uniform(dense_shape, dtype=dtypes.float32)
            mats_locs = array_ops.where(mats > 1.0 - sparsity)

            sparse_matrices = sparse_csr_matrix_ops.dense_to_csr_sparse_matrix(
                mats, mats_locs)
            dense_matrices = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
                sparse_matrices, type=dtypes.float32)
            nnz = math_ops.reduce_sum(
                sparse_csr_matrix_ops.sparse_matrix_nnz(sparse_matrices))
            ratio = math_ops.cast(nnz, dtypes.float32) / np.prod(dense_shape)

            with session.Session(
                config=config_pb2.ConfigProto(
                    intra_op_parallelism_threads=num_threads)) as sess:
              nnz_value, ratio_value = self.evaluate((nnz, ratio))
              device_str = "cpu" if device == CPU else "gpu"
              name_template = (
                  "dense_to_sparse_matrix_%s_N_%d_batch_size_%d_num_threads_%d")
              self.run_op_benchmark(
                  sess,
                  sparse_matrices.op,
                  name=name_template %
                  (device_str, dense_shape[-1], batch_size, num_threads),
                  extras={
                      "percentage_nonzero": ratio_value,
                      "num_nonzero": nnz_value,
                  },
                  min_iters=50)
              name_template = (
                  "sparse_matrix_to_dense_%s_N_%d_batch_size_%d_num_threads_%d")
              self.run_op_benchmark(
                  sess,
                  dense_matrices.op,
                  name=name_template %
                  (device_str, dense_shape[-1], batch_size, num_threads),
                  extras={
                      "percentage_nonzero": ratio_value,
                      "num_nonzero": nnz_value,
                  },
                  min_iters=50)

  def benchmark_sparse_cholesky(self):
    # TODO(anudhyan): Use conversions from SparseTensor instead of to get this
    # benchmark working for larger matrices. For this to work without GPU, we
    # need to write CPU kernels for SparseTensor conversions.
    num_rows = 500
    density = 0.01
    # pylint: disable=g-long-lambda
    sparsify = lambda m: array_ops.where(m > 1. - density, m,
                                         array_ops.zeros_like(m))
    # pylint: enable=g-long-lambda

    for batch_size in [1, 16]:
      for num_threads in [1, 4, 12]:
        dense_shape = [batch_size, num_rows, num_rows]

        with ops.Graph().as_default(), ops.device(CPU):
          # Create a "random" SPD matrix, by choosing each entry of A between
          # 0 and 1 at the specified density, and computing 0.5(A + At) + n*I.
          # This ensures diagonal dominance which implies positive-definiteness.
          dense_matrix = sparsify(
              random_ops.random_uniform(dense_shape, dtype=dtypes.float32))
          spd_dense_matrix = (
              0.5 *
              (dense_matrix + array_ops.transpose(dense_matrix, perm=[0, 2, 1]))
              + num_rows *
              linalg_ops.eye(dense_shape[-1], batch_shape=[batch_size]))

          # Convert to SparseMatrix and invoke Sparse Cholesky factorization
          # with AMD Ordering.
          sparse_matrix = dense_to_csr_sparse_matrix(spd_dense_matrix)
          ordering_amd = sparse_csr_matrix_ops.sparse_matrix_ordering_amd(
              sparse_matrix)
          cholesky_sparse_matrix = (
              sparse_csr_matrix_ops.sparse_matrix_sparse_cholesky(
                  sparse_matrix, ordering_amd, type=dtypes.float32))

          nnz = math_ops.reduce_sum(
              sparse_csr_matrix_ops.sparse_matrix_nnz(sparse_matrix))
          ratio = math_ops.cast(nnz, dtypes.float32) / np.prod(dense_shape)
          ordering_amd_name_template = (
              "sparse_matrix_ordering_amd_cpu_N_%d_batch_size_%d_threads_%d")
          sparse_cholesky_name_template = (
              "sparse_matrix_sparse_cholesky_cpu_N_%d_batch_size_%d_threads_%d")
          with session.Session(
              config=config_pb2.ConfigProto(
                  intra_op_parallelism_threads=num_threads)) as sess:
            nnz_value, ratio_value = self.evaluate((nnz, ratio))
            self.run_op_benchmark(
                sess,
                ordering_amd.op,
                name=ordering_amd_name_template %
                (dense_shape[-1], batch_size, num_threads),
                extras={
                    "percentage_nonzero": ratio_value,
                    "num_nonzero": nnz_value
                },
                min_iters=25)
            self.run_op_benchmark(
                sess,
                cholesky_sparse_matrix.op,
                name=sparse_cholesky_name_template %
                (dense_shape[-1], batch_size, num_threads),
                extras={
                    "percentage_nonzero": ratio_value,
                    "num_nonzero": nnz_value
                },
                min_iters=25)


if __name__ == "__main__":
  test.main()
