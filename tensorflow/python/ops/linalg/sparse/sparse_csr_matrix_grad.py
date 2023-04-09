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
"""CSR Sparse Matrix Gradients."""

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops


@ops.RegisterGradient("DenseToCSRSparseMatrix")
def _DenseToCSRSparseMatrixGrad(op, grad):
  """Gradient for dense_to_csr_sparse_matrix op."""
  grad_values = (
      sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
          grad, type=op.get_attr("T")))
  # inputs to fw op were: params, indices.
  return (grad_values, None)


@ops.RegisterGradient("CSRSparseMatrixToDense")
def _CSRSparseMatrixToDenseGrad(op, grad):
  """Gradient for csr_sparse_matrix_to_dense op."""
  coo_sparse_tensor = sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(
      op.inputs[0], type=grad.dtype)
  return sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
      indices=coo_sparse_tensor.indices,
      values=array_ops.gather_nd(grad, coo_sparse_tensor.indices),
      dense_shape=grad.shape)


@ops.RegisterGradient("SparseTensorToCSRSparseMatrix")
def _SparseTensorToCSRSparseMatrixGrad(op, grad):
  """Gradient for sparse_tensor_to_csr_sparse_matrix op."""
  grad_values = sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(
      grad, type=op.get_attr("T")).values
  return (None, grad_values, None)


@ops.RegisterGradient("CSRSparseMatrixToSparseTensor")
def _CSRSparseMatrixToSparseTensorGrad(op, *grads):
  """Gradient for csr_sparse_matrix_to_sparse_tensor op."""
  return sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
      indices=op.outputs[0], values=grads[1], dense_shape=op.outputs[2])


ops.NotDifferentiable("SparseMatrixNNZ")

ops.NotDifferentiable("SparseMatrixZeros")


def _PruneSparseTensor(unpruned, pruned_pattern):
  """Helper function to prune COO sparse tensor.

  Given two sparse tensors 'unpruned' and 'pruned_pattern', generates another
  sparse tensor with indices and values fron 'unpruned' only if its indices also
  occur in pruned_pattern.

  Args:
    unpruned: COO matrix with unpruned indices
    pruned_pattern: COO matrix with pruned pattern.

  TODO(tabakg): This is far from optimal. Consider a C++ implementation.

  Returns:
    Indices, values, and dense_shape of the pruned matrix.
  """
  pruned_indices = sparse_ops.sparse_reshape(
      pruned_pattern, shape=(-1,)).indices[..., 0]
  unpruned_indices = sparse_ops.sparse_reshape(
      unpruned, shape=(-1,)).indices[..., 0]
  best_match = array_ops.searchsorted(unpruned_indices, pruned_indices)
  keep_indices = array_ops.gather(
      best_match,
      array_ops.where(
          math_ops.equal(
              array_ops.gather(unpruned_indices, best_match), pruned_indices)))
  return (array_ops.gather_nd(unpruned.indices, keep_indices),
          array_ops.gather_nd(unpruned.values,
                              keep_indices), pruned_pattern.dense_shape)


def _PruneCSRMatrix(unpruned, pruned_pattern):
  """TODO(tabakg): Consider re-writing in C++."""
  _, dtype = sparse_csr_matrix_ops.dense_shape_and_type(pruned_pattern)
  coo_unpruned = sparse_tensor.SparseTensor(
      *sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(
          unpruned, type=dtype))
  coo_pruned_pattern = sparse_tensor.SparseTensor(
      *sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(
          pruned_pattern, type=dtype))
  return sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
      *_PruneSparseTensor(coo_unpruned, coo_pruned_pattern))


@ops.RegisterGradient("SparseMatrixAdd")
def _SparseMatrixAddGrad(op, grad):
  """Gradient for sparse_matrix_add op."""
  # input to sparse_matrix_add is (a, b, alpha, beta)
  # with a, b CSR and alpha beta scalars.
  # output is: alpha * a + beta * b

  # d(a*A + b*B)/dA . grad = a * grad

  # May have gotten the transposes wrong below.
  # d(a*A + b*B)/da . grad = tr(A' . grad)

  # For now, only implement gradients w.r.t. A and B.
  # TODO(ebrevdo): Implement reduce_sum for SparseMatrix so that we
  # can implement gradients w.r.t. a and b.
  (a_csr, b_csr, alpha, beta) = op.inputs
  return (sparse_csr_matrix_ops.sparse_matrix_mul(
      _PruneCSRMatrix(grad, a_csr), alpha),
          sparse_csr_matrix_ops.sparse_matrix_mul(
              _PruneCSRMatrix(grad, b_csr), beta), None, None)


def _PrunedDenseMatrixMultiplication(a,
                                     b,
                                     indices,
                                     transpose_a=False,
                                     adjoint_a=False,
                                     transpose_b=False,
                                     adjoint_b=False):
  """Multiplies two dense matrices at selected indices.

  The two inputs `a` and `b` must have matching rank (2 or 3). If using rank 3,
  the first rank is used for the batch number. The last two dimensions should
  also be compatible for matrix multiplication.

  TODO(tabakg): Consider C++ implementation. There is also a more efficient way
  to handle transposes here.

  Args:
    a: The left dense matrix (or batched matrices).
    b: The right dense matrix (or batched matrices).
    indices: The selected output indices where values should be produced. Other
      indices will be pruned (not computed in the first place). Indices are
      specified as a tensor of shape (length, rank), where length is the number
      of entries and rank is the rank of the dense inputs (2 or 3).
    transpose_a: Whether to transpose a.
    adjoint_a: Whether to take the conjugate transpose of a.
    transpose_b: Whether to transpose b.
    adjoint_b: Whether to take the conjugate transpose of b.

  Returns:
    A CSR matrix.
  """
  transpose_a = transpose_a or adjoint_a
  transpose_b = transpose_b or adjoint_b

  a = math_ops.conj(a) if adjoint_a else a
  b = math_ops.conj(b) if adjoint_b else b

  rank = len(a.shape)
  dense_shape = (a.shape[-1] if transpose_a else a.shape[-2],
                 b.shape[-2] if transpose_b else b.shape[-1])
  if rank == 2:
    rows = indices[:, 0]
    cols = indices[:, 1]
    transpose = array_ops.transpose
    gather_op = array_ops.gather
  elif rank == 3:
    dense_shape = (a.shape[0],) + dense_shape
    rows = indices[:, :2]
    cols = array_ops_stack.stack([indices[:, 0], indices[:, 2]], axis=1)
    transpose = lambda x: array_ops.transpose(x, perm=[0, 2, 1])
    gather_op = array_ops.gather_nd

  a_rows = gather_op(transpose(a) if transpose_a else a, indices=rows)
  b_cols = gather_op(b if transpose_b else transpose(b), indices=cols)
  values = math_ops.reduce_sum(a_rows * b_cols, axis=1)

  return sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
      indices=indices, values=values, dense_shape=dense_shape)


@ops.RegisterGradient("SparseMatrixTranspose")
def _SparseMatrixTransposeGrad(op, grad):
  """Gradient for sparse_matrix_transpose op."""
  return sparse_csr_matrix_ops.sparse_matrix_transpose(
      grad, type=op.get_attr("type"), conjugate=op.get_attr("conjugate"))


@ops.RegisterGradient("SparseMatrixSoftmax")
def _SparseMatrixSoftmaxGrad(op, grad_softmax):
  """Gradient for sparse_matrix_softmax op."""
  softmax = op.outputs[0]
  return sparse_csr_matrix_ops.sparse_matrix_softmax_grad(
      softmax, grad_softmax, type=op.get_attr("type"))


@ops.RegisterGradient("SparseMatrixMatMul")
def _SparseMatrixMatMulGrad(op, grad):
  """Gradient for sparse_matrix_mat_mul op."""
  # input to sparse_matrix_mat_mul is (A, B) with CSR A and dense B.
  # Output is dense:
  #   C = opA(A) . opB(B) if transpose_output = false
  #   C = (opA(A) . opB(B))' = opB(B)' . opA(A)' if transpose_output = true.
  # where opA = transpose if transpose_a = True else identity
  # and   opB = transpose if transpose_b = True else identity

  t_a = op.get_attr("transpose_a")
  t_b = op.get_attr("transpose_b")
  adj_a = op.get_attr("adjoint_a")
  adj_b = op.get_attr("adjoint_b")
  transpose_output = op.get_attr("transpose_output")
  conjugate_output = op.get_attr("conjugate_output")
  a = op.inputs[0]  # sparse matrix
  b = op.inputs[1]  # dense matrix
  conj = math_ops.conj
  sparse_matmul = sparse_csr_matrix_ops.sparse_matrix_mat_mul

  def matmul(x, y, **kwargs):  # pylint: disable=invalid-name
    return _PrunedDenseMatrixMultiplication(
        x,
        y,
        indices=sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(
            a, type=x.dtype).indices,
        **kwargs)

  if conjugate_output:
    grad = conj(grad)
  if not transpose_output:
    # C = opA(A) . opB(B)
    if not adj_a and not adj_b:
      a = conj(a)
      b = conj(b)
      if not t_a:
        grad_a = matmul(grad, b, transpose_b=not t_b)
      else:
        grad_a = matmul(b, grad, transpose_a=t_b, transpose_b=True)
      grad_b = sparse_matmul(a, grad, transpose_a=not t_a, transpose_output=t_b)
    elif not t_a and not t_b:
      if not adj_a:
        grad_a = matmul(grad, b, adjoint_b=not adj_b)
      else:
        grad_a = matmul(b, grad, adjoint_a=adj_b, adjoint_b=True)
      grad_b = sparse_matmul(
          a,
          grad,
          adjoint_a=not adj_a,
          transpose_output=adj_b,
          conjugate_output=adj_b)
    elif adj_a and t_b:
      grad_a = matmul(b, grad, transpose_a=True, adjoint_b=True)
      grad_b = sparse_matmul(a, grad, transpose_output=True)
    elif t_a and adj_b:
      grad_a = matmul(b, grad, transpose_a=True, transpose_b=True)
      grad_b = sparse_matmul(
          conj(a), grad, transpose_output=True, conjugate_output=True)
  else:
    # C = (opA(A) . opB(B))' =  opB(B)' . opA(A)'
    if not adj_a and not adj_b:
      a = conj(a)
      b = conj(b)
      if not t_a:
        grad_a = matmul(grad, b, transpose_a=True, transpose_b=not t_b)
      else:
        grad_a = matmul(b, grad, transpose_a=t_b)
      grad_b = sparse_matmul(
          a, grad, transpose_a=not t_a, transpose_b=True, transpose_output=t_b)
    elif not t_a and not t_b:
      if not adj_a:
        grad_a = matmul(grad, b, transpose_a=True, adjoint_b=not adj_b)
      else:
        grad_a = matmul(b, conj(grad), adjoint_a=adj_b)
      grad_b = sparse_matmul(
          a,
          grad,
          adjoint_a=not adj_a,
          transpose_b=True,
          transpose_output=adj_b,
          conjugate_output=adj_b)
    elif adj_a and t_b:
      grad_a = matmul(b, conj(grad), transpose_a=True)
      grad_b = sparse_matmul(a, grad, transpose_b=True, transpose_output=True)
    elif t_a and adj_b:
      grad_a = matmul(b, grad, transpose_a=True)
      grad_b = sparse_matmul(a, grad, adjoint_b=True, transpose_output=True)

  return (grad_a, grad_b)


@ops.RegisterGradient("SparseMatrixSparseMatMul")
def _SparseMatrixSparseMatMulGrad(op, grad):
  """Gradient for sparse_matrix_sparse_mat_mul op."""
  t_a = op.get_attr("transpose_a")
  t_b = op.get_attr("transpose_b")
  adj_a = op.get_attr("adjoint_a")
  adj_b = op.get_attr("adjoint_b")
  dtype = op.get_attr("type")

  # input to sparse_matrix_sparse_mat_mul is (A, B) with CSR A and B.
  # Output is CSR:
  #   C = opA(A) . opB(B)
  # where opA = transpose if transpose_a = True else identity
  # and   opB = transpose if transpose_b = True else identity
  a = op.inputs[0]
  b = op.inputs[1]
  conj = math_ops.conj
  matmul = sparse_csr_matrix_ops.sparse_matrix_sparse_mat_mul
  if not t_a and not t_b:
    if not adj_a:
      if not adj_b:
        grad_a = matmul(grad, b, adjoint_b=True, type=dtype)
        grad_b = matmul(a, grad, adjoint_a=True, type=dtype)
      else:
        grad_a = matmul(grad, b, type=dtype)
        grad_b = matmul(grad, a, adjoint_a=True, type=dtype)
    else:
      if not adj_b:
        grad_a = matmul(b, grad, adjoint_b=True, type=dtype)
        grad_b = matmul(a, grad, type=dtype)
      else:
        grad_a = matmul(b, grad, adjoint_a=True, adjoint_b=True, type=dtype)
        grad_b = matmul(grad, a, adjoint_a=True, adjoint_b=True, type=dtype)
  elif not adj_a and not adj_b:
    if not t_a and t_b:
      grad_a = matmul(grad, conj(b), type=dtype)
      grad_b = matmul(grad, conj(a), transpose_a=True, type=dtype)
    elif t_a and not t_b:
      grad_a = matmul(conj(b), grad, transpose_b=True, type=dtype)
      grad_b = matmul(conj(a), grad, type=dtype)
    else:
      grad_a = matmul(b, grad, adjoint_a=True, transpose_b=True, type=dtype)
      grad_b = matmul(grad, a, transpose_a=True, adjoint_b=True, type=dtype)
  elif adj_a and t_b:
    grad_a = matmul(b, grad, transpose_a=True, adjoint_b=True, type=dtype)
    grad_b = matmul(grad, a, transpose_a=True, transpose_b=True, type=dtype)
  elif t_a and adj_b:
    grad_a = matmul(b, grad, transpose_a=True, transpose_b=True, type=dtype)
    grad_b = matmul(grad, a, adjoint_a=True, transpose_b=True, type=dtype)

  # TODO(tabakg): There should be a C++ function for sparse-sparse
  # multiplication with pre-determined indices, instead of pruning after the
  # multiplication.
  return (_PruneCSRMatrix(grad_a, a), _PruneCSRMatrix(grad_b, b))


@ops.RegisterGradient("SparseMatrixMul")
def _SparseMatrixMulGrad(op, grad):
  """Gradient for sparse_matrix_mul op."""
  # input to sparse_matrix_mul is (A, B) with CSR A and dense B.
  # Output is CSR:
  #   C = A .* B
  del op
  del grad
  raise NotImplementedError
