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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
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
  del op  # Unused
  return sparse_csr_matrix_ops.dense_to_csr_sparse_matrix(
      grad, array_ops.stop_gradient(array_ops.where(math_ops.abs(grad) > 0)))


ops.NotDifferentiable("SparseMatrixNNZ")

ops.NotDifferentiable("SparseMatrixZeros")


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
  (_, _, alpha, beta) = op.inputs
  return (sparse_csr_matrix_ops.sparse_matrix_mul(grad, alpha),
          sparse_csr_matrix_ops.sparse_matrix_mul(grad, beta), None, None)


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
  matmul = math_ops.matmul

  if conjugate_output:
    grad = conj(grad)
  if not transpose_output:
    # C = opA(A) . opB(B)
    if not adj_a and not adj_b:
      a = conj(a)
      b = conj(b)
      if not t_a:
        grad_a_dense = matmul(grad, b, transpose_b=not t_b)
      else:
        grad_a_dense = matmul(b, grad, transpose_a=t_b, transpose_b=True)
      grad_b = sparse_matmul(a, grad, transpose_a=not t_a, transpose_output=t_b)
    elif not t_a and not t_b:
      if not adj_a:
        grad_a_dense = matmul(grad, b, adjoint_b=not adj_b)
      else:
        grad_a_dense = matmul(b, grad, adjoint_a=adj_b, adjoint_b=True)
      grad_b = sparse_matmul(
          a,
          grad,
          adjoint_a=not adj_a,
          transpose_output=adj_b,
          conjugate_output=adj_b)
    elif adj_a and t_b:
      grad_a_dense = matmul(b, grad, transpose_a=True, adjoint_b=True)
      grad_b = sparse_matmul(a, grad, transpose_output=True)
    elif t_a and adj_b:
      grad_a_dense = matmul(b, grad, transpose_a=True, transpose_b=True)
      grad_b = sparse_matmul(
          conj(a), grad, transpose_output=True, conjugate_output=True)
  else:
    # C = (opA(A) . opB(B))' =  opB(B)' . opA(A)'
    if not adj_a and not adj_b:
      a = conj(a)
      b = conj(b)
      if not t_a:
        grad_a_dense = matmul(grad, b, transpose_a=True, transpose_b=not t_b)
      else:
        grad_a_dense = matmul(b, grad, transpose_a=t_b)
      grad_b = sparse_matmul(
          a, grad, transpose_a=not t_a, transpose_b=True, transpose_output=t_b)
    elif not t_a and not t_b:
      if not adj_a:
        grad_a_dense = matmul(grad, b, transpose_a=True, adjoint_b=not adj_b)
      else:
        grad_a_dense = matmul(b, conj(grad), adjoint_a=adj_b)
      grad_b = sparse_matmul(
          a,
          grad,
          adjoint_a=not adj_a,
          transpose_b=True,
          transpose_output=adj_b,
          conjugate_output=adj_b)
    elif adj_a and t_b:
      grad_a_dense = matmul(b, conj(grad), transpose_a=True)
      grad_b = sparse_matmul(a, grad, transpose_b=True, transpose_output=True)
    elif t_a and adj_b:
      grad_a_dense = matmul(b, grad, transpose_a=True)
      grad_b = sparse_matmul(a, grad, adjoint_b=True, transpose_output=True)

  grad_a = sparse_csr_matrix_ops.dense_to_csr_sparse_matrix(
      grad_a_dense, array_ops.where(math_ops.abs(grad_a_dense) > 0))
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

  return (grad_a, grad_b)


@ops.RegisterGradient("SparseMatrixMul")
def _SparseMatrixMulGrad(op, grad):
  """Gradient for sparse_matrix_mul op."""
  # input to sparse_matrix_mul is (A, B) with CSR A and dense B.
  # Output is CSR:
  #   C = A .* B
  del op
  del grad
  raise NotImplementedError
