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
"""Lanczos algorithms."""

# TODO(rmlarsen): Add implementation of symmetric Lanczos algorithm.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.contrib.solvers.python.ops import util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops


def lanczos_bidiag(operator,
                   k,
                   orthogonalize=True,
                   starting_vector=None,
                   name="lanczos_bidiag"):
  """Computes a Lanczos bidiagonalization for a linear operator.

  Computes matrices `U` of shape `[m, k+1]`, `V` of shape `[n, k]` and lower
  bidiagonal matrix `B` of shape `[k+1, k]`, that satisfy the equations
  `A * V = U * B` and `A' * U[:, :-1] = V * B[:-1, :]'`.

  The columns of `U` are orthonormal and form a basis for the Krylov subspace
  `K(A*A', U[:,0])`.

  The columns of `V` are orthonormal and form a basis for the Krylov subspace
  `K(A'*A, A' U[:,0])`.

  Args:
    operator: An object representing a linear operator with attributes:
      - shape: Either a list of integers or a 1-D `Tensor` of type `int32` of
        length 2. `shape[0]` is the dimension on the domain of the operator,
        `shape[1]` is the dimension of the co-domain of the operator. On other
        words, if operator represents an M x N matrix A, `shape` must contain
        `[M, N]`.
      - dtype: The datatype of input to and output from `apply` and
        `apply_adjoint`.
      - apply: Callable object taking a vector `x` as input and returning a
        vector with the result of applying the operator to `x`, i.e. if
       `operator` represents matrix `A`, `apply` should return `A * x`.
      - apply_adjoint: Callable object taking a vector `x` as input and
        returning a vector with the result of applying the adjoint operator
        to `x`, i.e. if `operator` represents matrix `A`, `apply_adjoint` should
        return `conj(transpose(A)) * x`.
    k: An integer or a scalar Tensor of type `int32`. Determines the maximum
      number of steps to run. If an invariant subspace is found, the algorithm
      may terminate before `k` steps have been run.
    orthogonalize: If `True`, perform full orthogonalization. If `False` no
      orthogonalization is performed.
    starting_vector: If not null, must be a `Tensor` of shape `[n]`.
    name: A name scope for the operation.

  Returns:
    output: A namedtuple representing a Lanczos bidiagonalization of
      `operator` with attributes:
      u: A rank-2 `Tensor` of type `operator.dtype` and shape
        `[operator.shape[0], k_actual+1]`, where `k_actual` is the number of
        steps run.
      v: A rank-2 `Tensor` of type `operator.dtype` and shape
        `[operator.shape[1], k_actual]`, where `k_actual` is the number of steps
        run.
      alpha: A rank-1 `Tensor` of type `operator.dtype` and shape `[k]`.
      beta: A rank-1 `Tensor` of type `operator.dtype` and shape `[k]`.
  """

  def tarray(size, dtype, name):
    return tensor_array_ops.TensorArray(
        dtype=dtype, size=size, tensor_array_name=name, clear_after_read=False)

  # Reads a row-vector at location i in tarray and returns it as a
  # column-vector.
  def read_colvec(tarray, i):
    return array_ops.expand_dims(tarray.read(i), -1)

  # Writes an column-vector as a row-vecor at location i in tarray.
  def write_colvec(tarray, colvec, i):
    return tarray.write(i, array_ops.squeeze(colvec))

  # Ephemeral class holding Lanczos bidiagonalization state:
  #   u = left Lanczos vectors
  #   v = right Lanczos vectors
  #   alpha = diagonal of B_k.
  #   beta = subdiagonal of B_k.
  # Notice that we store the left and right Lanczos vectors as the _rows_
  # of u and v. This is done because tensors are stored row-major and
  # TensorArray only supports packing along dimension 0.
  lanzcos_bidiag_state = collections.namedtuple("LanczosBidiagState",
                                                ["u", "v", "alpha", "beta"])

  def update_state(old, i, u, v, alpha, beta):
    return lanzcos_bidiag_state(
        write_colvec(old.u, u, i + 1),
        write_colvec(old.v, v, i),
        old.alpha.write(i, alpha), old.beta.write(i, beta))

  def gram_schmidt_step(j, basis, v):
    """Makes v orthogonal to the j'th vector in basis."""
    v_shape = v.get_shape()
    basis_vec = read_colvec(basis, j)
    v -= math_ops.matmul(basis_vec, v, adjoint_a=True) * basis_vec
    v.set_shape(v_shape)
    return j + 1, basis, v

  def orthogonalize_once(i, basis, v):
    j = constant_op.constant(0, dtype=dtypes.int32)
    _, _, v = control_flow_ops.while_loop(lambda j, basis, v: j < i,
                                          gram_schmidt_step, [j, basis, v])
    return util.l2normalize(v)

  # Iterated modified Gram-Schmidt orthogonalization adapted from PROPACK.
  # TODO(rmlarsen): This is possibly the slowest implementation of
  # iterated Gram-Schmidt orthogonalization since the abacus. Move to C++.
  def orthogonalize_(i, basis, v):
    v_norm = util.l2norm(v)
    v_new, v_new_norm = orthogonalize_once(i, basis, v)
    # If the norm decreases more than 1/sqrt(2), run a second
    # round of MGS. See proof in:
    #   B. N. Parlett, ``The Symmetric Eigenvalue Problem'',
    #   Prentice-Hall, Englewood Cliffs, NJ, 1980. pp. 105-109
    return control_flow_ops.cond(v_new_norm < 0.7071 * v_norm,
                                 lambda: orthogonalize_once(i, basis, v),
                                 lambda: (v_new, v_new_norm))

  def stopping_criterion(i, _):
    # TODO(rmlarsen): Stop if an invariant subspace is detected.
    return i < k

  def lanczos_bidiag_step(i, ls):
    """Extends the Lanczos bidiagonalization ls by one step."""
    u = read_colvec(ls.u, i)
    r = operator.apply_adjoint(u)
    # The shape inference doesn't work across cond, save and reapply the shape.
    r_shape = r.get_shape()
    r = control_flow_ops.cond(
        i > 0, lambda: r - ls.beta.read(i - 1) * read_colvec(ls.v, i - 1),
        lambda: r)
    r.set_shape(r_shape)
    if orthogonalize:
      v, alpha = orthogonalize_(i - 1, ls.v, r)
    else:
      v, alpha = util.l2normalize(r)
    p = operator.apply(v) - alpha * u
    if orthogonalize:
      u, beta = orthogonalize_(i, ls.u, p)
    else:
      u, beta = util.l2normalize(p)

    return i + 1, update_state(ls, i, u, v, alpha, beta)

  with ops.name_scope(name):
    dtype = operator.dtype
    if starting_vector is None:
      starting_vector = random_ops.random_uniform(
          operator.shape[:1], -1, 1, dtype=dtype)
    u0, _ = util.l2normalize(starting_vector)
    ls = lanzcos_bidiag_state(
        u=write_colvec(tarray(k + 1, dtype, "u"), u0, 0),
        v=tarray(k, dtype, "v"),
        alpha=tarray(k, dtype, "alpha"),
        beta=tarray(k, dtype, "beta"))
    i = constant_op.constant(0, dtype=dtypes.int32)
    _, ls = control_flow_ops.while_loop(stopping_criterion, lanczos_bidiag_step,
                                        [i, ls])
    return lanzcos_bidiag_state(
        array_ops.matrix_transpose(ls.u.stack()),
        array_ops.matrix_transpose(ls.v.stack()),
        ls.alpha.stack(), ls.beta.stack())


# TODO(rmlarsen): Implement C++ ops for handling bidiagonal matrices
# efficiently. Such a module should provide
#    - multiplication,
#    - linear system solution by back-substitution,
#    - QR factorization,
#    - SVD.
def bidiag_matmul(matrix, alpha, beta, adjoint_b=False, name="bidiag_matmul"):
  """Multiplies a matrix by a bidiagonal matrix.

  alpha and beta are length k vectors representing the diagonal and first lower
  subdiagonal of (K+1) x K matrix B.
  If adjoint_b is False, computes A * B as follows:

    A * B =  A[:, :-1] * diag(alpha) + A[:, 1:] * diag(beta)

  If  adjoint_b is True, computes A * B[:-1, :]' as follows

    A * B[:-1, :]' =
      A * diag(alpha) + [zeros(m,1), A[:, :-1] * diag(beta[:-1])]

  Args:
    matrix: A rank-2 `Tensor` representing matrix A.
    alpha: A rank-1 `Tensor` representing the diagonal of B.
    beta: A rank-1 `Tensor` representing the lower subdiagonal diagonal of B.
    adjoint_b: `bool` determining what to compute.
    name: A name scope for the operation.

  Returns:
    If `adjoint_b` is False the `A * B` is returned.
    If `adjoint_b` is True the `A * B'` is returned.
  """
  with ops.name_scope(name):
    alpha = array_ops.expand_dims(alpha, 0)
    if adjoint_b is False:
      beta = array_ops.expand_dims(beta, 0)
      return matrix[:, :-1] * alpha + matrix[:, 1:] * beta
    else:
      beta = array_ops.expand_dims(beta[:-1], 0)
      shape = array_ops.shape(matrix)
      zero_column = array_ops.expand_dims(
          array_ops.zeros(
              shape[:1], dtype=matrix.dtype), 1)
      return matrix * alpha + array_ops.concat_v2(
          [zero_column, matrix[:, :-1] * beta], 1)
