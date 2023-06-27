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
"""Registrations for LinearOperator.cholesky."""

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_block_diag
from tensorflow.python.ops.linalg import linear_operator_composition
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_kronecker
from tensorflow.python.ops.linalg import linear_operator_lower_triangular
from tensorflow.python.ops.linalg import linear_operator_util

LinearOperatorLowerTriangular = (
    linear_operator_lower_triangular.LinearOperatorLowerTriangular)


# By default, compute the Cholesky of the dense matrix, and return a
# LowerTriangular operator. Methods below specialize this registration.
@linear_operator_algebra.RegisterCholesky(linear_operator.LinearOperator)
def _cholesky_linear_operator(linop):
  return LinearOperatorLowerTriangular(
      linalg_ops.cholesky(linop.to_dense()),
      is_non_singular=True,
      is_self_adjoint=False,
      is_square=True)


def _is_llt_product(linop):
  """Determines if linop = L @ L.H for L = LinearOperatorLowerTriangular."""
  if len(linop.operators) != 2:
    return False
  if not linear_operator_util.is_aat_form(linop.operators):
    return False
  return isinstance(linop.operators[0], LinearOperatorLowerTriangular)


@linear_operator_algebra.RegisterCholesky(
    linear_operator_composition.LinearOperatorComposition)
def _cholesky_linear_operator_composition(linop):
  """Computes Cholesky(LinearOperatorComposition)."""
  # L @ L.H will be handled with special code below. Why is L @ L.H the most
  # important special case?
  # Note that Diag @ Diag.H  and Diag @ TriL and TriL @ Diag are already
  # compressed to Diag or TriL by diag matmul
  # registration. Similarly for Identity and ScaledIdentity.
  # So these would not appear in a LinearOperatorComposition unless explicitly
  # constructed as such. So the most important thing to check is L @ L.H.
  if not _is_llt_product(linop):
    return LinearOperatorLowerTriangular(
        linalg_ops.cholesky(linop.to_dense()),
        is_non_singular=True,
        is_self_adjoint=False,
        is_square=True)

  left_op = linop.operators[0]

  # left_op.is_positive_definite ==> op already has positive diag. So return it.
  if left_op.is_positive_definite:
    return left_op

  # Recall that the base class has already verified linop.is_positive_definite,
  # else linop.cholesky() would have raised.
  # So in particular, we know the diagonal has nonzero entries.
  # In the generic case, we make op have positive diag by dividing each row
  # by the sign of the diag. This is equivalent to setting A = L @ D where D is
  # diag(sign(1 / L.diag_part())). Then A is lower triangular with positive diag
  # and A @ A^H = L @ D @ D^H @ L^H = L @ L^H = linop.
  # This also works for complex L, since sign(x + iy) = exp(i * angle(x + iy)).
  diag_sign = array_ops.expand_dims(math_ops.sign(left_op.diag_part()), axis=-2)
  return LinearOperatorLowerTriangular(
      tril=left_op.tril / diag_sign,
      is_non_singular=left_op.is_non_singular,
      # L.is_self_adjoint ==> L is diagonal ==> L @ D is diagonal ==> SA
      # L.is_self_adjoint is False ==> L not diagonal ==> L @ D not diag ...
      is_self_adjoint=left_op.is_self_adjoint,
      # L.is_positive_definite ==> L has positive diag ==> L = L @ D
      #   ==> (L @ D).is_positive_definite.
      # L.is_positive_definite is False could result in L @ D being PD or not..
      # Consider L = [[1, 0], [-2, 1]] and quadratic form with x = [1, 1].
      # Note we will already return left_op if left_op.is_positive_definite
      # above, but to be explicit write this below.
      is_positive_definite=True if left_op.is_positive_definite else None,
      is_square=True,
  )


@linear_operator_algebra.RegisterCholesky(
    linear_operator_diag.LinearOperatorDiag)
def _cholesky_diag(diag_operator):
  return linear_operator_diag.LinearOperatorDiag(
      math_ops.sqrt(diag_operator.diag),
      is_non_singular=True,
      is_self_adjoint=True,
      is_positive_definite=True,
      is_square=True)


@linear_operator_algebra.RegisterCholesky(
    linear_operator_identity.LinearOperatorIdentity)
def _cholesky_identity(identity_operator):
  return linear_operator_identity.LinearOperatorIdentity(
      num_rows=identity_operator._num_rows,  # pylint: disable=protected-access
      batch_shape=identity_operator.batch_shape,
      dtype=identity_operator.dtype,
      is_non_singular=True,
      is_self_adjoint=True,
      is_positive_definite=True,
      is_square=True)


@linear_operator_algebra.RegisterCholesky(
    linear_operator_identity.LinearOperatorScaledIdentity)
def _cholesky_scaled_identity(identity_operator):
  return linear_operator_identity.LinearOperatorScaledIdentity(
      num_rows=identity_operator._num_rows,  # pylint: disable=protected-access
      multiplier=math_ops.sqrt(identity_operator.multiplier),
      is_non_singular=True,
      is_self_adjoint=True,
      is_positive_definite=True,
      is_square=True)


@linear_operator_algebra.RegisterCholesky(
    linear_operator_block_diag.LinearOperatorBlockDiag)
def _cholesky_block_diag(block_diag_operator):
  # We take the cholesky of each block on the diagonal.
  return linear_operator_block_diag.LinearOperatorBlockDiag(
      operators=[
          operator.cholesky() for operator in block_diag_operator.operators],
      is_non_singular=True,
      is_self_adjoint=None,  # Let the operators passed in decide.
      is_square=True)


@linear_operator_algebra.RegisterCholesky(
    linear_operator_kronecker.LinearOperatorKronecker)
def _cholesky_kronecker(kronecker_operator):
  # Cholesky decomposition of a Kronecker product is the Kronecker product
  # of cholesky decompositions.
  return linear_operator_kronecker.LinearOperatorKronecker(
      operators=[
          operator.cholesky() for operator in kronecker_operator.operators],
      is_non_singular=True,
      is_self_adjoint=None,  # Let the operators passed in decide.
      is_square=True)
