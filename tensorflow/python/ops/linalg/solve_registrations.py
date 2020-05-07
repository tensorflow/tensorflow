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
"""Registrations for LinearOperator.solve."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_circulant
from tensorflow.python.ops.linalg import linear_operator_composition
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_inversion
from tensorflow.python.ops.linalg import linear_operator_lower_triangular
from tensorflow.python.ops.linalg import registrations_util


# By default, use a LinearOperatorComposition to delay the computation.
@linear_operator_algebra.RegisterSolve(
    linear_operator.LinearOperator, linear_operator.LinearOperator)
def _solve_linear_operator(linop_a, linop_b):
  """Generic solve of two `LinearOperator`s."""
  is_square = registrations_util.is_square(linop_a, linop_b)
  is_non_singular = None
  is_self_adjoint = None
  is_positive_definite = None

  if is_square:
    is_non_singular = registrations_util.combined_non_singular_hint(
        linop_a, linop_b)
  elif is_square is False:  # pylint:disable=g-bool-id-comparison
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False

  return linear_operator_composition.LinearOperatorComposition(
      operators=[
          linear_operator_inversion.LinearOperatorInversion(linop_a),
          linop_b
      ],
      is_non_singular=is_non_singular,
      is_self_adjoint=is_self_adjoint,
      is_positive_definite=is_positive_definite,
      is_square=is_square,
  )


@linear_operator_algebra.RegisterSolve(
    linear_operator_inversion.LinearOperatorInversion,
    linear_operator.LinearOperator)
def _solve_inverse_linear_operator(linop_a, linop_b):
  """Solve inverse of generic `LinearOperator`s."""
  return linop_a.operator.matmul(linop_b)


# Identity
@linear_operator_algebra.RegisterSolve(
    linear_operator_identity.LinearOperatorIdentity,
    linear_operator.LinearOperator)
def _solve_linear_operator_identity_left(identity, linop):
  del identity
  return linop


@linear_operator_algebra.RegisterSolve(
    linear_operator.LinearOperator,
    linear_operator_identity.LinearOperatorIdentity)
def _solve_linear_operator_identity_right(linop, identity):
  del identity
  return linop.inverse()


@linear_operator_algebra.RegisterSolve(
    linear_operator_identity.LinearOperatorScaledIdentity,
    linear_operator_identity.LinearOperatorScaledIdentity)
def _solve_linear_operator_scaled_identity(linop_a, linop_b):
  """Solve of two ScaledIdentity `LinearOperators`."""
  return linear_operator_identity.LinearOperatorScaledIdentity(
      num_rows=linop_a.domain_dimension_tensor(),
      multiplier=linop_b.multiplier / linop_a.multiplier,
      is_non_singular=registrations_util.combined_non_singular_hint(
          linop_a, linop_b),
      is_self_adjoint=registrations_util.combined_commuting_self_adjoint_hint(
          linop_a, linop_b),
      is_positive_definite=(
          registrations_util.combined_commuting_positive_definite_hint(
              linop_a, linop_b)),
      is_square=True)


# Diag.


@linear_operator_algebra.RegisterSolve(
    linear_operator_diag.LinearOperatorDiag,
    linear_operator_diag.LinearOperatorDiag)
def _solve_linear_operator_diag(linop_a, linop_b):
  return linear_operator_diag.LinearOperatorDiag(
      diag=linop_b.diag / linop_a.diag,
      is_non_singular=registrations_util.combined_non_singular_hint(
          linop_a, linop_b),
      is_self_adjoint=registrations_util.combined_commuting_self_adjoint_hint(
          linop_a, linop_b),
      is_positive_definite=(
          registrations_util.combined_commuting_positive_definite_hint(
              linop_a, linop_b)),
      is_square=True)


@linear_operator_algebra.RegisterSolve(
    linear_operator_diag.LinearOperatorDiag,
    linear_operator_identity.LinearOperatorScaledIdentity)
def _solve_linear_operator_diag_scaled_identity_right(
    linop_diag, linop_scaled_identity):
  return linear_operator_diag.LinearOperatorDiag(
      diag=linop_scaled_identity.multiplier / linop_diag.diag,
      is_non_singular=registrations_util.combined_non_singular_hint(
          linop_diag, linop_scaled_identity),
      is_self_adjoint=registrations_util.combined_commuting_self_adjoint_hint(
          linop_diag, linop_scaled_identity),
      is_positive_definite=(
          registrations_util.combined_commuting_positive_definite_hint(
              linop_diag, linop_scaled_identity)),
      is_square=True)


@linear_operator_algebra.RegisterSolve(
    linear_operator_identity.LinearOperatorScaledIdentity,
    linear_operator_diag.LinearOperatorDiag)
def _solve_linear_operator_diag_scaled_identity_left(
    linop_scaled_identity, linop_diag):
  return linear_operator_diag.LinearOperatorDiag(
      diag=linop_diag.diag / linop_scaled_identity.multiplier,
      is_non_singular=registrations_util.combined_non_singular_hint(
          linop_diag, linop_scaled_identity),
      is_self_adjoint=registrations_util.combined_commuting_self_adjoint_hint(
          linop_diag, linop_scaled_identity),
      is_positive_definite=(
          registrations_util.combined_commuting_positive_definite_hint(
              linop_diag, linop_scaled_identity)),
      is_square=True)


@linear_operator_algebra.RegisterSolve(
    linear_operator_diag.LinearOperatorDiag,
    linear_operator_lower_triangular.LinearOperatorLowerTriangular)
def _solve_linear_operator_diag_tril(linop_diag, linop_triangular):
  return linear_operator_lower_triangular.LinearOperatorLowerTriangular(
      tril=linop_triangular.to_dense() / linop_diag.diag[..., None],
      is_non_singular=registrations_util.combined_non_singular_hint(
          linop_diag, linop_triangular),
      # This is safe to do since the Triangular matrix is only self-adjoint
      # when it is a diagonal matrix, and hence commutes.
      is_self_adjoint=registrations_util.combined_commuting_self_adjoint_hint(
          linop_diag, linop_triangular),
      is_positive_definite=None,
      is_square=True)


# Circulant.


@linear_operator_algebra.RegisterSolve(
    linear_operator_circulant.LinearOperatorCirculant,
    linear_operator_circulant.LinearOperatorCirculant)
def _solve_linear_operator_circulant_circulant(linop_a, linop_b):
  return linear_operator_circulant.LinearOperatorCirculant(
      spectrum=linop_b.spectrum / linop_a.spectrum,
      is_non_singular=registrations_util.combined_non_singular_hint(
          linop_a, linop_b),
      is_self_adjoint=registrations_util.combined_commuting_self_adjoint_hint(
          linop_a, linop_b),
      is_positive_definite=(
          registrations_util.combined_commuting_positive_definite_hint(
              linop_a, linop_b)),
      is_square=True)
