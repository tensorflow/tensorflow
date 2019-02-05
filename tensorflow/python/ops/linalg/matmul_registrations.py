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
"""Registrations for LinearOperator.matmul."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_circulant
from tensorflow.python.ops.linalg import linear_operator_composition
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_lower_triangular
from tensorflow.python.ops.linalg import linear_operator_zeros


def _combined_self_adjoint_hint(operator_a, operator_b):
  """Get combined hint for self-adjoint-ness."""
  # Note: only use this method in the commuting case.
  # The property is preserved under composition when the operators commute.
  if operator_a.is_self_adjoint and operator_b.is_self_adjoint:
    return True

  # The property is not preserved when an operator with the property is composed
  # with an operator without the property.
  if ((operator_a.is_self_adjoint is True and
       operator_b.is_self_adjoint is False) or
      (operator_a.is_self_adjoint is False and
       operator_b.is_self_adjoint is True)):
    return False

  # The property is not known when operators are not known to have the property
  # or both operators don't have the property (the property for the complement
  # class is not closed under composition).
  return None


def _is_square(operator_a, operator_b):
  """Return a hint to whether the composition is square."""
  if operator_a.is_square and operator_b.is_square:
    return True
  if operator_a.is_square is False and operator_b.is_square is False:
    # Let A have shape [B, M, N], B have shape [B, N, L].
    m = operator_a.range_dimension
    l = operator_b.domain_dimension
    if m is not None and l is not None:
      return m == l

    return None


def _combined_positive_definite_hint(operator_a, operator_b):
  """Get combined PD hint for compositions."""
  # Note: Positive definiteness is only guaranteed to be preserved
  # when the operators commute and are symmetric. Only use this method in
  # commuting cases.

  if (operator_a.is_positive_definite is True and
      operator_a.is_self_adjoint is True and
      operator_b.is_positive_definite is True and
      operator_b.is_self_adjoint is True):
    return True

  return None


def _combined_non_singular_hint(operator_a, operator_b):
  """Get combined hint for when ."""
  # If either operator is not-invertible the composition isn't.
  if (operator_a.is_non_singular is False or
      operator_b.is_non_singular is False):
    return False

  return operator_a.is_non_singular and operator_b.is_non_singular


# By default, use a LinearOperatorComposition to delay the computation.
@linear_operator_algebra.RegisterMatmul(
    linear_operator.LinearOperator, linear_operator.LinearOperator)
def _matmul_linear_operator(linop_a, linop_b):
  """Generic matmul of two `LinearOperator`s."""
  is_square = _is_square(linop_a, linop_b)
  is_non_singular = None
  is_self_adjoint = None
  is_positive_definite = None

  if is_square:
    is_non_singular = _combined_non_singular_hint(linop_a, linop_b)
    is_self_adjoint = _combined_self_adjoint_hint(linop_a, linop_b)
  elif is_square is False:
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False

  return linear_operator_composition.LinearOperatorComposition(
      operators=[linop_a, linop_b],
      is_non_singular=is_non_singular,
      is_self_adjoint=is_self_adjoint,
      is_positive_definite=is_positive_definite,
      is_square=is_square,
  )

# Identity


@linear_operator_algebra.RegisterMatmul(
    linear_operator_identity.LinearOperatorIdentity,
    linear_operator.LinearOperator)
def _matmul_linear_operator_identity_left(identity, linop):
  del identity
  return linop


@linear_operator_algebra.RegisterMatmul(
    linear_operator.LinearOperator,
    linear_operator_identity.LinearOperatorIdentity)
def _matmul_linear_operator_identity_right(linop, identity):
  del identity
  return linop


# Zeros


@linear_operator_algebra.RegisterMatmul(
    linear_operator.LinearOperator,
    linear_operator_zeros.LinearOperatorZeros)
def _matmul_linear_operator_zeros_right(linop, zeros):
  if not zeros.is_square or not linop.is_square:
    raise ValueError("Matmul with non-square `LinearOperator`s or non-square "
                     "`LinearOperatorZeros` not supported at this time.")
  return zeros


@linear_operator_algebra.RegisterMatmul(
    linear_operator_zeros.LinearOperatorZeros,
    linear_operator.LinearOperator)
def _matmul_linear_operator_zeros_left(zeros, linop):
  if not zeros.is_square or not linop.is_square:
    raise ValueError("Matmul with non-square `LinearOperator`s or non-square "
                     "`LinearOperatorZeros` not supported at this time.")
  return zeros


# Diag.


@linear_operator_algebra.RegisterMatmul(
    linear_operator_diag.LinearOperatorDiag,
    linear_operator_diag.LinearOperatorDiag)
def _matmul_linear_operator_diag(linop_a, linop_b):
  return linear_operator_diag.LinearOperatorDiag(
      diag=linop_a.diag * linop_b.diag,
      is_non_singular=_combined_non_singular_hint(linop_a, linop_b),
      is_self_adjoint=_combined_self_adjoint_hint(
          linop_a, linop_b),
      is_positive_definite=_combined_positive_definite_hint(
          linop_a, linop_b),
      is_square=True)


@linear_operator_algebra.RegisterMatmul(
    linear_operator_diag.LinearOperatorDiag,
    linear_operator_identity.LinearOperatorScaledIdentity)
def _matmul_linear_operator_diag_scaled_identity_right(
    linop_diag, linop_scaled_identity):
  return linear_operator_diag.LinearOperatorDiag(
      diag=linop_diag.diag * linop_scaled_identity.multiplier,
      is_non_singular=_combined_non_singular_hint(
          linop_diag, linop_scaled_identity),
      is_self_adjoint=_combined_self_adjoint_hint(
          linop_diag, linop_scaled_identity),
      is_positive_definite=_combined_positive_definite_hint(
          linop_diag, linop_scaled_identity),
      is_square=True)


@linear_operator_algebra.RegisterMatmul(
    linear_operator_identity.LinearOperatorScaledIdentity,
    linear_operator_diag.LinearOperatorDiag)
def _matmul_linear_operator_diag_scaled_identity_left(
    linop_scaled_identity, linop_diag):
  return linear_operator_diag.LinearOperatorDiag(
      diag=linop_diag.diag * linop_scaled_identity.multiplier,
      is_non_singular=_combined_non_singular_hint(
          linop_diag, linop_scaled_identity),
      is_self_adjoint=_combined_self_adjoint_hint(
          linop_diag, linop_scaled_identity),
      is_positive_definite=_combined_positive_definite_hint(
          linop_diag, linop_scaled_identity),
      is_square=True)


@linear_operator_algebra.RegisterMatmul(
    linear_operator_diag.LinearOperatorDiag,
    linear_operator_lower_triangular.LinearOperatorLowerTriangular)
def _matmul_linear_operator_diag_tril(linop_diag, linop_triangular):
  return linear_operator_lower_triangular.LinearOperatorLowerTriangular(
      tril=linop_diag.diag[..., None] * linop_triangular.to_dense(),
      is_non_singular=_combined_non_singular_hint(
          linop_diag, linop_triangular),
      # This is safe to do since the Triangular matrix is only self-adjoint
      # when it is a diagonal matrix, and hence commutes.
      is_self_adjoint=_combined_self_adjoint_hint(
          linop_diag, linop_triangular),
      is_positive_definite=None,
      is_square=True)


@linear_operator_algebra.RegisterMatmul(
    linear_operator_lower_triangular.LinearOperatorLowerTriangular,
    linear_operator_diag.LinearOperatorDiag)
def _matmul_linear_operator_tril_diag(linop_triangular, linop_diag):
  return linear_operator_lower_triangular.LinearOperatorLowerTriangular(
      tril=linop_triangular.to_dense() * linop_diag.diag,
      is_non_singular=_combined_non_singular_hint(
          linop_diag, linop_triangular),
      # This is safe to do since the Triangular matrix is only self-adjoint
      # when it is a diagonal matrix, and hence commutes.
      is_self_adjoint=_combined_self_adjoint_hint(
          linop_diag, linop_triangular),
      is_positive_definite=None,
      is_square=True)

# Circulant.


@linear_operator_algebra.RegisterMatmul(
    linear_operator_circulant.LinearOperatorCirculant,
    linear_operator_circulant.LinearOperatorCirculant)
def _matmul_linear_operator_circulant_circulant(linop_a, linop_b):
  return linear_operator_circulant.LinearOperatorCirculant(
      spectrum=linop_a.spectrum * linop_b.spectrum,
      is_non_singular=_combined_non_singular_hint(linop_a, linop_b),
      is_self_adjoint=_combined_self_adjoint_hint(linop_a, linop_b),
      is_positive_definite=_combined_positive_definite_hint(
          linop_a, linop_b),
      is_square=True)
