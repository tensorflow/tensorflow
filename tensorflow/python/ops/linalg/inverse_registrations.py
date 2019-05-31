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
"""Registrations for LinearOperator.inverse."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_block_diag
from tensorflow.python.ops.linalg import linear_operator_circulant
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_householder
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_inversion
from tensorflow.python.ops.linalg import linear_operator_kronecker


# By default, return LinearOperatorInversion which switched the .matmul
# and .solve methods.
@linear_operator_algebra.RegisterInverse(linear_operator.LinearOperator)
def _inverse_linear_operator(linop):
  return linear_operator_inversion.LinearOperatorInversion(
      linop,
      is_non_singular=linop.is_non_singular,
      is_self_adjoint=linop.is_self_adjoint,
      is_positive_definite=linop.is_positive_definite,
      is_square=linop.is_square)


@linear_operator_algebra.RegisterInverse(
    linear_operator_inversion.LinearOperatorInversion)
def _inverse_inverse_linear_operator(linop_inversion):
  return linop_inversion.operator


@linear_operator_algebra.RegisterInverse(
    linear_operator_diag.LinearOperatorDiag)
def _inverse_diag(diag_operator):
  return linear_operator_diag.LinearOperatorDiag(
      1. / diag_operator.diag,
      is_non_singular=diag_operator.is_non_singular,
      is_self_adjoint=diag_operator.is_self_adjoint,
      is_positive_definite=diag_operator.is_positive_definite,
      is_square=True)


@linear_operator_algebra.RegisterInverse(
    linear_operator_identity.LinearOperatorIdentity)
def _inverse_identity(identity_operator):
  return identity_operator


@linear_operator_algebra.RegisterInverse(
    linear_operator_identity.LinearOperatorScaledIdentity)
def _inverse_scaled_identity(identity_operator):
  return linear_operator_identity.LinearOperatorScaledIdentity(
      num_rows=identity_operator._num_rows,  # pylint: disable=protected-access
      multiplier=1. / identity_operator.multiplier,
      is_non_singular=identity_operator.is_non_singular,
      is_self_adjoint=True,
      is_positive_definite=identity_operator.is_positive_definite,
      is_square=True)


@linear_operator_algebra.RegisterInverse(
    linear_operator_block_diag.LinearOperatorBlockDiag)
def _inverse_block_diag(block_diag_operator):
  # We take the inverse of each block on the diagonal.
  return linear_operator_block_diag.LinearOperatorBlockDiag(
      operators=[
          operator.inverse() for operator in block_diag_operator.operators],
      is_non_singular=block_diag_operator.is_non_singular,
      is_self_adjoint=block_diag_operator.is_self_adjoint,
      is_positive_definite=block_diag_operator.is_positive_definite,
      is_square=True)


@linear_operator_algebra.RegisterInverse(
    linear_operator_kronecker.LinearOperatorKronecker)
def _inverse_kronecker(kronecker_operator):
  # Inverse decomposition of a Kronecker product is the Kronecker product
  # of inverse decompositions.
  return linear_operator_kronecker.LinearOperatorKronecker(
      operators=[
          operator.inverse() for operator in kronecker_operator.operators],
      is_non_singular=kronecker_operator.is_non_singular,
      is_self_adjoint=kronecker_operator.is_self_adjoint,
      is_positive_definite=kronecker_operator.is_positive_definite,
      is_square=True)


@linear_operator_algebra.RegisterInverse(
    linear_operator_circulant.LinearOperatorCirculant)
def _inverse_circulant(circulant_operator):
  # Inverting the spectrum is sufficient to get the inverse.
  return linear_operator_circulant.LinearOperatorCirculant(
      spectrum=1. / circulant_operator.spectrum,
      is_non_singular=circulant_operator.is_non_singular,
      is_self_adjoint=circulant_operator.is_self_adjoint,
      is_positive_definite=circulant_operator.is_positive_definite,
      is_square=True)


@linear_operator_algebra.RegisterInverse(
    linear_operator_householder.LinearOperatorHouseholder)
def _inverse_householder(householder_operator):
  return householder_operator

