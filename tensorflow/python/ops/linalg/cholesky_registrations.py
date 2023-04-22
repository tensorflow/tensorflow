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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_block_diag
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_kronecker
from tensorflow.python.ops.linalg import linear_operator_lower_triangular


# By default, compute the Cholesky of the dense matrix, and return a
# LowerTriangular operator. Methods below specialize this registration.
@linear_operator_algebra.RegisterCholesky(linear_operator.LinearOperator)
def _cholesky_linear_operator(linop):
  return linear_operator_lower_triangular.LinearOperatorLowerTriangular(
      linalg_ops.cholesky(linop.to_dense()),
      is_non_singular=True,
      is_self_adjoint=False,
      is_square=True)


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
      is_self_adjoint=False,
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
      is_self_adjoint=False,
      is_square=True)
