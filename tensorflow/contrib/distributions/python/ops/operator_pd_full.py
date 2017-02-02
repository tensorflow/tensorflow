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
"""Symmetric positive definite (PD) Operator defined by a full matrix."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.contrib.distributions.python.ops import operator_pd_cholesky
from tensorflow.python.framework import ops
from tensorflow.python.ops import linalg_ops


__all__ = [
    "OperatorPDFull",
]


class OperatorPDFull(operator_pd_cholesky.OperatorPDCholesky):
  """Class representing a (batch) of positive definite matrices `A`.

  This class provides access to functions of a batch of symmetric positive
  definite (PD) matrices `A` in `R^{k x k}` defined by dense matrices.
  Determinants and solves are `O(k^3)`.

  In practice, this operator represents a (batch) matrix `A` with shape
  `[N1,...,Nb, k, k]` for some `b >= 0`.  The first `b` indices designate a
  batch member.  For every batch member `(n1,...,nb)`, `A[n1,...,nb, : :]` is
  a `k x k` matrix.

  Since `A` is (batch) positive definite, it has a (or several) square roots `S`
  such that `A = SS^T`.

  For example,

  ```python
  distributions = tf.contrib.distributions
  matrix = [[1.0, 0.5], [1.0, 2.0]]
  operator = OperatorPDFull(matrix)
  operator.log_det()

  # Compute the quadratic form x^T A^{-1} x for vector x.
  x = [1.0, 2.0]
  operator.inv_quadratic_form(x)

  # Matrix multiplication by the square root, S w.
  # If w is iid normal, S w has covariance A.
  w = [[1.0], [2.0]]
  operator.sqrt_matmul(w)
  ```

  The above three methods, `log_det`, `inv_quadratic_form`, and
  `sqrt_matmul` provide "all" that is necessary to use a covariance matrix
  in a multi-variate normal distribution.  See the class `MVNOperatorPD`.
  """

  def __init__(self, matrix, verify_pd=True, name="OperatorPDFull"):
    """Initialize an OperatorPDFull.

    Args:
      matrix:  Shape `[N1,...,Nb, k, k]` tensor with `b >= 0`, `k >= 1`.  The
        last two dimensions should be `k x k` symmetric positive definite
        matrices.
      verify_pd: Whether to check that `matrix` is symmetric positive definite.
        If `verify_pd` is `False`, correct behavior is not guaranteed.
      name:  A name to prepend to all ops created by this class.
    """
    with ops.name_scope(name):
      with ops.name_scope("init", values=[matrix]):
        matrix = ops.convert_to_tensor(matrix)
        # Check symmetric here.  Positivity will be verified by checking the
        # diagonal of the Cholesky factor inside the parent class.  The Cholesky
        # factorization linalg_ops.cholesky() does not always fail for non PSD
        # matrices, so don't rely on that.
        if verify_pd:
          matrix = distribution_util.assert_symmetric(matrix)
        chol = linalg_ops.cholesky(matrix)
        super(OperatorPDFull, self).__init__(chol, verify_pd=verify_pd)
