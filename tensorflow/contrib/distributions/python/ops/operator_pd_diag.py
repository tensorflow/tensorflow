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
"""Positive definite Operator defined with diagonal covariance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

from tensorflow.contrib.distributions.python.ops import operator_pd
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

__all__ = [
    "OperatorPDDiag",
    "OperatorPDSqrtDiag",
]


@six.add_metaclass(abc.ABCMeta)
class OperatorPDDiagBase(operator_pd.OperatorPDBase):
  """Base class for diagonal operators."""

  def __init__(self, diag, verify_pd=True, name="OperatorPDDiagBase"):
    self._verify_pd = verify_pd
    self._name = name
    with ops.name_scope(name):
      with ops.name_scope("init", values=[diag]):
        self._diag = self._check_diag(diag)

  def _check_diag(self, diag):
    """Verify that `diag` is positive."""
    diag = ops.convert_to_tensor(diag, name="diag")
    if not self.verify_pd:
      return diag
    deps = [check_ops.assert_positive(diag)]
    return control_flow_ops.with_dependencies(deps, diag)

  @property
  def name(self):
    """String name identifying this `Operator`."""
    return self._name

  @property
  def verify_pd(self):
    """Whether to verify that this `Operator` is positive definite."""
    return self._verify_pd

  @property
  def dtype(self):
    """Data type of matrix elements of `A`."""
    return self._diag.dtype

  @property
  def inputs(self):
    """Initialization arguments."""
    return [self._diag]

  def get_shape(self):
    """`TensorShape` giving static shape."""
    # If d_shape = [5, 3], we return [5, 3, 3].
    d_shape = self._diag.get_shape()
    return d_shape.concatenate(d_shape[-1:])

  def _shape(self):
    d_shape = array_ops.shape(self._diag)
    k = array_ops.gather(d_shape, array_ops.size(d_shape) - 1)
    return array_ops.concat_v2((d_shape, [k]), 0)

  @abc.abstractmethod
  def _batch_log_det(self):
    pass

  @abc.abstractmethod
  def _inv_quadratic_form_on_vectors(self, x):
    pass

  @abc.abstractmethod
  def _batch_matmul(self, x, transpose_x=False):
    pass

  @abc.abstractmethod
  def _batch_sqrt_matmul(self, x, transpose_x=False):
    pass

  @abc.abstractmethod
  def _batch_solve(self, rhs):
    pass

  @abc.abstractmethod
  def _batch_sqrt_solve(self, rhs):
    pass

  @abc.abstractmethod
  def _to_dense(self):
    pass

  @abc.abstractmethod
  def _sqrt_to_dense(self):
    pass

  @abc.abstractmethod
  def _add_to_tensor(self, mat):
    pass


class OperatorPDDiag(OperatorPDDiagBase):
  """Class representing a (batch) of positive definite matrices `A`.

  This class provides access to functions of a batch of symmetric positive
  definite (PD) matrices `A` in `R^{k x k}`.

  In this case, `A` is diagonal and is defined by a provided tensor `diag`,
  `A_{ii} = diag[i]`.

  Determinants, solves, and storage are `O(k)`.

  In practice, this operator represents a (batch) matrix `A` with shape
  `[N1,...,Nn, k, k]` for some `n >= 0`.  The first `n` indices designate a
  batch member.  For every batch member `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  a `k x k` matrix.

  For example,

  ```python
  distributions = tf.contrib.distributions
  diag = [1.0, 2.0]
  operator = OperatorPDDiag(diag)
  operator.det()  # ==> (1 * 2)

  # Compute the quadratic form x^T A^{-1} x for vector x.
  x = [1.0, 2.0]
  operator.inv_quadratic_form_on_vectors(x)

  # Matrix multiplication by the square root, S w, with A = S S^T.
  # Recall A is diagonal, and so then is S, with  S_{ij} = sqrt(A_{ij}).
  # If w is iid normal, S w has covariance A.
  w = [[1.0],
       [2.0]]
  operator.sqrt_matmul(w)
  ```

  The above three methods, `log_det`, `inv_quadratic_form_on_vectors`, and
  `sqrt_matmul` provide "all" that is necessary to use a covariance matrix
  in a multi-variate normal distribution.  See the class
  `MultivariateNormalDiag`.
  """

  def __init__(self, diag, verify_pd=True, name="OperatorPDDiag"):
    """Initialize an OperatorPDDiag.

    Args:
      diag:  Shape `[N1,...,Nn, k]` positive tensor with `n >= 0`, `k >= 1`.
      verify_pd: Whether to check `diag` is positive.
      name:  A name to prepend to all ops created by this class.
    """
    super(OperatorPDDiag, self).__init__(
        diag, verify_pd=verify_pd, name=name)

  def _batch_log_det(self):
    return math_ops.reduce_sum(
        math_ops.log(math_ops.abs(self._diag)), reduction_indices=[-1])

  def _inv_quadratic_form_on_vectors(self, x):
    return self._iqfov_via_solve(x)

  def _batch_matmul(self, x, transpose_x=False):
    if transpose_x:
      x = array_ops.matrix_transpose(x)
    diag_mat = array_ops.expand_dims(self._diag, -1)
    return diag_mat * x

  def _batch_sqrt_matmul(self, x, transpose_x=False):
    if transpose_x:
      x = array_ops.matrix_transpose(x)
    diag_mat = array_ops.expand_dims(self._diag, -1)
    return math_ops.sqrt(diag_mat) * x

  def _batch_solve(self, rhs):
    diag_mat = array_ops.expand_dims(self._diag, -1)
    return rhs / diag_mat

  def _batch_sqrt_solve(self, rhs):
    diag_mat = array_ops.expand_dims(self._diag, -1)
    return rhs / math_ops.sqrt(diag_mat)

  def _to_dense(self):
    return array_ops.matrix_diag(self._diag)

  def _sqrt_to_dense(self):
    return array_ops.matrix_diag(math_ops.sqrt(self._diag))

  def _add_to_tensor(self, mat):
    mat_diag = array_ops.matrix_diag_part(mat)
    new_diag = self._diag + mat_diag
    return array_ops.matrix_set_diag(mat, new_diag)


class OperatorPDSqrtDiag(OperatorPDDiagBase):
  """Class representing a (batch) of positive definite matrices `A`.

  This class provides access to functions of a batch of symmetric positive
  definite (PD) matrices `A` in `R^{k x k}` defined by their square root,
  `S`, such that `A = SS^T`.

  In this case, `S` is diagonal and is defined by a provided tensor `diag`,
  `S_{ii} = diag[i]`.  As a result, `A` is diagonal with `A_{ii} = diag[i]**2`.

  Determinants, solves, and storage are `O(k)`.

  In practice, this operator represents a (batch) matrix `A` with shape
  `[N1,...,Nn, k, k]` for some `n >= 0`.  The first `n` indices designate a
  batch member.  For every batch member `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  a `k x k` matrix.

  For example,

  ```python
  distributions = tf.contrib.distributions
  diag = [1.0, 2.0]
  operator = OperatorPDSqrtDiag(diag)
  operator.det()  # ==> (1 * 2)**2

  # Compute the quadratic form x^T A^{-1} x for vector x.
  x = [1.0, 2.0]
  operator.inv_quadratic_form_on_vectors(x)

  # Matrix multiplication by the square root, S w.
  # If w is iid normal, S w has covariance A.
  w = [[1.0], [2.0]]
  operator.sqrt_matmul(w)
  ```

  The above three methods, `log_det`, `inv_quadratic_form_on_vectors`, and
  `sqrt_matmul` provide "all" that is necessary to use a covariance matrix
  in a multi-variate normal distribution.  See the class
  `MultivariateNormalDiag`.
  """

  def __init__(self, diag, verify_pd=True, name="OperatorPDSqrtDiag"):
    """Initialize an OperatorPDSqrtDiag.

    Args:
      diag:  Shape `[N1,...,Nn, k]` positive tensor with `n >= 0`, `k >= 1`.
      verify_pd: Whether to check `diag` is positive.
      name:  A name to prepend to all ops created by this class.
    """
    super(OperatorPDSqrtDiag, self).__init__(
        diag, verify_pd=verify_pd, name=name)

  def _batch_log_det(self):
    return 2 * math_ops.reduce_sum(
        math_ops.log(math_ops.abs(self._diag)),
        reduction_indices=[-1])

  def _inv_quadratic_form_on_vectors(self, x):
    # This Operator is defined in terms of diagonal entries of the sqrt.
    return self._iqfov_via_sqrt_solve(x)

  def _batch_matmul(self, x, transpose_x=False):
    if transpose_x:
      x = array_ops.matrix_transpose(x)
    diag_mat = array_ops.expand_dims(self._diag, -1)
    return math_ops.square(diag_mat) * x

  def _batch_sqrt_matmul(self, x, transpose_x=False):
    if transpose_x:
      x = array_ops.matrix_transpose(x)
    diag_mat = array_ops.expand_dims(self._diag, -1)
    return diag_mat * x

  def _batch_solve(self, rhs):
    diag_mat = array_ops.expand_dims(self._diag, -1)
    return rhs / math_ops.square(diag_mat)

  def _batch_sqrt_solve(self, rhs):
    diag_mat = array_ops.expand_dims(self._diag, -1)
    return rhs / diag_mat

  def _to_dense(self):
    return array_ops.matrix_diag(math_ops.square(self._diag))

  def _sqrt_to_dense(self):
    return array_ops.matrix_diag(self._diag)

  def _add_to_tensor(self, mat):
    mat_diag = array_ops.matrix_diag_part(mat)
    new_diag = math_ops.square(self._diag) + mat_diag
    return array_ops.matrix_set_diag(mat, new_diag)
