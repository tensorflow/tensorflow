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
"""Identity operator in `R^k`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.contrib.distributions.python.ops import operator_pd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops


class OperatorPDIdentity(operator_pd.OperatorPDBase):
  """Identity operator in `R^k`:  `Ax = x`.

  This provides an efficient implementation of the identity as an `OperatorPD`.
  Storage, solves, and matmul are all `O(1)`, independent of batch size.

  In order to be a drop-in replacement for other operators, shape and dtype
  of arguments (e.g. to `matmul`) are checked statically as though this operator
  was an instantiated matrix.

  Dynamic shape checks of arguments are not done since that could impede
  performance.
  """

  def __init__(self, shape, dtype, verify_pd=True, name="OperatorPDIdentity"):
    """Initialize an `OperatorPDIdentity`.

    Args:
      shape:  `int32` rank 1 `Tensor` of length at least 2, and with the last
        two entries equal (since this is a square matrix).
      dtype:  Data type of the matrix that this operator represents.
      verify_pd:  `Boolean`, if `True`, asserts are added to the initialization
        args to ensure they define this operator as a square (batch) matrix.
      name:  Name to prepend to `Ops`.
    """

    # Grab static shape if available now.
    with ops.name_scope(name):
      with ops.name_scope("init", values=[shape]):
        self._dtype = dtypes.as_dtype(dtype)
        self._verify_pd = verify_pd
        self._name = name

        # Store the static shape (if possible) right now before adding the
        # asserts, since the asserts prevent .constant_value from working.
        shape = ops.convert_to_tensor(shape, name="shape")
        self._get_shape = tensor_shape.TensorShape(
            tensor_util.constant_value(shape))
        self._shape_arg = self._check_shape(shape)

  def _check_shape(self, shape):
    """Check that the init arg `shape` defines a valid operator."""
    shape = ops.convert_to_tensor(shape, name="shape")
    if not self._verify_pd:
      return shape

    # Further checks are equivalent to verification that this is positive
    # definite.  Why?  Because the further checks simply check that this is a
    # square matrix, and combining the fact that this is square (and thus maps
    # a vector space R^k onto itself), with the behavior of .matmul(), this must
    # be the identity operator.
    rank = array_ops.size(shape)
    assert_matrix = check_ops.assert_less_equal(2, rank)
    with ops.control_dependencies([assert_matrix]):
      last_dim = array_ops.gather(shape, rank - 1)
      second_to_last_dim = array_ops.gather(shape, rank - 2)
      assert_square = check_ops.assert_equal(last_dim, second_to_last_dim)
      return control_flow_ops.with_dependencies([assert_matrix, assert_square],
                                                shape)

  def _check_x(self, x):
    """Static check that the argument `x` is proper `shape`, `dtype`."""
    # x is a typical argument e.g. to matmul or solve.  In both cases, x should
    # have the same type/shape since this is a square matrix.  These checks are
    # ususally not needed since we ususally have some tensor backing this
    # distribution, and the calls to tf.matmul do a shape/type check.
    #
    # Static checks only for efficiency, the identity should be fast.
    #
    # Why check at all?  Because we want this operator to be swappable for a
    # real Operator.
    if self.dtype != x.dtype:
      raise TypeError(
          "Expected argument \"x\" to have same dtype as this operator (%s).  "
          "Found: %s" % (self.dtype, x.dtype))

    x_shape = x.get_shape()
    self_shape = self.get_shape()
    found_msg = (
        "Found: operator.shape = %s,  x.shape = %s" % (self_shape, x_shape))
    if x_shape.ndims is not None and self_shape.ndims is not None:
      if x_shape.ndims != self_shape.ndims:
        raise ValueError(
            "Expected argument \"x\" to have same tensor rank as this "
            "operator. " + found_msg)
      if x_shape.is_fully_defined() and self_shape.is_fully_defined():
        if x_shape[-2] != self_shape[-1]:
          raise ValueError(
              "Incompatible shapes for matrix-matrix operation.  " + found_msg)

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
    return self._dtype

  def _add_to_tensor(self, mat):
    # Add to a tensor in O(k) time!
    mat_diag = array_ops.matrix_diag_part(mat)
    new_diag = constant_op.constant(1, dtype=self.dtype) + mat_diag
    return array_ops.matrix_set_diag(mat, new_diag)

  def _inv_quadratic_form_on_vectors(self, x):
    self._check_x(x)
    return self._iqfov_via_sqrt_solve(x)

  @property
  def inputs(self):
    """List of tensors that were provided as initialization inputs."""
    return [self._shape]

  def get_shape(self):
    """Static `TensorShape` of entire operator.

    If this operator represents the batch matrix `A` with
    `A.shape = [N1,...,Nn, k, k]`, then this returns
    `TensorShape([N1,...,Nn, k, k])`

    Returns:
      `TensorShape`, statically determined, may be undefined.
    """
    return self._get_shape

  def _shape(self):
    return self._shape_arg

  def _det(self):
    det = array_ops.ones(self.batch_shape(), dtype=self.dtype)
    det.set_shape(self.get_batch_shape())
    return det

  def _batch_log_det(self):
    log_det = array_ops.zeros(self.batch_shape(), dtype=self.dtype)
    log_det.set_shape(self.get_batch_shape())
    return log_det

  def _batch_sqrt_log_det(self):
    s_log_det = array_ops.zeros(self.batch_shape(), dtype=self.dtype)
    s_log_det.set_shape(self.get_batch_shape())
    return s_log_det

  def _batch_matmul(self, x, transpose_x=False):
    if transpose_x:
      x = array_ops.matrix_transpose(x)
    self._check_x(x)
    return x

  def _batch_sqrt_matmul(self, x, transpose_x=False):
    return self._batch_matmul(x, transpose_x=transpose_x)

  def _batch_solve(self, rhs):
    self._check_x(rhs)
    return rhs

  def _batch_sqrt_solve(self, rhs):
    self._check_x(rhs)
    return rhs

  def _to_dense(self):
    diag = array_ops.ones(self.vector_shape(), dtype=self.dtype)
    dense = array_ops.matrix_diag(diag)
    dense.set_shape(self.get_shape())
    return dense

  def _sqrt_to_dense(self):
    return self.to_dense()
