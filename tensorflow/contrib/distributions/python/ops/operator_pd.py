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
"""Base class for symmetric positive definite operator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


__all__ = [
    'OperatorPDBase',
]


@six.add_metaclass(abc.ABCMeta)
class OperatorPDBase(object):
  """Class representing a (batch) of positive definite matrices `A`.

  This class provides access to functions of a (batch) symmetric positive
  definite (PD) matrix, without the need to materialize them.  In other words,
  this provides means to do "matrix free" computations.

  For example, `my_operator.matmul(x)` computes the result of matrix
  multiplication, and this class is free to do this computation with or without
  ever materializing a matrix.

  In practice, this operator represents a (batch) matrix `A` with shape
  `[N1,...,Nb, k, k]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(n1,...,nb)`, `A[n1,...,nb, : :]` is
  a `k x k` matrix.  Again, this matrix `A` may not be materialized, but for
  purposes of broadcasting this shape will be relevant.

  Since `A` is (batch) positive definite, it has a (or several) square roots `S`
  such that `A = SS^T`.

  For example, if `MyOperator` inherits from `OperatorPDBase`, the user can do

  ```python
  operator = MyOperator(...)  # Initialize with some tensors.
  operator.log_det()

  # Compute the quadratic form x^T A^{-1} x for vector x.
  x = ... # some shape [..., k] tensor
  operator.inv_quadratic_form(x)

  # Matrix multiplication by the square root, S w.
  # If w is iid normal, S w has covariance A.
  w = ... # some shape [..., k, L] tensor, L >= 1
  operator.sqrt_matmul(w)
  ```

  The above three methods, `log_det`, `inv_quadratic_form`, and
  `sqrt_matmul` provide "all" that is necessary to use a covariance matrix
  in a multi-variate normal distribution.  See the class `MVNOperatorPD`.
  """

  @abc.abstractproperty
  def name(self):
    """String name identifying this `Operator`."""
    # return self._name
    pass

  @abc.abstractproperty
  def verify_pd(self):
    """Whether to verify that this `Operator` is positive definite."""
    # return self._verify_pd
    pass

  @abc.abstractproperty
  def dtype(self):
    """Data type of matrix elements of `A`."""
    pass

  def inv_quadratic_form(self, x, name='inv_quadratic_form'):
    """Compute the quadratic form: x^T A^{-1} x.

    Args:
      x: `Tensor` with shape broadcastable to `[N1,...,Nb, k]` and same `dtype`
        as self.
      name:  A name scope to use for ops added by this method.

    Returns:
      `Tensor` holding the square of the norm induced by inverse of `A`.  For
      every broadcast batch member.
    """
    # with ops.name_scope(self.name):
    #   with ops.op_scope([x] + self.inputs, name):
    #     # ... your code here
    pass

  def det(self, name='det'):
    """Determinant for every batch member.

    Args:
      name:  A name scope to use for ops added by this method.

    Returns:
      Determinant for every batch member.
    """
    # Derived classes are encouraged to implement log_det() (since it is
    # usually more stable), and then det() comes for free.
    with ops.name_scope(self.name):
      with ops.op_scope(self.inputs, name):
        return math_ops.exp(self.log_det())

  def log_det(self, name='log_det'):
    """Log of the determinant for every batch member.

    Args:
      name:  A name scope to use for ops added by this method.

    Returns:
      Logarithm of determinant for every batch member.
    """
    # with ops.name_scope(self.name):
    #   with ops.op_scope(self.inputs, name):
    #     # ... your code here
    pass

  @abc.abstractproperty
  def inputs(self):
    """List of tensors that were provided as initialization inputs."""
    pass

  def sqrt_matmul(self, x, name='sqrt_matmul'):
    """Left (batch) matmul `x` by a sqrt of this matrix:  `Sx` where `A = S S^T.

    Args:
      x: `Tensor` with shape broadcastable to `[N1,...,Nb, k]` and same `dtype`
        as self.
      name:  A name scope to use for ops added by this method.

    Returns:
      Shape `[N1,...,Nb, k]` `Tensor` holding the product `S x`.
    """
    # with ops.name_scope(self.name):
    #   with ops.op_scope([x] + self.inputs, name):
    #     # ... your code here
    pass

  @abc.abstractmethod
  def get_shape(self):
    """`TensorShape` giving static shape."""
    pass

  def get_batch_shape(self):
    """`TensorShape` with batch shape."""
    return self.get_shape()[:-2]

  def get_vector_shape(self):
    """`TensorShape` of vectors this operator will work with."""
    return self.get_shape()[:-1]

  @abc.abstractmethod
  def shape(self, name='shape'):
    """Equivalent to `tf.shape(A).`  Equal to `[N1,...,Nb, k, k]`, `b >= 0`.

    Args:
      name:  A name scope to use for ops added by this method.

    Returns:
      `int32` `Tensor`
    """
    # with ops.name_scope(self.name):
    #   with ops.op_scope(self.inputs, name):
    #     # ... your code here
    pass

  def rank(self, name='rank'):
    """Tensor rank.  Equivalent to `tf.rank(A)`.  Will equal `b + 2`.

    If this operator represents the batch matrix `A` with
    `A.shape = [N1,...,Nb, k, k]`, the `rank` is `b + 2`.

    Args:
      name:  A name scope to use for ops added by this method.

    Returns:
      `int32` `Tensor`
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with ops.name_scope(self.name):
      with ops.op_scope(self.inputs, name):
        return array_ops.shape(self.shape())[0]

  def batch_shape(self, name='batch_shape'):
    """Shape of batches associated with this operator.

    If this operator represents the batch matrix `A` with
    `A.shape = [N1,...,Nb, k, k]`, the `batch_shape` is `[N1,...,Nb]`.

    Args:
      name:  A name scope to use for ops added by this method.

    Returns:
      `int32` `Tensor`
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with ops.name_scope(self.name):
      with ops.op_scope(self.inputs, name):
        end = array_ops.pack([self.rank() - 2])
        return array_ops.slice(self.shape(), [0], end)

  def vector_shape(self, name='vector_shape'):
    """Shape of (batch) vectors that this (batch) matrix will multiply.

    If this operator represents the batch matrix `A` with
    `A.shape = [N1,...,Nb, k, k]`, the `vector_shape` is `[N1,...,Nb, k]`.

    Args:
      name:  A name scope to use for ops added by this method.

    Returns:
      `int32` `Tensor`
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with ops.name_scope(self.name):
      with ops.op_scope(self.inputs, name):
        return array_ops.slice(self.shape(), [0], [self.rank() - 1])

  def vector_space_dimension(self, name='vector_space_dimension'):
    """Dimension of vector space on which this acts.  The `k` in `R^k`.

    If this operator represents the batch matrix `A` with
    `A.shape = [N1,...,Nb, k, k]`, the `vector_space_dimension` is `k`.

    Args:
      name:  A name scope to use for ops added by this method.

    Returns:
      `int32` `Tensor`
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with ops.name_scope(self.name):
      with ops.op_scope(self.inputs, name):
        return array_ops.gather(self.shape(), self.rank() - 1)

  def matmul(self, x, name='matmul'):
    """Left multiply `x` by this operator.

    Args:
      x: Shape `[N1,...,Nb, k, L]` `Tensor` with same `dtype` as this operator
      name:  A name to give this `Op`.

    Returns:
      A result equivalent to `tf.batch_matmul(self.to_dense(), x)`.
    """
    # with ops.name_scope(self.name):
    #   with ops.op_scope([x] + self.inputs, name):
    #     # ... your code here
    raise NotImplementedError('This operator has no batch_matmul Op.')

  def to_dense(self, name='to_dense'):
    """Return a dense (batch) matrix representing this operator."""
    # with ops.name_scope(self.name):
    #   with ops.op_scope(self.inputs, name):
    #     # ... your code here
    raise NotImplementedError('This operator has no dense representation.')

  def to_dense_sqrt(self, name='to_dense_sqrt'):
    """Return a dense (batch) matrix representing sqrt of this operator."""
    # with ops.name_scope(self.name):
    #   with ops.op_scope(self.inputs, name):
    #     # ... your code here
    raise NotImplementedError('This operator has no dense sqrt representation.')
