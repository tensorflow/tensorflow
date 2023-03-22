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
"""Base class for linear operators."""

import abc
import contextlib

import numpy as np

from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.ops.linalg import slicing
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export

__all__ = ["LinearOperator"]


# TODO(langmore) Use matrix_solve_ls for singular or non-square matrices.
@tf_export("linalg.LinearOperator")
class LinearOperator(
    module.Module, composite_tensor.CompositeTensor, metaclass=abc.ABCMeta):
  """Base class defining a [batch of] linear operator[s].

  Subclasses of `LinearOperator` provide access to common methods on a
  (batch) matrix, without the need to materialize the matrix.  This allows:

  * Matrix free computations
  * Operators that take advantage of special structure, while providing a
    consistent API to users.

  #### Subclassing

  To enable a public method, subclasses should implement the leading-underscore
  version of the method.  The argument signature should be identical except for
  the omission of `name="..."`.  For example, to enable
  `matmul(x, adjoint=False, name="matmul")` a subclass should implement
  `_matmul(x, adjoint=False)`.

  #### Performance contract

  Subclasses should only implement the assert methods
  (e.g. `assert_non_singular`) if they can be done in less than `O(N^3)`
  time.

  Class docstrings should contain an explanation of computational complexity.
  Since this is a high-performance library, attention should be paid to detail,
  and explanations can include constants as well as Big-O notation.

  #### Shape compatibility

  `LinearOperator` subclasses should operate on a [batch] matrix with
  compatible shape.  Class docstrings should define what is meant by compatible
  shape.  Some subclasses may not support batching.

  Examples:

  `x` is a batch matrix with compatible shape for `matmul` if

  ```
  operator.shape = [B1,...,Bb] + [M, N],  b >= 0,
  x.shape =   [B1,...,Bb] + [N, R]
  ```

  `rhs` is a batch matrix with compatible shape for `solve` if

  ```
  operator.shape = [B1,...,Bb] + [M, N],  b >= 0,
  rhs.shape =   [B1,...,Bb] + [M, R]
  ```

  #### Example docstring for subclasses.

  This operator acts like a (batch) matrix `A` with shape
  `[B1,...,Bb, M, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `m x n` matrix.  Again, this matrix `A` may not be materialized, but for
  purposes of identifying and working with compatible arguments the shape is
  relevant.

  Examples:

  ```python
  some_tensor = ... shape = ????
  operator = MyLinOp(some_tensor)

  operator.shape()
  ==> [2, 4, 4]

  operator.log_abs_determinant()
  ==> Shape [2] Tensor

  x = ... Shape [2, 4, 5] Tensor

  operator.matmul(x)
  ==> Shape [2, 4, 5] Tensor
  ```

  #### Shape compatibility

  This operator acts on batch matrices with compatible shape.
  FILL IN WHAT IS MEANT BY COMPATIBLE SHAPE

  #### Performance

  FILL THIS IN

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning:

  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.

  #### Initialization parameters

  All subclasses of `LinearOperator` are expected to pass a `parameters`
  argument to `super().__init__()`.  This should be a `dict` containing
  the unadulterated arguments passed to the subclass `__init__`.  For example,
  `MyLinearOperator` with an initializer should look like:

  ```python
  def __init__(self, operator, is_square=False, name=None):
     parameters = dict(
         operator=operator,
         is_square=is_square,
         name=name
     )
     ...
     super().__init__(..., parameters=parameters)
  ```

   Users can then access `my_linear_operator.parameters` to see all arguments
   passed to its initializer.
  """

  # TODO(b/143910018) Remove graph_parents in V3.
  @deprecation.deprecated_args(None, "Do not pass `graph_parents`.  They will "
                               " no longer be used.", "graph_parents")
  def __init__(self,
               dtype,
               graph_parents=None,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name=None,
               parameters=None):
    """Initialize the `LinearOperator`.

    **This is a private method for subclass use.**
    **Subclasses should copy-paste this `__init__` documentation.**

    Args:
      dtype: The type of the this `LinearOperator`.  Arguments to `matmul` and
        `solve` will have to be this type.
      graph_parents: (Deprecated) Python list of graph prerequisites of this
        `LinearOperator` Typically tensors that are passed during initialization
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `dtype` is real, this is equivalent to being symmetric.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name: A name for this `LinearOperator`.
      parameters: Python `dict` of parameters used to instantiate this
        `LinearOperator`.

    Raises:
      ValueError:  If any member of graph_parents is `None` or not a `Tensor`.
      ValueError:  If hints are set incorrectly.
    """
    # Check and auto-set flags.
    if is_positive_definite:
      if is_non_singular is False:
        raise ValueError("A positive definite matrix is always non-singular.")
      is_non_singular = True

    if is_non_singular:
      if is_square is False:
        raise ValueError("A non-singular matrix is always square.")
      is_square = True

    if is_self_adjoint:
      if is_square is False:
        raise ValueError("A self-adjoint matrix is always square.")
      is_square = True

    self._is_square_set_or_implied_by_hints = is_square

    if graph_parents is not None:
      self._set_graph_parents(graph_parents)
    else:
      self._graph_parents = []
    self._dtype = dtypes.as_dtype(dtype).base_dtype if dtype else dtype
    self._is_non_singular = is_non_singular
    self._is_self_adjoint = is_self_adjoint
    self._is_positive_definite = is_positive_definite
    self._parameters = self._no_dependency(parameters)
    self._parameters_sanitized = False
    self._name = name or type(self).__name__

  @contextlib.contextmanager
  def _name_scope(self, name=None):  # pylint: disable=method-hidden
    """Helper function to standardize op scope."""
    full_name = self.name
    if name is not None:
      full_name += "/" + name
    with ops.name_scope(full_name) as scope:
      yield scope

  @property
  def parameters(self):
    """Dictionary of parameters used to instantiate this `LinearOperator`."""
    return dict(self._parameters)

  @property
  def dtype(self):
    """The `DType` of `Tensor`s handled by this `LinearOperator`."""
    return self._dtype

  @property
  def name(self):
    """Name prepended to all ops created by this `LinearOperator`."""
    return self._name

  @property
  @deprecation.deprecated(None, "Do not call `graph_parents`.")
  def graph_parents(self):
    """List of graph dependencies of this `LinearOperator`."""
    return self._graph_parents

  @property
  def is_non_singular(self):
    return self._is_non_singular

  @property
  def is_self_adjoint(self):
    return self._is_self_adjoint

  @property
  def is_positive_definite(self):
    return self._is_positive_definite

  @property
  def is_square(self):
    """Return `True/False` depending on if this operator is square."""
    # Static checks done after __init__.  Why?  Because domain/range dimension
    # sometimes requires lots of work done in the derived class after init.
    auto_square_check = self.domain_dimension == self.range_dimension
    if self._is_square_set_or_implied_by_hints is False and auto_square_check:
      raise ValueError(
          "User set is_square hint to False, but the operator was square.")
    if self._is_square_set_or_implied_by_hints is None:
      return auto_square_check

    return self._is_square_set_or_implied_by_hints

  @abc.abstractmethod
  def _shape(self):
    # Write this in derived class to enable all static shape methods.
    raise NotImplementedError("_shape is not implemented.")

  @property
  def shape(self):
    """`TensorShape` of this `LinearOperator`.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns
    `TensorShape([B1,...,Bb, M, N])`, equivalent to `A.shape`.

    Returns:
      `TensorShape`, statically determined, may be undefined.
    """
    return self._shape()

  def _shape_tensor(self):
    # This is not an abstractmethod, since we want derived classes to be able to
    # override this with optional kwargs, which can reduce the number of
    # `convert_to_tensor` calls.  See derived classes for examples.
    raise NotImplementedError("_shape_tensor is not implemented.")

  def shape_tensor(self, name="shape_tensor"):
    """Shape of this `LinearOperator`, determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
    `[B1,...,Bb, M, N]`, equivalent to `tf.shape(A)`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      # Prefer to use statically defined shape if available.
      if self.shape.is_fully_defined():
        return linear_operator_util.shape_tensor(self.shape.as_list())
      else:
        return self._shape_tensor()

  @property
  def batch_shape(self):
    """`TensorShape` of batch dimensions of this `LinearOperator`.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns
    `TensorShape([B1,...,Bb])`, equivalent to `A.shape[:-2]`

    Returns:
      `TensorShape`, statically determined, may be undefined.
    """
    # Derived classes get this "for free" once .shape is implemented.
    return self.shape[:-2]

  def batch_shape_tensor(self, name="batch_shape_tensor"):
    """Shape of batch dimensions of this operator, determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
    `[B1,...,Bb]`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._batch_shape_tensor()

  def _batch_shape_tensor(self, shape=None):
    # `shape` may be passed in if this can be pre-computed in a
    # more efficient manner, e.g. without excessive Tensor conversions.
    if self.batch_shape.is_fully_defined():
      return linear_operator_util.shape_tensor(
          self.batch_shape.as_list(), name="batch_shape")
    else:
      shape = self.shape_tensor() if shape is None else shape
      return shape[:-2]

  @property
  def tensor_rank(self, name="tensor_rank"):
    """Rank (in the sense of tensors) of matrix corresponding to this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

    Args:
      name:  A name for this `Op`.

    Returns:
      Python integer, or None if the tensor rank is undefined.
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with self._name_scope(name):  # pylint: disable=not-callable
      return self.shape.ndims

  def tensor_rank_tensor(self, name="tensor_rank_tensor"):
    """Rank (in the sense of tensors) of matrix corresponding to this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`, determined at runtime.
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._tensor_rank_tensor()

  def _tensor_rank_tensor(self, shape=None):
    # `shape` may be passed in if this can be pre-computed in a
    # more efficient manner, e.g. without excessive Tensor conversions.
    if self.tensor_rank is not None:
      return tensor_conversion.convert_to_tensor_v2_with_dispatch(
          self.tensor_rank
      )
    else:
      shape = self.shape_tensor() if shape is None else shape
      return array_ops.size(shape)

  @property
  def domain_dimension(self):
    """Dimension (in the sense of vector spaces) of the domain of this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

    Returns:
      `Dimension` object.
    """
    # Derived classes get this "for free" once .shape is implemented.
    if self.shape.rank is None:
      return tensor_shape.Dimension(None)
    else:
      return self.shape.dims[-1]

  def domain_dimension_tensor(self, name="domain_dimension_tensor"):
    """Dimension (in the sense of vector spaces) of the domain of this operator.

    Determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._domain_dimension_tensor()

  def _domain_dimension_tensor(self, shape=None):
    # `shape` may be passed in if this can be pre-computed in a
    # more efficient manner, e.g. without excessive Tensor conversions.
    dim_value = tensor_shape.dimension_value(self.domain_dimension)
    if dim_value is not None:
      return tensor_conversion.convert_to_tensor_v2_with_dispatch(dim_value)
    else:
      shape = self.shape_tensor() if shape is None else shape
      return shape[-1]

  @property
  def range_dimension(self):
    """Dimension (in the sense of vector spaces) of the range of this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

    Returns:
      `Dimension` object.
    """
    # Derived classes get this "for free" once .shape is implemented.
    if self.shape.dims:
      return self.shape.dims[-2]
    else:
      return tensor_shape.Dimension(None)

  def range_dimension_tensor(self, name="range_dimension_tensor"):
    """Dimension (in the sense of vector spaces) of the range of this operator.

    Determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._range_dimension_tensor()

  def _range_dimension_tensor(self, shape=None):
    # `shape` may be passed in if this can be pre-computed in a
    # more efficient manner, e.g. without excessive Tensor conversions.
    dim_value = tensor_shape.dimension_value(self.range_dimension)
    if dim_value is not None:
      return tensor_conversion.convert_to_tensor_v2_with_dispatch(dim_value)
    else:
      shape = self.shape_tensor() if shape is None else shape
      return shape[-2]

  def _assert_non_singular(self):
    """Private default implementation of _assert_non_singular."""
    logging.warn(
        "Using (possibly slow) default implementation of assert_non_singular."
        "  Requires conversion to a dense matrix and O(N^3) operations.")
    if self._can_use_cholesky():
      return self.assert_positive_definite()
    else:
      singular_values = linalg_ops.svd(self.to_dense(), compute_uv=False)
      # TODO(langmore) Add .eig and .cond as methods.
      cond = (math_ops.reduce_max(singular_values, axis=-1) /
              math_ops.reduce_min(singular_values, axis=-1))
      return check_ops.assert_less(
          cond,
          self._max_condition_number_to_be_non_singular(),
          message="Singular matrix up to precision epsilon.")

  def _max_condition_number_to_be_non_singular(self):
    """Return the maximum condition number that we consider nonsingular."""
    with ops.name_scope("max_nonsingular_condition_number"):
      dtype_eps = np.finfo(self.dtype.as_numpy_dtype).eps
      eps = math_ops.cast(
          math_ops.reduce_max([
              100.,
              math_ops.cast(self.range_dimension_tensor(), self.dtype),
              math_ops.cast(self.domain_dimension_tensor(), self.dtype)
          ]), self.dtype) * dtype_eps
      return 1. / eps

  def assert_non_singular(self, name="assert_non_singular"):
    """Returns an `Op` that asserts this operator is non singular.

    This operator is considered non-singular if

    ```
    ConditionNumber < max{100, range_dimension, domain_dimension} * eps,
    eps := np.finfo(self.dtype.as_numpy_dtype).eps
    ```

    Args:
      name:  A string name to prepend to created ops.

    Returns:
      An `Assert` `Op`, that, when run, will raise an `InvalidArgumentError` if
        the operator is singular.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._assert_non_singular()

  def _assert_positive_definite(self):
    """Default implementation of _assert_positive_definite."""
    logging.warn(
        "Using (possibly slow) default implementation of "
        "assert_positive_definite."
        "  Requires conversion to a dense matrix and O(N^3) operations.")
    # If the operator is self-adjoint, then checking that
    # Cholesky decomposition succeeds + results in positive diag is necessary
    # and sufficient.
    if self.is_self_adjoint:
      return check_ops.assert_positive(
          array_ops.matrix_diag_part(linalg_ops.cholesky(self.to_dense())),
          message="Matrix was not positive definite.")
    # We have no generic check for positive definite.
    raise NotImplementedError("assert_positive_definite is not implemented.")

  def assert_positive_definite(self, name="assert_positive_definite"):
    """Returns an `Op` that asserts this operator is positive definite.

    Here, positive definite means that the quadratic form `x^H A x` has positive
    real part for all nonzero `x`.  Note that we do not require the operator to
    be self-adjoint to be positive definite.

    Args:
      name:  A name to give this `Op`.

    Returns:
      An `Assert` `Op`, that, when run, will raise an `InvalidArgumentError` if
        the operator is not positive definite.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._assert_positive_definite()

  def _assert_self_adjoint(self):
    dense = self.to_dense()
    logging.warn(
        "Using (possibly slow) default implementation of assert_self_adjoint."
        "  Requires conversion to a dense matrix.")
    return check_ops.assert_equal(
        dense,
        linalg.adjoint(dense),
        message="Matrix was not equal to its adjoint.")

  def assert_self_adjoint(self, name="assert_self_adjoint"):
    """Returns an `Op` that asserts this operator is self-adjoint.

    Here we check that this operator is *exactly* equal to its hermitian
    transpose.

    Args:
      name:  A string name to prepend to created ops.

    Returns:
      An `Assert` `Op`, that, when run, will raise an `InvalidArgumentError` if
        the operator is not self-adjoint.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._assert_self_adjoint()

  def _check_input_dtype(self, arg):
    """Check that arg.dtype == self.dtype."""
    if arg.dtype.base_dtype != self.dtype:
      raise TypeError(
          "Expected argument to have dtype %s.  Found: %s in tensor %s" %
          (self.dtype, arg.dtype, arg))

  @abc.abstractmethod
  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    raise NotImplementedError("_matmul is not implemented.")

  def matmul(self, x, adjoint=False, adjoint_arg=False, name="matmul"):
    """Transform [batch] matrix `x` with left multiplication:  `x --> Ax`.

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)
    operator.shape = [..., M, N]

    X = ... # shape [..., N, R], batch matrix, R > 0.

    Y = operator.matmul(X)
    Y.shape
    ==> [..., M, R]

    Y[..., :, r] = sum_j A[..., :, j] X[j, r]
    ```

    Args:
      x: `LinearOperator` or `Tensor` with compatible shape and same `dtype` as
        `self`. See class docstring for definition of compatibility.
      adjoint: Python `bool`.  If `True`, left multiply by the adjoint: `A^H x`.
      adjoint_arg:  Python `bool`.  If `True`, compute `A x^H` where `x^H` is
        the hermitian transpose (transposition and complex conjugation).
      name:  A name for this `Op`.

    Returns:
      A `LinearOperator` or `Tensor` with shape `[..., M, R]` and same `dtype`
        as `self`.
    """
    if isinstance(x, LinearOperator):
      left_operator = self.adjoint() if adjoint else self
      right_operator = x.adjoint() if adjoint_arg else x

      if (right_operator.range_dimension is not None and
          left_operator.domain_dimension is not None and
          right_operator.range_dimension != left_operator.domain_dimension):
        raise ValueError(
            "Operators are incompatible. Expected `x` to have dimension"
            " {} but got {}.".format(
                left_operator.domain_dimension, right_operator.range_dimension))
      with self._name_scope(name):  # pylint: disable=not-callable
        return linear_operator_algebra.matmul(left_operator, right_operator)

    with self._name_scope(name):  # pylint: disable=not-callable
      x = tensor_conversion.convert_to_tensor_v2_with_dispatch(x, name="x")
      self._check_input_dtype(x)

      self_dim = -2 if adjoint else -1
      arg_dim = -1 if adjoint_arg else -2
      tensor_shape.dimension_at_index(
          self.shape, self_dim).assert_is_compatible_with(
              x.shape[arg_dim])

      return self._matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)

  def __matmul__(self, other):
    return self.matmul(other)

  def _matvec(self, x, adjoint=False):
    x_mat = array_ops.expand_dims(x, axis=-1)
    y_mat = self.matmul(x_mat, adjoint=adjoint)
    return array_ops.squeeze(y_mat, axis=-1)

  def matvec(self, x, adjoint=False, name="matvec"):
    """Transform [batch] vector `x` with left multiplication:  `x --> Ax`.

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)

    X = ... # shape [..., N], batch vector

    Y = operator.matvec(X)
    Y.shape
    ==> [..., M]

    Y[..., :] = sum_j A[..., :, j] X[..., j]
    ```

    Args:
      x: `Tensor` with compatible shape and same `dtype` as `self`.
        `x` is treated as a [batch] vector meaning for every set of leading
        dimensions, the last dimension defines a vector.
        See class docstring for definition of compatibility.
      adjoint: Python `bool`.  If `True`, left multiply by the adjoint: `A^H x`.
      name:  A name for this `Op`.

    Returns:
      A `Tensor` with shape `[..., M]` and same `dtype` as `self`.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      x = tensor_conversion.convert_to_tensor_v2_with_dispatch(x, name="x")
      self._check_input_dtype(x)
      self_dim = -2 if adjoint else -1
      tensor_shape.dimension_at_index(
          self.shape, self_dim).assert_is_compatible_with(x.shape[-1])
      return self._matvec(x, adjoint=adjoint)

  def _determinant(self):
    logging.warn(
        "Using (possibly slow) default implementation of determinant."
        "  Requires conversion to a dense matrix and O(N^3) operations.")
    if self._can_use_cholesky():
      return math_ops.exp(self.log_abs_determinant())
    return linalg_ops.matrix_determinant(self.to_dense())

  def determinant(self, name="det"):
    """Determinant for every batch member.

    Args:
      name:  A name for this `Op`.

    Returns:
      `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.

    Raises:
      NotImplementedError:  If `self.is_square` is `False`.
    """
    if self.is_square is False:
      raise NotImplementedError(
          "Determinant not implemented for an operator that is expected to "
          "not be square.")
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._determinant()

  def _log_abs_determinant(self):
    logging.warn(
        "Using (possibly slow) default implementation of determinant."
        "  Requires conversion to a dense matrix and O(N^3) operations.")
    if self._can_use_cholesky():
      diag = array_ops.matrix_diag_part(linalg_ops.cholesky(self.to_dense()))
      return 2 * math_ops.reduce_sum(math_ops.log(diag), axis=[-1])
    _, log_abs_det = linalg.slogdet(self.to_dense())
    return log_abs_det

  def log_abs_determinant(self, name="log_abs_det"):
    """Log absolute value of determinant for every batch member.

    Args:
      name:  A name for this `Op`.

    Returns:
      `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.

    Raises:
      NotImplementedError:  If `self.is_square` is `False`.
    """
    if self.is_square is False:
      raise NotImplementedError(
          "Determinant not implemented for an operator that is expected to "
          "not be square.")
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._log_abs_determinant()

  def _dense_solve(self, rhs, adjoint=False, adjoint_arg=False):
    """Solve by conversion to a dense matrix."""
    if self.is_square is False:  # pylint: disable=g-bool-id-comparison
      raise NotImplementedError(
          "Solve is not yet implemented for non-square operators.")
    rhs = linalg.adjoint(rhs) if adjoint_arg else rhs
    if self._can_use_cholesky():
      return linalg_ops.cholesky_solve(
          linalg_ops.cholesky(self.to_dense()), rhs)
    return linear_operator_util.matrix_solve_with_broadcast(
        self.to_dense(), rhs, adjoint=adjoint)

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    """Default implementation of _solve."""
    logging.warn(
        "Using (possibly slow) default implementation of solve."
        "  Requires conversion to a dense matrix and O(N^3) operations.")
    return self._dense_solve(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)

  def solve(self, rhs, adjoint=False, adjoint_arg=False, name="solve"):
    """Solve (exact or approx) `R` (batch) systems of equations: `A X = rhs`.

    The returned `Tensor` will be close to an exact solution if `A` is well
    conditioned. Otherwise closeness will vary. See class docstring for details.

    Examples:

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)
    operator.shape = [..., M, N]

    # Solve R > 0 linear systems for every member of the batch.
    RHS = ... # shape [..., M, R]

    X = operator.solve(RHS)
    # X[..., :, r] is the solution to the r'th linear system
    # sum_j A[..., :, j] X[..., j, r] = RHS[..., :, r]

    operator.matmul(X)
    ==> RHS
    ```

    Args:
      rhs: `Tensor` with same `dtype` as this operator and compatible shape.
        `rhs` is treated like a [batch] matrix meaning for every set of leading
        dimensions, the last two dimensions defines a matrix.
        See class docstring for definition of compatibility.
      adjoint: Python `bool`.  If `True`, solve the system involving the adjoint
        of this `LinearOperator`:  `A^H X = rhs`.
      adjoint_arg:  Python `bool`.  If `True`, solve `A X = rhs^H` where `rhs^H`
        is the hermitian transpose (transposition and complex conjugation).
      name:  A name scope to use for ops added by this method.

    Returns:
      `Tensor` with shape `[...,N, R]` and same `dtype` as `rhs`.

    Raises:
      NotImplementedError:  If `self.is_non_singular` or `is_square` is False.
    """
    if self.is_non_singular is False:
      raise NotImplementedError(
          "Exact solve not implemented for an operator that is expected to "
          "be singular.")
    if self.is_square is False:
      raise NotImplementedError(
          "Exact solve not implemented for an operator that is expected to "
          "not be square.")
    if isinstance(rhs, LinearOperator):
      left_operator = self.adjoint() if adjoint else self
      right_operator = rhs.adjoint() if adjoint_arg else rhs

      if (right_operator.range_dimension is not None and
          left_operator.domain_dimension is not None and
          right_operator.range_dimension != left_operator.domain_dimension):
        raise ValueError(
            "Operators are incompatible. Expected `rhs` to have dimension"
            " {} but got {}.".format(
                left_operator.domain_dimension, right_operator.range_dimension))
      with self._name_scope(name):  # pylint: disable=not-callable
        return linear_operator_algebra.solve(left_operator, right_operator)

    with self._name_scope(name):  # pylint: disable=not-callable
      rhs = tensor_conversion.convert_to_tensor_v2_with_dispatch(
          rhs, name="rhs"
      )
      self._check_input_dtype(rhs)

      self_dim = -1 if adjoint else -2
      arg_dim = -1 if adjoint_arg else -2
      tensor_shape.dimension_at_index(
          self.shape, self_dim).assert_is_compatible_with(
              rhs.shape[arg_dim])

      return self._solve(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)

  def _solvevec(self, rhs, adjoint=False):
    """Default implementation of _solvevec."""
    rhs_mat = array_ops.expand_dims(rhs, axis=-1)
    solution_mat = self.solve(rhs_mat, adjoint=adjoint)
    return array_ops.squeeze(solution_mat, axis=-1)

  def solvevec(self, rhs, adjoint=False, name="solve"):
    """Solve single equation with best effort: `A X = rhs`.

    The returned `Tensor` will be close to an exact solution if `A` is well
    conditioned. Otherwise closeness will vary. See class docstring for details.

    Examples:

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)
    operator.shape = [..., M, N]

    # Solve one linear system for every member of the batch.
    RHS = ... # shape [..., M]

    X = operator.solvevec(RHS)
    # X is the solution to the linear system
    # sum_j A[..., :, j] X[..., j] = RHS[..., :]

    operator.matvec(X)
    ==> RHS
    ```

    Args:
      rhs: `Tensor` with same `dtype` as this operator.
        `rhs` is treated like a [batch] vector meaning for every set of leading
        dimensions, the last dimension defines a vector.  See class docstring
        for definition of compatibility regarding batch dimensions.
      adjoint: Python `bool`.  If `True`, solve the system involving the adjoint
        of this `LinearOperator`:  `A^H X = rhs`.
      name:  A name scope to use for ops added by this method.

    Returns:
      `Tensor` with shape `[...,N]` and same `dtype` as `rhs`.

    Raises:
      NotImplementedError:  If `self.is_non_singular` or `is_square` is False.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      rhs = tensor_conversion.convert_to_tensor_v2_with_dispatch(
          rhs, name="rhs"
      )
      self._check_input_dtype(rhs)
      self_dim = -1 if adjoint else -2
      tensor_shape.dimension_at_index(
          self.shape, self_dim).assert_is_compatible_with(rhs.shape[-1])

      return self._solvevec(rhs, adjoint=adjoint)

  def adjoint(self, name="adjoint"):
    """Returns the adjoint of the current `LinearOperator`.

    Given `A` representing this `LinearOperator`, return `A*`.
    Note that calling `self.adjoint()` and `self.H` are equivalent.

    Args:
      name:  A name for this `Op`.

    Returns:
      `LinearOperator` which represents the adjoint of this `LinearOperator`.
    """
    if self.is_self_adjoint is True:  # pylint: disable=g-bool-id-comparison
      return self
    with self._name_scope(name):  # pylint: disable=not-callable
      return linear_operator_algebra.adjoint(self)

  # self.H is equivalent to self.adjoint().
  H = property(adjoint, None)

  def inverse(self, name="inverse"):
    """Returns the Inverse of this `LinearOperator`.

    Given `A` representing this `LinearOperator`, return a `LinearOperator`
    representing `A^-1`.

    Args:
      name: A name scope to use for ops added by this method.

    Returns:
      `LinearOperator` representing inverse of this matrix.

    Raises:
      ValueError: When the `LinearOperator` is not hinted to be `non_singular`.
    """
    if self.is_square is False:  # pylint: disable=g-bool-id-comparison
      raise ValueError("Cannot take the Inverse: This operator represents "
                       "a non square matrix.")
    if self.is_non_singular is False:  # pylint: disable=g-bool-id-comparison
      raise ValueError("Cannot take the Inverse: This operator represents "
                       "a singular matrix.")

    with self._name_scope(name):  # pylint: disable=not-callable
      return linear_operator_algebra.inverse(self)

  def cholesky(self, name="cholesky"):
    """Returns a Cholesky factor as a `LinearOperator`.

    Given `A` representing this `LinearOperator`, if `A` is positive definite
    self-adjoint, return `L`, where `A = L L^T`, i.e. the cholesky
    decomposition.

    Args:
      name:  A name for this `Op`.

    Returns:
      `LinearOperator` which represents the lower triangular matrix
      in the Cholesky decomposition.

    Raises:
      ValueError: When the `LinearOperator` is not hinted to be positive
        definite and self adjoint.
    """

    if not self._can_use_cholesky():
      raise ValueError("Cannot take the Cholesky decomposition: "
                       "Not a positive definite self adjoint matrix.")
    with self._name_scope(name):  # pylint: disable=not-callable
      return linear_operator_algebra.cholesky(self)

  def _to_dense(self):
    """Generic and often inefficient implementation.  Override often."""
    if self.batch_shape.is_fully_defined():
      batch_shape = self.batch_shape
    else:
      batch_shape = self.batch_shape_tensor()

    dim_value = tensor_shape.dimension_value(self.domain_dimension)
    if dim_value is not None:
      n = dim_value
    else:
      n = self.domain_dimension_tensor()

    eye = linalg_ops.eye(num_rows=n, batch_shape=batch_shape, dtype=self.dtype)
    return self.matmul(eye)

  def to_dense(self, name="to_dense"):
    """Return a dense (batch) matrix representing this operator."""
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._to_dense()

  def _diag_part(self):
    """Generic and often inefficient implementation.  Override often."""
    return array_ops.matrix_diag_part(self.to_dense())

  def diag_part(self, name="diag_part"):
    """Efficiently get the [batch] diagonal part of this operator.

    If this operator has shape `[B1,...,Bb, M, N]`, this returns a
    `Tensor` `diagonal`, of shape `[B1,...,Bb, min(M, N)]`, where
    `diagonal[b1,...,bb, i] = self.to_dense()[b1,...,bb, i, i]`.

    ```
    my_operator = LinearOperatorDiag([1., 2.])

    # Efficiently get the diagonal
    my_operator.diag_part()
    ==> [1., 2.]

    # Equivalent, but inefficient method
    tf.linalg.diag_part(my_operator.to_dense())
    ==> [1., 2.]
    ```

    Args:
      name:  A name for this `Op`.

    Returns:
      diag_part:  A `Tensor` of same `dtype` as self.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._diag_part()

  def _trace(self):
    return math_ops.reduce_sum(self.diag_part(), axis=-1)

  def trace(self, name="trace"):
    """Trace of the linear operator, equal to sum of `self.diag_part()`.

    If the operator is square, this is also the sum of the eigenvalues.

    Args:
      name:  A name for this `Op`.

    Returns:
      Shape `[B1,...,Bb]` `Tensor` of same `dtype` as `self`.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._trace()

  def _add_to_tensor(self, x):
    # Override if a more efficient implementation is available.
    return self.to_dense() + x

  def add_to_tensor(self, x, name="add_to_tensor"):
    """Add matrix represented by this operator to `x`.  Equivalent to `A + x`.

    Args:
      x:  `Tensor` with same `dtype` and shape broadcastable to `self.shape`.
      name:  A name to give this `Op`.

    Returns:
      A `Tensor` with broadcast shape and same `dtype` as `self`.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      x = tensor_conversion.convert_to_tensor_v2_with_dispatch(x, name="x")
      self._check_input_dtype(x)
      return self._add_to_tensor(x)

  def _eigvals(self):
    return linalg_ops.self_adjoint_eigvals(self.to_dense())

  def eigvals(self, name="eigvals"):
    """Returns the eigenvalues of this linear operator.

    If the operator is marked as self-adjoint (via `is_self_adjoint`)
    this computation can be more efficient.

    Note: This currently only supports self-adjoint operators.

    Args:
      name:  A name for this `Op`.

    Returns:
      Shape `[B1,...,Bb, N]` `Tensor` of same `dtype` as `self`.
    """
    if not self.is_self_adjoint:
      raise NotImplementedError("Only self-adjoint matrices are supported.")
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._eigvals()

  def _cond(self):
    if not self.is_self_adjoint:
      # In general the condition number is the ratio of the
      # absolute value of the largest and smallest singular values.
      vals = linalg_ops.svd(self.to_dense(), compute_uv=False)
    else:
      # For self-adjoint matrices, and in general normal matrices,
      # we can use eigenvalues.
      vals = math_ops.abs(self._eigvals())

    return (math_ops.reduce_max(vals, axis=-1) /
            math_ops.reduce_min(vals, axis=-1))

  def cond(self, name="cond"):
    """Returns the condition number of this linear operator.

    Args:
      name:  A name for this `Op`.

    Returns:
      Shape `[B1,...,Bb]` `Tensor` of same `dtype` as `self`.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._cond()

  def _can_use_cholesky(self):
    return self.is_self_adjoint and self.is_positive_definite

  def _set_graph_parents(self, graph_parents):
    """Set self._graph_parents.  Called during derived class init.

    This method allows derived classes to set graph_parents, without triggering
    a deprecation warning (which is invoked if `graph_parents` is passed during
    `__init__`.

    Args:
      graph_parents: Iterable over Tensors.
    """
    # TODO(b/143910018) Remove this function in V3.
    graph_parents = [] if graph_parents is None else graph_parents
    for i, t in enumerate(graph_parents):
      if t is None or not (linear_operator_util.is_ref(t) or
                           tensor_util.is_tf_type(t)):
        raise ValueError("Graph parent item %d is not a Tensor; %s." % (i, t))
    self._graph_parents = graph_parents

  @property
  def _composite_tensor_fields(self):
    """A tuple of parameter names to rebuild the `LinearOperator`.

    The tuple contains the names of kwargs to the `LinearOperator`'s constructor
    that the `TypeSpec` needs to rebuild the `LinearOperator` instance.

    "is_non_singular", "is_self_adjoint", "is_positive_definite", and
    "is_square" are common to all `LinearOperator` subclasses and may be
    omitted.
    """
    return ()

  @property
  def _composite_tensor_prefer_static_fields(self):
    """A tuple of names referring to parameters that may be treated statically.

    This is a subset of `_composite_tensor_fields`, and contains the names of
    of `Tensor`-like args to the `LinearOperator`s constructor that may be
    stored as static values, if they are statically known. These are typically
    shapes or axis values.
    """
    return ()

  @property
  def _type_spec(self):
    # This property will be overwritten by the `@make_composite_tensor`
    # decorator. However, we need it so that a valid subclass of the `ABCMeta`
    # class `CompositeTensor` can be constructed and passed to the
    # `@make_composite_tensor` decorator.
    pass

  def _convert_variables_to_tensors(self):
    """Recursively converts ResourceVariables in the LinearOperator to Tensors.

    The usage of `self._type_spec._from_components` violates the contract of
    `CompositeTensor`, since it is called on a different nested structure
    (one containing only `Tensor`s) than `self.type_spec` specifies (one that
    may contain `ResourceVariable`s). Since `LinearOperator`'s
    `_from_components` method just passes the contents of the nested structure
    to `__init__` to rebuild the operator, and any `LinearOperator` that may be
    instantiated with `ResourceVariables` may also be instantiated with
    `Tensor`s, this usage is valid.

    Returns:
      tensor_operator: `self` with all internal Variables converted to Tensors.
    """
    # pylint: disable=protected-access
    components = self._type_spec._to_components(self)
    tensor_components = variable_utils.convert_variables_to_tensors(
        components)
    return self._type_spec._from_components(tensor_components)
    # pylint: enable=protected-access

  def __getitem__(self, slices):
    return slicing.batch_slice(self, params_overrides={}, slices=slices)

  @property
  def _experimental_parameter_ndims_to_matrix_ndims(self):
    """A dict of names to number of dimensions contributing to an operator.

    This is a dictionary of parameter names to `int`s specifying the
    number of right-most dimensions contributing to the **matrix** shape of the
    densified operator.
    If the parameter is a `Tensor`, this is mapped to an `int`.
    If the parameter is a `LinearOperator` (called `A`), this specifies the
    number of batch dimensions of `A` contributing to this `LinearOperator`s
    matrix shape.
    If the parameter is a structure, this is a structure of the same type of
    `int`s.
    """
    return ()


class _LinearOperatorSpec(type_spec.BatchableTypeSpec):
  """A tf.TypeSpec for `LinearOperator` objects."""

  __slots__ = ("_param_specs", "_non_tensor_params", "_prefer_static_fields")

  def __init__(self, param_specs, non_tensor_params, prefer_static_fields):
    """Initializes a new `_LinearOperatorSpec`.

    Args:
      param_specs: Python `dict` of `tf.TypeSpec` instances that describe
        kwargs to the `LinearOperator`'s constructor that are `Tensor`-like or
        `CompositeTensor` subclasses.
      non_tensor_params: Python `dict` containing non-`Tensor` and non-
        `CompositeTensor` kwargs to the `LinearOperator`'s constructor.
      prefer_static_fields: Python `tuple` of strings corresponding to the names
        of `Tensor`-like args to the `LinearOperator`s constructor that may be
        stored as static values, if known. These are typically shapes, indices,
        or axis values.
    """
    self._param_specs = param_specs
    self._non_tensor_params = non_tensor_params
    self._prefer_static_fields = prefer_static_fields

  @classmethod
  def from_operator(cls, operator):
    """Builds a `_LinearOperatorSpec` from a `LinearOperator` instance.

    Args:
      operator: An instance of `LinearOperator`.

    Returns:
      linear_operator_spec: An instance of `_LinearOperatorSpec` to be used as
        the `TypeSpec` of `operator`.
    """
    validation_fields = ("is_non_singular", "is_self_adjoint",
                         "is_positive_definite", "is_square")
    kwargs = _extract_attrs(
        operator,
        keys=set(operator._composite_tensor_fields + validation_fields))  # pylint: disable=protected-access

    non_tensor_params = {}
    param_specs = {}
    for k, v in list(kwargs.items()):
      type_spec_or_v = _extract_type_spec_recursively(v)
      is_tensor = [isinstance(x, type_spec.TypeSpec)
                   for x in nest.flatten(type_spec_or_v)]
      if all(is_tensor):
        param_specs[k] = type_spec_or_v
      elif not any(is_tensor):
        non_tensor_params[k] = v
      else:
        raise NotImplementedError(f"Field {k} contains a mix of `Tensor` and "
                                  f" non-`Tensor` values.")

    return cls(
        param_specs=param_specs,
        non_tensor_params=non_tensor_params,
        prefer_static_fields=operator._composite_tensor_prefer_static_fields)  # pylint: disable=protected-access

  def _to_components(self, obj):
    return _extract_attrs(obj, keys=list(self._param_specs))

  def _from_components(self, components):
    kwargs = dict(self._non_tensor_params, **components)
    return self.value_type(**kwargs)

  @property
  def _component_specs(self):
    return self._param_specs

  def _serialize(self):
    return (self._param_specs,
            self._non_tensor_params,
            self._prefer_static_fields)

  def _copy(self, **overrides):
    kwargs = {
        "param_specs": self._param_specs,
        "non_tensor_params": self._non_tensor_params,
        "prefer_static_fields": self._prefer_static_fields
    }
    kwargs.update(overrides)
    return type(self)(**kwargs)

  def _batch(self, batch_size):
    """Returns a TypeSpec representing a batch of objects with this TypeSpec."""
    return self._copy(
        param_specs=nest.map_structure(
            lambda spec: spec._batch(batch_size),  # pylint: disable=protected-access
            self._param_specs))

  def _unbatch(self, batch_size):
    """Returns a TypeSpec representing a single element of this TypeSpec."""
    return self._copy(
        param_specs=nest.map_structure(
            lambda spec: spec._unbatch(),  # pylint: disable=protected-access
            self._param_specs))


def make_composite_tensor(cls, module_name="tf.linalg"):
  """Class decorator to convert `LinearOperator`s to `CompositeTensor`."""

  spec_name = "{}Spec".format(cls.__name__)
  spec_type = type(spec_name, (_LinearOperatorSpec,), {"value_type": cls})
  type_spec_registry.register("{}.{}".format(module_name, spec_name))(spec_type)
  cls._type_spec = property(spec_type.from_operator)  # pylint: disable=protected-access
  return cls


def _extract_attrs(op, keys):
  """Extract constructor kwargs to reconstruct `op`.

  Args:
    op: A `LinearOperator` instance.
    keys: A Python `tuple` of strings indicating the names of the constructor
      kwargs to extract from `op`.

  Returns:
    kwargs: A Python `dict` of kwargs to `op`'s constructor, keyed by `keys`.
  """

  kwargs = {}
  not_found = object()
  for k in keys:
    srcs = [
        getattr(op, k, not_found), getattr(op, "_" + k, not_found),
        getattr(op, "parameters", {}).get(k, not_found),
    ]
    if any(v is not not_found for v in srcs):
      kwargs[k] = [v for v in srcs if v is not not_found][0]
    else:
      raise ValueError(
          f"Could not determine an appropriate value for field `{k}` in object "
          f" `{op}`. Looked for \n"
          f" 1. an attr called `{k}`,\n"
          f" 2. an attr called `_{k}`,\n"
          f" 3. an entry in `op.parameters` with key '{k}'.")
    if k in op._composite_tensor_prefer_static_fields and kwargs[k] is not None:  # pylint: disable=protected-access
      if tensor_util.is_tensor(kwargs[k]):
        static_val = tensor_util.constant_value(kwargs[k])
        if static_val is not None:
          kwargs[k] = static_val
    if isinstance(kwargs[k], (np.ndarray, np.generic)):
      kwargs[k] = kwargs[k].tolist()
  return kwargs


def _extract_type_spec_recursively(value):
  """Return (collection of) `TypeSpec`(s) for `value` if it includes `Tensor`s.

  If `value` is a `Tensor` or `CompositeTensor`, return its `TypeSpec`. If
  `value` is a collection containing `Tensor` values, recursively supplant them
  with their respective `TypeSpec`s in a collection of parallel stucture.

  If `value` is none of the above, return it unchanged.

  Args:
    value: a Python `object` to (possibly) turn into a (collection of)
    `tf.TypeSpec`(s).

  Returns:
    spec: the `TypeSpec` or collection of `TypeSpec`s corresponding to `value`
    or `value`, if no `Tensor`s are found.
  """
  if isinstance(value, composite_tensor.CompositeTensor):
    return value._type_spec  # pylint: disable=protected-access
  if isinstance(value, variables.Variable):
    return resource_variable_ops.VariableSpec(
        value.shape, dtype=value.dtype, trainable=value.trainable)
  if tensor_util.is_tensor(value):
    return tensor_spec.TensorSpec(value.shape, value.dtype)
  # Unwrap trackable data structures to comply with `Type_Spec._serialize`
  # requirements. `ListWrapper`s are converted to `list`s, and for other
  # trackable data structures, the `__wrapped__` attribute is used.
  if isinstance(value, list):
    return list(_extract_type_spec_recursively(v) for v in value)
  if isinstance(value, data_structures.TrackableDataStructure):
    return _extract_type_spec_recursively(value.__wrapped__)
  if isinstance(value, tuple):
    return type(value)(_extract_type_spec_recursively(x) for x in value)
  if isinstance(value, dict):
    return type(value)((k, _extract_type_spec_recursively(v))
                       for k, v in value.items())
  return value


# Overrides for tf.linalg functions. This allows a LinearOperator to be used in
# place of a Tensor.
# For instance tf.trace(linop) and linop.trace() both work.


@dispatch.dispatch_for_types(linalg.adjoint, LinearOperator)
def _adjoint(matrix, name=None):
  return matrix.adjoint(name)


@dispatch.dispatch_for_types(linalg.cholesky, LinearOperator)
def _cholesky(input, name=None):   # pylint:disable=redefined-builtin
  return input.cholesky(name)


# The signature has to match with the one in python/op/array_ops.py,
# so we have k, padding_value, and align even though we don't use them here.
# pylint:disable=unused-argument
@dispatch.dispatch_for_types(linalg.diag_part, LinearOperator)
def _diag_part(
    input,  # pylint:disable=redefined-builtin
    name="diag_part",
    k=0,
    padding_value=0,
    align="RIGHT_LEFT"):
  return input.diag_part(name)
# pylint:enable=unused-argument


@dispatch.dispatch_for_types(linalg.det, LinearOperator)
def _det(input, name=None):  # pylint:disable=redefined-builtin
  return input.determinant(name)


@dispatch.dispatch_for_types(linalg.inv, LinearOperator)
def _inverse(input, adjoint=False, name=None):   # pylint:disable=redefined-builtin
  inv = input.inverse(name)
  if adjoint:
    inv = inv.adjoint()
  return inv


@dispatch.dispatch_for_types(linalg.logdet, LinearOperator)
def _logdet(matrix, name=None):
  if matrix.is_positive_definite and matrix.is_self_adjoint:
    return matrix.log_abs_determinant(name)
  raise ValueError("Expected matrix to be self-adjoint positive definite.")


@dispatch.dispatch_for_types(math_ops.matmul, LinearOperator)
def _matmul(  # pylint:disable=missing-docstring
    a,
    b,
    transpose_a=False,
    transpose_b=False,
    adjoint_a=False,
    adjoint_b=False,
    a_is_sparse=False,
    b_is_sparse=False,
    output_type=None,  # pylint: disable=unused-argument
    name=None):
  if transpose_a or transpose_b:
    raise ValueError("Transposing not supported at this time.")
  if a_is_sparse or b_is_sparse:
    raise ValueError("Sparse methods not supported at this time.")
  if not isinstance(a, LinearOperator):
    # We use the identity (B^HA^H)^H =  AB
    adjoint_matmul = b.matmul(
        a,
        adjoint=(not adjoint_b),
        adjoint_arg=(not adjoint_a),
        name=name)
    return linalg.adjoint(adjoint_matmul)
  return a.matmul(
      b, adjoint=adjoint_a, adjoint_arg=adjoint_b, name=name)


@dispatch.dispatch_for_types(linalg.solve, LinearOperator)
def _solve(
    matrix,
    rhs,
    adjoint=False,
    name=None):
  if not isinstance(matrix, LinearOperator):
    raise ValueError("Passing in `matrix` as a Tensor and `rhs` as a "
                     "LinearOperator is not supported.")
  return matrix.solve(rhs, adjoint=adjoint, name=name)


@dispatch.dispatch_for_types(linalg.trace, LinearOperator)
def _trace(x, name=None):
  return x.trace(name)
