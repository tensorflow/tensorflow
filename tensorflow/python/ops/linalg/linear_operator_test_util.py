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
"""Utilities for testing `LinearOperator` and sub-classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import itertools
import numpy as np
import six

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import test


class OperatorShapesInfo(object):
  """Object encoding expected shape for a test.

  Encodes the expected shape of a matrix for a test. Also
  allows additional metadata for the test harness.
  """

  def __init__(self, shape, **kwargs):
    self.shape = shape
    self.__dict__.update(kwargs)


class CheckTapeSafeSkipOptions(object):

  # Skip checking this particular method.
  DETERMINANT = "determinant"
  DIAG_PART = "diag_part"
  LOG_ABS_DETERMINANT = "log_abs_determinant"
  TRACE = "trace"


@six.add_metaclass(abc.ABCMeta)  # pylint: disable=no-init
class LinearOperatorDerivedClassTest(test.TestCase):
  """Tests for derived classes.

  Subclasses should implement every abstractmethod, and this will enable all
  test methods to work.
  """

  # Absolute/relative tolerance for tests.
  _atol = {
      dtypes.float16: 1e-3,
      dtypes.float32: 1e-6,
      dtypes.float64: 1e-12,
      dtypes.complex64: 1e-6,
      dtypes.complex128: 1e-12
  }

  _rtol = {
      dtypes.float16: 1e-3,
      dtypes.float32: 1e-6,
      dtypes.float64: 1e-12,
      dtypes.complex64: 1e-6,
      dtypes.complex128: 1e-12
  }

  def assertAC(self, x, y):
    """Derived classes can set _atol, _rtol to get different tolerance."""
    dtype = dtypes.as_dtype(x.dtype)
    atol = self._atol[dtype]
    rtol = self._rtol[dtype]
    self.assertAllClose(x, y, atol=atol, rtol=rtol)

  @staticmethod
  def adjoint_options():
    return [False, True]

  @staticmethod
  def adjoint_arg_options():
    return [False, True]

  @staticmethod
  def dtypes_to_test():
    # TODO(langmore) Test tf.float16 once tf.linalg.solve works in 16bit.
    return [dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128]

  @staticmethod
  def use_placeholder_options():
    return [False, True]

  @staticmethod
  def operator_shapes_infos():
    """Returns list of OperatorShapesInfo, encapsulating the shape to test."""
    raise NotImplementedError("operator_shapes_infos has not been implemented.")

  @abc.abstractmethod
  def operator_and_matrix(
      self, shapes_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    """Build a batch matrix and an Operator that should have similar behavior.

    Every operator acts like a (batch) matrix.  This method returns both
    together, and is used by tests.

    Args:
      shapes_info: `OperatorShapesInfo`, encoding shape information about the
        operator.
      dtype:  Numpy dtype.  Data type of returned array/operator.
      use_placeholder:  Python bool.  If True, initialize the operator with a
        placeholder of undefined shape and correct dtype.
      ensure_self_adjoint_and_pd: If `True`,
        construct this operator to be Hermitian Positive Definite, as well
        as ensuring the hints `is_positive_definite` and `is_self_adjoint`
        are set.
        This is useful for testing methods such as `cholesky`.

    Returns:
      operator:  `LinearOperator` subclass instance.
      mat:  `Tensor` representing operator.
    """
    # Create a matrix as a numpy array with desired shape/dtype.
    # Create a LinearOperator that should have the same behavior as the matrix.
    raise NotImplementedError("Not implemented yet.")

  @abc.abstractmethod
  def make_rhs(self, operator, adjoint, with_batch=True):
    """Make a rhs appropriate for calling operator.solve(rhs).

    Args:
      operator:  A `LinearOperator`
      adjoint:  Python `bool`.  If `True`, we are making a 'rhs' value for the
        adjoint operator.
      with_batch: Python `bool`. If `True`, create `rhs` with the same batch
        shape as operator, and otherwise create a matrix without any batch
        shape.

    Returns:
      A `Tensor`
    """
    raise NotImplementedError("make_rhs is not defined.")

  @abc.abstractmethod
  def make_x(self, operator, adjoint, with_batch=True):
    """Make an 'x' appropriate for calling operator.matmul(x).

    Args:
      operator:  A `LinearOperator`
      adjoint:  Python `bool`.  If `True`, we are making an 'x' value for the
        adjoint operator.
      with_batch: Python `bool`. If `True`, create `x` with the same batch shape
        as operator, and otherwise create a matrix without any batch shape.

    Returns:
      A `Tensor`
    """
    raise NotImplementedError("make_x is not defined.")

  @staticmethod
  def skip_these_tests():
    """List of test names to skip."""
    # Subclasses should over-ride if they want to skip some tests.
    # To skip "test_foo", add "foo" to this list.
    return []

  def assertRaisesError(self, msg):
    """assertRaisesRegexp or OpError, depending on context.executing_eagerly."""
    if context.executing_eagerly():
      return self.assertRaisesRegexp(Exception, msg)
    return self.assertRaisesOpError(msg)

  def check_tape_safe(self, operator, skip_options=None):
    """Check gradients are not None w.r.t. operator.variables.

    Meant to be called from the derived class.

    This ensures grads are not w.r.t every variable in operator.variables.  If
    more fine-grained testing is needed, a custom test should be written.

    Args:
      operator: LinearOperator.  Exact checks done will depend on hints.
      skip_options: Optional list of CheckTapeSafeSkipOptions.
        Makes this test skip particular checks.
    """
    skip_options = skip_options or []

    if not operator.variables:
      raise AssertionError("`operator.variables` was empty")

    def _assert_not_none(iterable):
      for item in iterable:
        self.assertIsNotNone(item)

    # Tape tests that can be run on every operator below.
    with backprop.GradientTape() as tape:
      _assert_not_none(tape.gradient(operator.to_dense(), operator.variables))

    with backprop.GradientTape() as tape:
      _assert_not_none(
          tape.gradient(operator.adjoint().to_dense(), operator.variables))

    x = math_ops.cast(
        array_ops.ones(shape=operator.H.shape_tensor()[:-1]), operator.dtype)

    with backprop.GradientTape() as tape:
      _assert_not_none(tape.gradient(operator.matvec(x), operator.variables))

    # Tests for square, but possibly non-singular operators below.
    if not operator.is_square:
      return

    for option in [
        CheckTapeSafeSkipOptions.DETERMINANT,
        CheckTapeSafeSkipOptions.LOG_ABS_DETERMINANT,
        CheckTapeSafeSkipOptions.DIAG_PART,
        CheckTapeSafeSkipOptions.TRACE,
    ]:
      with backprop.GradientTape() as tape:
        if option not in skip_options:
          _assert_not_none(
              tape.gradient(getattr(operator, option)(), operator.variables))

    # Tests for non-singular operators below.
    if operator.is_non_singular is False:  # pylint: disable=g-bool-id-comparison
      return

    with backprop.GradientTape() as tape:
      _assert_not_none(
          tape.gradient(operator.inverse().to_dense(), operator.variables))

    with backprop.GradientTape() as tape:
      _assert_not_none(tape.gradient(operator.solvevec(x), operator.variables))

    # Tests for SPD operators below.
    if not (operator.is_self_adjoint and operator.is_positive_definite):
      return

    with backprop.GradientTape() as tape:
      _assert_not_none(
          tape.gradient(operator.cholesky().to_dense(), operator.variables))


# pylint:disable=missing-docstring


def _test_to_dense(use_placeholder, shapes_info, dtype):
  def test_to_dense(self):
    with self.session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)
      op_dense = operator.to_dense()
      if not use_placeholder:
        self.assertAllEqual(shapes_info.shape, op_dense.shape)
      op_dense_v, mat_v = sess.run([op_dense, mat])
      self.assertAC(op_dense_v, mat_v)
  return test_to_dense


def _test_det(use_placeholder, shapes_info, dtype):
  def test_det(self):
    with self.session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)
      op_det = operator.determinant()
      if not use_placeholder:
        self.assertAllEqual(shapes_info.shape[:-2], op_det.shape)
      op_det_v, mat_det_v = sess.run(
          [op_det, linalg_ops.matrix_determinant(mat)])
      self.assertAC(op_det_v, mat_det_v)
  return test_det


def _test_log_abs_det(use_placeholder, shapes_info, dtype):
  def test_log_abs_det(self):
    with self.session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)
      op_log_abs_det = operator.log_abs_determinant()
      _, mat_log_abs_det = linalg.slogdet(mat)
      if not use_placeholder:
        self.assertAllEqual(
            shapes_info.shape[:-2], op_log_abs_det.shape)
      op_log_abs_det_v, mat_log_abs_det_v = sess.run(
          [op_log_abs_det, mat_log_abs_det])
      self.assertAC(op_log_abs_det_v, mat_log_abs_det_v)
  return test_log_abs_det


def _test_matmul_base(
    self,
    use_placeholder,
    shapes_info,
    dtype,
    adjoint,
    adjoint_arg,
    with_batch):
  # If batch dimensions are omitted, but there are
  # no batch dimensions for the linear operator, then
  # skip the test case. This is already checked with
  # with_batch=True.
  if not with_batch and len(shapes_info.shape) <= 2:
    return
  with self.session(graph=ops.Graph()) as sess:
    sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
    operator, mat = self.operator_and_matrix(
        shapes_info, dtype, use_placeholder=use_placeholder)
    x = self.make_x(
        operator, adjoint=adjoint, with_batch=with_batch)
    # If adjoint_arg, compute A X^H^H = A X.
    if adjoint_arg:
      op_matmul = operator.matmul(
          linalg.adjoint(x),
          adjoint=adjoint,
          adjoint_arg=adjoint_arg)
    else:
      op_matmul = operator.matmul(x, adjoint=adjoint)
    mat_matmul = math_ops.matmul(mat, x, adjoint_a=adjoint)
    if not use_placeholder:
      self.assertAllEqual(op_matmul.shape,
                          mat_matmul.shape)
    op_matmul_v, mat_matmul_v = sess.run(
        [op_matmul, mat_matmul])
    self.assertAC(op_matmul_v, mat_matmul_v)


def _test_matmul(
    use_placeholder,
    shapes_info,
    dtype,
    adjoint,
    adjoint_arg):
  def test_matmul(self):
    _test_matmul_base(
        self,
        use_placeholder,
        shapes_info,
        dtype,
        adjoint,
        adjoint_arg,
        with_batch=True)
  return test_matmul


def _test_matmul_with_broadcast(
    use_placeholder,
    shapes_info,
    dtype,
    adjoint,
    adjoint_arg):
  def test_matmul_with_broadcast(self):
    _test_matmul_base(
        self,
        use_placeholder,
        shapes_info,
        dtype,
        adjoint,
        adjoint_arg,
        with_batch=True)
  return test_matmul_with_broadcast


def _test_adjoint(use_placeholder, shapes_info, dtype):
  def test_adjoint(self):
    with self.test_session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)
      op_adjoint = operator.adjoint().to_dense()
      op_adjoint_h = operator.H.to_dense()
      mat_adjoint = linalg.adjoint(mat)
      op_adjoint_v, op_adjoint_h_v, mat_adjoint_v = sess.run(
          [op_adjoint, op_adjoint_h, mat_adjoint])
      self.assertAC(mat_adjoint_v, op_adjoint_v)
      self.assertAC(mat_adjoint_v, op_adjoint_h_v)
  return test_adjoint


def _test_cholesky(use_placeholder, shapes_info, dtype):
  def test_cholesky(self):
    with self.test_session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder,
          ensure_self_adjoint_and_pd=True)
      op_chol = operator.cholesky().to_dense()
      mat_chol = linalg_ops.cholesky(mat)
      op_chol_v, mat_chol_v = sess.run([op_chol, mat_chol])
      self.assertAC(mat_chol_v, op_chol_v)
  return test_cholesky


def _test_solve_base(
    self,
    use_placeholder,
    shapes_info,
    dtype,
    adjoint,
    adjoint_arg,
    with_batch):
  # If batch dimensions are omitted, but there are
  # no batch dimensions for the linear operator, then
  # skip the test case. This is already checked with
  # with_batch=True.
  if not with_batch and len(shapes_info.shape) <= 2:
    return
  with self.session(graph=ops.Graph()) as sess:
    sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
    operator, mat = self.operator_and_matrix(
        shapes_info, dtype, use_placeholder=use_placeholder)
    rhs = self.make_rhs(
        operator, adjoint=adjoint, with_batch=with_batch)
    # If adjoint_arg, solve A X = (rhs^H)^H = rhs.
    if adjoint_arg:
      op_solve = operator.solve(
          linalg.adjoint(rhs),
          adjoint=adjoint,
          adjoint_arg=adjoint_arg)
    else:
      op_solve = operator.solve(
          rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)
    mat_solve = linear_operator_util.matrix_solve_with_broadcast(
        mat, rhs, adjoint=adjoint)
    if not use_placeholder:
      self.assertAllEqual(op_solve.shape,
                          mat_solve.shape)
    op_solve_v, mat_solve_v = sess.run([op_solve, mat_solve])
    self.assertAC(op_solve_v, mat_solve_v)


def _test_solve(
    use_placeholder, shapes_info, dtype, adjoint, adjoint_arg):
  def test_solve(self):
    _test_solve_base(
        self,
        use_placeholder,
        shapes_info,
        dtype,
        adjoint,
        adjoint_arg,
        with_batch=True)
  return test_solve


def _test_solve_with_broadcast(
    use_placeholder, shapes_info, dtype, adjoint, adjoint_arg):
  def test_solve_with_broadcast(self):
    _test_solve_base(
        self,
        use_placeholder,
        shapes_info,
        dtype,
        adjoint,
        adjoint_arg,
        with_batch=False)
  return test_solve_with_broadcast


def _test_inverse(use_placeholder, shapes_info, dtype):
  def test_inverse(self):
    with self.session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)
      op_inverse_v, mat_inverse_v = sess.run([
          operator.inverse().to_dense(), linalg.inv(mat)])
      self.assertAC(op_inverse_v, mat_inverse_v)
  return test_inverse


def _test_trace(use_placeholder, shapes_info, dtype):
  def test_trace(self):
    with self.session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)
      op_trace = operator.trace()
      mat_trace = math_ops.trace(mat)
      if not use_placeholder:
        self.assertAllEqual(op_trace.shape, mat_trace.shape)
      op_trace_v, mat_trace_v = sess.run([op_trace, mat_trace])
      self.assertAC(op_trace_v, mat_trace_v)
  return test_trace


def _test_add_to_tensor(use_placeholder, shapes_info, dtype):
  def test_add_to_tensor(self):
    with self.session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)
      op_plus_2mat = operator.add_to_tensor(2 * mat)

      if not use_placeholder:
        self.assertAllEqual(shapes_info.shape, op_plus_2mat.shape)

      op_plus_2mat_v, mat_v = sess.run([op_plus_2mat, mat])

      self.assertAC(op_plus_2mat_v, 3 * mat_v)
  return test_add_to_tensor


def _test_diag_part(use_placeholder, shapes_info, dtype):
  def test_diag_part(self):
    with self.session(graph=ops.Graph()) as sess:
      sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)
      op_diag_part = operator.diag_part()
      mat_diag_part = array_ops.matrix_diag_part(mat)

      if not use_placeholder:
        self.assertAllEqual(mat_diag_part.shape,
                            op_diag_part.shape)

      op_diag_part_, mat_diag_part_ = sess.run(
          [op_diag_part, mat_diag_part])

      self.assertAC(op_diag_part_, mat_diag_part_)
  return test_diag_part

# pylint:enable=missing-docstring


def add_tests(test_cls):
  """Add tests for LinearOperator methods."""
  test_name_dict = {
      "add_to_tensor": _test_add_to_tensor,
      "cholesky": _test_cholesky,
      "det": _test_det,
      "diag_part": _test_diag_part,
      "inverse": _test_inverse,
      "log_abs_det": _test_log_abs_det,
      "matmul": _test_matmul,
      "matmul_with_broadcast": _test_matmul_with_broadcast,
      "solve": _test_solve,
      "solve_with_broadcast": _test_solve_with_broadcast,
      "to_dense": _test_to_dense,
      "trace": _test_trace,
  }
  tests_with_adjoint_args = [
      "matmul",
      "matmul_with_broadcast",
      "solve",
      "solve_with_broadcast",
  ]

  for name, test_template_fn in test_name_dict.items():
    if name in test_cls.skip_these_tests():
      continue

    for dtype, use_placeholder, shape_info in itertools.product(
        test_cls.dtypes_to_test(),
        test_cls.use_placeholder_options(),
        test_cls.operator_shapes_infos()):
      base_test_name = "_".join([
          "test", name, "_shape={},dtype={},use_placeholder={}".format(
              shape_info.shape, dtype, use_placeholder)])
      if name in tests_with_adjoint_args:
        for adjoint in test_cls.adjoint_options():
          for adjoint_arg in test_cls.adjoint_arg_options():
            test_name = base_test_name + ",adjoint={},adjoint_arg={}".format(
                adjoint, adjoint_arg)
            if hasattr(test_cls, test_name):
              raise RuntimeError("Test %s defined more than once" % test_name)
            setattr(
                test_cls,
                test_name,
                test_util.run_deprecated_v1(test_template_fn(
                    use_placeholder,
                    shape_info,
                    dtype,
                    adjoint,
                    adjoint_arg)))
      else:
        if hasattr(test_cls, base_test_name):
          raise RuntimeError("Test %s defined more than once" % base_test_name)
        setattr(
            test_cls,
            base_test_name,
            test_util.run_deprecated_v1(test_template_fn(
                use_placeholder, shape_info, dtype)))


@six.add_metaclass(abc.ABCMeta)
class SquareLinearOperatorDerivedClassTest(LinearOperatorDerivedClassTest):
  """Base test class appropriate for square operators.

  Sub-classes must still define all abstractmethods from
  LinearOperatorDerivedClassTest that are not defined here.
  """

  @staticmethod
  def operator_shapes_infos():
    shapes_info = OperatorShapesInfo
    # non-batch operators (n, n) and batch operators.
    return [
        shapes_info((0, 0)),
        shapes_info((1, 1)),
        shapes_info((1, 3, 3)),
        shapes_info((3, 4, 4)),
        shapes_info((2, 1, 4, 4))]

  def make_rhs(self, operator, adjoint, with_batch=True):
    # This operator is square, so rhs and x will have same shape.
    # adjoint value makes no difference because the operator shape doesn't
    # change since it is square, but be pedantic.
    return self.make_x(operator, adjoint=not adjoint, with_batch=with_batch)

  def make_x(self, operator, adjoint, with_batch=True):
    # Value of adjoint makes no difference because the operator is square.
    # Return the number of systems to solve, R, equal to 1 or 2.
    r = self._get_num_systems(operator)
    # If operator.shape = [B1,...,Bb, N, N] this returns a random matrix of
    # shape [B1,...,Bb, N, R], R = 1 or 2.
    if operator.shape.is_fully_defined():
      batch_shape = operator.batch_shape.as_list()
      n = operator.domain_dimension.value
      if with_batch:
        x_shape = batch_shape + [n, r]
      else:
        x_shape = [n, r]
    else:
      batch_shape = operator.batch_shape_tensor()
      n = operator.domain_dimension_tensor()
      if with_batch:
        x_shape = array_ops.concat((batch_shape, [n, r]), 0)
      else:
        x_shape = [n, r]

    return random_normal(x_shape, dtype=operator.dtype)

  def _get_num_systems(self, operator):
    """Get some number, either 1 or 2, depending on operator."""
    if operator.tensor_rank is None or operator.tensor_rank % 2:
      return 1
    else:
      return 2


@six.add_metaclass(abc.ABCMeta)
class NonSquareLinearOperatorDerivedClassTest(LinearOperatorDerivedClassTest):
  """Base test class appropriate for generic rectangular operators.

  Square shapes are never tested by this class, so if you want to test your
  operator with a square shape, create two test classes, the other subclassing
  SquareLinearOperatorFullMatrixTest.

  Sub-classes must still define all abstractmethods from
  LinearOperatorDerivedClassTest that are not defined here.
  """

  @staticmethod
  def skip_these_tests():
    """List of test names to skip."""
    return [
        "cholesky",
        "inverse",
        "solve",
        "solve_with_broadcast",
        "det",
        "log_abs_det"
    ]

  @staticmethod
  def operator_shapes_infos():
    shapes_info = OperatorShapesInfo
    # non-batch operators (n, n) and batch operators.
    return [
        shapes_info((2, 1)),
        shapes_info((1, 2)),
        shapes_info((1, 3, 2)),
        shapes_info((3, 3, 4)),
        shapes_info((2, 1, 2, 4))]

  def make_rhs(self, operator, adjoint, with_batch=True):
    # TODO(langmore) Add once we're testing solve_ls.
    raise NotImplementedError(
        "make_rhs not implemented because we don't test solve")

  def make_x(self, operator, adjoint, with_batch=True):
    # Return the number of systems for the argument 'x' for .matmul(x)
    r = self._get_num_systems(operator)
    # If operator.shape = [B1,...,Bb, M, N] this returns a random matrix of
    # shape [B1,...,Bb, N, R], R = 1 or 2.
    if operator.shape.is_fully_defined():
      batch_shape = operator.batch_shape.as_list()
      if adjoint:
        n = operator.range_dimension.value
      else:
        n = operator.domain_dimension.value
      if with_batch:
        x_shape = batch_shape + [n, r]
      else:
        x_shape = [n, r]
    else:
      batch_shape = operator.batch_shape_tensor()
      if adjoint:
        n = operator.range_dimension_tensor()
      else:
        n = operator.domain_dimension_tensor()
      if with_batch:
        x_shape = array_ops.concat((batch_shape, [n, r]), 0)
      else:
        x_shape = [n, r]

    return random_normal(x_shape, dtype=operator.dtype)

  def _get_num_systems(self, operator):
    """Get some number, either 1 or 2, depending on operator."""
    if operator.tensor_rank is None or operator.tensor_rank % 2:
      return 1
    else:
      return 2


def random_positive_definite_matrix(shape, dtype, force_well_conditioned=False):
  """[batch] positive definite matrix.

  Args:
    shape:  `TensorShape` or Python list.  Shape of the returned matrix.
    dtype:  `TensorFlow` `dtype` or Python dtype.
    force_well_conditioned:  Python bool.  If `True`, returned matrix has
      eigenvalues with modulus in `(1, 4)`.  Otherwise, eigenvalues are
      chi-squared random variables.

  Returns:
    `Tensor` with desired shape and dtype.
  """
  dtype = dtypes.as_dtype(dtype)
  if not tensor_util.is_tensor(shape):
    shape = tensor_shape.TensorShape(shape)
    # Matrix must be square.
    shape.dims[-1].assert_is_compatible_with(shape.dims[-2])

  with ops.name_scope("random_positive_definite_matrix"):
    tril = random_tril_matrix(
        shape, dtype, force_well_conditioned=force_well_conditioned)
    return math_ops.matmul(tril, tril, adjoint_b=True)


def random_tril_matrix(shape,
                       dtype,
                       force_well_conditioned=False,
                       remove_upper=True):
  """[batch] lower triangular matrix.

  Args:
    shape:  `TensorShape` or Python `list`.  Shape of the returned matrix.
    dtype:  `TensorFlow` `dtype` or Python dtype
    force_well_conditioned:  Python `bool`. If `True`, returned matrix will have
      eigenvalues with modulus in `(1, 2)`.  Otherwise, eigenvalues are unit
      normal random variables.
    remove_upper:  Python `bool`.
      If `True`, zero out the strictly upper triangle.
      If `False`, the lower triangle of returned matrix will have desired
      properties, but will not have the strictly upper triangle zero'd out.

  Returns:
    `Tensor` with desired shape and dtype.
  """
  with ops.name_scope("random_tril_matrix"):
    # Totally random matrix.  Has no nice properties.
    tril = random_normal(shape, dtype=dtype)
    if remove_upper:
      tril = array_ops.matrix_band_part(tril, -1, 0)

    # Create a diagonal with entries having modulus in [1, 2].
    if force_well_conditioned:
      maxval = ops.convert_to_tensor(np.sqrt(2.), dtype=dtype.real_dtype)
      diag = random_sign_uniform(
          shape[:-1], dtype=dtype, minval=1., maxval=maxval)
      tril = array_ops.matrix_set_diag(tril, diag)

    return tril


def random_normal(shape, mean=0.0, stddev=1.0, dtype=dtypes.float32, seed=None):
  """Tensor with (possibly complex) Gaussian entries.

  Samples are distributed like

  ```
  N(mean, stddev^2), if dtype is real,
  X + iY,  where X, Y ~ N(mean, stddev^2) if dtype is complex.
  ```

  Args:
    shape:  `TensorShape` or Python list.  Shape of the returned tensor.
    mean:  `Tensor` giving mean of normal to sample from.
    stddev:  `Tensor` giving stdev of normal to sample from.
    dtype:  `TensorFlow` `dtype` or numpy dtype
    seed:  Python integer seed for the RNG.

  Returns:
    `Tensor` with desired shape and dtype.
  """
  dtype = dtypes.as_dtype(dtype)

  with ops.name_scope("random_normal"):
    samples = random_ops.random_normal(
        shape, mean=mean, stddev=stddev, dtype=dtype.real_dtype, seed=seed)
    if dtype.is_complex:
      if seed is not None:
        seed += 1234
      more_samples = random_ops.random_normal(
          shape, mean=mean, stddev=stddev, dtype=dtype.real_dtype, seed=seed)
      samples = math_ops.complex(samples, more_samples)
    return samples


def random_uniform(shape,
                   minval=None,
                   maxval=None,
                   dtype=dtypes.float32,
                   seed=None):
  """Tensor with (possibly complex) Uniform entries.

  Samples are distributed like

  ```
  Uniform[minval, maxval], if dtype is real,
  X + iY,  where X, Y ~ Uniform[minval, maxval], if dtype is complex.
  ```

  Args:
    shape:  `TensorShape` or Python list.  Shape of the returned tensor.
    minval:  `0-D` `Tensor` giving the minimum values.
    maxval:  `0-D` `Tensor` giving the maximum values.
    dtype:  `TensorFlow` `dtype` or Python dtype
    seed:  Python integer seed for the RNG.

  Returns:
    `Tensor` with desired shape and dtype.
  """
  dtype = dtypes.as_dtype(dtype)

  with ops.name_scope("random_uniform"):
    samples = random_ops.random_uniform(
        shape, dtype=dtype.real_dtype, minval=minval, maxval=maxval, seed=seed)
    if dtype.is_complex:
      if seed is not None:
        seed += 12345
      more_samples = random_ops.random_uniform(
          shape,
          dtype=dtype.real_dtype,
          minval=minval,
          maxval=maxval,
          seed=seed)
      samples = math_ops.complex(samples, more_samples)
    return samples


def random_sign_uniform(shape,
                        minval=None,
                        maxval=None,
                        dtype=dtypes.float32,
                        seed=None):
  """Tensor with (possibly complex) random entries from a "sign Uniform".

  Letting `Z` be a random variable equal to `-1` and `1` with equal probability,
  Samples from this `Op` are distributed like

  ```
  Z * X, where X ~ Uniform[minval, maxval], if dtype is real,
  Z * (X + iY),  where X, Y ~ Uniform[minval, maxval], if dtype is complex.
  ```

  Args:
    shape:  `TensorShape` or Python list.  Shape of the returned tensor.
    minval:  `0-D` `Tensor` giving the minimum values.
    maxval:  `0-D` `Tensor` giving the maximum values.
    dtype:  `TensorFlow` `dtype` or Python dtype
    seed:  Python integer seed for the RNG.

  Returns:
    `Tensor` with desired shape and dtype.
  """
  dtype = dtypes.as_dtype(dtype)

  with ops.name_scope("random_sign_uniform"):
    unsigned_samples = random_uniform(
        shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed)
    if seed is not None:
      seed += 12
    signs = math_ops.sign(
        random_ops.random_uniform(shape, minval=-1., maxval=1., seed=seed))
    return unsigned_samples * math_ops.cast(signs, unsigned_samples.dtype)


def random_normal_correlated_columns(shape,
                                     mean=0.0,
                                     stddev=1.0,
                                     dtype=dtypes.float32,
                                     eps=1e-4,
                                     seed=None):
  """Batch matrix with (possibly complex) Gaussian entries and correlated cols.

  Returns random batch matrix `A` with specified element-wise `mean`, `stddev`,
  living close to an embedded hyperplane.

  Suppose `shape[-2:] = (M, N)`.

  If `M < N`, `A` is a random `M x N` [batch] matrix with iid Gaussian entries.

  If `M >= N`, then the colums of `A` will be made almost dependent as follows:

  ```
  L = random normal N x N-1 matrix, mean = 0, stddev = 1 / sqrt(N - 1)
  B = random normal M x N-1 matrix, mean = 0, stddev = stddev.

  G = (L B^H)^H, a random normal M x N matrix, living on N-1 dim hyperplane
  E = a random normal M x N matrix, mean = 0, stddev = eps
  mu = a constant M x N matrix, equal to the argument "mean"

  A = G + E + mu
  ```

  Args:
    shape:  Python list of integers.
      Shape of the returned tensor.  Must be at least length two.
    mean:  `Tensor` giving mean of normal to sample from.
    stddev:  `Tensor` giving stdev of normal to sample from.
    dtype:  `TensorFlow` `dtype` or numpy dtype
    eps:  Distance each column is perturbed from the low-dimensional subspace.
    seed:  Python integer seed for the RNG.

  Returns:
    `Tensor` with desired shape and dtype.

  Raises:
    ValueError:  If `shape` is not at least length 2.
  """
  dtype = dtypes.as_dtype(dtype)

  if len(shape) < 2:
    raise ValueError(
        "Argument shape must be at least length 2.  Found: %s" % shape)

  # Shape is the final shape, e.g. [..., M, N]
  shape = list(shape)
  batch_shape = shape[:-2]
  m, n = shape[-2:]

  # If there is only one column, "they" are by definition correlated.
  if n < 2 or n < m:
    return random_normal(
        shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)

  # Shape of the matrix with only n - 1 columns that we will embed in higher
  # dimensional space.
  smaller_shape = batch_shape + [m, n - 1]

  # Shape of the embedding matrix, mapping batch matrices
  # from [..., N-1, M] to [..., N, M]
  embedding_mat_shape = batch_shape + [n, n - 1]

  # This stddev for the embedding_mat ensures final result has correct stddev.
  stddev_mat = 1 / np.sqrt(n - 1)

  with ops.name_scope("random_normal_correlated_columns"):
    smaller_mat = random_normal(
        smaller_shape, mean=0.0, stddev=stddev_mat, dtype=dtype, seed=seed)

    if seed is not None:
      seed += 1287

    embedding_mat = random_normal(embedding_mat_shape, dtype=dtype, seed=seed)

    embedded_t = math_ops.matmul(embedding_mat, smaller_mat, transpose_b=True)
    embedded = array_ops.matrix_transpose(embedded_t)

    mean_mat = array_ops.ones_like(embedded) * mean

    return embedded + random_normal(shape, stddev=eps, dtype=dtype) + mean_mat
