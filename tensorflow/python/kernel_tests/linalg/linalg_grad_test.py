# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.linalg_grad."""

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test as test_lib


def _AddTest(test, op_name, testcase_name, fn):
  test_name = '_'.join(['test', op_name, testcase_name])
  if hasattr(test, test_name):
    raise RuntimeError('Test %s defined more than once' % test_name)
  setattr(test, test_name, fn)


class ShapeTest(test_lib.TestCase):

  @test_util.run_deprecated_v1
  def testBatchGradientUnknownSize(self):
    with self.cached_session():
      batch_size = constant_op.constant(3)
      matrix_size = constant_op.constant(4)
      batch_identity = array_ops.tile(
          array_ops.expand_dims(
              array_ops.diag(array_ops.ones([matrix_size])), 0),
          [batch_size, 1, 1])
      determinants = linalg_ops.matrix_determinant(batch_identity)
      reduced = math_ops.reduce_sum(determinants)
      sum_grad = gradients_impl.gradients(reduced, batch_identity)[0]
      self.assertAllClose(batch_identity, self.evaluate(sum_grad))


class MatrixUnaryFunctorGradientTest(test_lib.TestCase):
  pass  # Filled in below

# TODO(b/417809163): re-enable this test when upstream issues are resolved
# see commit msg for details
# def _GetMatrixUnaryFunctorGradientTest(functor_, dtype_, shape_, **kwargs_):
#
#  @test_util.enable_control_flow_v2
#  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
#  @test_util.run_without_tensor_float_32(
#      'Tests `tf.linalg.expm`, which call matmul. Additionally, calls ops '
#      'which do matmul in their gradient, such as MatrixSolve.')
#  def Test(self):

#    def RandomInput():
#      np.random.seed(1)
#      return np.random.uniform(
#          low=-1.0, high=1.0,
#          size=np.prod(shape_)).reshape(shape_).astype(dtype_)

#    if functor_.__name__ == 'matrix_square_root':
#      # Square the input matrix to ensure that its matrix square root exists
#      f = lambda x: functor_(math_ops.matmul(x, x), **kwargs_)
#    else:
#      f = functor_

#    # Optimal stepsize for central difference is O(epsilon^{1/3}).
#    epsilon = np.finfo(dtype_).eps
#    delta = epsilon**(1.0 / 3.0)
# tolerance obtained by looking at actual differences using
# np.linalg.norm(theoretical-numerical, np.inf) on -mavx build
#    tol = 1e-6 if dtype_ == np.float64 else 0.05

#    theoretical, numerical = gradient_checker_v2.compute_gradient(
#        f, [RandomInput()], delta=delta)
#    self.assertAllClose(theoretical, numerical, atol=tol, rtol=tol)

#  return Test


class MatrixBinaryFunctorGradientTest(test_lib.TestCase):
  pass  # Filled in below


def _GetMatrixBinaryFunctorGradientTest(functor_,
                                        dtype_,
                                        shape_,
                                        float32_tol_fudge=1.0,
                                        **kwargs_):

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  @test_util.run_without_tensor_float_32(
      'Tests `tf.linalg.lstsq`, which call matmul. Additionally, calls ops '
      'which do matmul in their gradient, such as MatrixSolveLs.')
  # TODO(b/164254522): With TensorFloat-32, some tests fails with extremely high
  # absolute and relative differences when calling assertAllClose. For example,
  # the test test_MatrixSolveLsGradient_float32_10_10_1e-06 of class
  # MatrixBinaryFunctorGradientTest fails with a max absolute difference of
  # 0.883 and a max relative difference of 736892. We should consider disabling
  # TensorFloat-32 within `tf.linalg.lstsq and perhaps other linear algebra
  # functions, even if TensorFloat-32 is allowed globally.
  def Test(self):

    def RandomInput():
      np.random.seed(1)
      return np.random.uniform(
          low=-1.0, high=1.0,
          size=np.prod(shape_)).reshape(shape_).astype(dtype_)

    fixed = RandomInput()

    # Optimal stepsize for central difference is O(epsilon^{1/3}).
    epsilon = np.finfo(dtype_).eps
    delta = epsilon**(1.0 / 3.0)
    # tolerance obtained by looking at actual differences using
    # np.linalg.norm(theoretical-numerical, np.inf) on -mavx build
    tol = 1e-6 if dtype_ == np.float64 else float32_tol_fudge * 0.05

    # check gradient w.r.t. left argument.
    theoretical, numerical = gradient_checker_v2.compute_gradient(
        lambda x: functor_(x, fixed, **kwargs_), [RandomInput()], delta=delta)
    self.assertAllClose(theoretical, numerical, atol=tol, rtol=tol)

    # check gradient w.r.t. right argument.
    theoretical, numerical = gradient_checker_v2.compute_gradient(
        lambda y: functor_(fixed, y, **kwargs_), [RandomInput()], delta=delta)
    self.assertAllClose(theoretical, numerical, atol=tol, rtol=tol)

  return Test


def _GetBandedTriangularSolveGradientTest(
    functor_,
    dtype_,
    shape_,
    float32_tol_fudge=1.0,  # pylint: disable=redefined-outer-name
    **kwargs_):

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def Test(self):
    n = shape_[-1]

    np.random.seed(1)
    # Make sure invertible.
    a_np = np.random.uniform(low=1.0, high=2.0, size=shape_).astype(dtype_)
    a = constant_op.constant(a_np)

    b_np = np.random.uniform(low=-1.0, high=1.0, size=[n, n]).astype(dtype_)
    b = constant_op.constant(b_np)

    epsilon = np.finfo(dtype_).eps
    delta = epsilon**(1.0 / 3.0)
    # tolerance obtained by looking at actual differences using
    # np.linalg.norm(theoretical-numerical, np.inf) on -mavx build
    tol = 1e-6 if dtype_ == np.float64 else float32_tol_fudge * 0.05

    # check gradient w.r.t. left argument.
    theoretical, numerical = gradient_checker_v2.compute_gradient(
        lambda x: functor_(x, b, **kwargs_), [a], delta=delta)
    self.assertAllClose(theoretical, numerical, atol=tol, rtol=tol)

    # check gradient w.r.t. right argument.
    theoretical, numerical = gradient_checker_v2.compute_gradient(
        lambda y: functor_(a, y, **kwargs_), [b], delta=delta)
    self.assertAllClose(theoretical, numerical, atol=tol, rtol=tol)

  return Test


class DetGradSingularMatrixTest(test_lib.TestCase):
  """Gradients of det and slogdet remain defined for singular matrices.

  The holomorphic derivative of det(A) is the cofactor matrix, so the
  reverse-mode gradient is its elementwise conjugate, conj(det(A)) * A^{-H}
  for invertible A. For real A this is the transpose of the classical
  adjugate, which extends continuously to singular A. These tests compare the
  analytic gradient against the numerical one via gradient_checker_v2 for
  real and complex, singular and invertible, and near-singular inputs, check
  values against the closed-form adjugate, and confirm that slogdet's
  backward pass stays finite for a singular matrix instead of raising.
  """

  @test_util.run_in_graph_and_eager_modes(use_gpu=False)
  def testDetGradSingular2x2(self):
    # det([[1, 2], [2, 4]]) == 0; its adjugate is [[4, -2], [-2, 1]].
    m = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64)
    theoretical, _ = gradient_checker_v2.compute_gradient(
        linalg_ops.matrix_determinant, [m]
    )
    self.assertAllClose([[4.0, -2.0, -2.0, 1.0]], theoretical[0])

  @test_util.run_in_graph_and_eager_modes(use_gpu=False)
  def testDetGradInvertible2x2(self):
    # The adjugate of the invertible A = [[1, 2], [3, 4]] is [[4, -3], [-2, 1]].
    m = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    theoretical, numerical = gradient_checker_v2.compute_gradient(
        linalg_ops.matrix_determinant, [m]
    )
    self.assertAllClose([[4.0, -3.0, -2.0, 1.0]], theoretical[0])
    self.assertAllClose(theoretical[0], numerical[0], atol=1e-6)

  @test_util.run_in_graph_and_eager_modes(use_gpu=False)
  def testDetGradSingularFloat32(self):
    m = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    theoretical, _ = gradient_checker_v2.compute_gradient(
        linalg_ops.matrix_determinant, [m]
    )
    self.assertAllClose([[4.0, -2.0, -2.0, 1.0]], theoretical[0], atol=1e-5)

  @test_util.run_in_graph_and_eager_modes(use_gpu=False)
  def testDetGradNearSingular(self):
    # A = [[1, 1], [1, 1 + eps]] has det = eps and condition number ~4/eps.
    # The gradient (adjugate transpose) is [[1 + eps, -1], [-1, 1]], which the
    # SVD-based evaluation must reproduce without the loss of accuracy that
    # forming eps * A^{-1} would incur at these condition numbers.
    for eps in (1e-6, 1e-9, 1e-12):
      m = constant_op.constant([[1.0, 1.0], [1.0, 1.0 + eps]], dtype=np.float64)
      with backprop.GradientTape() as tape:
        tape.watch(m)
        d = linalg_ops.matrix_determinant(m)
      grad = tape.gradient(d, m)
      self.assertAllClose(
          self.evaluate(grad), [[1.0 + eps, -1.0], [-1.0, 1.0]], atol=1e-9
      )

  @test_util.run_in_graph_and_eager_modes(use_gpu=False)
  def testDetGradNearSingularNumerical(self):
    # The analytic gradient must also agree with the numerical one for a
    # near-singular matrix (det = 1e-6, condition number ~4e6).
    m = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-6]], dtype=np.float64)
    theoretical, numerical = gradient_checker_v2.compute_gradient(
        linalg_ops.matrix_determinant, [m]
    )
    self.assertAllClose(theoretical[0], numerical[0], atol=1e-6)

  @test_util.run_in_graph_and_eager_modes(use_gpu=False)
  def testDetGradBatchWithSingular(self):
    # A batch mixing a singular and an invertible matrix must produce the
    # per-matrix adjugate for both entries.
    m = constant_op.constant(
        [[[1.0, 2.0], [2.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64
    )
    with backprop.GradientTape() as tape:
      tape.watch(m)
      d = linalg_ops.matrix_determinant(m)
      total = math_ops.reduce_sum(d)
    grad = tape.gradient(total, m)
    self.assertIsNotNone(grad)
    self.assertAllClose(
        self.evaluate(grad),
        [[[4.0, -2.0], [-2.0, 1.0]], [[4.0, -3.0], [-2.0, 1.0]]],
        atol=1e-10,
    )

  @test_util.run_in_graph_and_eager_modes(use_gpu=False)
  def testDetGradComplexInvertible(self):
    # The reverse-mode gradient of det is conj(det(A)) * A^{-H}; compare the
    # analytic gradient against both that closed form and the numerical
    # Jacobian for complex64 and complex128.
    for dtype, tol in ((np.complex64, 2e-2), (np.complex128, 1e-6)):
      a = np.array(
          [[1.0 + 1.0j, 2.0 - 1.0j], [3.0 + 0.5j, 4.0 + 2.0j]], dtype=dtype
      )
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          linalg_ops.matrix_determinant, [a]
      )
      self.assertAllClose(theoretical[0], numerical[0], atol=tol)
      m = constant_op.constant(a)
      with backprop.GradientTape() as tape:
        tape.watch(m)
        d = linalg_ops.matrix_determinant(m)
      grad = tape.gradient(d, m)
      expected = math_ops.conj(d) * linalg_ops.matrix_inverse(m, adjoint=True)
      self.assertAllClose(
          self.evaluate(grad), self.evaluate(expected), atol=tol
      )

  @test_util.run_in_graph_and_eager_modes(use_gpu=False)
  def testDetGradComplexSingular(self):
    # [[1+i, 2+2i], [2-i, 4-2i]] has det 0 (second row is conj-scaled first);
    # the analytic gradient must match the numerical one instead of raising.
    for dtype, tol in ((np.complex64, 2e-2), (np.complex128, 1e-6)):
      a = np.array(
          [[1.0 + 1.0j, 2.0 + 2.0j], [2.0 - 1.0j, 4.0 - 2.0j]], dtype=dtype
      )
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          linalg_ops.matrix_determinant, [a]
      )
      self.assertAllClose(theoretical[0], numerical[0], atol=tol)

  @test_util.run_in_graph_and_eager_modes(use_gpu=False)
  def testSlogdetGradSingularDoesNotCrash(self):
    # log|det(A)| is -inf for a singular A, so its gradient is undefined, but
    # the backward pass must stay finite instead of raising.
    m = constant_op.constant([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64)
    with backprop.GradientTape() as tape:
      tape.watch(m)
      _, log_abs_det = gen_linalg_ops.log_matrix_determinant(m)
    grad = tape.gradient(log_abs_det, m)
    self.assertIsNotNone(grad)
    self.assertAllEqual(
        self.evaluate(math_ops.is_finite(grad)), [[True, True], [True, True]]
    )

  @test_util.run_in_graph_and_eager_modes(use_gpu=False)
  def testSlogdetGradInvertibleUnchanged(self):
    # For invertible A the gradient of log|det(A)| is A^{-H}; the
    # pseudoinverse-based path must reproduce it for real and complex inputs.
    a_real = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    a_complex = np.array(
        [[1.0 + 1.0j, 2.0 - 1.0j], [3.0 + 0.5j, 4.0 + 2.0j]],
        dtype=np.complex128,
    )
    for a in (a_real, a_complex):
      m = constant_op.constant(a)
      with backprop.GradientTape() as tape:
        tape.watch(m)
        _, log_abs_det = gen_linalg_ops.log_matrix_determinant(m)
      grad = tape.gradient(log_abs_det, m)
      expected = linalg_ops.matrix_inverse(m, adjoint=True)
      self.assertAllClose(
          self.evaluate(grad), self.evaluate(expected), atol=1e-10
      )

  @test_util.run_in_graph_and_eager_modes(use_gpu=False)
  def testSlogdetGradComplexSingularDoesNotCrash(self):
    m = constant_op.constant(
        [[1.0 + 1.0j, 2.0 + 2.0j], [2.0 - 1.0j, 4.0 - 2.0j]],
        dtype=np.complex128,
    )
    with backprop.GradientTape() as tape:
      tape.watch(m)
      _, log_abs_det = gen_linalg_ops.log_matrix_determinant(m)
    grad = tape.gradient(log_abs_det, m)
    self.assertIsNotNone(grad)
    self.assertAllEqual(
        self.evaluate(math_ops.is_finite(math_ops.abs(grad))),
        [[True, True], [True, True]],
    )


if __name__ == '__main__':
  # Tests for gradients of binary matrix operations.
  for dtype in np.float32, np.float64:
    for size in 2, 5, 10:
      # We skip the rank 4, size 10 case: it is slow and conceptually covered
      # by the other cases.
      for extra in [(), (2,), (3,)] + [(3, 2)] * (size < 10):
        for adjoint in False, True:
          shape = extra + (size, size)
          name = '%s_%s_adj_%s' % (dtype.__name__, '_'.join(map(
              str, shape)), str(adjoint))
          _AddTest(
              MatrixBinaryFunctorGradientTest, 'MatrixSolveGradient', name,
              _GetMatrixBinaryFunctorGradientTest(
                  linalg_ops.matrix_solve, dtype, shape, adjoint=adjoint))

          for lower in True, False:
            name = '%s_low_%s' % (name, lower)
            _AddTest(
                MatrixBinaryFunctorGradientTest,
                'MatrixTriangularSolveGradient', name,
                _GetMatrixBinaryFunctorGradientTest(
                    linalg_ops.matrix_triangular_solve,
                    dtype,
                    shape,
                    float32_tol_fudge=4.0,
                    adjoint=adjoint,
                    lower=lower))

            band_shape = extra + (size // 2 + 1, size)
            name = '%s_%s_adj_%s_low_%s' % (dtype.__name__, '_'.join(
                map(str, band_shape)), str(adjoint), lower)
            _AddTest(
                MatrixBinaryFunctorGradientTest,
                'BandedTriangularSolveGradient', name,
                _GetBandedTriangularSolveGradientTest(
                    linalg_ops.banded_triangular_solve,
                    dtype,
                    band_shape,
                    float32_tol_fudge=4.0,
                    adjoint=adjoint,
                    lower=lower))

  # Tests for gradients of unary matrix operations.
  for dtype in np.float32, np.float64:
    for size in 2, 5, 10:
      # We skip the rank 4, size 10 case: it is slow and conceptually covered
      # by the other cases.
      for extra in [(), (2,), (3,)] + [(3, 2)] * (size < 10):
        shape = extra + (size, size)
        name = '%s_%s' % (dtype.__name__, '_'.join(map(str, shape)))
        # _AddTest(
        #     MatrixUnaryFunctorGradientTest, 'MatrixInverseGradient', name,
        #     _GetMatrixUnaryFunctorGradientTest(linalg_ops.matrix_inverse,
        #                                        dtype, shape))
        #        _AddTest(
        #            MatrixUnaryFunctorGradientTest,
        #            'MatrixAdjointInverseGradient', name,
        #            _GetMatrixUnaryFunctorGradientTest(
        #                lambda x: linalg_ops.matrix_inverse(x, adjoint=True),
        #                dtype, shape))

        #        if True:  # not test_lib.is_built_with_rocm():
        # TODO(b/417809163):
        # re-enable this test when upstream issues are resolved
        # see commit msg for details
        # _AddTest(
        #     MatrixUnaryFunctorGradientTest, 'MatrixExponentialGradient', name,
        #     _GetMatrixUnaryFunctorGradientTest(linalg_impl.matrix_exponential,
        #                                         dtype, shape))
        #        _AddTest(
        #            MatrixUnaryFunctorGradientTest,
        #            'MatrixDeterminantGradient', name,
        #            _GetMatrixUnaryFunctorGradientTest(linalg_ops.matrix_determinant,
        #                                               dtype, shape))
        #        _AddTest(
        #            MatrixUnaryFunctorGradientTest,
        #            'LogMatrixDeterminantGradient',
        #            name,
        #            _GetMatrixUnaryFunctorGradientTest(lambda x:
        #                linalg_ops.log_matrix_determinant(x)[1], dtype, shape))

        # The numerical Jacobian is consistently invalid for these four shapes
        # because the matrix square root of the perturbed input doesn't exist
        if shape in {(2, 5, 5), (3, 5, 5), (3, 10, 10), (3, 2, 5, 5)}:
          # Alternative shape that consistently produces a valid numerical
          # Jacobian
          shape = extra + (size + 1, size + 1)
          name = '%s_%s' % (dtype.__name__, '_'.join(map(str, shape)))
  #        _AddTest(
  #            MatrixUnaryFunctorGradientTest, 'MatrixSquareRootGradient', name,
  #            _GetMatrixUnaryFunctorGradientTest(linalg_ops.matrix_square_root,
  #                                               dtype, shape))

  # Tests for gradients of matrix_solve_ls
  for dtype in np.float32, np.float64:
    for rows in 2, 5, 10:
      for cols in 2, 5, 10:
        for l2_regularization in 1e-6, 0.001, 1.0:
          shape = (rows, cols)
          name = '%s_%s_%s' % (dtype.__name__, '_'.join(map(
              str, shape)), l2_regularization)
          float32_tol_fudge = 5.1 if l2_regularization == 1e-6 else 4.0
          _AddTest(
              MatrixBinaryFunctorGradientTest,
              'MatrixSolveLsGradient',
              name,
              # pylint: disable=long-lambda,g-long-lambda
              _GetMatrixBinaryFunctorGradientTest(
                  (lambda a, b, l=l2_regularization: linalg_ops.matrix_solve_ls(
                      a, b, l)), dtype, shape, float32_tol_fudge))

  test_lib.main()
