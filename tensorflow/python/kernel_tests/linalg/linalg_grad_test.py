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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
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
