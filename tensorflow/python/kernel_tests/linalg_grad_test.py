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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
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

  def testBatchGradientUnknownSize(self):
    with self.test_session():
      batch_size = constant_op.constant(3)
      matrix_size = constant_op.constant(4)
      batch_identity = array_ops.tile(
          array_ops.expand_dims(
              array_ops.diag(array_ops.ones([matrix_size])), 0),
          [batch_size, 1, 1])
      determinants = linalg_ops.matrix_determinant(batch_identity)
      reduced = math_ops.reduce_sum(determinants)
      sum_grad = gradients_impl.gradients(reduced, batch_identity)[0]
      self.assertAllClose(batch_identity.eval(), sum_grad.eval())


class MatrixUnaryFunctorGradientTest(test_lib.TestCase):
  pass  # Filled in below


def _GetMatrixUnaryFunctorGradientTest(functor_, dtype_, shape_, **kwargs_):

  def Test(self):
    with self.test_session(use_gpu=True):
      np.random.seed(1)
      a_np = np.random.uniform(
          low=-1.0, high=1.0,
          size=np.prod(shape_)).reshape(shape_).astype(dtype_)
      a = constant_op.constant(a_np)
      b = functor_(a, **kwargs_)

      # Optimal stepsize for central difference is O(epsilon^{1/3}).
      epsilon = np.finfo(dtype_).eps
      delta = epsilon**(1.0 / 3.0)
      # tolerance obtained by looking at actual differences using
      # np.linalg.norm(theoretical-numerical, np.inf) on -mavx build
      tol = 1e-6 if dtype_ == np.float64 else 0.05

      theoretical, numerical = gradient_checker.compute_gradient(
          a,
          a.get_shape().as_list(),
          b,
          b.get_shape().as_list(),
          x_init_value=a_np,
          delta=delta)
      self.assertAllClose(theoretical, numerical, atol=tol, rtol=tol)

  return Test


class MatrixBinaryFunctorGradientTest(test_lib.TestCase):
  pass  # Filled in below


def _GetMatrixBinaryFunctorGradientTest(functor_,
                                        dtype_,
                                        shape_,
                                        float32_tol_fudge=1.0,
                                        **kwargs_):

  def Test(self):
    # rocBLAS on ROCm stack causes accuracy failure
    if test_lib.is_built_with_rocm():
      use_gpu = False
    else:
    # TODO(rmlarsen): Debug illegal address bug on CUDA and re-enable
    # GPU test for matrix_solve.
      use_gpu = False if functor_ == linalg_ops.matrix_solve else True

    with self.test_session(use_gpu=use_gpu):
      np.random.seed(1)
      a_np = np.random.uniform(
          low=-1.0, high=1.0,
          size=np.prod(shape_)).reshape(shape_).astype(dtype_)
      a = constant_op.constant(a_np)

      b_np = np.random.uniform(
          low=-1.0, high=1.0,
          size=np.prod(shape_)).reshape(shape_).astype(dtype_)
      b = constant_op.constant(b_np)
      c = functor_(a, b, **kwargs_)

      # Optimal stepsize for central difference is O(epsilon^{1/3}).
      epsilon = np.finfo(dtype_).eps
      delta = epsilon**(1.0 / 3.0)
      # tolerance obtained by looking at actual differences using
      # np.linalg.norm(theoretical-numerical, np.inf) on -mavx build
      tol = 1e-6 if dtype_ == np.float64 else float32_tol_fudge * 0.04
      # The gradients for a and b may be of very different magnitudes,
      # so to not get spurious failures we test them separately.
      for factor, factor_init in [a, a_np], [b, b_np]:
        theoretical, numerical = gradient_checker.compute_gradient(
            factor,
            factor.get_shape().as_list(),
            c,
            c.get_shape().as_list(),
            x_init_value=factor_init,
            delta=delta)
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
          name = '%s_%s_adj_%s' % (dtype.__name__, '_'.join(map(str, shape)),
                                   str(adjoint))
          _AddTest(MatrixBinaryFunctorGradientTest, 'MatrixSolveGradient', name,
                   _GetMatrixBinaryFunctorGradientTest(
                       linalg_ops.matrix_solve, dtype, shape, adjoint=adjoint))

          for lower in True, False:
            name = '%s_low_%s' % (name, lower)
            _AddTest(MatrixBinaryFunctorGradientTest,
                     'MatrixTriangularSolveGradient', name,
                     _GetMatrixBinaryFunctorGradientTest(
                         linalg_ops.matrix_triangular_solve,
                         dtype,
                         shape,
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
        _AddTest(MatrixUnaryFunctorGradientTest, 'MatrixInverseGradient', name,
                 _GetMatrixUnaryFunctorGradientTest(linalg_ops.matrix_inverse,
                                                    dtype, shape))
        _AddTest(
            MatrixUnaryFunctorGradientTest, 'MatrixDeterminantGradient', name,
            _GetMatrixUnaryFunctorGradientTest(linalg_ops.matrix_determinant,
                                               dtype, shape))
        _AddTest(
            MatrixUnaryFunctorGradientTest, 'LogMatrixDeterminantGradient',
            name,
            _GetMatrixUnaryFunctorGradientTest(
                lambda x: linalg_ops.log_matrix_determinant(x)[1],
                dtype, shape))

  # Tests for gradients of matrix_solve_ls
  for dtype in np.float32, np.float64:
    for rows in 2, 5, 10:
      for cols in 2, 5, 10:
        for l2_regularization in 1e-6, 0.001, 1.0:
          shape = (rows, cols)
          name = '%s_%s_%s' % (dtype.__name__, '_'.join(map(str, shape)),
                               l2_regularization)
          _AddTest(
              MatrixBinaryFunctorGradientTest,
              'MatrixSolveLsGradient',
              name,
              # pylint: disable=long-lambda,g-long-lambda
              _GetMatrixBinaryFunctorGradientTest(
                  (lambda a, b, l=l2_regularization:
                   linalg_ops.matrix_solve_ls(a, b, l)),
                  dtype,
                  shape,
                  float32_tol_fudge=4.0))

  test_lib.main()
