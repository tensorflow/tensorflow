# Copyright 2015 Google Inc. All Rights Reserved.
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
import tensorflow as tf


class MatrixUnaryFunctorGradientTest(tf.test.TestCase):
  pass  # Filled in below


def _GetMatrixUnaryFunctorGradientTest(functor_, batch_functor_, dtype_,
                                       shape_):

  def Test(self):
    with self.test_session():
      np.random.seed(1)
      m = np.random.uniform(low=1.0,
                            high=100.0,
                            size=np.prod(shape_)).reshape(shape_).astype(dtype_)
      a = tf.constant(m)
      if len(shape_) == 2 and functor_ is not None:
        b = functor_(a)
      elif batch_functor_ is not None:
        b = batch_functor_(a)
      else:
        return

      # Optimal stepsize for central difference is O(epsilon^{1/3}).
      epsilon = np.finfo(dtype_).eps
      delta = epsilon**(1.0 / 3.0)
      # tolerance obtained by looking at actual differences using
      # np.linalg.norm(theoretical-numerical, np.inf) on -mavx build
      tol = 1e-3

      theoretical, numerical = tf.test.compute_gradient(a,
                                                        a.get_shape().as_list(),
                                                        b,
                                                        b.get_shape().as_list(),
                                                        delta=delta)
      self.assertAllClose(theoretical, numerical, atol=tol, rtol=tol)

  return Test


class MatrixBinaryFunctorGradientTest(tf.test.TestCase):
  pass  # Filled in below


def _GetMatrixBinaryFunctorGradientTest(functor_, batch_functor_, dtype_,
                                        shape_):

  def Test(self):
    with self.test_session():
      np.random.seed(1)
      m = np.random.uniform(low=1.0,
                            high=100.0,
                            size=np.prod(shape_)).reshape(shape_).astype(dtype_)
      a = tf.constant(m)

      n = np.random.uniform(low=1.0,
                            high=100.0,
                            size=np.prod(shape_)).reshape(shape_).astype(dtype_)
      b = tf.constant(n)

      if len(shape_) == 2 and functor_ is not None:
        c = functor_(a, b)
      elif batch_functor_ is not None:
        c = batch_functor_(a, b)
      else:
        return

      # Optimal stepsize for central difference is O(epsilon^{1/3}).
      epsilon = np.finfo(dtype_).eps
      delta = epsilon**(1.0 / 3.0)
      # tolerance obtained by looking at actual differences using
      # np.linalg.norm(theoretical-numerical, np.inf) on -mavx build
      tol = 1e-3

      # The gradients for a and b may be of very different magnitudes,
      # so to not get spurious failures we test them separately.
      for factor in a, b:
        theoretical, numerical = tf.test.compute_gradient(
            factor,
            factor.get_shape().as_list(),
            c,
            c.get_shape().as_list(),
            delta=delta)
        self.assertAllClose(theoretical, numerical, atol=tol, rtol=tol)

  return Test


if __name__ == '__main__':
  # TODO(rmlarsen,irving): Reenable float32 once tolerances are fixed
  # The test used to loop over (np.float, np.double), both of which are float64.

  # Tests for gradients of binary matrix operations.
  for dtype in (np.float64,):
    for size in 2, 3, 5, 10:
      # We skip the rank 4, size 10 case: it is slow and conceptually covered
      # by the other cases.
      for extra in [(), (2,), (3,)] + [(3, 2)] * (size < 10):
        shape = extra + (size, size)
        name = '%s_%s' % (dtype.__name__, '_'.join(map(str, shape)))
        setattr(MatrixUnaryFunctorGradientTest,
                'testMatrixInverseGradient_' + name,
                _GetMatrixUnaryFunctorGradientTest(tf.matrix_inverse,
                                                   tf.batch_matrix_inverse,
                                                   dtype, shape))
        setattr(MatrixBinaryFunctorGradientTest,
                'testMatrixSolveGradient_' + name,
                _GetMatrixBinaryFunctorGradientTest(tf.matrix_solve,
                                                    tf.batch_matrix_solve,
                                                    dtype, shape))

  # Tests for gradients of unary matrix operations.
  for dtype in (np.float64,):
    for size in 2, 5, 10:
      # increase this list to check batch version
      for extra in [()]:
        shape = extra + (size, size)
        name = '%s_%s' % (dtype.__name__, '_'.join(map(str, shape)))
        setattr(MatrixUnaryFunctorGradientTest,
                'testMatrixUnaryFunctorGradient_' + name,
                _GetMatrixUnaryFunctorGradientTest(tf.matrix_determinant,
                                                   tf.batch_matrix_determinant,
                                                   dtype, shape))

  tf.test.main()
