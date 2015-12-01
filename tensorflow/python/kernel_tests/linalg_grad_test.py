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

import tensorflow.python.platform

import numpy as np
import tensorflow as tf


class MatrixInverseGradientTest(tf.test.TestCase):
  pass  # Filled in below


def _GetMatrixInverseGradientTest(dtype_, shape_):

  def Test(self):
    with self.test_session():
      np.random.seed(1)
      m = np.random.uniform(low=1.0,
                            high=100.0,
                            size=np.prod(shape_)).reshape(shape_).astype(dtype_)
      a = tf.constant(m)
      epsilon = np.finfo(dtype_).eps
      # Optimal stepsize for central difference is O(epsilon^{1/3}).
      delta = epsilon**(1.0 / 3.0)
      tol = 1e-3

      if len(shape_) == 2:
        ainv = tf.matrix_inverse(a)
      else:
        ainv = tf.batch_matrix_inverse(a)

      theoretical, numerical = tf.test.compute_gradient(a,
                                                        shape_,
                                                        ainv,
                                                        shape_,
                                                        delta=delta)
      self.assertAllClose(theoretical, numerical, atol=tol, rtol=tol)

  return Test


class MatrixDeterminantGradientTest(tf.test.TestCase):
  pass  # Filled in below


def _GetMatrixDeterminantGradientTest(dtype_, shape_):

  def Test(self):
    with self.test_session():
      np.random.seed(1)
      m = np.random.uniform(low=1.0,
                            high=100.0,
                            size=np.prod(shape_)).reshape(shape_).astype(dtype_)
      a = tf.constant(m)
      epsilon = np.finfo(dtype_).eps
      # Optimal stepsize for central difference is O(epsilon^{1/3}).
      delta = epsilon**(1.0 / 3.0)

      # tolerance obtained by looking at actual differences using
      # np.linalg.norm(theoretical-numerical, np.inf) on -mavx build

      tol = 1e-3

      if len(shape_) == 2:
        c = tf.matrix_determinant(a)
      else:
        c = tf.batch_matrix_determinant(a)

      out_shape = shape_[:-2]  # last two dimensions hold matrices
      theoretical, numerical = tf.test.compute_gradient(a,
                                                        shape_,
                                                        c,
                                                        out_shape,
                                                        delta=delta)

      self.assertAllClose(theoretical, numerical, atol=tol, rtol=tol)

  return Test


if __name__ == '__main__':
  # TODO(rmlarsen,irving): Reenable float32 once tolerances are fixed
  # The test used to loop over (np.float, np.double), both of which are float64.
  for dtype in (np.float64,):
    for size in 2, 3, 5, 10:
      # We skip the rank 4, size 10 case: it is slow and conceptually covered
      # by the other cases.
      for extra in [(), (2,), (3,)] + [(3, 2)] * (size < 10):
        shape = extra + (size, size)
        name = '%s_%s' % (dtype.__name__, '_'.join(map(str, shape)))
        setattr(MatrixInverseGradientTest, 'testMatrixInverseGradient_' + name,
                _GetMatrixInverseGradientTest(dtype, shape))

  for dtype in (np.float64,):
    for size in 2, 5, 10:
      # increase this list to check batch version
      for extra in [()]:
        shape = extra+(size, size)
        name = '%s_%s' % (dtype.__name__, '_'.join(map(str, shape)))
        setattr(MatrixDeterminantGradientTest,
                'testMatrixDeterminantGradient_' + name,
                _GetMatrixDeterminantGradientTest(dtype, shape))
  tf.test.main()
