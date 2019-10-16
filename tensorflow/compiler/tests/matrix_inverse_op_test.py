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
"""Tests for tensorflow.ops.math_ops.matrix_inverse."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class InverseOpTest(xla_test.XLATestCase):

  def _verifyInverse(self, x, np_type):
    for adjoint in False, True:
      y = x.astype(np_type)
      with self.session() as sess:
        # Verify that x^{-1} * x == Identity matrix.
        p = array_ops.placeholder(dtypes.as_dtype(y.dtype), y.shape, name="x")
        with self.test_scope():
          inv = linalg_ops.matrix_inverse(p, adjoint=adjoint)
          tf_ans = math_ops.matmul(inv, p, adjoint_b=adjoint)
          np_ans = np.identity(y.shape[-1])
          if x.ndim > 2:
            tiling = list(y.shape)
            tiling[-2:] = [1, 1]
            np_ans = np.tile(np_ans, tiling)
        out = sess.run(tf_ans, feed_dict={p: y})
        self.assertAllClose(np_ans, out, rtol=1e-3, atol=1e-3)
        self.assertShapeEqual(y, tf_ans)

  def _verifyInverseReal(self, x):
    for np_type in self.float_types & {np.float64, np.float32}:
      self._verifyInverse(x, np_type)

  def _makeBatch(self, matrix1, matrix2):
    matrix_batch = np.concatenate(
        [np.expand_dims(matrix1, 0),
         np.expand_dims(matrix2, 0)])
    matrix_batch = np.tile(matrix_batch, [2, 3, 1, 1])
    return matrix_batch

  def testNonsymmetric(self):
    # 2x2 matrices
    matrix1 = np.array([[1., 2.], [3., 4.]])
    matrix2 = np.array([[1., 3.], [3., 5.]])
    self._verifyInverseReal(matrix1)
    self._verifyInverseReal(matrix2)
    # A multidimensional batch of 2x2 matrices
    self._verifyInverseReal(self._makeBatch(matrix1, matrix2))

  def testSymmetricPositiveDefinite(self):
    # 2x2 matrices
    matrix1 = np.array([[2., 1.], [1., 2.]])
    matrix2 = np.array([[3., -1.], [-1., 3.]])
    self._verifyInverseReal(matrix1)
    self._verifyInverseReal(matrix2)
    # A multidimensional batch of 2x2 matrices
    self._verifyInverseReal(self._makeBatch(matrix1, matrix2))

  def testEmpty(self):
    self._verifyInverseReal(np.empty([0, 2, 2]))
    self._verifyInverseReal(np.empty([2, 0, 0]))


if __name__ == "__main__":
  googletest.main()
