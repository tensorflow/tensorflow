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
"""Tests for XLA implementation of tf.linalg.solve."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import googletest


class MatrixSolveOpTest(xla_test.XLATestCase):

  def _verifySolve(self, x, y, batch_dims=None):
    for np_type in self.float_types & {np.float32, np.float64}:
      if np_type == np.float32:
        tol = 1e-4
      else:
        tol = 1e-12
      for adjoint in False, True:
        a = x.astype(np_type)
        b = y.astype(np_type)
        a_np = np.transpose(a) if adjoint else a
        if batch_dims is not None:
          a = np.tile(a, batch_dims + [1, 1])
          a_np = np.tile(a_np, batch_dims + [1, 1])
          b = np.tile(b, batch_dims + [1, 1])
        np_ans = np.linalg.solve(a_np, b)
        for use_placeholder in False, True:
          with self.session() as sess:
            if use_placeholder:
              with self.test_scope():
                a_ph = array_ops.placeholder(dtypes.as_dtype(np_type))
                b_ph = array_ops.placeholder(dtypes.as_dtype(np_type))
                tf_ans = linalg_ops.matrix_solve(a_ph, b_ph, adjoint=adjoint)
              out = sess.run(tf_ans, {a_ph: a, b_ph: b})
            else:
              with self.test_scope():
                tf_ans = linalg_ops.matrix_solve(a, b, adjoint=adjoint)
              out = sess.run(tf_ans)
              self.assertEqual(tf_ans.get_shape(), out.shape)
            self.assertEqual(np_ans.shape, out.shape)
            self.assertAllClose(np_ans, out, atol=tol, rtol=tol)

  def _generateMatrix(self, m, n):
    return np.random.normal(-5, 5, m * n).reshape([m, n])

  def testSolve(self):
    for n in 1, 2, 4, 9:
      matrix = self._generateMatrix(n, n)
      for nrhs in 1, 2, n:
        rhs = self._generateMatrix(n, nrhs)
        self._verifySolve(matrix, rhs)

  def testSolveBatch(self):
    for n in 2, 5:
      matrix = self._generateMatrix(n, n)
      for nrhs in 1, n:
        rhs = self._generateMatrix(n, nrhs)
        for batch_dims in [[2], [2, 2], [7, 4]]:
          self._verifySolve(matrix, rhs, batch_dims=batch_dims)

  def testConcurrent(self):
    with self.session() as sess:
      all_ops = []
      for adjoint_ in False, True:
        lhs1 = random_ops.random_normal([3, 3], seed=42)
        lhs2 = random_ops.random_normal([3, 3], seed=42)
        rhs1 = random_ops.random_normal([3, 3], seed=42)
        rhs2 = random_ops.random_normal([3, 3], seed=42)
        with self.test_scope():
          s1 = linalg_ops.matrix_solve(lhs1, rhs1, adjoint=adjoint_)
          s2 = linalg_ops.matrix_solve(lhs2, rhs2, adjoint=adjoint_)
        all_ops += [s1, s2]
      val = sess.run(all_ops)
      self.assertAllEqual(val[0], val[1])
      self.assertAllEqual(val[2], val[3])


if __name__ == "__main__":
  googletest.main()
