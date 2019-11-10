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

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import googletest


class MatrixSolveOpTest(xla_test.XLATestCase, parameterized.TestCase):

  def _verifySolve(self, x, y, adjoint):
    for np_type in self.float_types & {np.float32, np.float64}:
      tol = 1e-4 if np_type == np.float32 else 1e-12
      a = x.astype(np_type)
      b = y.astype(np_type)
      np_ans = np.linalg.solve(np.swapaxes(a, -2, -1) if adjoint else a, b)
      with self.session() as sess:
        with self.test_scope():
          tf_ans = linalg_ops.matrix_solve(a, b, adjoint=adjoint)
        out = sess.run(tf_ans)
        self.assertEqual(tf_ans.shape, out.shape)
        self.assertEqual(np_ans.shape, out.shape)
        self.assertAllClose(np_ans, out, atol=tol, rtol=tol)

  @parameterized.named_parameters(
      ("Scalar", 1, 1, [], [], False),
      ("Vector", 5, 1, [], [], False),
      ("MultipleRHS", 5, 4, [], [], False),
      ("Adjoint", 5, 4, [], [], True),
      ("BatchedScalar", 1, 4, [2], [2], False),
      ("BatchedVector", 5, 4, [2], [2], False),
      ("BatchedRank2", 5, 4, [7, 4], [7, 4], False),
      ("BatchedAdjoint", 5, 4, [7, 4], [7, 4], True),
  )
  def testSolve(self, n, nrhs, batch_dims, rhs_batch_dims, adjoint):
    matrix = np.random.normal(-5.0, 5.0, batch_dims + [n, n])
    rhs = np.random.normal(-5.0, 5.0, rhs_batch_dims + [n, nrhs])
    self._verifySolve(matrix, rhs, adjoint=adjoint)

  @parameterized.named_parameters(
      ("Simple", False),
      ("Adjoint", True),
  )
  def testConcurrent(self, adjoint):
    with self.session() as sess:
      lhs1 = random_ops.random_normal([3, 3], seed=42)
      lhs2 = random_ops.random_normal([3, 3], seed=42)
      rhs1 = random_ops.random_normal([3, 3], seed=42)
      rhs2 = random_ops.random_normal([3, 3], seed=42)
      with self.test_scope():
        s1 = linalg_ops.matrix_solve(lhs1, rhs1, adjoint=adjoint)
        s2 = linalg_ops.matrix_solve(lhs2, rhs2, adjoint=adjoint)
      self.assertAllEqual(*sess.run([s1, s2]))


if __name__ == "__main__":
  googletest.main()
