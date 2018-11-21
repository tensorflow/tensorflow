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
"""Tests for tensorflow.ops.math_ops.matrix_solve."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


class MatrixSolveOpTest(test.TestCase):

  def _verifySolve(self, x, y, batch_dims=None):
    for np_type in [np.float32, np.float64, np.complex64, np.complex128]:
      if np_type == np.float32 or np_type == np.complex64:
        tol = 1e-5
      else:
        tol = 1e-12
      for adjoint in False, True:
        if np_type is [np.float32, np.float64]:
          a = x.real().astype(np_type)
          b = y.real().astype(np_type)
        else:
          a = x.astype(np_type)
          b = y.astype(np_type)
          a_np = np.conj(np.transpose(a)) if adjoint else a
        if batch_dims is not None:
          a = np.tile(a, batch_dims + [1, 1])
          a_np = np.tile(a_np, batch_dims + [1, 1])
          b = np.tile(b, batch_dims + [1, 1])
        np_ans = np.linalg.solve(a_np, b)
        for use_placeholder in False, True:
          with self.cached_session(use_gpu=True) as sess:
            if use_placeholder:
              a_ph = array_ops.placeholder(dtypes.as_dtype(np_type))
              b_ph = array_ops.placeholder(dtypes.as_dtype(np_type))
              tf_ans = linalg_ops.matrix_solve(a_ph, b_ph, adjoint=adjoint)
              out = sess.run(tf_ans, {a_ph: a, b_ph: b})
            else:
              tf_ans = linalg_ops.matrix_solve(a, b, adjoint=adjoint)
              out = self.evaluate(tf_ans)
              self.assertEqual(tf_ans.get_shape(), out.shape)
            self.assertEqual(np_ans.shape, out.shape)
            self.assertAllClose(np_ans, out, atol=tol, rtol=tol)

  def _generateMatrix(self, m, n):
    matrix = (np.random.normal(-5, 5,
                               m * n).astype(np.complex128).reshape([m, n]))
    matrix.imag = (np.random.normal(-5, 5, m * n).astype(np.complex128).reshape(
        [m, n]))
    return matrix

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

  def testNonSquareMatrix(self):
    # When the solve of a non-square matrix is attempted we should return
    # an error
    with self.session(use_gpu=True):
      with self.assertRaises(ValueError):
        matrix = constant_op.constant([[1., 2., 3.], [3., 4., 5.]])
        linalg_ops.matrix_solve(matrix, matrix)

  def testWrongDimensions(self):
    # The matrix and right-hand sides should have the same number of rows.
    with self.session(use_gpu=True):
      matrix = constant_op.constant([[1., 0.], [0., 1.]])
      rhs = constant_op.constant([[1., 0.]])
      with self.assertRaises(ValueError):
        linalg_ops.matrix_solve(matrix, rhs)

  def testNotInvertible(self):
    # The input should be invertible.
    with self.session(use_gpu=True):
      with self.assertRaisesOpError("Input matrix is not invertible."):
        # All rows of the matrix below add to zero
        matrix = constant_op.constant([[1., 0., -1.], [-1., 1., 0.],
                                       [0., -1., 1.]])
        linalg_ops.matrix_solve(matrix, matrix).eval()

  def testConcurrent(self):
    with self.session(use_gpu=True) as sess:
      all_ops = []
      for adjoint_ in False, True:
        lhs1 = random_ops.random_normal([3, 3], seed=42)
        lhs2 = random_ops.random_normal([3, 3], seed=42)
        rhs1 = random_ops.random_normal([3, 3], seed=42)
        rhs2 = random_ops.random_normal([3, 3], seed=42)
        s1 = linalg_ops.matrix_solve(lhs1, rhs1, adjoint=adjoint_)
        s2 = linalg_ops.matrix_solve(lhs2, rhs2, adjoint=adjoint_)
        all_ops += [s1, s2]
      val = self.evaluate(all_ops)
      self.assertAllEqual(val[0], val[1])
      self.assertAllEqual(val[2], val[3])


class MatrixSolveBenchmark(test.Benchmark):

  matrix_shapes = [
      (4, 4),
      (10, 10),
      (16, 16),
      (101, 101),
      (256, 256),
      (1001, 1001),
      (1024, 1024),
      (2048, 2048),
      (513, 4, 4),
      (513, 16, 16),
      (513, 256, 256),
  ]

  def _GenerateTestData(self, matrix_shape, num_rhs):
    batch_shape = matrix_shape[:-2]
    matrix_shape = matrix_shape[-2:]
    assert matrix_shape[0] == matrix_shape[1]
    n = matrix_shape[0]
    matrix = (np.ones(matrix_shape).astype(np.float32) /
              (2.0 * n) + np.diag(np.ones(n).astype(np.float32)))
    rhs = np.ones([n, num_rhs]).astype(np.float32)
    matrix = variables.Variable(
        np.tile(matrix, batch_shape + (1, 1)), trainable=False)
    rhs = variables.Variable(
        np.tile(rhs, batch_shape + (1, 1)), trainable=False)
    return matrix, rhs

  def benchmarkMatrixSolveOp(self):
    run_gpu_test = test.is_gpu_available(True)
    for adjoint in False, True:
      for matrix_shape in self.matrix_shapes:
        for num_rhs in 1, 2, matrix_shape[-1]:

          with ops.Graph().as_default(), \
              session.Session(config=benchmark.benchmark_config()) as sess, \
              ops.device("/cpu:0"):
            matrix, rhs = self._GenerateTestData(matrix_shape, num_rhs)
            x = linalg_ops.matrix_solve(matrix, rhs, adjoint=adjoint)
            variables.global_variables_initializer().run()
            self.run_op_benchmark(
                sess,
                control_flow_ops.group(x),
                min_iters=25,
                store_memory_usage=False,
                name=("matrix_solve_cpu_shape_{matrix_shape}_num_rhs_{num_rhs}_"
                      "adjoint_{adjoint}").format(
                          matrix_shape=matrix_shape,
                          num_rhs=num_rhs,
                          adjoint=adjoint))

          if run_gpu_test:
            with ops.Graph().as_default(), \
                session.Session(config=benchmark.benchmark_config()) as sess, \
                ops.device("/gpu:0"):
              matrix, rhs = self._GenerateTestData(matrix_shape, num_rhs)
              x = linalg_ops.matrix_solve(matrix, rhs, adjoint=adjoint)
              variables.global_variables_initializer().run()
              self.run_op_benchmark(
                  sess,
                  control_flow_ops.group(x),
                  min_iters=25,
                  store_memory_usage=False,
                  name=("matrix_solve_gpu_shape_{matrix_shape}_num_rhs_"
                        "{num_rhs}_adjoint_{adjoint}").format(
                            matrix_shape=matrix_shape, num_rhs=num_rhs,
                            adjoint=adjoint))


if __name__ == "__main__":
  test.main()
