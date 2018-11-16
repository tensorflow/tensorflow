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
"""Tests for tensorflow.ops.tf.MatrixDeterminant."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


class DeterminantOpTest(test.TestCase):

  def _compareDeterminantBase(self, matrix_x, tf_ans):
    out = self.evaluate(tf_ans)
    shape = matrix_x.shape
    if shape[-1] == 0 and shape[-2] == 0:
      np_ans = np.ones(shape[:-2]).astype(matrix_x.dtype)
    else:
      np_ans = np.array(np.linalg.det(matrix_x)).astype(matrix_x.dtype)
    self.assertShapeEqual(np_ans, tf_ans)
    self.assertAllClose(np_ans, out, atol=5e-5)

  def _compareLogDeterminantBase(self, matrix_x, tf_ans):
    sign_tf, abs_log_det_tf = tf_ans
    shape = matrix_x.shape
    if shape[-1] == 0 or shape[-2] == 0:
      np_sign, np_ans = (1.0, np.zeros(shape[:-2]).astype(matrix_x.dtype))
    else:
      np_sign, np_ans = np.linalg.slogdet(matrix_x)
      np_ans = np_ans.astype(matrix_x.dtype)

    self.assertShapeEqual(np_ans, abs_log_det_tf)
    sign_tf_val = self.evaluate(sign_tf)
    abs_log_det_tf_val = self.evaluate(abs_log_det_tf)
    self.assertAllClose(
        sign_tf_val * np.exp(abs_log_det_tf_val),
        np_sign * np.exp(np_ans),
        atol=5e-5)

  def _compareDeterminant(self, matrix_x):
    with self.cached_session(use_gpu=True):
      self._compareDeterminantBase(matrix_x,
                                   linalg_ops.matrix_determinant(matrix_x))
      self._compareLogDeterminantBase(
          matrix_x, gen_linalg_ops.log_matrix_determinant(matrix_x))

  def testBasic(self):
    # 2x2 matrices
    self._compareDeterminant(np.array([[2., 3.], [3., 4.]]).astype(np.float32))
    self._compareDeterminant(np.array([[0., 0.], [0., 0.]]).astype(np.float32))
    # 5x5 matrices (Eigen forces LU decomposition)
    self._compareDeterminant(
        np.array([[2., 3., 4., 5., 6.], [3., 4., 9., 2., 0.], [
            2., 5., 8., 3., 8.
        ], [1., 6., 7., 4., 7.], [2., 3., 4., 5., 6.]]).astype(np.float32))
    # A multidimensional batch of 2x2 matrices
    self._compareDeterminant(np.random.rand(3, 4, 5, 2, 2).astype(np.float32))

  def testBasicDouble(self):
    # 2x2 matrices
    self._compareDeterminant(np.array([[2., 3.], [3., 4.]]).astype(np.float64))
    self._compareDeterminant(np.array([[0., 0.], [0., 0.]]).astype(np.float64))
    # 5x5 matrices (Eigen forces LU decomposition)
    self._compareDeterminant(
        np.array([[2., 3., 4., 5., 6.], [3., 4., 9., 2., 0.], [
            2., 5., 8., 3., 8.
        ], [1., 6., 7., 4., 7.], [2., 3., 4., 5., 6.]]).astype(np.float64))
    # A multidimensional batch of 2x2 matrices
    self._compareDeterminant(np.random.rand(3, 4, 5, 2, 2).astype(np.float64))

  def testBasicComplex64(self):
    # 2x2 matrices
    self._compareDeterminant(
        np.array([[2., 3.], [3., 4.]]).astype(np.complex64))
    self._compareDeterminant(
        np.array([[0., 0.], [0., 0.]]).astype(np.complex64))
    self._compareDeterminant(
        np.array([[1. + 1.j, 1. - 1.j], [-1. + 1.j, -1. - 1.j]]).astype(
            np.complex64))
    # 5x5 matrices (Eigen forces LU decomposition)
    self._compareDeterminant(
        np.array([[2., 3., 4., 5., 6.], [3., 4., 9., 2., 0.], [
            2., 5., 8., 3., 8.
        ], [1., 6., 7., 4., 7.], [2., 3., 4., 5., 6.]]).astype(np.complex64))
    # A multidimensional batch of 2x2 matrices
    self._compareDeterminant(np.random.rand(3, 4, 5, 2, 2).astype(np.complex64))

  def testBasicComplex128(self):
    # 2x2 matrices
    self._compareDeterminant(
        np.array([[2., 3.], [3., 4.]]).astype(np.complex128))
    self._compareDeterminant(
        np.array([[0., 0.], [0., 0.]]).astype(np.complex128))
    self._compareDeterminant(
        np.array([[1. + 1.j, 1. - 1.j], [-1. + 1.j, -1. - 1.j]]).astype(
            np.complex128))
    # 5x5 matrices (Eigen forces LU decomposition)
    self._compareDeterminant(
        np.array([[2., 3., 4., 5., 6.], [3., 4., 9., 2., 0.], [
            2., 5., 8., 3., 8.
        ], [1., 6., 7., 4., 7.], [2., 3., 4., 5., 6.]]).astype(np.complex128))
    # A multidimensional batch of 2x2 matrices
    self._compareDeterminant(
        np.random.rand(3, 4, 5, 2, 2).astype(np.complex128))

  def testInfiniteDeterminant(self):
    max_double = np.finfo("d").max
    huge_matrix = np.array([[max_double, 0.0], [0.0, max_double]])
    self._compareDeterminant(huge_matrix)

  def testNonSquareMatrix(self):
    # When the determinant of a non-square matrix is attempted we should return
    # an error
    with self.assertRaises(ValueError):
      linalg_ops.matrix_determinant(
          np.array([[1., 2., 3.], [3., 5., 4.]]).astype(np.float32))

  def testWrongDimensions(self):
    # The input to the determinant should be a 2-dimensional tensor.
    tensor1 = constant_op.constant([1., 2.])
    with self.assertRaises(ValueError):
      linalg_ops.matrix_determinant(tensor1)

  def testEmpty(self):
    self._compareDeterminant(np.empty([0, 2, 2]))
    self._compareDeterminant(np.empty([2, 0, 0]))

  def testConcurrentExecutesWithoutError(self):
    with self.session(use_gpu=True) as sess:
      matrix1 = random_ops.random_normal([5, 5], seed=42)
      matrix2 = random_ops.random_normal([5, 5], seed=42)
      det1 = linalg_ops.matrix_determinant(matrix1)
      det2 = linalg_ops.matrix_determinant(matrix2)
      det1_val, det2_val = sess.run([det1, det2])
      self.assertEqual(det1_val, det2_val)


class MatrixDeterminantBenchmark(test.Benchmark):

  shapes = [
      (4, 4),
      (10, 10),
      (16, 16),
      (101, 101),
      (256, 256),
      (1000, 1000),
      (1024, 1024),
      (2048, 2048),
      (513, 4, 4),
      (513, 16, 16),
      (513, 256, 256),
  ]

  def _GenerateMatrix(self, shape):
    batch_shape = shape[:-2]
    shape = shape[-2:]
    assert shape[0] == shape[1]
    n = shape[0]
    matrix = np.ones(shape).astype(np.float32) / (
        2.0 * n) + np.diag(np.ones(n).astype(np.float32))
    return variables.Variable(np.tile(matrix, batch_shape + (1, 1)))

  def benchmarkMatrixDeterminantOp(self):
    for shape in self.shapes:
      with ops.Graph().as_default(), session.Session(
          config=benchmark.benchmark_config()) as sess, ops.device("/cpu:0"):
        matrix = self._GenerateMatrix(shape)
        d = linalg_ops.matrix_determinant(matrix)
        variables.global_variables_initializer().run()
        self.run_op_benchmark(
            sess,
            control_flow_ops.group(
                d,),
            min_iters=25,
            name="matrix_determinant_cpu_{shape}".format(shape=shape))

      if test.is_gpu_available(True):
        with ops.Graph().as_default(), session.Session(
            config=benchmark.benchmark_config()) as sess, ops.device("/gpu:0"):
          matrix = self._GenerateMatrix(shape)
          d = linalg_ops.matrix_determinant(matrix)
          variables.global_variables_initializer().run()
          self.run_op_benchmark(
              sess,
              control_flow_ops.group(
                  d,),
              min_iters=25,
              name="matrix_determinant_gpu_{shape}".format(shape=shape))


if __name__ == "__main__":
  test.main()
