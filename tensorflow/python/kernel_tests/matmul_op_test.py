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

"""Tests for tensorflow.ops.math_ops.matmul."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops


class MatMulTest(tf.test.TestCase):

  def _testCpuMatmul(self, x, y, transpose_x=False, transpose_y=False):
    x_mat = np.matrix(x).T if transpose_x else np.matrix(x)
    y_mat = np.matrix(y).T if transpose_y else np.matrix(y)
    np_ans = x_mat * y_mat
    with self.test_session(use_gpu=False):
      tf_ans = tf.matmul(x, y, transpose_x, transpose_y).eval()
    self.assertAllClose(np_ans, tf_ans)
    self.assertAllEqual(np_ans.shape, tf_ans.shape)

  def _testGpuMatmul(self, x, y, transpose_x=False, transpose_y=False):
    x_mat = np.matrix(x).T if transpose_x else np.matrix(x)
    y_mat = np.matrix(y).T if transpose_y else np.matrix(y)
    np_ans = x_mat * y_mat
    with self.test_session(use_gpu=True):
      tf_ans = tf.matmul(x, y, transpose_x, transpose_y).eval()
    self.assertAllClose(np_ans, tf_ans)
    self.assertAllEqual(np_ans.shape, tf_ans.shape)

  def _randMatrix(self, rows, cols, dtype):
    if dtype is np.complex64:
      real = self._randMatrix(rows, cols, np.float32)
      imag = self._randMatrix(rows, cols, np.float32)
      return real + np.complex(0, 1) * imag
    else:
      return np.random.uniform(low=1.0, high=100.0, size=rows * cols).reshape(
          [rows, cols]).astype(dtype)

  # Basic test:
  #   [ [1],
  #     [2],
  #     [3],   *  [1, 2]
  #     [4] ]
  def testFloatBasic(self):
    x = np.arange(1., 5.).reshape([4, 1]).astype(np.float32)
    y = np.arange(1., 3.).reshape([1, 2]).astype(np.float32)
    self._testCpuMatmul(x, y)
    self._testGpuMatmul(x, y)

  def testDoubleBasic(self):
    x = np.arange(1., 5.).reshape([4, 1]).astype(np.float64)
    y = np.arange(1., 3.).reshape([1, 2]).astype(np.float64)
    self._testCpuMatmul(x, y)

  def testInt32Basic(self):
    x = np.arange(1., 5.).reshape([4, 1]).astype(np.int32)
    y = np.arange(1., 3.).reshape([1, 2]).astype(np.int32)
    self._testCpuMatmul(x, y)

  def testSComplexBasic(self):
    x = np.arange(1., 5.).reshape([4, 1]).astype(np.complex64)
    y = np.arange(1., 3.).reshape([1, 2]).astype(np.complex64)
    self._testCpuMatmul(x, y)

  # Tests testing random sized matrices.
  def testFloatRandom(self):
    for _ in range(10):
      n, k, m = np.random.randint(1, 100, size=3)
      x = self._randMatrix(n, k, np.float32)
      y = self._randMatrix(k, m, np.float32)
      self._testCpuMatmul(x, y)
      self._testGpuMatmul(x, y)

  def testDoubleRandom(self):
    for _ in range(10):
      n, k, m = np.random.randint(1, 100, size=3)
      x = self._randMatrix(n, k, np.float64)
      y = self._randMatrix(k, m, np.float64)
      self._testCpuMatmul(x, y)

  def testInt32Random(self):
    for _ in range(10):
      n, k, m = np.random.randint(1, 100, size=3)
      x = self._randMatrix(n, k, np.int32)
      y = self._randMatrix(k, m, np.int32)
      self._testCpuMatmul(x, y)

  def testSComplexRandom(self):
    for _ in range(10):
      n, k, m = np.random.randint(1, 100, size=3)
      x = self._randMatrix(n, k, np.complex64)
      y = self._randMatrix(k, m, np.complex64)
      self._testCpuMatmul(x, y)

  # Test the cases that transpose the matrices before multiplying.
  # NOTE(keveman): The cases where only one of the inputs is
  # transposed are covered by tf.matmul's gradient function.
  def testFloatRandomTransposeBoth(self):
    for _ in range(10):
      n, k, m = np.random.randint(1, 100, size=3)
      x = self._randMatrix(k, n, np.float32)
      y = self._randMatrix(m, k, np.float32)
      self._testCpuMatmul(x, y, True, True)
      self._testGpuMatmul(x, y, True, True)

  def testDoubleRandomTransposeBoth(self):
    for _ in range(10):
      n, k, m = np.random.randint(1, 100, size=3)
      x = self._randMatrix(k, n, np.float64)
      y = self._randMatrix(m, k, np.float64)
      self._testCpuMatmul(x, y, True, True)

  def testMatMul_OutEmpty_A(self):
    n, k, m = 0, 8, 3
    x = self._randMatrix(n, k, np.float32)
    y = self._randMatrix(k, m, np.float32)
    self._testCpuMatmul(x, y)
    self._testGpuMatmul(x, y)

  def testMatMul_OutEmpty_B(self):
    n, k, m = 3, 8, 0
    x = self._randMatrix(n, k, np.float32)
    y = self._randMatrix(k, m, np.float32)
    self._testCpuMatmul(x, y)
    self._testGpuMatmul(x, y)

  def testMatMul_Inputs_Empty(self):
    n, k, m = 3, 0, 4
    x = self._randMatrix(n, k, np.float32)
    y = self._randMatrix(k, m, np.float32)
    self._testCpuMatmul(x, y)
    self._testGpuMatmul(x, y)

  def testShapeErrors(self):
    a = tf.placeholder(tf.float32, [32, 37])
    b = tf.placeholder(tf.float32, [36, 2])
    c = tf.placeholder(tf.float32, [37])
    with self.assertRaisesRegexp(
        ValueError, "Dimensions 37 and 36 are not compatible"):
      tf.matmul(a, b)
    with self.assertRaisesRegexp(ValueError, "must have rank 2"):
      tf.matmul(a, c)


# TODO(zhifengc): Figures out how to test matmul gradients on GPU.
class MatMulGradientTest(tf.test.TestCase):

  def testGradientInput0(self):
    with self.test_session(use_gpu=False):
      x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2],
                   dtype=tf.float64, name="x")
      y = tf.constant([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
                   shape=[2, 4], dtype=tf.float64, name="y")
      m = tf.matmul(x, y, name="matmul")
      err = tf.test.compute_gradient_error(x, [3, 2], m, [3, 4])
    print("matmul input0 gradient err = ", err)
    self.assertLess(err, 1e-10)

  def testGradientInput1(self):
    with self.test_session(use_gpu=False):
      x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2],
                   dtype=tf.float64, name="x")
      y = tf.constant([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
                   shape=[2, 4], dtype=tf.float64, name="y")
      m = tf.matmul(x, y, name="matmul")
      err = tf.test.compute_gradient_error(y, [2, 4], m, [3, 4])
    print("matmul input1 gradient err = ", err)
    self.assertLess(err, 1e-10)

  def _VerifyInput0(self, transpose_a, transpose_b):
    shape_x = [3, 2]
    shape_y = [2, 4]
    if transpose_a:
      shape_x = list(reversed(shape_x))
    if transpose_b:
      shape_y = list(reversed(shape_y))
    with self.test_session(use_gpu=False):
      x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=shape_x,
                   dtype=tf.float64, name="x")
      y = tf.constant([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
                   shape=shape_y, dtype=tf.float64, name="y")
      m = tf.matmul(x, y, transpose_a, transpose_b, name="matmul")
      err = tf.test.compute_gradient_error(x, shape_x, m, [3, 4])
    print("matmul input0 gradient err = ", err)
    self.assertLess(err, 1e-10)

  def testGradientInput0WithTranspose(self):
    self._VerifyInput0(transpose_a=True, transpose_b=False)
    self._VerifyInput0(transpose_a=False, transpose_b=True)
    self._VerifyInput0(transpose_a=True, transpose_b=True)

  def _VerifyInput1(self, transpose_a, transpose_b):
    shape_x = [3, 2]
    shape_y = [2, 4]
    if transpose_a:
      shape_x = list(reversed(shape_x))
    if transpose_b:
      shape_y = list(reversed(shape_y))
    with self.test_session(use_gpu=False):
      x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=shape_x,
                   dtype=tf.float64, name="x")
      y = tf.constant([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
                   shape=shape_y, dtype=tf.float64, name="y")
      m = tf.matmul(x, y, transpose_a, transpose_b, name="matmul")
      err = tf.test.compute_gradient_error(y, shape_y, m, [3, 4])
    print("matmul input1 gradient err = ", err)
    self.assertLess(err, 1e-10)

  def testGradientInput1WithTranspose(self):
    self._VerifyInput1(transpose_a=True, transpose_b=False)
    self._VerifyInput1(transpose_a=False, transpose_b=True)
    self._VerifyInput1(transpose_a=True, transpose_b=True)


class MatMulStatsTest(tf.test.TestCase):

  def testSimpleStatistics(self):
    g = tf.Graph()
    with g.as_default():
      a = tf.Variable(tf.random_normal([25, 16]))
      b = tf.Variable(tf.random_normal([16, 9]))
      tf.matmul(a, b)
      for op in g.get_operations():
        flops = ops.get_stats_for_node_def(g, op.node_def, "flops").value
        if op.name == "MatMul":
          self.assertEqual(7200, flops)

  def testTransposedStatistics(self):
    g = tf.Graph()
    with g.as_default():
      a = tf.Variable(tf.random_normal([16, 25]))
      b = tf.Variable(tf.random_normal([16, 9]))
      tf.matmul(a, b, transpose_a=True)
      for op in g.get_operations():
        flops = ops.get_stats_for_node_def(g, op.node_def, "flops").value
        if op.name == "MatMul":
          self.assertEqual(7200, flops)


if __name__ == "__main__":
  tf.test.main()
