# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.gen_linalg_ops.matrix_exponential."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def np_expm(x):
  """Slow but accurate Taylor series matrix exponential."""
  y = np.zeros(x.shape, dtype=x.dtype)
  xn = np.eye(x.shape[0], dtype=x.dtype)
  for n in range(40):
    y += xn / float(math.factorial(n))
    xn = np.dot(xn, x)
  return y


class ExponentialOpTest(test.TestCase):

  def _verifyExponential(self, x, np_type):
    inp = x.astype(np_type)
    with self.test_session(use_gpu=True):
      tf_ans = gen_linalg_ops._matrix_exponential(inp)
      if x.size == 0:
        np_ans = np.empty(x.shape, dtype=np_type)
      else:
        if x.ndim > 2:
          np_ans = np.zeros(inp.shape, dtype=np_type)
          for i in itertools.product(*[range(x) for x in inp.shape[:-2]]):
            np_ans[i] = np_expm(inp[i])
        else:
          np_ans = np_expm(inp)
      out = tf_ans.eval()
      self.assertAllClose(np_ans, out, rtol=1e-4, atol=1e-3)

  def _verifyExponentialReal(self, x):
    for np_type in [np.float32, np.float64]:
      self._verifyExponential(x, np_type)

  def _verifyExponentialComplex(self, x):
    for np_type in [np.complex64, np.complex128]:
      self._verifyExponential(x, np_type)

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
    self._verifyExponentialReal(matrix1)
    self._verifyExponentialReal(matrix2)
    # A multidimensional batch of 2x2 matrices
    self._verifyExponentialReal(self._makeBatch(matrix1, matrix2))
    # Complex
    matrix1 = matrix1.astype(np.complex64)
    matrix1 += 1j * matrix1
    matrix2 = matrix2.astype(np.complex64)
    matrix2 += 1j * matrix2
    self._verifyExponentialComplex(matrix1)
    self._verifyExponentialComplex(matrix2)
    # Complex batch
    self._verifyExponentialComplex(self._makeBatch(matrix1, matrix2))

  def testSymmetricPositiveDefinite(self):
    # 2x2 matrices
    matrix1 = np.array([[2., 1.], [1., 2.]])
    matrix2 = np.array([[3., -1.], [-1., 3.]])
    self._verifyExponentialReal(matrix1)
    self._verifyExponentialReal(matrix2)
    # A multidimensional batch of 2x2 matrices
    self._verifyExponentialReal(self._makeBatch(matrix1, matrix2))
    # Complex
    matrix1 = matrix1.astype(np.complex64)
    matrix1 += 1j * matrix1
    matrix2 = matrix2.astype(np.complex64)
    matrix2 += 1j * matrix2
    self._verifyExponentialComplex(matrix1)
    self._verifyExponentialComplex(matrix2)
    # Complex batch
    self._verifyExponentialComplex(self._makeBatch(matrix1, matrix2))

  def testNonSquareMatrix(self):
    # When the exponential of a non-square matrix is attempted we should return
    # an error
    with self.assertRaises(ValueError):
      gen_linalg_ops._matrix_exponential(np.array([[1., 2., 3.], [3., 4., 5.]]))

  def testWrongDimensions(self):
    # The input to the exponential should be at least a 2-dimensional tensor.
    tensor3 = constant_op.constant([1., 2.])
    with self.assertRaises(ValueError):
      gen_linalg_ops._matrix_exponential(tensor3)

  def testEmpty(self):
    self._verifyExponentialReal(np.empty([0, 2, 2]))
    self._verifyExponentialReal(np.empty([2, 0, 0]))

  def testRandomSmallAndLarge(self):
    np.random.seed(42)
    for dtype in np.float32, np.float64, np.complex64, np.complex128:
      for batch_dims in [(), (1,), (3,), (2, 2)]:
        for size in 8, 31, 32:
          shape = batch_dims + (size, size)
          matrix = np.random.uniform(
              low=-1.0, high=1.0,
              size=np.prod(shape)).reshape(shape).astype(dtype)
          self._verifyExponentialReal(matrix)

  def testConcurrentExecutesWithoutError(self):
    with self.test_session(use_gpu=True) as sess:
      matrix1 = random_ops.random_normal([5, 5], seed=42)
      matrix2 = random_ops.random_normal([5, 5], seed=42)
      expm1 = gen_linalg_ops._matrix_exponential(matrix1)
      expm2 = gen_linalg_ops._matrix_exponential(matrix2)
      expm = sess.run([expm1, expm2])
      self.assertAllEqual(expm[0], expm[1])


class MatrixExponentialBenchmark(test.Benchmark):

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

  def benchmarkMatrixExponentialOp(self):
    for shape in self.shapes:
      with ops.Graph().as_default(), \
          session.Session() as sess, \
          ops.device("/cpu:0"):
        matrix = self._GenerateMatrix(shape)
        expm = gen_linalg_ops._matrix_exponential(matrix)
        variables.global_variables_initializer().run()
        self.run_op_benchmark(
            sess,
            control_flow_ops.group(expm),
            min_iters=25,
            name="matrix_exponential_cpu_{shape}".format(
                shape=shape))


if __name__ == "__main__":
  test.main()
