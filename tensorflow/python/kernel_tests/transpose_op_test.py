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

"""Functional tests for Transpose op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np
import tensorflow as tf


class TransposeTest(tf.test.TestCase):

  def _np_transpose(self, x, perm):
    ret = np.copy(x)
    ret = ret.transpose(perm)
    return ret

  def _compareCpu(self, x, p):
    np_ans = self._np_transpose(x, p)
    with self.test_session(use_gpu=False):
      inx = tf.convert_to_tensor(x)
      y = tf.transpose(inx, p)
      tf_ans = y.eval()
      self.assertAllEqual(np_ans, tf_ans)
      self.assertShapeEqual(np_ans, y)

      jacob_t = None
      # Gradient check on CPU.
      xs = list(np.shape(x))
      ys = list(np.shape(tf_ans))
      if x.dtype == np.float32:
        jacob_t, jacob_n = tf.test.compute_gradient(inx, xs, y, ys, x, 1e-2)
        self.assertAllClose(jacob_t, jacob_n, 1e-3, 1e-3)
      elif x.dtype == np.float64:
        jacob_t, jacob_n = tf.test.compute_gradient(inx, xs, y, ys, x, 1e-2)
        self.assertAllClose(jacob_t, jacob_n, 1e-6, 1e-6)

      return tf_ans, jacob_t

  def _compareGpu(self, x, p):
    np_ans = self._np_transpose(x, p)
    with self.test_session(use_gpu=True):
      inx = tf.convert_to_tensor(x)
      y = tf.transpose(inx, p)
      tf_ans = y.eval()

      self.assertAllEqual(np_ans, tf_ans)
      self.assertShapeEqual(np_ans, y)

      jacob_t = None
      # Gradient check on GPU.
      xs = list(np.shape(x))
      ys = list(np.shape(tf_ans))
      if x.dtype == np.float32:
        jacob_t, jacob_n = tf.test.compute_gradient(inx, xs, y, ys, x, 1e-2)
        self.assertAllClose(jacob_t, jacob_n, 1e-3, 1e-3)
      elif x.dtype == np.float64:
        jacob_t, jacob_n = tf.test.compute_gradient(inx, xs, y, ys, x, 1e-2)
        self.assertAllClose(jacob_t, jacob_n, 1e-6, 1e-6)

      return tf_ans, jacob_t

  def _compare(self, x, use_gpu=False):
    n = np.ndim(x)
    # generate all permutations of [0, 1, ... n-1] in random order.
    all_perm = np.random.permutation(
        [p for p in itertools.permutations(range(n))]).astype(np.int32)
    for p in all_perm[0:2]:
      self._compareCpu(x, p)
      if use_gpu:
        self._compareGpu(x, p)

  def _compare_cpu_gpu(self, x):
    n = np.ndim(x)
    # generate all permutation of [0, 1, ... n-1] in random order,
    # choose the first two.
    perms = itertools.permutations(range(n))
    for _ in range(2):
      p = np.random.permutation(next(perms)).astype(np.int32)
      tf_a_cpu, tf_g_cpu = self._compareCpu(x, p)
      tf_a_gpu, tf_g_gpu = self._compareGpu(x, p)
      assert tf_g_cpu is not None
      assert tf_g_gpu is not None
      if x.dtype == np.float32:
        self.assertAllClose(tf_a_cpu, tf_a_gpu, 1e-3, 1e-3)
        self.assertAllClose(tf_g_cpu, tf_g_gpu, 1e-3, 1e-3)
      elif x.dtype == np.float64:
        self.assertAllClose(tf_a_cpu, tf_a_gpu, 1e-6, 1e-6)
        self.assertAllClose(tf_g_cpu, tf_g_gpu, 1e-6, 1e-6)

  def _testBoth(self, x):
    self._compare(x, use_gpu=False)
    self._compare(x, use_gpu=True)

  def testRank1(self):
    self._compareCpu(np.arange(0., 2), [0])

  def test1D(self):
    vector = np.arange(0, 2).reshape((1, 1, 1, 2, 1))
    self._compare(vector, use_gpu=False)
    self._compare(vector, use_gpu=True)

  def testNop(self):
    self._compareCpu(np.arange(0, 6).reshape([3, 2]).astype(np.float32), [0, 1])

  def testSimple(self):
    self._compareCpu(np.arange(0, 8).reshape([2, 4]).astype(np.float32),
                     np.array([1, 0]).astype(np.int32))

  def testHalf(self):
    self._compare(np.arange(0, 21).reshape([3, 7]).astype(np.float16))
    self._compare(
        np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.float16))
    self._compare(
        np.arange(0, 16).reshape([1, 2, 1, 2, 1, 2, 1, 2]).astype(np.float16))

  def testFloat(self):
    self._compare_cpu_gpu(np.arange(0, 21).reshape([3, 7]).astype(np.float32))
    self._compare_cpu_gpu(
        np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.float32))
    self._compare_cpu_gpu(
        np.arange(0, 16).reshape([1, 2, 1, 2, 1, 2, 1, 2]).astype(np.float32))

  def testDouble(self):
    self._compare_cpu_gpu(np.arange(0, 21).reshape([3, 7]).astype(np.float64))
    self._compare_cpu_gpu(
        np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.float64))
    self._compare_cpu_gpu(
        np.arange(0, 16).reshape([1, 2, 1, 2, 1, 2, 1, 2]).astype(np.float64))

  def testComplex64(self):
    self._testBoth(np.complex(1, 2) *
                   np.arange(0, 21).reshape([3, 7]).astype(np.complex64))
    self._testBoth(np.complex(1, 2) *
                   np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.complex64))
    self._testBoth(
        np.complex(1, 2) *
        np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(np.complex64))

  def testComplex128(self):
    self._testBoth(np.complex(1, 2) *
                   np.arange(0, 21).reshape([3, 7]).astype(np.complex128))
    self._testBoth(np.complex(1, 2) *
                   np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.complex128))
    self._testBoth(
        np.complex(1, 2) *
        np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(np.complex128))

  def testInt8(self):
    self._testBoth(np.arange(0, 21).reshape([3, 7]).astype(np.int8))
    self._testBoth(np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.int8))
    self._testBoth(np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(
        np.int8))

  def testInt16(self):
    self._testBoth(np.arange(0, 21).reshape([3, 7]).astype(np.int16))
    self._testBoth(np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.int16))
    self._testBoth(np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(
        np.int16))

  def testInt32(self):
    self._testBoth(np.arange(0, 21).reshape([3, 7]).astype(np.int32))
    self._testBoth(np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.int32))
    self._testBoth(np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(
        np.int32))

  def testInt64(self):
    self._testBoth(np.arange(0, 21).reshape([3, 7]).astype(np.int64))
    self._testBoth(np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.int64))
    self._testBoth(np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(
        np.int64))

  def testTranspose2DAuto(self):
    x_np = [[1, 2, 3], [4, 5, 6]]
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        x_tf = tf.transpose(x_np).eval()
        self.assertAllEqual(x_tf, [[1, 4], [2, 5], [3, 6]])

  def testTransposeShapes(self):
    self.assertEqual([], tf.transpose(
        tf.placeholder(tf.int32, shape=[])).get_shape().dims)
    self.assertEqual([100], tf.transpose(
        tf.placeholder(tf.int32, shape=[100])).get_shape().dims)
    self.assertEqual([37, 100], tf.transpose(
        tf.placeholder(tf.int32, shape=[100, 37])).get_shape().dims)
    self.assertEqual([100, 37], tf.transpose(
        tf.placeholder(tf.int32, shape=[100, 37]), [0, 1]).get_shape().dims)
    self.assertEqual([15, 37, 100], tf.transpose(
        tf.placeholder(tf.int32, shape=[100, 37, 15])).get_shape().dims)
    self.assertEqual([15, 100, 37], tf.transpose(
        tf.placeholder(tf.int32,
                       shape=[100, 37, 15]), [2, 0, 1]).get_shape().dims)
    self.assertEqual(tf.TensorShape(None), tf.transpose(
        tf.placeholder(tf.int32)).get_shape())

  def testNullTensor(self):
    with self.test_session():
      x = tf.constant([], dtype=tf.float32, shape=[1, 4, 0])
      xt = tf.transpose(x, [0, 2, 1]).eval()
      self.assertAllEqual(xt.shape, (1, 0, 4))

  def _testError(self, x, p, err):
    with self.test_session():
      with self.assertRaisesOpError(err):
        tf.transpose(x, p).eval()

  def testError(self):
    with self.assertRaises(ValueError):
      tf.transpose(np.arange(0., 30).reshape([2, 3, 5]), [[0, 1], [2, 3]])
    self._testError(np.arange(0., 2 ** 11).reshape([2] * 11),
                    np.arange(11),
                    "not implemented")
    with self.assertRaises(ValueError):
      tf.transpose(np.arange(0., 30).reshape([2, 3, 5]), [0, 1, 3])
    self._testError(np.arange(0., 30).reshape([2, 3, 5]),
                    [0, 1, 1],
                    "2 is missing")

if __name__ == "__main__":
  tf.test.main()
