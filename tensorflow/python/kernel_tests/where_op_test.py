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
"""Tests for tensorflow.ops.reverse_sequence_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import sys

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


class WhereOpTest(test.TestCase):

  def _testWhere(self, x, truth, expected_err_re=None, fn=array_ops.where):
    with self.cached_session(use_gpu=True):
      ans = fn(x)
      self.assertEqual([None, x.ndim], ans.get_shape().as_list())
      if expected_err_re is None:
        tf_ans = self.evaluate(ans)
        self.assertAllClose(tf_ans, truth, atol=1e-10)
      else:
        with self.assertRaisesOpError(expected_err_re):
          self.evaluate(ans)

  def _testWrongNumbers(self, fn=array_ops.where):
    with self.session(use_gpu=True):
      with self.assertRaises(ValueError):
        fn([False, True], [1, 2], None)
      with self.assertRaises(ValueError):
        fn([False, True], None, [1, 2])

  def _testBasicVec(self, fn=array_ops.where):
    x = np.asarray([True, False])
    truth = np.asarray([[0]], dtype=np.int64)
    self._testWhere(x, truth, None, fn)

    x = np.asarray([False, True, False])
    truth = np.asarray([[1]], dtype=np.int64)
    self._testWhere(x, truth, None, fn)

    x = np.asarray([False, False, True, False, True])
    truth = np.asarray([[2], [4]], dtype=np.int64)
    self._testWhere(x, truth, None, fn)

  def _testRandomVec(self, fn=array_ops.where):
    x = np.random.rand(1000000) > 0.5
    truth = np.vstack([np.where(x)[0].astype(np.int64)]).T
    self._testWhere(x, truth, None, fn)

  def _testBasicMat(self, fn=array_ops.where):
    x = np.asarray([[True, False], [True, False]])

    # Ensure RowMajor mode
    truth = np.asarray([[0, 0], [1, 0]], dtype=np.int64)

    self._testWhere(x, truth, None, fn)

  def _testBasic3Tensor(self, fn=array_ops.where):
    x = np.asarray([[[True, False], [True, False]],
                    [[False, True], [False, True]],
                    [[False, False], [False, True]]])

    # Ensure RowMajor mode
    truth = np.asarray(
        [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1], [2, 1, 1]], dtype=np.int64)

    self._testWhere(x, truth, None, fn)

  def _testRandom(self, dtype, expected_err_re=None, fn=array_ops.where):
    shape = [127, 33, 53]
    x = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    x = (np.random.randn(*shape) > 0).astype(dtype)
    truth = np.where(np.abs(x) > 0)  # Tuples of indices by axis.
    truth = np.vstack(truth).T  # Convert to [num_true, indices].
    self._testWhere(x, truth, expected_err_re, fn)

  def _testThreeArgument(self, fn=array_ops.where):
    x = np.array([[-2, 3, -1], [1, -3, -3]])
    np_val = np.where(x > 0, x * x, -x)
    with self.test_session(use_gpu=True):
      tf_val = fn(constant_op.constant(x) > 0, x * x, -x).eval()
    self.assertAllEqual(tf_val, np_val)

  def testWrongNumbers(self):
    self._testWrongNumbers()

  @test_util.run_deprecated_v1
  def testBasicVec(self):
    self._testBasicVec()

  @test_util.run_deprecated_v1
  def testRandomVec(self):
    self._testRandomVec()

  @test_util.run_deprecated_v1
  def testBasicMat(self):
    self._testBasicMat()

  @test_util.run_deprecated_v1
  def testBasic3Tensor(self):
    self._testBasic3Tensor()

  @test_util.run_deprecated_v1
  def testRandomBool(self):
    self._testRandom(np.bool)

  @test_util.run_deprecated_v1
  def testRandomInt32(self):
    self._testRandom(np.int32)

  @test_util.run_deprecated_v1
  def testRandomInt64(self):
    self._testRandom(np.int64)

  @test_util.run_deprecated_v1
  def testRandomFloat(self):
    self._testRandom(np.float32)

  @test_util.run_deprecated_v1
  def testRandomDouble(self):
    self._testRandom(np.float64)

  @test_util.run_deprecated_v1
  def testRandomComplex64(self):
    self._testRandom(np.complex64)

  @test_util.run_deprecated_v1
  def testRandomComplex128(self):
    self._testRandom(np.complex128)

  @test_util.run_deprecated_v1
  def testRandomUint8(self):
    self._testRandom(np.uint8)

  @test_util.run_deprecated_v1
  def testRandomInt8(self):
    self._testRandom(np.int8)

  @test_util.run_deprecated_v1
  def testRandomInt16(self):
    self._testRandom(np.int16)

  @test_util.run_deprecated_v1
  def testThreeArgument(self):
    self._testThreeArgument()

  def testV2WrongNumbers(self):
    self._testWrongNumbers(array_ops.where_v2)

  def testV2BasicVec(self):
    self._testBasicVec(array_ops.where_v2)

  def testV2RandomVec(self):
    self._testRandomVec(array_ops.where_v2)

  def testV2BasicMat(self):
    self._testBasicMat(array_ops.where_v2)

  def testV2Basic3Tensor(self):
    self._testBasic3Tensor(array_ops.where_v2)

  def testV2RandomBool(self):
    self._testRandom(np.bool, None, array_ops.where_v2)

  def testV2RandomInt32(self):
    self._testRandom(np.int32, None, array_ops.where_v2)

  def testV2RandomInt64(self):
    self._testRandom(np.int64, None, array_ops.where_v2)

  def testV2RandomFloat(self):
    self._testRandom(np.float32, None, array_ops.where_v2)

  def testV2RandomDouble(self):
    self._testRandom(np.float64, None, array_ops.where_v2)

  def testV2RandomComplex64(self):
    self._testRandom(np.complex64, None, array_ops.where_v2)

  def testV2RandomComplex128(self):
    self._testRandom(np.complex128, None, array_ops.where_v2)

  def testV2RandomUint8(self):
    self._testRandom(np.uint8, None, array_ops.where_v2)

  def testV2RandomInt8(self):
    self._testRandom(np.int8, None, array_ops.where_v2)

  def testV2RandomInt16(self):
    self._testRandom(np.int16, None, array_ops.where_v2)

  def testV2ThreeArgument(self):
    self._testThreeArgument(array_ops.where_v2)

  def testV2Broadcasting(self):
    f = np.random.normal(0, 1, (3, 5, 1, 1))
    x = np.zeros((7, 11))
    y = np.ones((7, 11))
    np_val = np.where(f < 0, x, y)
    with self.test_session(use_gpu=True):
      tf_val = self.evaluate(
          array_ops.where_v2(constant_op.constant(f) < 0, x, y))
    self.assertAllEqual(tf_val, np_val)

  def testV2ScalarBroadcasting(self):
    x = np.zeros((7, 11))
    y = np.ones((7, 11))
    np_val = np.where(True, x, y)
    with self.test_session(use_gpu=True):
      tf_val = self.evaluate(array_ops.where_v2(
          constant_op.constant(True, dtype=dtypes.bool), x, y))
    self.assertAllEqual(tf_val, np_val)

  def testV2VectorBroadcasting(self):
    x = np.zeros(7)
    y = np.ones(7)
    np_val = np.where([True], x, y)
    with self.test_session(use_gpu=True):
      tf_val = self.evaluate(array_ops.where_v2(
          constant_op.constant([True], dtype=dtypes.bool), x, y))
    self.assertAllEqual(tf_val, np_val)

  def testV2PredBroadcasting(self):
    pred = np.array([1, 0, 0]).reshape((3, 1))
    x = np.random.randn(3, 4)
    y = np.random.randn(3, 4)
    np_val = np.where(pred, x, y)
    with self.test_session(use_gpu=True):
      tf_val = self.evaluate(array_ops.where_v2(pred, x, y))
    self.assertAllClose(tf_val, np_val)

  @test_util.run_deprecated_v1
  def testBatchSelect(self):
    x = np.array([[-2, 3, -1] * 64, [1, -3, -3] * 64] * 8192)  # [16384, 192]
    c_mat = np.array([[False] * 192, [True] * 192] * 8192)  # [16384, 192]
    c_vec = np.array([False, True] * 8192)  # [16384]
    np_val = np.where(c_mat, x * x, -x)
    with self.session(use_gpu=True):
      tf_val = array_ops.where(c_vec, x * x, -x).eval()
    self.assertAllEqual(tf_val, np_val)


class WhereBenchmark(test.Benchmark):

  def benchmarkWhere(self):
    for (m, n, p, use_gpu) in itertools.product(
        [10],
        [10, 100, 1000, 10000, 100000, 1000000],
        [0.01, 0.5, 0.99],
        [False, True]):
      name = "m_%d_n_%d_p_%g_use_gpu_%s" % (m, n, p, use_gpu)
      device = "/%s:0" % ("gpu" if use_gpu else "cpu")
      with ops.Graph().as_default():
        with ops.device(device):
          x = random_ops.random_uniform((m, n), dtype=dtypes.float32) <= p
          v = resource_variable_ops.ResourceVariable(x)
          op = array_ops.where(v)
        with session.Session(config=benchmark.benchmark_config()) as sess:
          v.initializer.run()
          r = self.run_op_benchmark(sess, op, min_iters=100, name=name)
          gb_processed_input = m * n / 1.0e9
          # approximate size of output: m*n*p int64s for each axis.
          gb_processed_output = 2 * 8 * m * n * p / 1.0e9
          gb_processed = gb_processed_input + gb_processed_output
          throughput = gb_processed / r["wall_time"]
          print("Benchmark: %s \t wall_time: %0.03g s \t "
                "Throughput: %0.03g GB/s" % (name, r["wall_time"], throughput))
          sys.stdout.flush()

  def benchmarkBatchSelect(self):
    for (m, n, use_gpu) in itertools.product([1000, 10000, 100000],
                                             [10, 100, 1000], [False, True]):
      name = "m_%d_n_%d_use_gpu_%s" % (m, n, use_gpu)
      device = "/%s:0" % ("gpu" if use_gpu else "cpu")
      with ops.Graph().as_default():
        with ops.device(device):
          x_gen = random_ops.random_uniform([m, n], dtype=dtypes.float32)
          y_gen = random_ops.random_uniform([m, n], dtype=dtypes.float32)
          c_gen = random_ops.random_uniform([m], dtype=dtypes.float32) <= 0.5
          x = resource_variable_ops.ResourceVariable(x_gen)
          y = resource_variable_ops.ResourceVariable(y_gen)
          c = resource_variable_ops.ResourceVariable(c_gen)
          op = array_ops.where(c, x, y)
        with session.Session(config=benchmark.benchmark_config()) as sess:
          x.initializer.run()
          y.initializer.run()
          c.initializer.run()
          r = self.run_op_benchmark(sess, op, min_iters=100, name=name)
          # approximate size of output: m*n*2 floats for each axis.
          gb_processed = m * n * 8 / 1.0e9
          throughput = gb_processed / r["wall_time"]
          print("Benchmark: %s \t wall_time: %0.03g s \t "
                "Throughput: %0.03g GB/s" % (name, r["wall_time"], throughput))
          sys.stdout.flush()


if __name__ == "__main__":
  test.main()
