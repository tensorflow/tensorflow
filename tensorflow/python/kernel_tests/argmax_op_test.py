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
"""Tests for tensorflow.ops.argmax_op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


class ArgMaxTest(test.TestCase):

  def _testArg(self,
               method,
               x,
               axis,
               expected_values,
               use_gpu=False,
               expected_err_re=None):
    with self.session(use_gpu=use_gpu):
      ans = method(x, axis=axis)
      if expected_err_re is None:
        tf_ans = self.evaluate(ans)
        # Defaults to int64 output.
        self.assertEqual(np.int64, tf_ans.dtype)
        self.assertAllEqual(tf_ans, expected_values)
        self.assertShapeEqual(expected_values, ans)
      else:
        with self.assertRaisesOpError(expected_err_re):
          self.evaluate(ans)

  def _testBothArg(self,
                   method,
                   x,
                   axis,
                   expected_values,
                   expected_err_re=None):
    self._testArg(method, x, axis, expected_values, True, expected_err_re)
    # Compilation time is too large with XLA/CPU autojit.
    if not test_util.is_xla_enabled():
      self._testArg(method, x, axis, expected_values, False, expected_err_re)

  def _testBasic(self, dtype):
    x = np.arange(200, dtype=dtype)
    np.random.shuffle(x)

    # Check that argmin and argmax match numpy along the primary axis
    self._testBothArg(math_ops.argmax, x, 0, x.argmax())
    self._testBothArg(math_ops.argmin, x, 0, x.argmin())

  def _testDim(self, dtype):
    shape = (3, 2, 4, 1, 5, 3, 2)
    x = np.arange(functools.reduce(lambda x, y: x * y, shape), dtype=dtype)
    np.random.shuffle(x)
    x = x.reshape(shape)

    # Check that argmin and argmax match numpy along all axes
    for axis in range(-7, 7):
      self._testBothArg(math_ops.argmax, x, axis, x.argmax(axis))
      self._testBothArg(math_ops.argmin, x, axis, x.argmin(axis))

  def _testLargeOutput(self, dtype):
    # Test case where output size is greater than 1024, which uses a different
    # codepath on the GPU.
    x = np.asarray(100 * np.random.randn(11, 10, 5, 11), dtype=dtype)
    self._testBothArg(math_ops.argmax, x, 2, x.argmax(2))
    self._testBothArg(math_ops.argmin, x, 2, x.argmin(2))

  def testFloat(self):
    self._testBasic(np.float32)
    self._testDim(np.float32)
    self._testLargeOutput(np.float32)

  def testFloatInt32Output(self):
    x = np.asarray(100 * np.random.randn(200), dtype=np.float32)
    expected_values = x.argmax()
    with self.session(use_gpu=True):
      ans = math_ops.argmax(x, axis=0, output_type=dtypes.int32)
      tf_ans = self.evaluate(ans)
      self.assertEqual(np.int32, tf_ans.dtype)
      # The values are equal when comparing int32 to int64 because
      # the values don't have a range that exceeds 32-bit integers.
      self.assertAllEqual(tf_ans, expected_values)
    expected_values = x.argmin()
    with self.session(use_gpu=True):
      ans = math_ops.argmin(x, axis=0, output_type=dtypes.int32)
      tf_ans = self.evaluate(ans)
      self.assertEqual(np.int32, tf_ans.dtype)
      self.assertAllEqual(tf_ans, expected_values)

  def testDouble(self):
    self._testBasic(np.float64)
    self._testDim(np.float64)
    self._testLargeOutput(np.float64)

  def testHalf(self):
    self._testBasic(np.float16)
    self._testDim(np.float16)
    self._testLargeOutput(np.float16)

  def testInt32(self):
    self._testBasic(np.int32)
    self._testDim(np.int32)

  def testInt64(self):
    self._testBasic(np.int64)
    self._testDim(np.int64)

  def testEmpty(self):
    with self.cached_session():
      for op in math_ops.argmin, math_ops.argmax:
        with self.assertRaisesOpError(
            r"Reduction axis 0 is empty in shape \[0\]"):
          op([], 0).eval()

  @test_util.run_deprecated_v1
  def testDefaultAxis(self):
    with self.cached_session():
      for op in math_ops.argmin, math_ops.argmax:
        ans = op([1]).eval()
        self.assertAllEqual(ans, 0)

  @test_util.run_deprecated_v1
  def testOutputEmpty(self):
    with self.cached_session():
      for op in math_ops.argmin, math_ops.argmax:
        ret = op(array_ops.zeros(shape=[1, 0, 2]), axis=-1).eval()
        self.assertEqual(ret.shape, (1, 0))


class ArgMaxBenchmark(test.Benchmark):

  def _RunSingleBenchmark(self, shape, dtype, bench_name):
    with session.Session(config=benchmark.benchmark_config()) as sess:
      num_dims = len(shape)
      var = variables.Variable(random_ops.random_uniform(shape, dtype=dtype))
      variables.variables_initializer([var]).run()
      for dim in range(num_dims):
        num_ops_in_group = 15
        op = control_flow_ops.group(*(math_ops.argmax(var, dimension=dim)
                                      for _ in range(num_ops_in_group)))
        op_name = "%s_%s_dim%d" % (bench_name, dtype.name, dim)
        num_bytes = num_ops_in_group * np.prod(shape) * dtype.size
        self.run_op_benchmark(sess, op, burn_iters=5, min_iters=20,
                              name=op_name, mbs=num_bytes / 1e6)

  def _runBenchmarksWithDtype(self, dtype):
    self._RunSingleBenchmark((2**17,), dtype, "1d")
    self._RunSingleBenchmark((2**13, 2**13), dtype, "square_2d")
    self._RunSingleBenchmark((2**5, 2**16), dtype, "rectangle1_2d")
    self._RunSingleBenchmark((2**16, 2**5), dtype, "rectangle2_2d")
    self._RunSingleBenchmark((2**8, 2**8, 2**8), dtype, "cube_3d")
    self._RunSingleBenchmark((2**16, 2**5, 2**5), dtype, "rectangle1_3d")
    self._RunSingleBenchmark((2**5, 2**16, 2**5), dtype, "rectangle2_3d")
    self._RunSingleBenchmark((2**5, 2**5, 2**16), dtype, "rectangle3_3d")

  def benchmarkFloat(self):
    self._runBenchmarksWithDtype(dtypes.float32)

  def benchmarkDouble(self):
    self._runBenchmarksWithDtype(dtypes.float64)

  def benchmarkHalf(self):
    self._runBenchmarksWithDtype(dtypes.float16)


if __name__ == "__main__":
  test.main()
