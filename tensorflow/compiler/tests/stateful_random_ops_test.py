# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for stateful random-number generation ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.client import device_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.random import util as \
random_test_util
from tensorflow.python.ops import gen_stateful_random_ops
from tensorflow.python.ops import stateful_random_ops as \
random
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def xla_device_name():
  devices = device_lib.list_local_devices()
  def find_type(device_type):
    for d in devices:
      if d.device_type == device_type:
        return d.name
    return None
  name = find_type("TPU") or find_type("XLA_GPU") or find_type("XLA_CPU")
  if name is None:
    raise ValueError(
        "Can't find any XLA device. Available devices:\n%s" % devices)
  return str(name)


class StatefulRandomOpsTest(xla_test.XLATestCase):
  """Test cases for stateful random-number generator operators."""

  _ints = [dtypes.int32, dtypes.uint32, dtypes.int64, dtypes.uint64]
  _floats = [dtypes.bfloat16, dtypes.float32]

  @test_util.run_v2_only
  def testSimple(self):
    """A simple test.
    """
    with ops.device(xla_device_name()):
      gen = random.Generator(seed=0, algorithm=random.RNG_ALG_THREEFRY)
      gen.normal(shape=(3,))
      gen.uniform(shape=(3,), minval=0, maxval=10, dtype=dtypes.uint32)
      gen.uniform_full_int(shape=(3,))

  @test_util.run_v2_only
  def testDefun(self):
    """Test for defun.
    """
    with ops.device(xla_device_name()):
      gen = random.Generator(seed=0, algorithm=random.RNG_ALG_THREEFRY)
      @def_function.function
      def f():
        x = gen.normal(shape=(3,))
        y = gen.uniform(shape=(3,), minval=0, maxval=10, dtype=dtypes.uint32)
        z = gen.uniform_full_int(shape=(3,))
        return (x, y, z)
      f()

  @test_util.run_v2_only
  def testThreefry2x32(self):
    """Tests ThreeFry2x32 conforms to known results.
    """
    # Based on
    # https://github.com/google/jax/blob/8565a3486adf16beb388b2364c9cd930d7a0d92d/tests/random_test.py#L65-L85
    # which is in turn based on
    # https://github.com/DEShawResearch/Random123-Boost/blob/65e3d874b67aa7b3e02d5ad8306462f52d2079c0/libs/random/test/test_threefry.cpp#L30-L32

    def uint32s_to_uint64(a, b):
      return b << 32 | a

    def verify(counter1, counter2, key1, key2, expect1, expect2):
      counter = uint32s_to_uint64(counter1, counter2)
      key = uint32s_to_uint64(key1, key2)
      random.get_global_generator().reset([counter, key])
      got = random.get_global_generator().uniform_full_int(
          shape=(2,), dtype=dtypes.uint32)
      expect = [expect1, expect2]
      self.assertAllEqual(expect, got)
      random.get_global_generator().reset([counter, key])
      got = random.get_global_generator().uniform_full_int(
          shape=(), dtype=dtypes.uint64)
      self.assertAllEqual(uint32s_to_uint64(*expect), got)

    with ops.device(xla_device_name()):
      random.reset_global_generator(seed=0, algorithm=random.RNG_ALG_THREEFRY)
      verify(0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x6b200159, 0x99ba4efe)
      verify(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
             0x1cb996fc, 0xbb002be7)
      verify(0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344,
             0xc4923a9c, 0x483df7a0)

  @test_util.run_v2_only
  def testNewState(self):
    """Tests that the new state is correct.
    """
    with ops.device(xla_device_name()):
      counter = 57
      key = 0x1234
      size = 46
      seed = [counter, key]
      gen = random.Generator(
          seed=seed, algorithm=random.RNG_ALG_THREEFRY)
      gen.uniform_full_int(shape=(size,), dtype=dtypes.uint32)
      self.assertAllEqual([counter+(size+1)//2, key], gen.state.read_value())
      gen.reset(seed=seed)
      gen.uniform_full_int(shape=(size,), dtype=dtypes.uint64)
      self.assertAllEqual([counter+size, key], gen.state.read_value())

  def _testRngIsNotConstant(self, rng, dtype):
    # Tests that 'rng' does not always return the same value.
    # The random-number generator, if working correctly, should produce the
    # same output multiple times with low probability.
    x = rng(dtype).numpy()
    y = rng(dtype).numpy()
    self.assertFalse(np.array_equal(x, y))

  @test_util.run_v2_only
  def testUniformIsNotConstant(self):
    with ops.device(xla_device_name()):
      gen = random.Generator(seed=1234, algorithm=random.RNG_ALG_THREEFRY)
      def rng(dtype):
        maxval = dtype.max
        # Workaround for b/125364959
        if dtype == dtypes.uint64:
          maxval = 10000000
        return gen.uniform(shape=[2], dtype=dtype, maxval=maxval)

      for dtype in self._ints + self._floats:
        self._testRngIsNotConstant(rng, dtype)

  @test_util.run_v2_only
  def testNormalIsNotConstant(self):
    with ops.device(xla_device_name()):
      gen = random.Generator(seed=1234, algorithm=random.RNG_ALG_THREEFRY)
      def rng(dtype):
        return gen.normal(shape=[2], dtype=dtype)

      for dtype in self._floats:
        self._testRngIsNotConstant(rng, dtype)

  @test_util.run_v2_only
  def testUniformIsInRange(self):
    minval = 2
    maxval = 33
    size = 1000
    with ops.device(xla_device_name()):
      for dtype in self._ints + self._floats:
        gen = random.Generator(seed=1234, algorithm=random.RNG_ALG_THREEFRY)
        x = gen.uniform(
            shape=[size], dtype=dtype, minval=minval, maxval=maxval).numpy()
        self.assertTrue(np.all(x >= minval))
        self.assertTrue(np.all(x <= maxval))

  @test_util.run_v2_only
  def testNormalIsFinite(self):
    with ops.device(xla_device_name()):
      gen = random.Generator(seed=1234, algorithm=random.RNG_ALG_THREEFRY)
      for dtype in self._floats:
        x = gen.normal(shape=[10000], dtype=dtype).numpy()
        self.assertTrue(np.all(np.isfinite(x)))

  @test_util.run_v2_only
  def testDistributionOfUniform(self):
    """Use Pearson's Chi-squared test to test for uniformity."""
    with ops.device(xla_device_name()):
      n = 1000
      seed = 12
      for dtype in self._ints + self._floats:
        gen = random.Generator(seed=seed, algorithm=random.RNG_ALG_THREEFRY)
        maxval = 1
        if dtype.is_integer:
          maxval = 100
        x = gen.uniform(shape=[n], maxval=maxval, dtype=dtype).numpy()
        if maxval > 1:
          # Normalize y to range [0, 1).
          x = x.astype(float) / maxval
        # Tests that the values are distributed amongst 10 bins with equal
        # probability. 16.92 is the Chi^2 value for 9 degrees of freedom with
        # p=0.05. This test is probabilistic and would be flaky if the random
        # seed were not fixed.
        val = random_test_util.chi_squared(x, 10)
        self.assertLess(val, 16.92)

  @test_util.run_v2_only
  def testDistributionOfNormal(self):
    """Use Anderson-Darling test to test distribution appears normal."""
    with ops.device(xla_device_name()):
      n = 1000
      for dtype in self._floats:
        gen = random.Generator(seed=1234, algorithm=random.RNG_ALG_THREEFRY)
        x = gen.normal(shape=[n], dtype=dtype).numpy()
        # The constant 2.492 is the 5% critical value for the Anderson-Darling
        # test where the mean and variance are known. This test is probabilistic
        # so to avoid flakiness the seed is fixed.
        self.assertLess(
            random_test_util.anderson_darling(x.astype(float)), 2.492)

  @test_util.run_v2_only
  def testTruncatedNormal(self):
    for dtype in self._floats:
      gen = random.Generator(seed=123)
      n = 10000000
      y = gen.truncated_normal(shape=[n], dtype=dtype).numpy()
      random_test_util.test_truncated_normal(
          self.assertEqual, self.assertAllClose, dtype, n, y)

  @test_util.run_v2_only
  def testErrors(self):
    """Tests that proper errors are raised.
    """
    shape = [2, 3]
    with ops.device(xla_device_name()):
      gen = random.Generator(seed=1234, algorithm=random.RNG_ALG_THREEFRY)
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          r"algorithm must be of shape \[\], not"):
        gen_stateful_random_ops.stateful_standard_normal_v2(
            gen.state.handle, [0, 0], shape)
      with self.assertRaisesWithPredicateMatch(
          TypeError, "Requested dtype: int64"):
        gen_stateful_random_ops.stateful_standard_normal_v2(
            gen.state.handle, 1.1, shape)
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          "Unsupported algorithm id"):
        gen_stateful_random_ops.stateful_standard_normal_v2(
            gen.state.handle, 123, shape)
      var = variables.Variable([0, 0], dtype=dtypes.uint32)
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          "Type mismatch for read of variable .* Expected int64; got"):
        gen_stateful_random_ops.stateful_standard_normal_v2(
            var.handle, random.RNG_ALG_THREEFRY, shape)
      var = variables.Variable([[0]], dtype=dtypes.int64)
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          "RNG state must have one and only one dimension, not"):
        gen_stateful_random_ops.stateful_standard_normal_v2(
            var.handle, random.RNG_ALG_THREEFRY, shape)
      var = variables.Variable([0], dtype=dtypes.int64)
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          "For the ThreeFry algorithm, the size of state must be at least"):
        gen_stateful_random_ops.stateful_standard_normal_v2(
            var.handle, random.RNG_ALG_THREEFRY, shape)


if __name__ == "__main__":
  test.main()
