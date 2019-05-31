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

from absl.testing import parameterized
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


ALGS = [random.RNG_ALG_PHILOX, random.RNG_ALG_THREEFRY]
INTS = [dtypes.int32, dtypes.uint32, dtypes.int64, dtypes.uint64]


# TODO(wangpeng): use parametrized tests to test both ThreeFry and Philox
class StatefulRandomOpsTest(xla_test.XLATestCase, parameterized.TestCase):
  """Test cases for stateful random-number generator operators."""

  _ints = INTS
  _floats = [dtypes.bfloat16, dtypes.float32]

  @parameterized.parameters(ALGS)
  @test_util.run_v2_only
  def testSimple(self, alg):
    """A simple test."""
    with ops.device(xla_device_name()):
      gen = random.Generator.from_seed(seed=0, alg=alg)
      gen.normal(shape=(3,))
      gen.uniform(shape=(3,), minval=0, maxval=10, dtype=dtypes.uint32)
      gen.uniform_full_int(shape=(3,))

  @parameterized.parameters(ALGS)
  @test_util.run_v2_only
  def testDefun(self, alg):
    """Test for defun."""
    with ops.device(xla_device_name()):
      gen = random.Generator.from_seed(seed=0, alg=alg)
      @def_function.function
      def f():
        x = gen.normal(shape=(3,))
        y = gen.uniform(shape=(3,), minval=0, maxval=10, dtype=dtypes.uint32)
        z = gen.uniform_full_int(shape=(3,))
        return (x, y, z)
      f()

  def _compareToKnownOutputs(self, g, counter, key, expect):
    """Compares against known outputs for specific counter and key inputs."""
    def uint32s_to_uint64(a, b):
      return b << 32 | a

    def uint32s_to_uint64s(ls):
      return [uint32s_to_uint64(ls[2 * i], ls[2 * i + 1])
              for i in range(len(ls) // 2)]

    ctr_len = len(counter)
    counter = uint32s_to_uint64s(counter)
    key = uint32s_to_uint64s(key)
    state = counter + key
    g.reset(state)
    got = g.uniform_full_int(shape=(ctr_len,), dtype=dtypes.uint32)
    self.assertAllEqual(expect, got)
    g.reset(state)
    got = g.uniform_full_int(shape=(ctr_len // 2,), dtype=dtypes.uint64)
    self.assertAllEqual(uint32s_to_uint64s(expect), got)

  @test_util.run_v2_only
  def testThreefry2x32(self):
    """Tests ThreeFry2x32 conforms to known results.
    """
    # Based on
    # https://github.com/google/jax/blob/8565a3486adf16beb388b2364c9cd930d7a0d92d/tests/random_test.py#L65-L85
    # which is in turn based on
    # https://github.com/DEShawResearch/Random123-Boost/blob/65e3d874b67aa7b3e02d5ad8306462f52d2079c0/libs/random/test/test_threefry.cpp#L30-L32

    with ops.device(xla_device_name()):
      g = random.Generator.from_seed(seed=0, alg=random.RNG_ALG_THREEFRY)
      self._compareToKnownOutputs(
          g,
          [0x00000000, 0x00000000], [0x00000000, 0x00000000],
          [0x6b200159, 0x99ba4efe])
      self._compareToKnownOutputs(
          g,
          [0xffffffff, 0xffffffff], [0xffffffff, 0xffffffff],
          [0x1cb996fc, 0xbb002be7])
      self._compareToKnownOutputs(
          g,
          [0x243f6a88, 0x85a308d3], [0x13198a2e, 0x03707344],
          [0xc4923a9c, 0x483df7a0])

  @test_util.run_v2_only
  def testPhilox4x32(self):
    """Tests Philox4x32 conforms to known results.
    """
    # Based on
    # https://github.com/DEShawResearch/Random123-Boost/blob/65e3d874b67aa7b3e02d5ad8306462f52d2079c0/libs/random/test/test_philox.cpp#L50-L52

    with ops.device(xla_device_name()):
      g = random.Generator.from_seed(seed=0, alg=random.RNG_ALG_PHILOX)
      self._compareToKnownOutputs(
          g,
          [0x00000000, 0x00000000, 0x00000000, 0x00000000],
          [0x00000000, 0x00000000],
          [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8])
      self._compareToKnownOutputs(
          g,
          [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff],
          [0xffffffff, 0xffffffff],
          [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd])
      self._compareToKnownOutputs(
          g,
          [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
          [0xa4093822, 0x299f31d0],
          [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1])

  @test_util.run_v2_only
  def testNewStateThreeFry(self):
    """Tests that the new state is correct (for ThreeFry).
    """
    with ops.device(xla_device_name()):
      counter = 57
      key = 0x1234
      size = 46
      state = [counter, key]
      gen = random.Generator(state=state, alg=random.RNG_ALG_THREEFRY)
      gen.uniform_full_int(shape=(size,), dtype=dtypes.uint32)
      self.assertAllEqual([counter+(size+1)//2, key], gen.state.read_value())
      gen.reset(state)
      gen.uniform_full_int(shape=(size,), dtype=dtypes.uint64)
      self.assertAllEqual([counter+size, key], gen.state.read_value())

  @test_util.run_v2_only
  def testNewStatePhilox(self):
    """Tests that the new state is correct (for Philox).
    """
    with ops.device(xla_device_name()):
      counter_low = 57
      counter_high = 283
      key = 0x1234
      size = 47
      state = [counter_low, counter_high, key]
      gen = random.Generator(state=state, alg=random.RNG_ALG_PHILOX)
      gen.uniform_full_int(shape=(size,), dtype=dtypes.uint32)
      self.assertAllEqual([counter_low+(size+3)//4, counter_high, key],
                          gen.state.read_value())
      gen.reset(state)
      gen.uniform_full_int(shape=(size,), dtype=dtypes.uint64)
      self.assertAllEqual([counter_low+(size+1)//2, counter_high, key],
                          gen.state.read_value())
      # Tests that large counter_low will correctly overflows to counter_high
      counter_low = -1  # same as 0xffffffffffffffff
      counter_high = 283
      size = 47
      state = [counter_low, counter_high, key]
      gen = random.Generator(state=state, alg=random.RNG_ALG_PHILOX)
      gen.uniform_full_int(shape=(size,), dtype=dtypes.uint32)
      self.assertAllEqual([(size+3)//4-1, counter_high+1, key],
                          gen.state.read_value())
      gen.reset(state)
      gen.uniform_full_int(shape=(size,), dtype=dtypes.uint64)
      self.assertAllEqual([(size+1)//2-1, counter_high+1, key],
                          gen.state.read_value())

  @parameterized.parameters(INTS)
  @test_util.run_v2_only
  def testXLAEqualsCPU(self, dtype):
    """Tests that XLA and CPU kernels generate the same integers."""
    seed = 1234
    shape = [315, 49]
    with ops.device("/device:CPU:0"):
      cpu = (random.Generator.from_seed(seed=seed, alg=random.RNG_ALG_PHILOX)
             .uniform_full_int(shape=shape, dtype=dtype))
    with ops.device(xla_device_name()):
      xla = (random.Generator.from_seed(seed=seed, alg=random.RNG_ALG_PHILOX)
             .uniform_full_int(shape=shape, dtype=dtype))
    self.assertAllEqual(cpu, xla)

  def _testRngIsNotConstant(self, rng, dtype):
    # Tests that 'rng' does not always return the same value.
    # The random-number generator, if working correctly, should produce the
    # same output multiple times with low probability.
    x = rng(dtype).numpy()
    y = rng(dtype).numpy()
    self.assertFalse(np.array_equal(x, y))

  @parameterized.parameters(ALGS)
  @test_util.run_v2_only
  def testUniformIsNotConstant(self, alg):
    with ops.device(xla_device_name()):
      gen = random.Generator.from_seed(seed=1234, alg=alg)
      def rng(dtype):
        maxval = dtype.max
        # Workaround for b/125364959
        if dtype == dtypes.uint64:
          maxval = 10000000
        return gen.uniform(shape=[2], dtype=dtype, maxval=maxval)

      for dtype in self._ints + self._floats:
        self._testRngIsNotConstant(rng, dtype)

  @parameterized.parameters(ALGS)
  @test_util.run_v2_only
  def testNormalIsNotConstant(self, alg):
    with ops.device(xla_device_name()):
      gen = random.Generator.from_seed(seed=1234, alg=alg)
      def rng(dtype):
        return gen.normal(shape=[2], dtype=dtype)

      for dtype in self._floats:
        self._testRngIsNotConstant(rng, dtype)

  @parameterized.parameters(ALGS)
  @test_util.run_v2_only
  def testUniformIsInRange(self, alg):
    minval = 2
    maxval = 33
    size = 1000
    with ops.device(xla_device_name()):
      for dtype in self._ints + self._floats:
        gen = random.Generator.from_seed(seed=1234, alg=alg)
        x = gen.uniform(
            shape=[size], dtype=dtype, minval=minval, maxval=maxval).numpy()
        self.assertTrue(np.all(x >= minval))
        self.assertTrue(np.all(x <= maxval))

  @parameterized.parameters(ALGS)
  @test_util.run_v2_only
  def testNormalIsFinite(self, alg):
    with ops.device(xla_device_name()):
      gen = random.Generator.from_seed(seed=1234, alg=alg)
      for dtype in self._floats:
        x = gen.normal(shape=[10000], dtype=dtype).numpy()
        self.assertTrue(np.all(np.isfinite(x)))

  @parameterized.parameters(ALGS)
  @test_util.run_v2_only
  def testDistributionOfUniform(self, alg):
    """Use Pearson's Chi-squared test to test for uniformity."""
    with ops.device(xla_device_name()):
      n = 1000
      seed = 12
      for dtype in self._ints + self._floats:
        gen = random.Generator.from_seed(seed=seed, alg=alg)
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

  @parameterized.parameters(ALGS)
  @test_util.run_v2_only
  def testDistributionOfNormal(self, alg):
    """Use Anderson-Darling test to test distribution appears normal."""
    with ops.device(xla_device_name()):
      n = 1000
      for dtype in self._floats:
        gen = random.Generator.from_seed(seed=1234, alg=alg)
        x = gen.normal(shape=[n], dtype=dtype).numpy()
        # The constant 2.492 is the 5% critical value for the Anderson-Darling
        # test where the mean and variance are known. This test is probabilistic
        # so to avoid flakiness the seed is fixed.
        self.assertLess(
            random_test_util.anderson_darling(x.astype(float)), 2.492)

  @parameterized.parameters(ALGS)
  @test_util.run_v2_only
  def testTruncatedNormal(self, alg):
    with ops.device(xla_device_name()):
      for dtype in self._floats:
        gen = random.Generator.from_seed(seed=123, alg=alg)
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
      gen = random.Generator.from_seed(seed=1234, alg=random.RNG_ALG_THREEFRY)
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
          "The size of the state must be at least"):
        gen_stateful_random_ops.stateful_standard_normal_v2(
            var.handle, random.RNG_ALG_THREEFRY, shape)
      var = variables.Variable([0, 0], dtype=dtypes.int64)
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          "The size of the state must be at least"):
        gen_stateful_random_ops.stateful_standard_normal_v2(
            var.handle, random.RNG_ALG_PHILOX, shape)


if __name__ == "__main__":
  test.main()
