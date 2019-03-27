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
"""Tests for stateful_random_ops.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.random import util as \
random_test_util
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import gen_stateful_random_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import stateful_random_ops as \
random
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


g_seeded = None
g_unseeded = None


class StatefulRandomOpsTest(test.TestCase):

  def testCreateRNGStateIntSeed(self):
    """Tests `create_rng_state` when `seed` is int."""
    # using leading 'F' to test overflow tolerance
    state = random.create_rng_state(0xFFFF222233334444FFAA666677778888,
                                    random.RNG_ALG_PHILOX)
    self.assertAllEqual(
        list(map(random._uint_to_int,
                 [0xFFAA666677778888, 0xFFFF222233334444] +
                 [0] * (random.PHILOX_STATE_SIZE - 2))),
        state)

  @test_util.run_v2_only
  def testNonDeterministicInts(self):
    """Tests that non_deterministic_ints returns different results every time.

    This test is flaky, but with very low probability of failing.
    """
    shape = [2, 3]
    dtype = dtypes.uint64
    a = random.non_deterministic_ints(shape=shape, dtype=dtype)
    self.assertAllEqual(shape, a.shape)
    self.assertEqual(dtype, a.dtype)
    b = random.non_deterministic_ints(shape, dtype=dtype)
    self.assertNotAllClose(a, b)

  @test_util.run_v2_only
  def testGeneratorCreationInDefun(self):
    """Tests creating a Generator in defun.

    The interaction between Generator creation and defun should be the same as
    tf.Variable.
    """
    seed = 1234
    shape = [2, 3]
    with ops.device("/device:CPU:0"):
      gen = random.Generator(seed=seed)
      expected_normal1 = gen.normal(shape)
      expected_normal2 = gen.normal(shape)
      @def_function.function
      def f():
        global g_seeded
        global g_unseeded
        # defun'ed function should only create variables once
        if g_seeded is None:
          g_seeded = random.Generator(seed=seed)
        if g_unseeded is None:
          g_unseeded = random.Generator()
        r = g_seeded.normal(shape)
        r = (r, g_unseeded.normal(shape))
        return r
      def check_results(expected_normal, v1, v2):
        self.assertAllEqual(expected_normal, v1)
        self.assertAllEqual(shape, v2.shape)
      check_results(expected_normal1, *f())
      check_results(expected_normal2, *f())

  @test_util.run_v1_only
  def testTF1(self):
    seed = 1234
    shape = [2, 3]
    expected_normal1 = constant_op.constant(
        [[0.9356609, 1.0854305, -0.93788373],
         [-0.50615472, 1.31697023, 0.71375787]], dtype=dtypes.float32)
    expected_normal2 = constant_op.constant(
        [[-0.3964749, 0.8369565, -0.30946946],
         [1.1206646, 1.00852597, -0.10185789]], dtype=dtypes.float32)
    with self.cached_session() as sess:
      gen1 = random.Generator(seed=seed)
      gen2 = random.Generator()
      sess.run((gen1._state_var.initializer, gen2._state_var.initializer))
      r1 = gen1.normal(shape)
      r2 = gen2.normal(shape)
      def f():
        return sess.run((r1, r2))
      def check_results(expected_normal, v1, v2):
        self.assertAllEqual(expected_normal, v1)
        self.assertAllEqual(shape, v2.shape)
      check_results(expected_normal1, *f())
      check_results(expected_normal2, *f())

  @test_util.run_v2_only
  @test_util.also_run_as_tf_function
  def testEagerAndDefun(self):
    """A simple test to make sure the op works in eager and defunned mode."""
    random.get_global_generator().normal((3,))

  @test_util.run_v2_only
  def testOpSeedSelectionAfterSetSeed(self):
    """Tests that op-seed selection is reset after reseting global generator.

    Fixing GitHub issue 9171:
    https://github.com/tensorflow/tensorflow/issues/9171
    """
    shape = (3,)
    random.get_global_generator().reset(1)
    a = random.get_global_generator().normal(shape)
    random.get_global_generator().reset(1)
    b = random.get_global_generator().normal(shape)
    self.assertAllEqual(a, b)

    # Now do the above again using accelerated ('defun'ed) computation
    @def_function.function
    def f():
      return random.get_global_generator().normal(shape)

    random.get_global_generator().reset(1)
    c = f()
    random.get_global_generator().reset(1)
    d = f()
    self.assertAllEqual(c, d)
    self.assertAllEqual(a, c)

  @test_util.run_v2_only
  def testOpSeedSelectionNotSensitive(self):
    """Test that op-seed selection is not sensitive to trivial changes.

    Test that op-seed selection is not sensitive to trivial computation
    (i.e. graph) changes.

    Fixing b/32087099
    """
    def f(include_print):
      shape = constant_op.constant([5])
      if include_print:
        shape = logging_ops.Print(shape, [shape])
      return random.get_global_generator().normal(shape)

    def compare(fst_includes_print, snd_includes_print):
      random.get_global_generator().reset(50)
      fst = f(fst_includes_print)
      random.get_global_generator().reset(50)
      snd = f(snd_includes_print)
      self.assertAllEqual(fst, snd)
      # Now do the above again using accelerated (defunned) 'f'.
      # Running 'f' with two different Boolean arguments should cause
      # two different graphs to be generated, hence demonstrating the
      # insensitivity to graph changes.
      f_acc = def_function.function(f)
      random.get_global_generator().reset(50)
      fst = f_acc(fst_includes_print)
      random.get_global_generator().reset(50)
      snd = f_acc(snd_includes_print)
      self.assertAllEqual(fst, snd)

    compare(False, False)
    compare(True, True)
    compare(True, False)

  def _sameAsOldRandomOps(self, device):
    def compare(dtype, old, new):
      seed1, seed2 = 79, 25
      # note how the two seeds for the old op correspond to the seed for the new
      # op
      with ops.device(device):
        gen = random.Generator(seed=[0, seed2, seed1])

      # create a graph for the old op in order to call it many times
      @def_function.function
      def run_old():
        with ops.device(device):
          return old(dtype, seed1, seed2)

      def run_new():
        with ops.device(device):
          return new(dtype, gen)

      for _ in range(100):
        self.assertAllEqual(run_old(), run_new())

    shape = constant_op.constant([4, 7])
    minval = 128
    maxval = 256

    # passing `dtype` around to compress go/gpylint-faq#cell-var-from-loop and
    # go/gpylint-faq#undefined-loop-variable
    def old_normal(dtype, seed1, seed2):
      return gen_random_ops.random_standard_normal(
          shape, dtype=dtype, seed=seed1, seed2=seed2)
    def new_normal(dtype, gen):
      return gen._standard_normal(shape, dtype=dtype)
    def old_uniform(dtype, seed1, seed2):
      minval2 = constant_op.constant(minval, dtype=dtype)
      maxval2 = constant_op.constant(maxval, dtype=dtype)
      return gen_random_ops.random_uniform_int(
          shape, minval=minval2, maxval=maxval2, seed=seed1, seed2=seed2)
    def new_uniform(dtype, gen):
      return gen.uniform(
          shape, minval=minval, maxval=maxval, dtype=dtype)

    for dtype in (dtypes.float16, dtypes.bfloat16, dtypes.float32,
                  dtypes.float64):
      compare(dtype, old_normal, new_normal)
    for dtype in [dtypes.int32, dtypes.int64]:
      compare(dtype, old_uniform, new_uniform)

  @test_util.run_v2_only
  def testCPUSameAsOldRandomOps(self):
    """Tests that the generated numbers are the same as the old random_ops.py.

    The CPU version.
    """
    self._sameAsOldRandomOps("/device:CPU:0")

  @test_util.run_v2_only
  @test_util.run_cuda_only
  def testGPUSameAsOldRandomOps(self):
    """Tests that the generated numbers are the same as the old random_ops.py.

    The GPU version.
    """
    self._sameAsOldRandomOps(test_util.gpu_device_name())

  @test_util.run_v2_only
  def testUniformIntIsInRange(self):
    minval = 2
    maxval = 33
    size = 1000
    gen = random.Generator(seed=1234)
    for dtype in [dtypes.int32, dtypes.int64]:
      x = gen.uniform(
          shape=[size], dtype=dtype, minval=minval, maxval=maxval).numpy()
      self.assertTrue(np.all(x >= minval))
      self.assertTrue(np.all(x < maxval))

  @test_util.run_v2_only
  def testNormalIsFinite(self):
    gen = random.Generator(seed=1234)
    for dtype in [dtypes.float32]:
      x = gen.normal(shape=[10000], dtype=dtype).numpy()
      self.assertTrue(np.all(np.isfinite(x)))

  @test_util.run_v2_only
  def testDistributionOfUniform(self):
    """Use Pearson's Chi-squared test to test for uniformity."""
    n = 1000
    seed = 12
    for dtype in [dtypes.int32, dtypes.int64]:
      gen = random.Generator(seed=seed)
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
    n = 1000
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      gen = random.Generator(seed=1234)
      x = gen.normal(shape=[n], dtype=dtype).numpy()
      # The constant 2.492 is the 5% critical value for the Anderson-Darling
      # test where the mean and variance are known. This test is probabilistic
      # so to avoid flakiness the seed is fixed.
      self.assertLess(
          random_test_util.anderson_darling(x.astype(float)), 2.492)

  @test_util.run_v2_only
  def testErrors(self):
    """Tests that proper errors are raised.
    """
    shape = [2, 3]
    gen = random.Generator(seed=1234)
    with self.assertRaisesWithPredicateMatch(
        errors.InvalidArgumentError,
        r"algorithm must be of shape \[\], not"):
      gen_stateful_random_ops.stateful_standard_normal_v2(
          gen.state.handle, [0, 0], shape)
    with self.assertRaisesWithPredicateMatch(
        TypeError, "Requested dtype: int64"):
      gen_stateful_random_ops.stateful_standard_normal_v2(
          gen.state.handle, 1.1, shape)
    with self.assertRaisesWithPredicateMatch(
        errors.InvalidArgumentError,
        "Unsupported algorithm id"):
      gen_stateful_random_ops.stateful_standard_normal_v2(
          gen.state.handle, 123, shape)
    var = variables.Variable([0, 0], dtype=dtypes.int32)
    with self.assertRaisesWithPredicateMatch(
        errors.InvalidArgumentError,
        "dtype of RNG state variable must be int64, not"):
      gen_stateful_random_ops.stateful_standard_normal_v2(
          var.handle, random.RNG_ALG_PHILOX, shape)
    var = variables.Variable([[0]], dtype=dtypes.int64)
    with self.assertRaisesWithPredicateMatch(
        errors.InvalidArgumentError,
        "RNG state must have one and only one dimension, not"):
      gen_stateful_random_ops.stateful_standard_normal_v2(
          var.handle, random.RNG_ALG_PHILOX, shape)
    var = variables.Variable([0], dtype=dtypes.int64)
    with self.assertRaisesWithPredicateMatch(
        errors.InvalidArgumentError,
        "For the Philox algorithm, the size of state must be at least"):
      gen_stateful_random_ops.stateful_standard_normal_v2(
          var.handle, random.RNG_ALG_PHILOX, shape)

  @test_util.run_v2_only
  def testStatefulStandardNormal(self):
    """Tests that the deprecated op 'StatefulStandardNormal' still works.
    """
    shape = constant_op.constant([4, 7])
    dtype = dtypes.float64
    seed = 1234
    algorithm = random.RNG_ALG_PHILOX
    state = random._make_state_from_seed(seed, algorithm)
    with ops.device("/device:CPU:0"):
      var1 = variables.Variable(
          np.concatenate((np.array([algorithm], dtype=random.STATE_TYPE),
                          state), axis=None),
          dtype=random.STATE_TYPE)
      var2 = variables.Variable(state, dtype=random.STATE_TYPE)
      for _ in range(100):
        t1 = gen_stateful_random_ops.stateful_standard_normal(
            var1.handle, shape, dtype)
        t2 = gen_stateful_random_ops.stateful_standard_normal_v2(
            var2.handle, algorithm, shape, dtype)
        self.assertAllEqual(t1, t2)

  @test_util.run_v2_only
  def testResetGlobalGeneratorBadWithDefun(self):
    """Demonstrates that reset_global_generator don't work properly with defun.
    """
    shape = (3,)

    @def_function.function
    def f():
      return random.get_global_generator().normal(shape)

    random.reset_global_generator(50)
    with self.assertRaisesWithPredicateMatch(
        errors.NotFoundError, "Resource .+ does not exist"):
      a = f()
      random.reset_global_generator(50)
      b = f()
      self.assertAllEqual(a, b)


if __name__ == "__main__":
  test.main()
