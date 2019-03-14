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
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
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

  @test_util.run_v2_only
  def testCPUSameAsOldRandomOps(self):
    """Tests that the generated numbers are the same as the old random_ops.py.

    The CPU version.
    """
    seed1, seed2 = 79, 25
    # note how the two seeds for the old op correspond to the seed for the new
    # op
    with ops.device("/device:CPU:0"):
      random.reset_global_generator([0, seed2, seed1])
    shape = constant_op.constant([4, 7])
    dtype = dtypes.float64

    # create a graph for the old op in order to call it many times
    @def_function.function
    def old():
      with ops.device("/device:CPU:0"):
        return gen_random_ops.random_standard_normal(
            shape, dtype=dtype, seed=seed1, seed2=seed2)

    def new():
      with ops.device("/device:CPU:0"):
        return random.get_global_generator().normal(shape, dtype=dtype)

    for _ in range(100):
      self.assertAllEqual(old(), new())

  @test_util.run_v2_only
  @test_util.run_cuda_only
  def testGPUSameAsOldRandomOps(self):
    """Tests that the generated numbers are the same as the old random_ops.py.

    The GPU version.
    """
    seed1, seed2 = 79, 25
    with ops.device(test_util.gpu_device_name()):
      random.reset_global_generator([0, seed2, seed1])
    shape = constant_op.constant([4, 7])
    dtype = dtypes.float64

    @def_function.function
    def old():
      with ops.device(test_util.gpu_device_name()):
        return gen_random_ops.random_standard_normal(
            shape, dtype=dtype, seed=seed1, seed2=seed2)

    def new():
      with ops.device(test_util.gpu_device_name()):
        return random.get_global_generator().normal(shape, dtype=dtype)

    for _ in range(100):
      self.assertAllEqual(old(), new())

  @test_util.run_v2_only
  def testStatefulStandardNormal(self):
    """Tests that op 'StatefulStandardNormal' still works.
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
        errors_impl.NotFoundError, "Resource .+ does not exist"):
      a = f()
      random.reset_global_generator(50)
      b = f()
      self.assertAllEqual(a, b)


if __name__ == "__main__":
  test.main()
