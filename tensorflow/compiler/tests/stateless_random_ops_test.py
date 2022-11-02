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
"""Tests for stateless random-number generation ops."""

import functools
import os
from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.client import device_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.random import util as \
random_test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops as stateless
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def xla_device():
  devices = device_lib.list_local_devices()

  def find_type(device_type):
    for d in devices:
      if d.device_type == device_type:
        return d
    return None

  d = find_type('TPU') or find_type('XLA_GPU') or find_type('XLA_CPU')
  if d is None:
    raise ValueError('Cannot find any XLA device. Available devices:\n%s' %
                     devices)
  return d


def _allowed_types(include_int=False):
  allowed_types = {
      dtypes.float64, dtypes.float32, dtypes.float16, dtypes.bfloat16
  }
  if include_int:
    allowed_types.update({dtypes.int32, dtypes.int64})
  return allowed_types


class StatelessRandomOpsTest(xla_test.XLATestCase, parameterized.TestCase):
  """Test cases for stateless random-number generator operators."""

  def _random_types(self, include_int=False):
    return self.all_tf_types & _allowed_types(include_int)

  @test_util.run_v2_only
  def testForcedCompile(self):
    """Tests whole-function forced-compilation.

    This test checks that stateless_random_* can be used in forced-compilation
    scenarios (e.g. TPU). The new version of stateless_random_* requires the
    intermediate tensor `alg` to be compile-time constant, so we need to check
    that this requirement won't prevent `seed` from depending on variables.
    """
    if config.list_logical_devices('TPU'):
      self.skipTest('To accommodate OSS, experimental_compile support for TPU '
                    'is not linked in.')
    # GPU doesn't support int32 variables, so we use int64.
    v = variables.Variable([1, 2], dtype=dtypes.int64)

    @def_function.function(experimental_compile=True)
    def f():
      key, counter = (
          gen_stateless_random_ops_v2.stateless_random_get_key_counter(
              seed=math_ops.cast(v.read_value(), dtypes.int32)))
      alg = gen_stateless_random_ops_v2.stateless_random_get_alg()
      return gen_stateless_random_ops_v2.stateless_random_normal_v2(
          shape=[], key=key, counter=counter, alg=alg)

    f()

  @test_util.run_v2_only
  def testGetKeyCounterAlg(self):
    seed = [1, 2]
    key, counter = gen_stateless_random_ops_v2.stateless_random_get_key_counter(
        seed)
    self.assertAllEqual(key.shape, [1])
    self.assertAllEqual(counter.shape, [2])
    alg = gen_stateless_random_ops_v2.stateless_random_get_alg()
    self.assertAllEqual(alg.shape, [])

  @parameterized.named_parameters(
      ('_%s_%s' % (op_id, alg_id), op, alg_group)  # pylint: disable=g-complex-comprehension
      for alg_id, alg_group in enumerate([
          [
              stateless.Algorithm.PHILOX, stateless.Algorithm.PHILOX.value,
              'philox'
          ],
          [
              stateless.Algorithm.THREEFRY, stateless.Algorithm.THREEFRY.value,
              'threefry'
          ],
          [
              stateless.Algorithm.AUTO_SELECT,
              stateless.Algorithm.AUTO_SELECT.value, 'auto_select', None
          ],
      ])
      for op_id, op in enumerate([
          stateless.stateless_random_normal,
          stateless.stateless_truncated_normal,
          functools.partial(
              stateless.stateless_random_uniform,
              dtype=dtypes.uint32,
              minval=None,
              maxval=None),
          functools.partial(
              stateless.stateless_random_uniform,
              dtype=dtypes.int32,
              maxval=100),
          functools.partial(
              stateless.stateless_random_uniform, dtype=dtypes.float32),
      ]))
  @test_util.run_v2_only
  def testAlg(self, op, alg_group):
    """Tests all values of `alg`."""
    if config.list_logical_devices('TPU') or config.list_logical_devices('GPU'):
      self.skipTest('Only _cpu tests linked in support for jit_compile on CPU.')
    seed = [1, 2]
    shape = [2, 3]
    outputs = []
    for alg in alg_group:
      with ops.device('CPU'):
        output = def_function.function(jit_compile=True)(op)(
            shape=shape, seed=seed, alg=alg)
      self.assertEqual(output.shape, shape)
      outputs.append(output)
    x = outputs[0]
    for y in outputs[1:]:
      self.assertAllEqual(x, y)

  def testLargeNormal(self):
    """Tests an OOM bug of StatelessRandomNormalV2 on TPU."""
    with self.session() as sess, self.test_scope():
      seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
      key, counter, alg = (gen_stateless_random_ops_v2.
                           stateless_random_get_key_counter_alg(seed_t))
      x = gen_stateless_random_ops_v2.stateless_random_normal_v2(
          shape=[1024, 32000], key=key, counter=counter, dtype=dtypes.float32,
          alg=alg)
      y = sess.run(x, {seed_t: [0x12345678, 0xabcdef1]})
      self.assertAllEqual([1024, 32000], y.shape)
      key, counter = (gen_stateless_random_ops_v2.
                      stateless_random_get_key_counter(seed_t))
      alg = gen_stateless_random_ops_v2.stateless_random_get_alg()
      x = gen_stateless_random_ops_v2.stateless_random_normal_v2(
          shape=[1024, 32000], key=key, counter=counter, dtype=dtypes.float32,
          alg=alg)
      y = sess.run(x, {seed_t: [0x12345678, 0xabcdef1]})
      self.assertAllEqual([1024, 32000], y.shape)

  @parameterized.named_parameters(
      (f'_{op_name}_{shape}_{dtype.name}', stateless_op, shape, dtype)  # pylint: disable=g-complex-comprehension
      for dtype in _allowed_types() for shape in ((), (3,), (2, 5))
      for op_name, stateless_op in (
          ('uniform', stateless.stateless_random_uniform),
          ('normal', stateless.stateless_random_normal),
      ))
  def testDeterminism(self, stateless_op, shape, dtype):
    # Stateless values should be equal iff the seeds are equal (roughly)
    seeds = [(x, y) for x in range(-2, 3) for y in range(-2, 3)] * 3  # pylint: disable=g-complex-comprehension
    with self.session(), self.test_scope():
      seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
      pure = stateless_op(shape, seed=seed_t, dtype=dtype)
      values = [(seed, pure.eval(feed_dict={seed_t: seed})) for seed in seeds]
      for s0, v0 in values:
        for s1, v1 in values:
          if s0 == s1:
            self.assertAllEqual(v0, v1)
          else:
            # The resolutions of float16 and bfloat16 are too low, so
            # in some cases (e.g. scalar shape) different seeds may
            # lead to the same output. So we skip those dtypes.
            if not (dtype in (dtypes.bfloat16, dtypes.float16) and shape == ()):  # pylint: disable=g-explicit-bool-comparison
              self.assertNotAllEqual(v0, v1)

  def testRandomUniformIsInRange(self):
    with self.session() as sess, self.test_scope():
      for dtype in self._random_types(include_int=True):
        maxval = 1
        if dtype.is_integer:
          maxval = 100
        seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
        x = stateless.stateless_random_uniform(
            shape=[1000], seed=seed_t, maxval=maxval, dtype=dtype)
        y = sess.run(x, {seed_t: [0x12345678, 0xabcdef1]})
        self.assertTrue(np.all(y >= 0))
        self.assertTrue(np.all(y < maxval))

  @parameterized.named_parameters(
      (f'_{alg.name}_{dtype.name}_{seed}', alg, dtype, seed)  # pylint: disable=g-complex-comprehension
      for seed in ([1, 2], [12, 23], [123, 456], [565656, 121212])
      for dtype in _allowed_types(include_int=True)
      for alg in list(stateless.Algorithm))
  def testDistributionOfStatelessRandomUniform(self, alg, dtype, seed):
    """Use Pearson's Chi-squared test to test for uniformity."""
    philox = stateless.Algorithm.PHILOX
    auto_select = stateless.Algorithm.AUTO_SELECT
    device = xla_device()
    if 'CPU' in device.device_type:
      device_type = 'CPU'
    elif 'GPU' in device.device_type:
      device_type = 'GPU'
    elif device.device_type == 'TPU':
      device_type = 'TPU'
    else:
      device_type = None
    bad_combos1 = [
        (dtypes.int32, [123, 456]),
        (dtypes.int64, [123, 456]),
        (dtypes.float16, [565656, 121212]),
        (dtypes.bfloat16, [1, 2]),
    ]
    bad_combos2 = [
        (dtypes.int32, [1, 2]),
        (dtypes.int32, [12, 23]),
    ]
    # TODO(b/244649364): Investigate why these combinations fail.
    if (device_type in ('CPU', 'GPU') and alg in (philox, auto_select) and
        (dtype, seed) in bad_combos1 or device_type == 'TPU' and
        (alg == philox and
         (dtype, seed) in bad_combos1 or alg == auto_select and
         (dtype, seed) in bad_combos2)):
      self.skipTest(
          'This (device, alg, dtype, seed) combination fails (b/244649364).')
    with self.session() as sess, self.test_scope():
      seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
      n = 1000
      maxval = 1
      if dtype.is_integer:
        maxval = 100
      x = stateless.stateless_random_uniform(
          shape=[n], seed=seed_t, maxval=maxval, dtype=dtype, alg=alg)
      y = sess.run(x, {seed_t: seed})
      # Convert y to float and normalize its value to range [0, 1) when
      # maxval != 1.
      y = y.astype(float) / maxval
      # Tests that the values are distributed amongst 10 bins with equal
      # probability. 16.92 is the Chi^2 value for 9 degrees of freedom with
      # p=0.05. This test is probabilistic and would be flaky if the random
      # seed were not fixed.
      self.assertLess(random_test_util.chi_squared(y, 10), 16.92)

  def testRandomNormalIsFinite(self):
    with self.session() as sess, self.test_scope():
      for dtype in self._random_types():
        seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
        x = stateless.stateless_random_normal(
            shape=[10000], seed=seed_t, dtype=dtype)
        y = sess.run(x, {seed_t: [0x12345678, 0xabcdef1]})
        self.assertTrue(np.all(np.isfinite(y)))

  @parameterized.named_parameters(
      (f'_{dtype.name}_{seed}', dtype, seed)  # pylint: disable=g-complex-comprehension
      for seed in ([1, 2], [12, 23], [123, 456], [25252, 314159])
      for dtype in _allowed_types())
  def testDistributionOfStatelessRandomNormal(self, dtype, seed):
    """Use Anderson-Darling test to test distribution appears normal."""
    with self.session() as sess, self.test_scope():
      seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
      n = 1000
      x = stateless.stateless_random_normal(shape=[n], seed=seed_t, dtype=dtype)
      y = sess.run(x, {seed_t: seed})
      # The constant 2.492 is the 5% critical value for the Anderson-Darling
      # test where the mean and variance are known. This test is probabilistic
      # so to avoid flakiness the seed is fixed.
      self.assertLess(random_test_util.anderson_darling(y.astype(float)), 2.492)

  @parameterized.named_parameters(
      (f'_{dtype.name}', dtype) for dtype in _allowed_types())
  def testTruncatedNormal(self, dtype):
    with self.session() as sess, self.test_scope():
      seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
      n = 10000000
      x = stateless.stateless_truncated_normal(
          shape=[n], seed=seed_t, dtype=dtype)
      y = sess.run(x, {seed_t: [0x12345678, 0xabcdef1]})
      is_megacore = 'megacore' in os.environ.get('TEST_TARGET', '').lower()
      if dtype == dtypes.float16:
        if is_megacore:
          mean_atol = 2e-3
        else:
          mean_atol = 7e-4
      else:
        mean_atol = 5e-4

      if dtype == dtypes.float16 and is_megacore:
        median_atol = 2e-3
      else:
        median_atol = 8e-4

      if dtype == dtypes.bfloat16:
        variance_rtol = 6e-3
      elif dtype == dtypes.float16:
        variance_rtol = 3e-3
      else:
        variance_rtol = 1e-3

      random_test_util.test_truncated_normal(
          self.assertEqual,
          self.assertAllClose,
          n,
          y,
          mean_atol=mean_atol,
          median_atol=median_atol,
          variance_rtol=variance_rtol)

  def _testParameterizedTruncatedNormal(self,
                                        means,
                                        stddevs,
                                        minvals,
                                        maxvals,
                                        variance_rtol=None):
    for dtype in self._random_types():
      with self.session() as sess, self.test_scope():
        seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
        n = int(10e7)
        x = stateless.stateless_parameterized_truncated_normal(
            shape=[n],
            seed=seed_t,
            means=means,
            stddevs=stddevs,
            minvals=minvals,
            maxvals=maxvals)
        y = sess.run(x, {seed_t: [0x12345678, 0xabcdef1]})
        if variance_rtol is None:
          variance_rtol = 6e-3 if dtype == dtypes.bfloat16 else 1e-3
        random_test_util.test_truncated_normal(
            self.assertEqual,
            self.assertAllClose,
            n,
            y,
            means=means,
            stddevs=stddevs,
            minvals=minvals,
            maxvals=maxvals,
            mean_atol=1e-3,
            median_atol=1e-3,
            variance_rtol=variance_rtol)

  def testParameterizedTruncatedNormalDefault(self):
    self._testParameterizedTruncatedNormal(0., 1., -2., 2.)

  def testParameterizedTruncatedNormalShifted(self):
    self._testParameterizedTruncatedNormal(-1., 1., -2., 2.)

  def testParameterizedTruncatedNormalRightTail(self):
    self._testParameterizedTruncatedNormal(0., 1., 4., 20., variance_rtol=2e-2)

  def testParameterizedTruncatedNormalLeftTail(self):
    self._testParameterizedTruncatedNormal(
        0., 1., -20., -4., variance_rtol=5e-2)

  def testParameterizedTruncatedNormalLeftTailTwoSidedBounds(self):
    self._testParameterizedTruncatedNormal(
        0., 1., -6., -3., variance_rtol=5e-2)

  def testParameterizedTruncatedNormalSmallStddev(self):
    self._testParameterizedTruncatedNormal(0., 0.1, 0.05, 0.10)

  def testParameterizedTruncatedNormalBroadcast(self):
    with self.session() as sess, self.test_scope():
      seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
      means = array_ops.zeros([2], dtype=dtypes.float32)
      stddevs = array_ops.ones([3, 1], dtype=dtypes.float32)
      minvals = -array_ops.ones([5, 1, 1], dtype=dtypes.float32)
      maxvals = array_ops.ones([7, 1, 1, 1], dtype=dtypes.float32)
      shape = [11, 7, 5, 3, 2]
      x = stateless.stateless_parameterized_truncated_normal(
          shape=shape,
          seed=seed_t,
          means=means,
          stddevs=stddevs,
          minvals=minvals,
          maxvals=maxvals)
      y = sess.run(x, {seed_t: [0x12345678, 0xabcdef1]})
      self.assertEqual((11, 7, 5, 3, 2), y.shape)


class StatelessRandomOpsBenchmark(test.Benchmark):
  """Microbenchmarks for the stateless random ops."""

  def _benchmarkUniform(self, name, dtype, use_xla_jit):

    def builder_fn():
      shape = (10, 1000, 1000)
      seed_var = variables.Variable((312, 456),
                                    dtype=dtypes.int32,
                                    name='input')
      random_t = stateless.stateless_random_uniform(
          shape, seed=seed_var, dtype=dtype)
      return '%s.shape%s' % (name, shape), [random_t]

    xla_test.Benchmark(self, builder_fn, use_xla_jit=use_xla_jit, device='cpu')

  def benchmarkUniformF16(self):
    self._benchmarkUniform(
        'uniform_f16', dtype=dtypes.float16, use_xla_jit=False)

  def benchmarkUniformF32(self):
    self._benchmarkUniform(
        'uniform_f32', dtype=dtypes.float32, use_xla_jit=False)

  def benchmarkUniformF64(self):
    self._benchmarkUniform(
        'uniform_f64', dtype=dtypes.float64, use_xla_jit=False)

  def benchmarkUniformF16XLA(self):
    self._benchmarkUniform(
        'uniform_f16', dtype=dtypes.float16, use_xla_jit=True)

  def benchmarkUniformF32XLA(self):
    self._benchmarkUniform(
        'uniform_f32', dtype=dtypes.float32, use_xla_jit=True)

  def benchmarkUniformF64XLA(self):
    self._benchmarkUniform(
        'uniform_f64', dtype=dtypes.float64, use_xla_jit=True)


if __name__ == '__main__':
  config.set_soft_device_placement(False)
  test.main()
