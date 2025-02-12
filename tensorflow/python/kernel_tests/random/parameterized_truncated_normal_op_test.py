# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for ParameterizedTruncatedNormalOp."""

import functools
import math
import timeit

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops as stateless
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def _get_stddev_inside_bounds_before_using_randn(gpu):
  # The boundary where the randn sampler is used varies between CPU and GPU.
  if gpu:
    return 1.3
  else:
    return 1.7


class TruncatedNormalMoments:
  memoized_moments = None
  mean = None
  stddev = None
  minval = None
  maxval = None

  def __init__(self, mean, stddev, minval, maxval):
    self.memoized_moments = [1.0]  # 0th moment
    self.mean = np.double(mean)
    self.stddev = np.double(stddev)
    # NOTE(ringwalt): The formula doesn't handle infinite values.
    self.minval = np.double(max(-10, minval))
    self.maxval = np.double(min(10, maxval))

  def __getitem__(self, moment):
    """Calculates the truncated normal moments.

    Args:
      moment: The number for the moment.

    Returns:
      The value for the given moment.

    Uses the recurrence relation described in:
        http://www.smp.uq.edu.au/people/YoniNazarathy/teaching_projects
            /studentWork/EricOrjebin_TruncatedNormalMoments.pdf
    """
    assert moment > 0
    # The test case must ensure it can import scipy.stats before this point.
    import scipy.stats  # pylint: disable=g-import-not-at-top
    dist = scipy.stats.norm(loc=self.mean, scale=self.stddev)
    for k in range(len(self.memoized_moments), moment + 1):
      m_k_minus_2 = self.memoized_moments[k - 2] if k > 1 else np.double(0.0)
      m_k_minus_1 = self.memoized_moments[k - 1]
      numerator = (np.power(self.maxval, k - 1) * dist.pdf(self.maxval) -
                   np.power(self.minval, k - 1) * dist.pdf(self.minval))
      denominator = dist.cdf(self.maxval) - dist.cdf(self.minval)
      m = ((k - 1) * self.stddev**2 * m_k_minus_2 + self.mean * m_k_minus_1 -
           self.stddev * numerator / denominator)
      assert abs(m) < 1e50  # ensure numerical accuracy
      self.memoized_moments.append(m)
    return self.memoized_moments[moment]


def calculate_moments(samples, max_moment):
  moments = [0.0] * (max_moment + 1)
  for k in range(len(moments)):
    moments[k] = np.mean(samples**k, axis=0)
  return moments


def z_test(real, expected, i, num_samples):
  numerical_error = 1e-6  # per-operation error
  moment_mean = expected[i]
  moment_squared = expected[2 * i]
  moment_var = moment_squared - moment_mean * moment_mean

  error_per_moment = i * numerical_error
  total_variance = moment_var / float(num_samples) + error_per_moment
  return abs((real[i] - moment_mean) / math.sqrt(total_variance))


class ParameterizedTruncatedNormalTest(test.TestCase):
  z_limit = 6.0

  # Stop at moment 10 to avoid numerical errors in the theoretical moments.
  max_moment = 10

  def validateMoments(self,
                      shape,
                      mean,
                      stddev,
                      minval,
                      maxval,
                      use_stateless=False,
                      seed=1618):
    try:
      # TruncatedNormalMoments requires scipy.stats.
      # Give up early if we are unable to import it.
      random_seed.set_random_seed(seed)
      with self.cached_session():
        if use_stateless:
          # Generate a seed that stateless ops can use.
          new_seed = random_ops.random_uniform([2],
                                               seed=seed,
                                               minval=0,
                                               maxval=(2**31 - 1),
                                               dtype=np.int32)
          samples = stateless.stateless_parameterized_truncated_normal(
              shape, new_seed, mean, stddev, minval, maxval).eval()
        else:
          samples = random_ops.parameterized_truncated_normal(
              shape, mean, stddev, minval, maxval).eval()
        assert (~np.isnan(samples)).all()
      moments = calculate_moments(samples, self.max_moment)
      expected_moments = TruncatedNormalMoments(mean, stddev, minval, maxval)
      num_samples = functools.reduce(lambda x, y: x * y, shape, 1)
      for i in range(1, len(moments)):
        self.assertLess(
            z_test(moments, expected_moments, i, num_samples), self.z_limit)
    except ImportError as e:
      tf_logging.warn("Cannot test truncated normal op: %s" % str(e))

  def validateKolmogorovSmirnov(self,
                                shape,
                                mean,
                                stddev,
                                minval,
                                maxval,
                                use_stateless=False,
                                seed=1618):
    try:
      import scipy.stats  # pylint: disable=g-import-not-at-top
      random_seed.set_random_seed(seed)
      with self.cached_session():
        if use_stateless:
          new_seed = random_ops.random_uniform([2],
                                               seed=seed,
                                               minval=0,
                                               maxval=(2**31 - 1),
                                               dtype=np.int32)
          samples = stateless.stateless_parameterized_truncated_normal(
              shape, new_seed, mean, stddev, minval, maxval).eval()
        else:
          samples = random_ops.parameterized_truncated_normal(
              shape, mean, stddev, minval, maxval).eval()

      assert (~np.isnan(samples)).all()
      minval = max(mean - stddev * 10, minval)
      maxval = min(mean + stddev * 10, maxval)
      dist = scipy.stats.norm(loc=mean, scale=stddev)
      cdf_min = dist.cdf(minval)
      cdf_max = dist.cdf(maxval)

      def truncated_cdf(x):
        return np.clip((dist.cdf(x) - cdf_min) / (cdf_max - cdf_min), 0.0, 1.0)

      pvalue = scipy.stats.kstest(samples, truncated_cdf)[1]
      self.assertGreater(pvalue, 1e-10)
    except ImportError as e:
      tf_logging.warn("Cannot test truncated normal op: %s" % str(e))

  @test_util.run_deprecated_v1
  def testDefaults(self):
    self.validateMoments([int(1e5)], 0.0, 1.0, -2.0, 2.0)
    self.validateMoments([int(1e5)], 0.0, 1.0, -2.0, 2.0, use_stateless=True)

  @test_util.run_deprecated_v1
  def testShifted(self):
    self.validateMoments([int(1e5)], -1.0, 1.0, -2.0, 2.0)
    self.validateMoments([int(1e5)], -1.0, 1.0, -2.0, 2.0, use_stateless=True)

  @test_util.run_deprecated_v1
  def testRightTail(self):
    self.validateMoments([int(1e5)], 0.0, 1.0, 4.0, np.inf)
    self.validateMoments([int(1e5)],
                         0.0,
                         1.0,
                         4.0,
                         np.inf,
                         use_stateless=True)

  @test_util.run_deprecated_v1
  def testLeftTail(self):
    self.validateMoments([int(1e5)], 0.0, 1.0, -np.inf, -4.0)
    self.validateMoments([int(1e5)],
                         0.0,
                         1.0,
                         -np.inf,
                         -4.0,
                         use_stateless=True)

  @test_util.run_deprecated_v1
  def testLeftTailTwoSidedBounds(self):
    self.validateMoments([int(1e5)], 0.0, 1.0, -6.0, -3.0)
    self.validateMoments([int(1e5)], 0.0, 1.0, -6.0, -3.0, use_stateless=True)

  @test_util.run_deprecated_v1
  @test_util.disable_xla("Low probability region")
  def testTwoSidedLeftTailShifted(self):
    self.validateKolmogorovSmirnov([int(1e5)], 6.0, 1.0, -1.0, 1.0)
    self.validateKolmogorovSmirnov([int(1e5)],
                                   6.0,
                                   1.0,
                                   -1.0,
                                   1.0,
                                   use_stateless=True)

  @test_util.run_deprecated_v1
  @test_util.disable_xla("Low probability region")
  def testRightTailShifted(self):
    self.validateMoments([int(1e5)], -5.0, 1.0, 2.0, np.inf)
    self.validateMoments([int(1e5)],
                         -5.0,
                         1.0,
                         2.0,
                         np.inf,
                         use_stateless=True)

  # Take the normal distribution around the mean, but truncating the left tail
  # far from the mean.
  @test_util.run_deprecated_v1
  def testTruncateOnLeft_entireTailOnRight(self):
    self.validateKolmogorovSmirnov([int(1e5)], 10.0, 1.0, 4.0, np.inf)
    self.validateKolmogorovSmirnov([int(1e5)],
                                   10.0,
                                   1.0,
                                   4.0,
                                   np.inf,
                                   use_stateless=True)

  # Take the normal distribution around the mean, but truncating the right tail.
  @test_util.run_deprecated_v1
  def testTruncateOnRight_entireTailOnLeft(self):
    self.validateKolmogorovSmirnov([int(1e5)], -8, 1.0, -np.inf, -4.0)
    self.validateKolmogorovSmirnov([int(1e5)],
                                   -8.,
                                   1.0,
                                   -np.inf,
                                   -4.0,
                                   use_stateless=True)

  @test_util.run_deprecated_v1
  def testSmallStddev(self):
    self.validateKolmogorovSmirnov([int(1e5)], 0.0, 0.1, 0.05, 0.10)
    self.validateKolmogorovSmirnov([int(1e5)],
                                   0.0,
                                   0.1,
                                   0.05,
                                   0.10,
                                   use_stateless=True)

  @test_util.run_deprecated_v1
  def testSamplingWithSmallStdDevFarFromBound(self):
    sample_op = random_ops.parameterized_truncated_normal(
        shape=(int(1e5),), means=0.8, stddevs=0.05, minvals=-1., maxvals=1.)
    new_seed = random_ops.random_uniform([2],
                                         seed=1234,
                                         minval=0,
                                         maxval=(2**31 - 1),
                                         dtype=np.int32)
    sample_op_stateless = stateless.stateless_parameterized_truncated_normal(
        shape=(int(1e5),),
        seed=new_seed,
        means=0.8,
        stddevs=0.05,
        minvals=-1.,
        maxvals=1.)

    with self.session() as sess:
      samples, samples_stateless = sess.run([sample_op, sample_op_stateless])
      # 0. is more than 16 standard deviations from the mean, and
      # should have a likelihood < 1e-57.
      assert (~np.isnan(samples)).all()
      assert (~np.isnan(samples_stateless)).all()
      self.assertAllGreater(samples, 0.)
      self.assertAllGreater(samples_stateless, 0.)

  def testShapeTypes(self):
    for shape_dtype in [np.int32, np.int64]:
      shape = np.array([1000], dtype=shape_dtype)
      sample_op = random_ops.parameterized_truncated_normal(
          shape=shape, means=0.0, stddevs=0.1, minvals=-1., maxvals=1.)
      new_seed = random_ops.random_uniform([2],
                                           seed=1234,
                                           minval=0,
                                           maxval=(2**31 - 1),
                                           dtype=np.int32)
      sample_op_stateless = stateless.stateless_parameterized_truncated_normal(
          shape=shape,
          seed=new_seed,
          means=0.0,
          stddevs=0.1,
          minvals=-1.,
          maxvals=1.)

      samples = self.evaluate(sample_op)
      stateless_samples = self.evaluate(sample_op_stateless)
      self.assertAllEqual(samples.shape, shape)
      self.assertAllEqual(stateless_samples.shape, shape)

  def testStatelessParameterizedTruncatedNormalHasGrads(self):
    mean = variables.Variable(0.01)
    stddev = variables.Variable(1.)
    minval = variables.Variable(-1.)
    maxval = variables.Variable(1.)

    with self.cached_session() as sess:
      with backprop.GradientTape(persistent=True) as tape:
        samples = stateless.stateless_parameterized_truncated_normal(
            [1], [1, 2], mean, stddev, minval, maxval)

      sess.run(variables.variables_initializer([mean, stddev, minval, maxval]))
      [mean_grad, std_grad], mean_actual_grad, std_actual_grad = sess.run([
          tape.gradient(samples, [mean, stddev]),
          array_ops.ones_like(mean),
          (samples - mean) / stddev])
      self.assertAllClose(mean_grad, mean_actual_grad)
      self.assertAllClose(std_grad, std_actual_grad[0])

      try:
        import scipy.stats  # pylint:disable=g-import-not-at-top
        truncnorm = scipy.stats.truncnorm(a=-1., b=1., loc=0., scale=1.)
        samples_np, [minval_grad, maxval_grad] = sess.run([
            samples, tape.gradient(samples, [minval, maxval])])

        sample_cdf = truncnorm.cdf(samples_np)
        # These come from the implicit reparameterization trick.
        scipy_maxval_grad = np.exp(
            0.5 * (samples_np ** 2 - ((1. - 0.01) / 1.) ** 2) +
            np.log(sample_cdf))

        scipy_minval_grad = np.exp(
            0.5 * (samples_np ** 2 - ((-1. - 0.01) / 1.) ** 2) +
            np.log1p(-sample_cdf))

        self.assertAllClose(minval_grad, scipy_minval_grad[0], rtol=1e-2)
        self.assertAllClose(maxval_grad, scipy_maxval_grad[0], rtol=1e-2)

      except ImportError as e:
        tf_logging.warn("Cannot test truncated normal op: %s" % str(e))

  @test_util.run_deprecated_v1
  def testSamplingAtRandnSwitchover(self):
    # The randn sampler is used as the bounds are moved farther from the mean,
    # and the probability of accepting a sample increases the farther the
    # bounds are from the mean.
    # This test asserts that at the point of switchover, both samplers are
    # working (not raising an error or returning nan) and returning the
    # expected moments.
    use_gpu = test.is_gpu_available()
    stddev_inside_bounds_before_using_randn = (
        _get_stddev_inside_bounds_before_using_randn(use_gpu))

    epsilon = 0.001
    self.validateMoments(
        shape=[int(1e6)],
        mean=0.,
        stddev=1.0,
        minval=-epsilon,
        maxval=stddev_inside_bounds_before_using_randn - epsilon)
    self.validateMoments(
        shape=[int(1e6)],
        mean=0.,
        stddev=1.0,
        minval=-epsilon,
        maxval=stddev_inside_bounds_before_using_randn + epsilon)

    self.validateMoments(
        shape=[int(1e6)],
        mean=0.,
        stddev=1.0,
        minval=-epsilon,
        maxval=stddev_inside_bounds_before_using_randn - epsilon,
        use_stateless=True)
    self.validateMoments(
        shape=[int(1e6)],
        mean=0.,
        stddev=1.0,
        minval=-epsilon,
        maxval=stddev_inside_bounds_before_using_randn + epsilon,
        use_stateless=True)


# Benchmarking code
def parameterized_vs_naive(shape, num_iters, use_gpu=False):
  np.random.seed(1618)  # Make it reproducible.

  # No CSE/CF.
  optimizer_options = config_pb2.OptimizerOptions(
      opt_level=config_pb2.OptimizerOptions.L0)
  config = config_pb2.ConfigProto(graph_options=config_pb2.GraphOptions(
      optimizer_options=optimizer_options))

  with session.Session(config=config) as sess:
    with ops.device("/cpu:0" if not use_gpu else None):
      param_op = control_flow_ops.group(
          random_ops.parameterized_truncated_normal(shape))
      naive_op = control_flow_ops.group(random_ops.truncated_normal(shape))

    # Burn-in to avoid session setup costs in the timing.
    sess.run(param_op)
    sess.run(param_op)
    param_dt = timeit.timeit(lambda: sess.run(param_op), number=num_iters)
    sess.run(naive_op)
    sess.run(naive_op)
    naive_dt = timeit.timeit(lambda: sess.run(naive_op), number=num_iters)
    return param_dt, naive_dt


def randn_sampler_switchover(shape, num_iters, use_gpu=False):
  # Benchmark by constructing samplers on the threshold of using the randn
  # rejection sampling and check that this threshold is set correctly by
  # benchmarking with bounds just above and below this threshold.
  # The uniform and randn samplers should have about the same performance
  # at this point.

  stddev_inside_bounds_before_using_randn = (
      _get_stddev_inside_bounds_before_using_randn(use_gpu))

  epsilon = 0.001

  np.random.seed(1618)  # Make it reproducible.

  # No CSE/CF.
  optimizer_options = config_pb2.OptimizerOptions(
      opt_level=config_pb2.OptimizerOptions.L0)
  config = config_pb2.ConfigProto(
      graph_options=config_pb2.GraphOptions(
          optimizer_options=optimizer_options))

  with session.Session(config=config) as sess:
    with ops.device("/cpu:0" if not use_gpu else "/gpu:0"):
      uniform_sampler_op = control_flow_ops.group(
          random_ops.parameterized_truncated_normal(
              shape,
              means=0.,
              stddevs=1.0,
              minvals=-stddev_inside_bounds_before_using_randn + epsilon,
              maxvals=0.01))
      randn_sampler_op = control_flow_ops.group(
          random_ops.parameterized_truncated_normal(
              shape,
              means=0.,
              stddevs=1.0,
              minvals=-stddev_inside_bounds_before_using_randn - epsilon,
              maxvals=0.01))

    # Burn-in to avoid session setup costs in the timing.
    sess.run(uniform_sampler_op)
    sess.run(uniform_sampler_op)
    uniform_dt = timeit.timeit(
        lambda: sess.run(uniform_sampler_op), number=num_iters)

    sess.run(randn_sampler_op)
    sess.run(randn_sampler_op)
    randn_dt = timeit.timeit(
        lambda: sess.run(randn_sampler_op), number=num_iters)

    return randn_dt, uniform_dt


class TruncatedNormalBenchmark(test.Benchmark):

  def benchmarkParameterizedOpVsNaiveOpCpu(self):
    self._benchmarkParameterizedOpVsNaiveOp(False)

  def benchmarkParameterizedOpVsNaiveOpGpu(self):
    self._benchmarkParameterizedOpVsNaiveOp(True)

  def _benchmarkParameterizedOpVsNaiveOp(self, use_gpu):
    num_iters = 50
    print(("Composition of new ParameterizedTruncatedNormalOp vs. "
           "naive TruncatedNormalOp [%d iters]") % num_iters)
    print("Shape\tsec(parameterized)\tsec(naive)\tspeedup")

    for shape in [[10000, 100], [1000, 1000], [1000000], [100, 100, 100],
                  [20, 20, 20, 20]]:
      p_dt, n_dt = parameterized_vs_naive(shape, num_iters, use_gpu)
      print("%s\t%.3f\t%.3f\t%.2f" % (shape, p_dt, n_dt, p_dt / n_dt))

      shape_str = "-".join(map(str, shape))
      self.report_benchmark(
          name="parameterized_shape" + shape_str,
          iters=num_iters,
          wall_time=p_dt)
      self.report_benchmark(
          name="naive_shape" + shape_str, iters=num_iters, wall_time=n_dt)

  def benchmarkRandnSamplerCPU(self):
    self._benchmarkRandnSampler(False)

  def benchmarkRandnSamplerGPU(self):
    self._benchmarkRandnSampler(True)

  def _benchmarkRandnSampler(self, use_gpu):
    num_iters = 100
    shape = [int(1e6)]
    randn_dt, uniform_dt = randn_sampler_switchover(shape, num_iters, use_gpu)

    print(("Randn Sampler vs uniform samplers [%d iters]\t%.4f\t%.4f") %
          (num_iters, randn_dt, uniform_dt))

    gpu_str = "_gpu" if use_gpu else "_cpu"
    self.report_benchmark(
        name="randn_sampler" + gpu_str, iters=num_iters, wall_time=randn_dt)
    self.report_benchmark(
        name="uniform_sampler" + gpu_str, iters=num_iters, wall_time=uniform_dt)


if __name__ == "__main__":
  test.main()
