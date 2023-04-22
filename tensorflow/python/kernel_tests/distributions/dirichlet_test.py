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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import dirichlet as dirichlet_lib
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def try_import(name):  # pylint: disable=invalid-name
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf_logging.warning("Could not import %s: %s" % (name, str(e)))
  return module


special = try_import("scipy.special")
stats = try_import("scipy.stats")


@test_util.run_all_in_graph_and_eager_modes
class DirichletTest(test.TestCase):

  def testSimpleShapes(self):
    alpha = np.random.rand(3)
    dist = dirichlet_lib.Dirichlet(alpha)
    self.assertEqual(3, self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tensor_shape.TensorShape([3]), dist.event_shape)
    self.assertEqual(tensor_shape.TensorShape([]), dist.batch_shape)

  def testComplexShapes(self):
    alpha = np.random.rand(3, 2, 2)
    dist = dirichlet_lib.Dirichlet(alpha)
    self.assertEqual(2, self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([3, 2], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tensor_shape.TensorShape([2]), dist.event_shape)
    self.assertEqual(tensor_shape.TensorShape([3, 2]), dist.batch_shape)

  def testConcentrationProperty(self):
    alpha = [[1., 2, 3]]
    dist = dirichlet_lib.Dirichlet(alpha)
    self.assertEqual([1, 3], dist.concentration.get_shape())
    self.assertAllClose(alpha, self.evaluate(dist.concentration))

  def testPdfXProper(self):
    alpha = [[1., 2, 3]]
    dist = dirichlet_lib.Dirichlet(alpha, validate_args=True)
    self.evaluate(dist.prob([.1, .3, .6]))
    self.evaluate(dist.prob([.2, .3, .5]))
    # Either condition can trigger.
    with self.assertRaisesOpError("samples must be positive"):
      self.evaluate(dist.prob([-1., 1.5, 0.5]))
    with self.assertRaisesOpError("samples must be positive"):
      self.evaluate(dist.prob([0., .1, .9]))
    with self.assertRaisesOpError("sample last-dimension must sum to `1`"):
      self.evaluate(dist.prob([.1, .2, .8]))

  def testLogPdfOnBoundaryIsFiniteWhenAlphaIsOne(self):
    # Test concentration = 1. for each dimension.
    concentration = 3 * np.ones((10, 10)).astype(np.float32)
    concentration[range(10), range(10)] = 1.
    x = 1 / 9. * np.ones((10, 10)).astype(np.float32)
    x[range(10), range(10)] = 0.
    dist = dirichlet_lib.Dirichlet(concentration)
    log_prob = self.evaluate(dist.log_prob(x))
    self.assertAllEqual(
        np.ones_like(log_prob, dtype=np.bool), np.isfinite(log_prob))

    # Test when concentration[k] = 1., and x is zero at various dimensions.
    dist = dirichlet_lib.Dirichlet(10 * [1.])
    log_prob = self.evaluate(dist.log_prob(x))
    self.assertAllEqual(
        np.ones_like(log_prob, dtype=np.bool), np.isfinite(log_prob))

  def testPdfZeroBatches(self):
    alpha = [1., 2]
    x = [.5, .5]
    dist = dirichlet_lib.Dirichlet(alpha)
    pdf = dist.prob(x)
    self.assertAllClose(1., self.evaluate(pdf))
    self.assertEqual((), pdf.get_shape())

  def testPdfZeroBatchesNontrivialX(self):
    alpha = [1., 2]
    x = [.3, .7]
    dist = dirichlet_lib.Dirichlet(alpha)
    pdf = dist.prob(x)
    self.assertAllClose(7. / 5, self.evaluate(pdf))
    self.assertEqual((), pdf.get_shape())

  def testPdfUniformZeroBatches(self):
    # Corresponds to a uniform distribution
    alpha = [1., 1, 1]
    x = [[.2, .5, .3], [.3, .4, .3]]
    dist = dirichlet_lib.Dirichlet(alpha)
    pdf = dist.prob(x)
    self.assertAllClose([2., 2.], self.evaluate(pdf))
    self.assertEqual((2), pdf.get_shape())

  def testPdfAlphaStretchedInBroadcastWhenSameRank(self):
    alpha = [[1., 2]]
    x = [[.5, .5], [.3, .7]]
    dist = dirichlet_lib.Dirichlet(alpha)
    pdf = dist.prob(x)
    self.assertAllClose([1., 7. / 5], self.evaluate(pdf))
    self.assertEqual((2), pdf.get_shape())

  def testPdfAlphaStretchedInBroadcastWhenLowerRank(self):
    alpha = [1., 2]
    x = [[.5, .5], [.2, .8]]
    pdf = dirichlet_lib.Dirichlet(alpha).prob(x)
    self.assertAllClose([1., 8. / 5], self.evaluate(pdf))
    self.assertEqual((2), pdf.get_shape())

  def testPdfXStretchedInBroadcastWhenSameRank(self):
    alpha = [[1., 2], [2., 3]]
    x = [[.5, .5]]
    pdf = dirichlet_lib.Dirichlet(alpha).prob(x)
    self.assertAllClose([1., 3. / 2], self.evaluate(pdf))
    self.assertEqual((2), pdf.get_shape())

  def testPdfXStretchedInBroadcastWhenLowerRank(self):
    alpha = [[1., 2], [2., 3]]
    x = [.5, .5]
    pdf = dirichlet_lib.Dirichlet(alpha).prob(x)
    self.assertAllClose([1., 3. / 2], self.evaluate(pdf))
    self.assertEqual((2), pdf.get_shape())

  def testMean(self):
    alpha = [1., 2, 3]
    dirichlet = dirichlet_lib.Dirichlet(concentration=alpha)
    self.assertEqual(dirichlet.mean().get_shape(), [3])
    if not stats:
      return
    expected_mean = stats.dirichlet.mean(alpha)
    self.assertAllClose(self.evaluate(dirichlet.mean()), expected_mean)

  def testCovarianceFromSampling(self):
    alpha = np.array([[1., 2, 3],
                      [2.5, 4, 0.01]], dtype=np.float32)
    dist = dirichlet_lib.Dirichlet(alpha)  # batch_shape=[2], event_shape=[3]
    x = dist.sample(int(250e3), seed=1)
    sample_mean = math_ops.reduce_mean(x, 0)
    x_centered = x - sample_mean[None, ...]
    sample_cov = math_ops.reduce_mean(math_ops.matmul(
        x_centered[..., None], x_centered[..., None, :]), 0)
    sample_var = array_ops.matrix_diag_part(sample_cov)
    sample_stddev = math_ops.sqrt(sample_var)

    [
        sample_mean_,
        sample_cov_,
        sample_var_,
        sample_stddev_,
        analytic_mean,
        analytic_cov,
        analytic_var,
        analytic_stddev,
    ] = self.evaluate([
        sample_mean,
        sample_cov,
        sample_var,
        sample_stddev,
        dist.mean(),
        dist.covariance(),
        dist.variance(),
        dist.stddev(),
    ])

    self.assertAllClose(sample_mean_, analytic_mean, atol=0.04, rtol=0.)
    self.assertAllClose(sample_cov_, analytic_cov, atol=0.06, rtol=0.)
    self.assertAllClose(sample_var_, analytic_var, atol=0.04, rtol=0.)
    self.assertAllClose(sample_stddev_, analytic_stddev, atol=0.02, rtol=0.)

  @test_util.run_without_tensor_float_32(
      "Calls Dirichlet.covariance, which calls matmul")
  def testVariance(self):
    alpha = [1., 2, 3]
    denominator = np.sum(alpha)**2 * (np.sum(alpha) + 1)
    dirichlet = dirichlet_lib.Dirichlet(concentration=alpha)
    self.assertEqual(dirichlet.covariance().get_shape(), (3, 3))
    if not stats:
      return
    expected_covariance = np.diag(stats.dirichlet.var(alpha))
    expected_covariance += [[0., -2, -3], [-2, 0, -6], [-3, -6, 0]
                           ] / denominator
    self.assertAllClose(
        self.evaluate(dirichlet.covariance()), expected_covariance)

  def testMode(self):
    alpha = np.array([1.1, 2, 3])
    expected_mode = (alpha - 1) / (np.sum(alpha) - 3)
    dirichlet = dirichlet_lib.Dirichlet(concentration=alpha)
    self.assertEqual(dirichlet.mode().get_shape(), [3])
    self.assertAllClose(self.evaluate(dirichlet.mode()), expected_mode)

  def testModeInvalid(self):
    alpha = np.array([1., 2, 3])
    dirichlet = dirichlet_lib.Dirichlet(
        concentration=alpha, allow_nan_stats=False)
    with self.assertRaisesOpError("Condition x < y.*"):
      self.evaluate(dirichlet.mode())

  def testModeEnableAllowNanStats(self):
    alpha = np.array([1., 2, 3])
    dirichlet = dirichlet_lib.Dirichlet(
        concentration=alpha, allow_nan_stats=True)
    expected_mode = np.zeros_like(alpha) + np.nan

    self.assertEqual(dirichlet.mode().get_shape(), [3])
    self.assertAllClose(self.evaluate(dirichlet.mode()), expected_mode)

  def testEntropy(self):
    alpha = [1., 2, 3]
    dirichlet = dirichlet_lib.Dirichlet(concentration=alpha)
    self.assertEqual(dirichlet.entropy().get_shape(), ())
    if not stats:
      return
    expected_entropy = stats.dirichlet.entropy(alpha)
    self.assertAllClose(self.evaluate(dirichlet.entropy()), expected_entropy)

  def testSample(self):
    alpha = [1., 2]
    dirichlet = dirichlet_lib.Dirichlet(alpha)
    n = constant_op.constant(100000)
    samples = dirichlet.sample(n)
    sample_values = self.evaluate(samples)
    self.assertEqual(sample_values.shape, (100000, 2))
    self.assertTrue(np.all(sample_values > 0.0))
    if not stats:
      return
    self.assertLess(
        stats.kstest(
            # Beta is a univariate distribution.
            sample_values[:, 0],
            stats.beta(a=1., b=2.).cdf)[0],
        0.01)

  def testDirichletFullyReparameterized(self):
    alpha = constant_op.constant([1.0, 2.0, 3.0])
    with backprop.GradientTape() as tape:
      tape.watch(alpha)
      dirichlet = dirichlet_lib.Dirichlet(alpha)
      samples = dirichlet.sample(100)
    grad_alpha = tape.gradient(samples, alpha)
    self.assertIsNotNone(grad_alpha)

  def testDirichletDirichletKL(self):
    conc1 = np.array([[1., 2., 3., 1.5, 2.5, 3.5],
                      [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]])
    conc2 = np.array([[0.5, 1., 1.5, 2., 2.5, 3.]])

    d1 = dirichlet_lib.Dirichlet(conc1)
    d2 = dirichlet_lib.Dirichlet(conc2)
    x = d1.sample(int(1e4), seed=0)
    kl_sample = math_ops.reduce_mean(d1.log_prob(x) - d2.log_prob(x), 0)
    kl_actual = kullback_leibler.kl_divergence(d1, d2)

    kl_sample_val = self.evaluate(kl_sample)
    kl_actual_val = self.evaluate(kl_actual)

    self.assertEqual(conc1.shape[:-1], kl_actual.get_shape())

    if not special:
      return

    kl_expected = (
        special.gammaln(np.sum(conc1, -1))
        - special.gammaln(np.sum(conc2, -1))
        - np.sum(special.gammaln(conc1) - special.gammaln(conc2), -1)
        + np.sum((conc1 - conc2) * (special.digamma(conc1) - special.digamma(
            np.sum(conc1, -1, keepdims=True))), -1))

    self.assertAllClose(kl_expected, kl_actual_val, atol=0., rtol=1e-6)
    self.assertAllClose(kl_sample_val, kl_actual_val, atol=0., rtol=1e-1)

    # Make sure KL(d1||d1) is 0
    kl_same = self.evaluate(kullback_leibler.kl_divergence(d1, d1))
    self.assertAllClose(kl_same, np.zeros_like(kl_expected))


if __name__ == "__main__":
  test.main()
