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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import dirichlet as dirichlet_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def try_import(name):  # pylint: disable=invalid-name
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf_logging.warning("Could not import %s: %s" % (name, str(e)))
  return module


stats = try_import("scipy.stats")


class DirichletTest(test.TestCase):

  def testSimpleShapes(self):
    with self.test_session():
      alpha = np.random.rand(3)
      dist = dirichlet_lib.Dirichlet(alpha)
      self.assertEqual(3, dist.event_shape_tensor().eval())
      self.assertAllEqual([], dist.batch_shape_tensor().eval())
      self.assertEqual(tensor_shape.TensorShape([3]), dist.event_shape)
      self.assertEqual(tensor_shape.TensorShape([]), dist.batch_shape)

  def testComplexShapes(self):
    with self.test_session():
      alpha = np.random.rand(3, 2, 2)
      dist = dirichlet_lib.Dirichlet(alpha)
      self.assertEqual(2, dist.event_shape_tensor().eval())
      self.assertAllEqual([3, 2], dist.batch_shape_tensor().eval())
      self.assertEqual(tensor_shape.TensorShape([2]), dist.event_shape)
      self.assertEqual(tensor_shape.TensorShape([3, 2]), dist.batch_shape)

  def testConcentrationProperty(self):
    alpha = [[1., 2, 3]]
    with self.test_session():
      dist = dirichlet_lib.Dirichlet(alpha)
      self.assertEqual([1, 3], dist.concentration.get_shape())
      self.assertAllClose(alpha, dist.concentration.eval())

  def testPdfXProper(self):
    alpha = [[1., 2, 3]]
    with self.test_session():
      dist = dirichlet_lib.Dirichlet(alpha, validate_args=True)
      dist.prob([.1, .3, .6]).eval()
      dist.prob([.2, .3, .5]).eval()
      # Either condition can trigger.
      with self.assertRaisesOpError("samples must be positive"):
        dist.prob([-1., 1.5, 0.5]).eval()
      with self.assertRaisesOpError("samples must be positive"):
        dist.prob([0., .1, .9]).eval()
      with self.assertRaisesOpError(
          "sample last-dimension must sum to `1`"):
        dist.prob([.1, .2, .8]).eval()

  def testPdfZeroBatches(self):
    with self.test_session():
      alpha = [1., 2]
      x = [.5, .5]
      dist = dirichlet_lib.Dirichlet(alpha)
      pdf = dist.prob(x)
      self.assertAllClose(1., pdf.eval())
      self.assertEqual((), pdf.get_shape())

  def testPdfZeroBatchesNontrivialX(self):
    with self.test_session():
      alpha = [1., 2]
      x = [.3, .7]
      dist = dirichlet_lib.Dirichlet(alpha)
      pdf = dist.prob(x)
      self.assertAllClose(7. / 5, pdf.eval())
      self.assertEqual((), pdf.get_shape())

  def testPdfUniformZeroBatches(self):
    with self.test_session():
      # Corresponds to a uniform distribution
      alpha = [1., 1, 1]
      x = [[.2, .5, .3], [.3, .4, .3]]
      dist = dirichlet_lib.Dirichlet(alpha)
      pdf = dist.prob(x)
      self.assertAllClose([2., 2.], pdf.eval())
      self.assertEqual((2), pdf.get_shape())

  def testPdfAlphaStretchedInBroadcastWhenSameRank(self):
    with self.test_session():
      alpha = [[1., 2]]
      x = [[.5, .5], [.3, .7]]
      dist = dirichlet_lib.Dirichlet(alpha)
      pdf = dist.prob(x)
      self.assertAllClose([1., 7. / 5], pdf.eval())
      self.assertEqual((2), pdf.get_shape())

  def testPdfAlphaStretchedInBroadcastWhenLowerRank(self):
    with self.test_session():
      alpha = [1., 2]
      x = [[.5, .5], [.2, .8]]
      pdf = dirichlet_lib.Dirichlet(alpha).prob(x)
      self.assertAllClose([1., 8. / 5], pdf.eval())
      self.assertEqual((2), pdf.get_shape())

  def testPdfXStretchedInBroadcastWhenSameRank(self):
    with self.test_session():
      alpha = [[1., 2], [2., 3]]
      x = [[.5, .5]]
      pdf = dirichlet_lib.Dirichlet(alpha).prob(x)
      self.assertAllClose([1., 3. / 2], pdf.eval())
      self.assertEqual((2), pdf.get_shape())

  def testPdfXStretchedInBroadcastWhenLowerRank(self):
    with self.test_session():
      alpha = [[1., 2], [2., 3]]
      x = [.5, .5]
      pdf = dirichlet_lib.Dirichlet(alpha).prob(x)
      self.assertAllClose([1., 3. / 2], pdf.eval())
      self.assertEqual((2), pdf.get_shape())

  def testMean(self):
    with self.test_session():
      alpha = [1., 2, 3]
      dirichlet = dirichlet_lib.Dirichlet(concentration=alpha)
      self.assertEqual(dirichlet.mean().get_shape(), [3])
      if not stats:
        return
      expected_mean = stats.dirichlet.mean(alpha)
      self.assertAllClose(dirichlet.mean().eval(), expected_mean)

  def testCovarianceFromSampling(self):
    alpha = np.array([[1., 2, 3],
                      [2.5, 4, 0.01]], dtype=np.float32)
    with self.test_session() as sess:
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
      ] = sess.run([
          sample_mean,
          sample_cov,
          sample_var,
          sample_stddev,
          dist.mean(),
          dist.covariance(),
          dist.variance(),
          dist.stddev(),
      ])
      self.assertAllClose(sample_mean_, analytic_mean, atol=0., rtol=0.04)
      self.assertAllClose(sample_cov_, analytic_cov, atol=0., rtol=0.06)
      self.assertAllClose(sample_var_, analytic_var, atol=0., rtol=0.03)
      self.assertAllClose(sample_stddev_, analytic_stddev, atol=0., rtol=0.02)

  def testVariance(self):
    with self.test_session():
      alpha = [1., 2, 3]
      denominator = np.sum(alpha)**2 * (np.sum(alpha) + 1)
      dirichlet = dirichlet_lib.Dirichlet(concentration=alpha)
      self.assertEqual(dirichlet.covariance().get_shape(), (3, 3))
      if not stats:
        return
      expected_covariance = np.diag(stats.dirichlet.var(alpha))
      expected_covariance += [[0., -2, -3], [-2, 0, -6],
                              [-3, -6, 0]] / denominator
      self.assertAllClose(dirichlet.covariance().eval(), expected_covariance)

  def testMode(self):
    with self.test_session():
      alpha = np.array([1.1, 2, 3])
      expected_mode = (alpha - 1) / (np.sum(alpha) - 3)
      dirichlet = dirichlet_lib.Dirichlet(concentration=alpha)
      self.assertEqual(dirichlet.mode().get_shape(), [3])
      self.assertAllClose(dirichlet.mode().eval(), expected_mode)

  def testModeInvalid(self):
    with self.test_session():
      alpha = np.array([1., 2, 3])
      dirichlet = dirichlet_lib.Dirichlet(concentration=alpha,
                                          allow_nan_stats=False)
      with self.assertRaisesOpError("Condition x < y.*"):
        dirichlet.mode().eval()

  def testModeEnableAllowNanStats(self):
    with self.test_session():
      alpha = np.array([1., 2, 3])
      dirichlet = dirichlet_lib.Dirichlet(concentration=alpha,
                                          allow_nan_stats=True)
      expected_mode = np.zeros_like(alpha) + np.nan

      self.assertEqual(dirichlet.mode().get_shape(), [3])
      self.assertAllClose(dirichlet.mode().eval(), expected_mode)

  def testEntropy(self):
    with self.test_session():
      alpha = [1., 2, 3]
      dirichlet = dirichlet_lib.Dirichlet(concentration=alpha)
      self.assertEqual(dirichlet.entropy().get_shape(), ())
      if not stats:
        return
      expected_entropy = stats.dirichlet.entropy(alpha)
      self.assertAllClose(dirichlet.entropy().eval(), expected_entropy)

  def testSample(self):
    with self.test_session():
      alpha = [1., 2]
      dirichlet = dirichlet_lib.Dirichlet(alpha)
      n = constant_op.constant(100000)
      samples = dirichlet.sample(n)
      sample_values = samples.eval()
      self.assertEqual(sample_values.shape, (100000, 2))
      self.assertTrue(np.all(sample_values > 0.0))
      if not stats:
        return
      self.assertLess(
          stats.kstest(
              # Beta is a univariate distribution.
              sample_values[:, 0],
              stats.beta(
                  a=1., b=2.).cdf)[0],
          0.01)


if __name__ == "__main__":
  test.main()
