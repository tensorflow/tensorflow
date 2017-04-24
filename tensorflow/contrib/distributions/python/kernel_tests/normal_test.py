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
"""Tests for initializers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
from scipy import stats

from tensorflow.contrib.distributions.python.ops import kullback_leibler
from tensorflow.contrib.distributions.python.ops import normal as normal_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class NormalTest(test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(123)

  def assertAllFinite(self, tensor):
    is_finite = np.isfinite(tensor.eval())
    all_true = np.ones_like(is_finite, dtype=np.bool)
    self.assertAllEqual(all_true, is_finite)

  def _testParamShapes(self, sample_shape, expected):
    with self.test_session():
      param_shapes = normal_lib.Normal.param_shapes(sample_shape)
      mu_shape, sigma_shape = param_shapes["loc"], param_shapes["scale"]
      self.assertAllEqual(expected, mu_shape.eval())
      self.assertAllEqual(expected, sigma_shape.eval())
      mu = array_ops.zeros(mu_shape)
      sigma = array_ops.ones(sigma_shape)
      self.assertAllEqual(
          expected,
          array_ops.shape(normal_lib.Normal(mu, sigma).sample()).eval())

  def _testParamStaticShapes(self, sample_shape, expected):
    param_shapes = normal_lib.Normal.param_static_shapes(sample_shape)
    mu_shape, sigma_shape = param_shapes["loc"], param_shapes["scale"]
    self.assertEqual(expected, mu_shape)
    self.assertEqual(expected, sigma_shape)

  def testParamShapes(self):
    sample_shape = [10, 3, 4]
    self._testParamShapes(sample_shape, sample_shape)
    self._testParamShapes(constant_op.constant(sample_shape), sample_shape)

  def testParamStaticShapes(self):
    sample_shape = [10, 3, 4]
    self._testParamStaticShapes(sample_shape, sample_shape)
    self._testParamStaticShapes(
        tensor_shape.TensorShape(sample_shape), sample_shape)

  def testNormalWithSoftplusScale(self):
    with self.test_session():
      mu = array_ops.zeros((10, 3))
      rho = array_ops.ones((10, 3)) * -2.
      normal = normal_lib.NormalWithSoftplusScale(loc=mu, scale=rho)
      self.assertAllEqual(mu.eval(), normal.loc.eval())
      self.assertAllEqual(nn_ops.softplus(rho).eval(), normal.scale.eval())

  def testNormalLogPDF(self):
    with self.test_session():
      batch_size = 6
      mu = constant_op.constant([3.0] * batch_size)
      sigma = constant_op.constant([math.sqrt(10.0)] * batch_size)
      x = np.array([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0], dtype=np.float32)
      normal = normal_lib.Normal(loc=mu, scale=sigma)
      expected_log_pdf = stats.norm(mu.eval(), sigma.eval()).logpdf(x)

      log_pdf = normal.log_prob(x)
      self.assertAllClose(expected_log_pdf, log_pdf.eval())
      self.assertAllEqual(normal.batch_shape_tensor().eval(),
                          log_pdf.get_shape())
      self.assertAllEqual(normal.batch_shape_tensor().eval(),
                          log_pdf.eval().shape)
      self.assertAllEqual(normal.batch_shape, log_pdf.get_shape())
      self.assertAllEqual(normal.batch_shape, log_pdf.eval().shape)

      pdf = normal.prob(x)
      self.assertAllClose(np.exp(expected_log_pdf), pdf.eval())
      self.assertAllEqual(normal.batch_shape_tensor().eval(), pdf.get_shape())
      self.assertAllEqual(normal.batch_shape_tensor().eval(), pdf.eval().shape)
      self.assertAllEqual(normal.batch_shape, pdf.get_shape())
      self.assertAllEqual(normal.batch_shape, pdf.eval().shape)

  def testNormalLogPDFMultidimensional(self):
    with self.test_session():
      batch_size = 6
      mu = constant_op.constant([[3.0, -3.0]] * batch_size)
      sigma = constant_op.constant([[math.sqrt(10.0), math.sqrt(15.0)]] *
                                   batch_size)
      x = np.array([[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0]], dtype=np.float32).T
      normal = normal_lib.Normal(loc=mu, scale=sigma)
      expected_log_pdf = stats.norm(mu.eval(), sigma.eval()).logpdf(x)

      log_pdf = normal.log_prob(x)
      log_pdf_values = log_pdf.eval()
      self.assertEqual(log_pdf.get_shape(), (6, 2))
      self.assertAllClose(expected_log_pdf, log_pdf_values)
      self.assertAllEqual(normal.batch_shape_tensor().eval(),
                          log_pdf.get_shape())
      self.assertAllEqual(normal.batch_shape_tensor().eval(),
                          log_pdf.eval().shape)
      self.assertAllEqual(normal.batch_shape, log_pdf.get_shape())
      self.assertAllEqual(normal.batch_shape, log_pdf.eval().shape)

      pdf = normal.prob(x)
      pdf_values = pdf.eval()
      self.assertEqual(pdf.get_shape(), (6, 2))
      self.assertAllClose(np.exp(expected_log_pdf), pdf_values)
      self.assertAllEqual(normal.batch_shape_tensor().eval(), pdf.get_shape())
      self.assertAllEqual(normal.batch_shape_tensor().eval(), pdf_values.shape)
      self.assertAllEqual(normal.batch_shape, pdf.get_shape())
      self.assertAllEqual(normal.batch_shape, pdf_values.shape)

  def testNormalCDF(self):
    with self.test_session():
      batch_size = 50
      mu = self._rng.randn(batch_size)
      sigma = self._rng.rand(batch_size) + 1.0
      x = np.linspace(-8.0, 8.0, batch_size).astype(np.float64)

      normal = normal_lib.Normal(loc=mu, scale=sigma)
      expected_cdf = stats.norm(mu, sigma).cdf(x)

      cdf = normal.cdf(x)
      self.assertAllClose(expected_cdf, cdf.eval(), atol=0)
      self.assertAllEqual(normal.batch_shape_tensor().eval(), cdf.get_shape())
      self.assertAllEqual(normal.batch_shape_tensor().eval(), cdf.eval().shape)
      self.assertAllEqual(normal.batch_shape, cdf.get_shape())
      self.assertAllEqual(normal.batch_shape, cdf.eval().shape)

  def testNormalSurvivalFunction(self):
    with self.test_session():
      batch_size = 50
      mu = self._rng.randn(batch_size)
      sigma = self._rng.rand(batch_size) + 1.0
      x = np.linspace(-8.0, 8.0, batch_size).astype(np.float64)

      normal = normal_lib.Normal(loc=mu, scale=sigma)
      expected_sf = stats.norm(mu, sigma).sf(x)

      sf = normal.survival_function(x)
      self.assertAllClose(expected_sf, sf.eval(), atol=0)
      self.assertAllEqual(normal.batch_shape_tensor().eval(), sf.get_shape())
      self.assertAllEqual(normal.batch_shape_tensor().eval(), sf.eval().shape)
      self.assertAllEqual(normal.batch_shape, sf.get_shape())
      self.assertAllEqual(normal.batch_shape, sf.eval().shape)

  def testNormalLogCDF(self):
    with self.test_session():
      batch_size = 50
      mu = self._rng.randn(batch_size)
      sigma = self._rng.rand(batch_size) + 1.0
      x = np.linspace(-100.0, 10.0, batch_size).astype(np.float64)

      normal = normal_lib.Normal(loc=mu, scale=sigma)
      expected_cdf = stats.norm(mu, sigma).logcdf(x)

      cdf = normal.log_cdf(x)
      self.assertAllClose(expected_cdf, cdf.eval(), atol=0, rtol=1e-5)
      self.assertAllEqual(normal.batch_shape_tensor().eval(), cdf.get_shape())
      self.assertAllEqual(normal.batch_shape_tensor().eval(), cdf.eval().shape)
      self.assertAllEqual(normal.batch_shape, cdf.get_shape())
      self.assertAllEqual(normal.batch_shape, cdf.eval().shape)

  def testFiniteGradientAtDifficultPoints(self):
    for dtype in [np.float32, np.float64]:
      g = ops.Graph()
      with g.as_default():
        mu = variables.Variable(dtype(0.0))
        sigma = variables.Variable(dtype(1.0))
        dist = normal_lib.Normal(loc=mu, scale=sigma)
        x = np.array([-100., -20., -5., 0., 5., 20., 100.]).astype(dtype)
        for func in [
            dist.cdf, dist.log_cdf, dist.survival_function,
            dist.log_survival_function, dist.log_prob, dist.prob
        ]:
          value = func(x)
          grads = gradients_impl.gradients(value, [mu, sigma])
          with self.test_session(graph=g):
            variables.global_variables_initializer().run()
            self.assertAllFinite(value)
            self.assertAllFinite(grads[0])
            self.assertAllFinite(grads[1])

  def testNormalLogSurvivalFunction(self):
    with self.test_session():
      batch_size = 50
      mu = self._rng.randn(batch_size)
      sigma = self._rng.rand(batch_size) + 1.0
      x = np.linspace(-10.0, 100.0, batch_size).astype(np.float64)

      normal = normal_lib.Normal(loc=mu, scale=sigma)
      expected_sf = stats.norm(mu, sigma).logsf(x)

      sf = normal.log_survival_function(x)
      self.assertAllClose(expected_sf, sf.eval(), atol=0, rtol=1e-5)
      self.assertAllEqual(normal.batch_shape_tensor().eval(), sf.get_shape())
      self.assertAllEqual(normal.batch_shape_tensor().eval(), sf.eval().shape)
      self.assertAllEqual(normal.batch_shape, sf.get_shape())
      self.assertAllEqual(normal.batch_shape, sf.eval().shape)

  def testNormalEntropyWithScalarInputs(self):
    # Scipy.stats.norm cannot deal with the shapes in the other test.
    with self.test_session():
      mu_v = 2.34
      sigma_v = 4.56
      normal = normal_lib.Normal(loc=mu_v, scale=sigma_v)

      # scipy.stats.norm cannot deal with these shapes.
      expected_entropy = stats.norm(mu_v, sigma_v).entropy()
      entropy = normal.entropy()
      self.assertAllClose(expected_entropy, entropy.eval())
      self.assertAllEqual(normal.batch_shape_tensor().eval(),
                          entropy.get_shape())
      self.assertAllEqual(normal.batch_shape_tensor().eval(),
                          entropy.eval().shape)
      self.assertAllEqual(normal.batch_shape, entropy.get_shape())
      self.assertAllEqual(normal.batch_shape, entropy.eval().shape)

  def testNormalEntropy(self):
    with self.test_session():
      mu_v = np.array([1.0, 1.0, 1.0])
      sigma_v = np.array([[1.0, 2.0, 3.0]]).T
      normal = normal_lib.Normal(loc=mu_v, scale=sigma_v)

      # scipy.stats.norm cannot deal with these shapes.
      sigma_broadcast = mu_v * sigma_v
      expected_entropy = 0.5 * np.log(2 * np.pi * np.exp(1) * sigma_broadcast**
                                      2)
      entropy = normal.entropy()
      np.testing.assert_allclose(expected_entropy, entropy.eval())
      self.assertAllEqual(normal.batch_shape_tensor().eval(),
                          entropy.get_shape())
      self.assertAllEqual(normal.batch_shape_tensor().eval(),
                          entropy.eval().shape)
      self.assertAllEqual(normal.batch_shape, entropy.get_shape())
      self.assertAllEqual(normal.batch_shape, entropy.eval().shape)

  def testNormalMeanAndMode(self):
    with self.test_session():
      # Mu will be broadcast to [7, 7, 7].
      mu = [7.]
      sigma = [11., 12., 13.]

      normal = normal_lib.Normal(loc=mu, scale=sigma)

      self.assertAllEqual((3,), normal.mean().get_shape())
      self.assertAllEqual([7., 7, 7], normal.mean().eval())

      self.assertAllEqual((3,), normal.mode().get_shape())
      self.assertAllEqual([7., 7, 7], normal.mode().eval())

  def testNormalQuantile(self):
    with self.test_session():
      batch_size = 52
      mu = self._rng.randn(batch_size)
      sigma = self._rng.rand(batch_size) + 1.0
      p = np.linspace(0., 1.0, batch_size - 2).astype(np.float64)
      # Quantile performs piecewise rational approximation so adding some
      # special input values to make sure we hit all the pieces.
      p = np.hstack((p, np.exp(-33), 1. - np.exp(-33)))

      normal = normal_lib.Normal(loc=mu, scale=sigma)
      expected_x = stats.norm(mu, sigma).ppf(p)
      x = normal.quantile(p)

      self.assertAllClose(expected_x, x.eval(), atol=0.)
      self.assertAllEqual(normal.batch_shape_tensor().eval(), x.get_shape())
      self.assertAllEqual(normal.batch_shape_tensor().eval(), x.eval().shape)
      self.assertAllEqual(normal.batch_shape, x.get_shape())
      self.assertAllEqual(normal.batch_shape, x.eval().shape)

  def _baseQuantileFiniteGradientAtDifficultPoints(self, dtype):
    g = ops.Graph()
    with g.as_default():
      mu = variables.Variable(dtype(0.0))
      sigma = variables.Variable(dtype(1.0))
      dist = normal_lib.Normal(loc=mu, scale=sigma)
      p = variables.Variable(
          np.array([0.,
                    np.exp(-32.), np.exp(-2.),
                    1. - np.exp(-2.), 1. - np.exp(-32.),
                    1.]).astype(dtype))

      value = dist.quantile(p)
      grads = gradients_impl.gradients(value, [mu, p])
      with self.test_session(graph=g):
        variables.global_variables_initializer().run()
        self.assertAllFinite(grads[0])
        self.assertAllFinite(grads[1])

  def testQuantileFiniteGradientAtDifficultPointsFloat32(self):
    self._baseQuantileFiniteGradientAtDifficultPoints(np.float32)

  def testQuantileFiniteGradientAtDifficultPointsFloat64(self):
    self._baseQuantileFiniteGradientAtDifficultPoints(np.float64)

  def testNormalVariance(self):
    with self.test_session():
      # sigma will be broadcast to [7, 7, 7]
      mu = [1., 2., 3.]
      sigma = [7.]

      normal = normal_lib.Normal(loc=mu, scale=sigma)

      self.assertAllEqual((3,), normal.variance().get_shape())
      self.assertAllEqual([49., 49, 49], normal.variance().eval())

  def testNormalStandardDeviation(self):
    with self.test_session():
      # sigma will be broadcast to [7, 7, 7]
      mu = [1., 2., 3.]
      sigma = [7.]

      normal = normal_lib.Normal(loc=mu, scale=sigma)

      self.assertAllEqual((3,), normal.stddev().get_shape())
      self.assertAllEqual([7., 7, 7], normal.stddev().eval())

  def testNormalSample(self):
    with self.test_session():
      mu = constant_op.constant(3.0)
      sigma = constant_op.constant(math.sqrt(3.0))
      mu_v = 3.0
      sigma_v = np.sqrt(3.0)
      n = constant_op.constant(100000)
      normal = normal_lib.Normal(loc=mu, scale=sigma)
      samples = normal.sample(n)
      sample_values = samples.eval()
      # Note that the standard error for the sample mean is ~ sigma / sqrt(n).
      # The sample variance similarly is dependent on sigma and n.
      # Thus, the tolerances below are very sensitive to number of samples
      # as well as the variances chosen.
      self.assertEqual(sample_values.shape, (100000,))
      self.assertAllClose(sample_values.mean(), mu_v, atol=1e-1)
      self.assertAllClose(sample_values.std(), sigma_v, atol=1e-1)

      expected_samples_shape = tensor_shape.TensorShape([n.eval()]).concatenate(
          tensor_shape.TensorShape(normal.batch_shape_tensor().eval()))

      self.assertAllEqual(expected_samples_shape, samples.get_shape())
      self.assertAllEqual(expected_samples_shape, sample_values.shape)

      expected_samples_shape = (tensor_shape.TensorShape(
          [n.eval()]).concatenate(normal.batch_shape))

      self.assertAllEqual(expected_samples_shape, samples.get_shape())
      self.assertAllEqual(expected_samples_shape, sample_values.shape)

  def testNormalSampleMultiDimensional(self):
    with self.test_session():
      batch_size = 2
      mu = constant_op.constant([[3.0, -3.0]] * batch_size)
      sigma = constant_op.constant([[math.sqrt(2.0), math.sqrt(3.0)]] *
                                   batch_size)
      mu_v = [3.0, -3.0]
      sigma_v = [np.sqrt(2.0), np.sqrt(3.0)]
      n = constant_op.constant(100000)
      normal = normal_lib.Normal(loc=mu, scale=sigma)
      samples = normal.sample(n)
      sample_values = samples.eval()
      # Note that the standard error for the sample mean is ~ sigma / sqrt(n).
      # The sample variance similarly is dependent on sigma and n.
      # Thus, the tolerances below are very sensitive to number of samples
      # as well as the variances chosen.
      self.assertEqual(samples.get_shape(), (100000, batch_size, 2))
      self.assertAllClose(sample_values[:, 0, 0].mean(), mu_v[0], atol=1e-1)
      self.assertAllClose(sample_values[:, 0, 0].std(), sigma_v[0], atol=1e-1)
      self.assertAllClose(sample_values[:, 0, 1].mean(), mu_v[1], atol=1e-1)
      self.assertAllClose(sample_values[:, 0, 1].std(), sigma_v[1], atol=1e-1)

      expected_samples_shape = tensor_shape.TensorShape([n.eval()]).concatenate(
          tensor_shape.TensorShape(normal.batch_shape_tensor().eval()))
      self.assertAllEqual(expected_samples_shape, samples.get_shape())
      self.assertAllEqual(expected_samples_shape, sample_values.shape)

      expected_samples_shape = (tensor_shape.TensorShape(
          [n.eval()]).concatenate(normal.batch_shape))
      self.assertAllEqual(expected_samples_shape, samples.get_shape())
      self.assertAllEqual(expected_samples_shape, sample_values.shape)

  def testNegativeSigmaFails(self):
    with self.test_session():
      normal = normal_lib.Normal(
          loc=[1.], scale=[-5.], validate_args=True, name="G")
      with self.assertRaisesOpError("Condition x > 0 did not hold"):
        normal.mean().eval()

  def testNormalShape(self):
    with self.test_session():
      mu = constant_op.constant([-3.0] * 5)
      sigma = constant_op.constant(11.0)
      normal = normal_lib.Normal(loc=mu, scale=sigma)

      self.assertEqual(normal.batch_shape_tensor().eval(), [5])
      self.assertEqual(normal.batch_shape, tensor_shape.TensorShape([5]))
      self.assertAllEqual(normal.event_shape_tensor().eval(), [])
      self.assertEqual(normal.event_shape, tensor_shape.TensorShape([]))

  def testNormalShapeWithPlaceholders(self):
    mu = array_ops.placeholder(dtype=dtypes.float32)
    sigma = array_ops.placeholder(dtype=dtypes.float32)
    normal = normal_lib.Normal(loc=mu, scale=sigma)

    with self.test_session() as sess:
      # get_batch_shape should return an "<unknown>" tensor.
      self.assertEqual(normal.batch_shape, tensor_shape.TensorShape(None))
      self.assertEqual(normal.event_shape, ())
      self.assertAllEqual(normal.event_shape_tensor().eval(), [])
      self.assertAllEqual(
          sess.run(normal.batch_shape_tensor(),
                   feed_dict={mu: 5.0,
                              sigma: [1.0, 2.0]}), [2])

  def testNormalNormalKL(self):
    with self.test_session() as sess:
      batch_size = 6
      mu_a = np.array([3.0] * batch_size)
      sigma_a = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
      mu_b = np.array([-3.0] * batch_size)
      sigma_b = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

      n_a = normal_lib.Normal(loc=mu_a, scale=sigma_a)
      n_b = normal_lib.Normal(loc=mu_b, scale=sigma_b)

      kl = kullback_leibler.kl(n_a, n_b)
      kl_val = sess.run(kl)

      kl_expected = ((mu_a - mu_b)**2 / (2 * sigma_b**2) + 0.5 * (
          (sigma_a**2 / sigma_b**2) - 1 - 2 * np.log(sigma_a / sigma_b)))

      self.assertEqual(kl.get_shape(), (batch_size,))
      self.assertAllClose(kl_val, kl_expected)


if __name__ == "__main__":
  test.main()
