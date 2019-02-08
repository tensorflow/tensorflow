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

import numpy as np
from scipy import stats
from tensorflow.contrib.distributions.python.ops import inverse_gamma
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class InverseGammaTest(test.TestCase):

  def testInverseGammaShape(self):
    with self.cached_session():
      alpha = constant_op.constant([3.0] * 5)
      beta = constant_op.constant(11.0)
      inv_gamma = inverse_gamma.InverseGamma(concentration=alpha, rate=beta)

      self.assertEqual(inv_gamma.batch_shape_tensor().eval(), (5,))
      self.assertEqual(inv_gamma.batch_shape,
                       tensor_shape.TensorShape([5]))
      self.assertAllEqual(inv_gamma.event_shape_tensor().eval(), [])
      self.assertEqual(inv_gamma.event_shape, tensor_shape.TensorShape(
          []))

  def testInverseGammaLogPDF(self):
    with self.cached_session():
      batch_size = 6
      alpha = constant_op.constant([2.0] * batch_size)
      beta = constant_op.constant([3.0] * batch_size)
      alpha_v = 2.0
      beta_v = 3.0
      x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)
      inv_gamma = inverse_gamma.InverseGamma(concentration=alpha, rate=beta)
      expected_log_pdf = stats.invgamma.logpdf(x, alpha_v, scale=beta_v)
      log_pdf = inv_gamma.log_prob(x)
      self.assertEqual(log_pdf.get_shape(), (6,))
      self.assertAllClose(log_pdf.eval(), expected_log_pdf)

      pdf = inv_gamma.prob(x)
      self.assertEqual(pdf.get_shape(), (6,))
      self.assertAllClose(pdf.eval(), np.exp(expected_log_pdf))

  def testInverseGammaLogPDFMultidimensional(self):
    with self.cached_session():
      batch_size = 6
      alpha = constant_op.constant([[2.0, 4.0]] * batch_size)
      beta = constant_op.constant([[3.0, 4.0]] * batch_size)
      alpha_v = np.array([2.0, 4.0])
      beta_v = np.array([3.0, 4.0])
      x = np.array([[2.5, 2.5, 4.0, 0.1, 1.0, 2.0]], dtype=np.float32).T
      inv_gamma = inverse_gamma.InverseGamma(concentration=alpha, rate=beta)
      expected_log_pdf = stats.invgamma.logpdf(x, alpha_v, scale=beta_v)
      log_pdf = inv_gamma.log_prob(x)
      log_pdf_values = log_pdf.eval()
      self.assertEqual(log_pdf.get_shape(), (6, 2))
      self.assertAllClose(log_pdf_values, expected_log_pdf)

      pdf = inv_gamma.prob(x)
      pdf_values = pdf.eval()
      self.assertEqual(pdf.get_shape(), (6, 2))
      self.assertAllClose(pdf_values, np.exp(expected_log_pdf))

  def testInverseGammaLogPDFMultidimensionalBroadcasting(self):
    with self.cached_session():
      batch_size = 6
      alpha = constant_op.constant([[2.0, 4.0]] * batch_size)
      beta = constant_op.constant(3.0)
      alpha_v = np.array([2.0, 4.0])
      beta_v = 3.0
      x = np.array([[2.5, 2.5, 4.0, 0.1, 1.0, 2.0]], dtype=np.float32).T
      inv_gamma = inverse_gamma.InverseGamma(concentration=alpha, rate=beta)
      expected_log_pdf = stats.invgamma.logpdf(x, alpha_v, scale=beta_v)
      log_pdf = inv_gamma.log_prob(x)
      log_pdf_values = log_pdf.eval()
      self.assertEqual(log_pdf.get_shape(), (6, 2))
      self.assertAllClose(log_pdf_values, expected_log_pdf)

      pdf = inv_gamma.prob(x)
      pdf_values = pdf.eval()
      self.assertEqual(pdf.get_shape(), (6, 2))
      self.assertAllClose(pdf_values, np.exp(expected_log_pdf))

  def testInverseGammaCDF(self):
    with self.cached_session():
      batch_size = 6
      alpha_v = 2.0
      beta_v = 3.0
      alpha = constant_op.constant([alpha_v] * batch_size)
      beta = constant_op.constant([beta_v] * batch_size)
      x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)

      inv_gamma = inverse_gamma.InverseGamma(concentration=alpha, rate=beta)
      expected_cdf = stats.invgamma.cdf(x, alpha_v, scale=beta_v)

      cdf = inv_gamma.cdf(x)
      self.assertEqual(cdf.get_shape(), (batch_size,))
      self.assertAllClose(cdf.eval(), expected_cdf)

  def testInverseGammaMode(self):
    with self.cached_session():
      alpha_v = np.array([5.5, 3.0, 2.5])
      beta_v = np.array([1.0, 4.0, 5.0])
      inv_gamma = inverse_gamma.InverseGamma(concentration=alpha_v, rate=beta_v)
      expected_modes = beta_v / (alpha_v + 1)
      self.assertEqual(inv_gamma.mode().get_shape(), (3,))
      self.assertAllClose(inv_gamma.mode().eval(), expected_modes)

  def testInverseGammaMeanAllDefined(self):
    with self.cached_session():
      alpha_v = np.array([5.5, 3.0, 2.5])
      beta_v = np.array([1.0, 4.0, 5.0])
      inv_gamma = inverse_gamma.InverseGamma(concentration=alpha_v, rate=beta_v)
      expected_means = stats.invgamma.mean(alpha_v, scale=beta_v)
      self.assertEqual(inv_gamma.mean().get_shape(), (3,))
      self.assertAllClose(inv_gamma.mean().eval(), expected_means)

  def testInverseGammaMeanAllowNanStats(self):
    with self.cached_session():
      # Mean will not be defined for the first entry.
      alpha_v = np.array([1.0, 3.0, 2.5])
      beta_v = np.array([1.0, 4.0, 5.0])
      inv_gamma = inverse_gamma.InverseGamma(
          concentration=alpha_v, rate=beta_v, allow_nan_stats=False)
      with self.assertRaisesOpError("x < y"):
        inv_gamma.mean().eval()

  def testInverseGammaMeanNanStats(self):
    with self.cached_session():
      # Mode will not be defined for the first two entries.
      alpha_v = np.array([0.5, 1.0, 3.0, 2.5])
      beta_v = np.array([1.0, 2.0, 4.0, 5.0])
      inv_gamma = inverse_gamma.InverseGamma(
          concentration=alpha_v, rate=beta_v, allow_nan_stats=True)
      expected_means = beta_v / (alpha_v - 1)
      expected_means[0] = np.nan
      expected_means[1] = np.nan
      self.assertEqual(inv_gamma.mean().get_shape(), (4,))
      self.assertAllClose(inv_gamma.mean().eval(), expected_means)

  def testInverseGammaVarianceAllDefined(self):
    with self.cached_session():
      alpha_v = np.array([7.0, 3.0, 2.5])
      beta_v = np.array([1.0, 4.0, 5.0])
      inv_gamma = inverse_gamma.InverseGamma(concentration=alpha_v, rate=beta_v)
      expected_variances = stats.invgamma.var(alpha_v, scale=beta_v)
      self.assertEqual(inv_gamma.variance().get_shape(), (3,))
      self.assertAllClose(inv_gamma.variance().eval(), expected_variances)

  def testInverseGammaVarianceAllowNanStats(self):
    with self.cached_session():
      alpha_v = np.array([1.5, 3.0, 2.5])
      beta_v = np.array([1.0, 4.0, 5.0])
      inv_gamma = inverse_gamma.InverseGamma(
          concentration=alpha_v, rate=beta_v, allow_nan_stats=False)
      with self.assertRaisesOpError("x < y"):
        inv_gamma.variance().eval()

  def testInverseGammaVarianceNanStats(self):
    with self.cached_session():
      alpha_v = np.array([1.5, 3.0, 2.5])
      beta_v = np.array([1.0, 4.0, 5.0])
      inv_gamma = inverse_gamma.InverseGamma(
          concentration=alpha_v, rate=beta_v, allow_nan_stats=True)
      expected_variances = stats.invgamma.var(alpha_v, scale=beta_v)
      expected_variances[0] = np.nan
      self.assertEqual(inv_gamma.variance().get_shape(), (3,))
      self.assertAllClose(inv_gamma.variance().eval(), expected_variances)

  def testInverseGammaEntropy(self):
    with self.cached_session():
      alpha_v = np.array([1.0, 3.0, 2.5])
      beta_v = np.array([1.0, 4.0, 5.0])
      expected_entropy = stats.invgamma.entropy(alpha_v, scale=beta_v)
      inv_gamma = inverse_gamma.InverseGamma(concentration=alpha_v, rate=beta_v)
      self.assertEqual(inv_gamma.entropy().get_shape(), (3,))
      self.assertAllClose(inv_gamma.entropy().eval(), expected_entropy)

  def testInverseGammaSample(self):
    with session.Session():
      alpha_v = 4.0
      beta_v = 3.0
      alpha = constant_op.constant(alpha_v)
      beta = constant_op.constant(beta_v)
      n = 100000
      inv_gamma = inverse_gamma.InverseGamma(concentration=alpha, rate=beta)
      samples = inv_gamma.sample(n, seed=137)
      sample_values = samples.eval()
      self.assertEqual(samples.get_shape(), (n,))
      self.assertEqual(sample_values.shape, (n,))
      self.assertAllClose(
          sample_values.mean(),
          stats.invgamma.mean(
              alpha_v, scale=beta_v),
          atol=.0025)
      self.assertAllClose(
          sample_values.var(),
          stats.invgamma.var(alpha_v, scale=beta_v),
          atol=.15)
      self.assertTrue(self._kstest(alpha_v, beta_v, sample_values))

  def testInverseGammaSampleMultiDimensional(self):
    with session.Session():
      alpha_v = np.array([np.arange(3, 103, dtype=np.float32)])  # 1 x 100
      beta_v = np.array([np.arange(1, 11, dtype=np.float32)]).T  # 10 x 1
      inv_gamma = inverse_gamma.InverseGamma(concentration=alpha_v, rate=beta_v)
      n = 10000
      samples = inv_gamma.sample(n, seed=137)
      sample_values = samples.eval()
      self.assertEqual(samples.get_shape(), (n, 10, 100))
      self.assertEqual(sample_values.shape, (n, 10, 100))
      zeros = np.zeros_like(alpha_v + beta_v)  # 10 x 100
      alpha_bc = alpha_v + zeros
      beta_bc = beta_v + zeros
      self.assertAllClose(
          sample_values.mean(axis=0),
          stats.invgamma.mean(
              alpha_bc, scale=beta_bc),
          atol=.25)
      self.assertAllClose(
          sample_values.var(axis=0),
          stats.invgamma.var(alpha_bc, scale=beta_bc),
          atol=4.5)
      fails = 0
      trials = 0
      for ai, a in enumerate(np.reshape(alpha_v, [-1])):
        for bi, b in enumerate(np.reshape(beta_v, [-1])):
          s = sample_values[:, bi, ai]
          trials += 1
          fails += 0 if self._kstest(a, b, s) else 1
      self.assertLess(fails, trials * 0.03)

  def _kstest(self, alpha, beta, samples):
    # Uses the Kolmogorov-Smirnov test for goodness of fit.
    ks, _ = stats.kstest(samples, stats.invgamma(alpha, scale=beta).cdf)
    # Return True when the test passes.
    return ks < 0.02

  def testInverseGammaPdfOfSampleMultiDims(self):
    with session.Session() as sess:
      inv_gamma = inverse_gamma.InverseGamma(
          concentration=[7., 11.],
          rate=[[5.], [6.]])
      num = 50000
      samples = inv_gamma.sample(num, seed=137)
      pdfs = inv_gamma.prob(samples)
      sample_vals, pdf_vals = sess.run([samples, pdfs])
      self.assertEqual(samples.get_shape(), (num, 2, 2))
      self.assertEqual(pdfs.get_shape(), (num, 2, 2))
      self.assertAllClose(
          stats.invgamma.mean(
              [[7., 11.], [7., 11.]], scale=np.array([[5., 5.], [6., 6.]])),
          sample_vals.mean(axis=0),
          atol=.1)
      self.assertAllClose(
          stats.invgamma.var([[7., 11.], [7., 11.]],
                             scale=np.array([[5., 5.], [6., 6.]])),
          sample_vals.var(axis=0),
          atol=.1)
      self._assertIntegral(sample_vals[:, 0, 0], pdf_vals[:, 0, 0], err=0.02)
      self._assertIntegral(sample_vals[:, 0, 1], pdf_vals[:, 0, 1], err=0.02)
      self._assertIntegral(sample_vals[:, 1, 0], pdf_vals[:, 1, 0], err=0.02)
      self._assertIntegral(sample_vals[:, 1, 1], pdf_vals[:, 1, 1], err=0.02)

  def _assertIntegral(self, sample_vals, pdf_vals, err=1e-3):
    s_p = zip(sample_vals, pdf_vals)
    prev = (0, 0)
    total = 0
    for k in sorted(s_p, key=lambda x: x[0]):
      pair_pdf = (k[1] + prev[1]) / 2
      total += (k[0] - prev[0]) * pair_pdf
      prev = k
    self.assertNear(1., total, err=err)

  def testInverseGammaNonPositiveInitializationParamsRaises(self):
    with self.cached_session():
      alpha_v = constant_op.constant(0.0, name="alpha")
      beta_v = constant_op.constant(1.0, name="beta")
      inv_gamma = inverse_gamma.InverseGamma(
          concentration=alpha_v, rate=beta_v, validate_args=True)
      with self.assertRaisesOpError("alpha"):
        inv_gamma.mean().eval()
      alpha_v = constant_op.constant(1.0, name="alpha")
      beta_v = constant_op.constant(0.0, name="beta")
      inv_gamma = inverse_gamma.InverseGamma(
          concentration=alpha_v, rate=beta_v, validate_args=True)
      with self.assertRaisesOpError("beta"):
        inv_gamma.mean().eval()

  def testInverseGammaWithSoftplusConcentrationRate(self):
    with self.cached_session():
      alpha = constant_op.constant([-0.1, -2.9], name="alpha")
      beta = constant_op.constant([1.0, -4.8], name="beta")
      inv_gamma = inverse_gamma.InverseGammaWithSoftplusConcentrationRate(
          concentration=alpha, rate=beta, validate_args=True)
      self.assertAllClose(nn_ops.softplus(alpha).eval(),
                          inv_gamma.concentration.eval())
      self.assertAllClose(nn_ops.softplus(beta).eval(),
                          inv_gamma.rate.eval())


if __name__ == "__main__":
  test.main()
