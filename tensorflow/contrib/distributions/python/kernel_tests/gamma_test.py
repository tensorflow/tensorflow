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
from tensorflow.contrib.distributions.python.ops import gamma as gamma_lib
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class GammaTest(test.TestCase):

  def testGammaShape(self):
    with self.test_session():
      alpha = constant_op.constant([3.0] * 5)
      beta = constant_op.constant(11.0)
      gamma = gamma_lib.Gamma(alpha=alpha, beta=beta)

      self.assertEqual(gamma.batch_shape().eval(), (5,))
      self.assertEqual(gamma.get_batch_shape(), tensor_shape.TensorShape([5]))
      self.assertAllEqual(gamma.event_shape().eval(), [])
      self.assertEqual(gamma.get_event_shape(), tensor_shape.TensorShape([]))

  def testGammaLogPDF(self):
    with self.test_session():
      batch_size = 6
      alpha = constant_op.constant([2.0] * batch_size)
      beta = constant_op.constant([3.0] * batch_size)
      alpha_v = 2.0
      beta_v = 3.0
      x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)
      gamma = gamma_lib.Gamma(alpha=alpha, beta=beta)
      expected_log_pdf = stats.gamma.logpdf(x, alpha_v, scale=1 / beta_v)
      log_pdf = gamma.log_pdf(x)
      self.assertEqual(log_pdf.get_shape(), (6,))
      self.assertAllClose(log_pdf.eval(), expected_log_pdf)

      pdf = gamma.pdf(x)
      self.assertEqual(pdf.get_shape(), (6,))
      self.assertAllClose(pdf.eval(), np.exp(expected_log_pdf))

  def testGammaLogPDFMultidimensional(self):
    with self.test_session():
      batch_size = 6
      alpha = constant_op.constant([[2.0, 4.0]] * batch_size)
      beta = constant_op.constant([[3.0, 4.0]] * batch_size)
      alpha_v = np.array([2.0, 4.0])
      beta_v = np.array([3.0, 4.0])
      x = np.array([[2.5, 2.5, 4.0, 0.1, 1.0, 2.0]], dtype=np.float32).T
      gamma = gamma_lib.Gamma(alpha=alpha, beta=beta)
      expected_log_pdf = stats.gamma.logpdf(x, alpha_v, scale=1 / beta_v)
      log_pdf = gamma.log_pdf(x)
      log_pdf_values = log_pdf.eval()
      self.assertEqual(log_pdf.get_shape(), (6, 2))
      self.assertAllClose(log_pdf_values, expected_log_pdf)

      pdf = gamma.pdf(x)
      pdf_values = pdf.eval()
      self.assertEqual(pdf.get_shape(), (6, 2))
      self.assertAllClose(pdf_values, np.exp(expected_log_pdf))

  def testGammaLogPDFMultidimensionalBroadcasting(self):
    with self.test_session():
      batch_size = 6
      alpha = constant_op.constant([[2.0, 4.0]] * batch_size)
      beta = constant_op.constant(3.0)
      alpha_v = np.array([2.0, 4.0])
      beta_v = 3.0
      x = np.array([[2.5, 2.5, 4.0, 0.1, 1.0, 2.0]], dtype=np.float32).T
      gamma = gamma_lib.Gamma(alpha=alpha, beta=beta)
      expected_log_pdf = stats.gamma.logpdf(x, alpha_v, scale=1 / beta_v)
      log_pdf = gamma.log_pdf(x)
      log_pdf_values = log_pdf.eval()
      self.assertEqual(log_pdf.get_shape(), (6, 2))
      self.assertAllClose(log_pdf_values, expected_log_pdf)

      pdf = gamma.pdf(x)
      pdf_values = pdf.eval()
      self.assertEqual(pdf.get_shape(), (6, 2))
      self.assertAllClose(pdf_values, np.exp(expected_log_pdf))

  def testGammaCDF(self):
    with self.test_session():
      batch_size = 6
      alpha = constant_op.constant([2.0] * batch_size)
      beta = constant_op.constant([3.0] * batch_size)
      alpha_v = 2.0
      beta_v = 3.0
      x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)

      gamma = gamma_lib.Gamma(alpha=alpha, beta=beta)
      expected_cdf = stats.gamma.cdf(x, alpha_v, scale=1 / beta_v)

      cdf = gamma.cdf(x)
      self.assertEqual(cdf.get_shape(), (6,))
      self.assertAllClose(cdf.eval(), expected_cdf)

  def testGammaMean(self):
    with self.test_session():
      alpha_v = np.array([1.0, 3.0, 2.5])
      beta_v = np.array([1.0, 4.0, 5.0])
      gamma = gamma_lib.Gamma(alpha=alpha_v, beta=beta_v)
      expected_means = stats.gamma.mean(alpha_v, scale=1 / beta_v)
      self.assertEqual(gamma.mean().get_shape(), (3,))
      self.assertAllClose(gamma.mean().eval(), expected_means)

  def testGammaModeAllowNanStatsIsFalseWorksWhenAllBatchMembersAreDefined(self):
    with self.test_session():
      alpha_v = np.array([5.5, 3.0, 2.5])
      beta_v = np.array([1.0, 4.0, 5.0])
      gamma = gamma_lib.Gamma(alpha=alpha_v, beta=beta_v)
      expected_modes = (alpha_v - 1) / beta_v
      self.assertEqual(gamma.mode().get_shape(), (3,))
      self.assertAllClose(gamma.mode().eval(), expected_modes)

  def testGammaModeAllowNanStatsFalseRaisesForUndefinedBatchMembers(self):
    with self.test_session():
      # Mode will not be defined for the first entry.
      alpha_v = np.array([0.5, 3.0, 2.5])
      beta_v = np.array([1.0, 4.0, 5.0])
      gamma = gamma_lib.Gamma(alpha=alpha_v, beta=beta_v, allow_nan_stats=False)
      with self.assertRaisesOpError("x < y"):
        gamma.mode().eval()

  def testGammaModeAllowNanStatsIsTrueReturnsNaNforUndefinedBatchMembers(self):
    with self.test_session():
      # Mode will not be defined for the first entry.
      alpha_v = np.array([0.5, 3.0, 2.5])
      beta_v = np.array([1.0, 4.0, 5.0])
      gamma = gamma_lib.Gamma(alpha=alpha_v, beta=beta_v, allow_nan_stats=True)
      expected_modes = (alpha_v - 1) / beta_v
      expected_modes[0] = np.nan
      self.assertEqual(gamma.mode().get_shape(), (3,))
      self.assertAllClose(gamma.mode().eval(), expected_modes)

  def testGammaVariance(self):
    with self.test_session():
      alpha_v = np.array([1.0, 3.0, 2.5])
      beta_v = np.array([1.0, 4.0, 5.0])
      gamma = gamma_lib.Gamma(alpha=alpha_v, beta=beta_v)
      expected_variances = stats.gamma.var(alpha_v, scale=1 / beta_v)
      self.assertEqual(gamma.variance().get_shape(), (3,))
      self.assertAllClose(gamma.variance().eval(), expected_variances)

  def testGammaStd(self):
    with self.test_session():
      alpha_v = np.array([1.0, 3.0, 2.5])
      beta_v = np.array([1.0, 4.0, 5.0])
      gamma = gamma_lib.Gamma(alpha=alpha_v, beta=beta_v)
      expected_std = stats.gamma.std(alpha_v, scale=1 / beta_v)
      self.assertEqual(gamma.std().get_shape(), (3,))
      self.assertAllClose(gamma.std().eval(), expected_std)

  def testGammaEntropy(self):
    with self.test_session():
      alpha_v = np.array([1.0, 3.0, 2.5])
      beta_v = np.array([1.0, 4.0, 5.0])
      expected_entropy = stats.gamma.entropy(alpha_v, scale=1 / beta_v)
      gamma = gamma_lib.Gamma(alpha=alpha_v, beta=beta_v)
      self.assertEqual(gamma.entropy().get_shape(), (3,))
      self.assertAllClose(gamma.entropy().eval(), expected_entropy)

  def testGammaSampleSmallAlpha(self):
    with session.Session():
      alpha_v = 0.05
      beta_v = 1.0
      alpha = constant_op.constant(alpha_v)
      beta = constant_op.constant(beta_v)
      n = 100000
      gamma = gamma_lib.Gamma(alpha=alpha, beta=beta)
      samples = gamma.sample(n, seed=137)
      sample_values = samples.eval()
      self.assertEqual(samples.get_shape(), (n,))
      self.assertEqual(sample_values.shape, (n,))
      self.assertAllClose(
          sample_values.mean(),
          stats.gamma.mean(
              alpha_v, scale=1 / beta_v),
          atol=.01)
      self.assertAllClose(
          sample_values.var(),
          stats.gamma.var(alpha_v, scale=1 / beta_v),
          atol=.15)
      self.assertTrue(self._kstest(alpha_v, beta_v, sample_values))

  def testGammaSample(self):
    with session.Session():
      alpha_v = 4.0
      beta_v = 3.0
      alpha = constant_op.constant(alpha_v)
      beta = constant_op.constant(beta_v)
      n = 100000
      gamma = gamma_lib.Gamma(alpha=alpha, beta=beta)
      samples = gamma.sample(n, seed=137)
      sample_values = samples.eval()
      self.assertEqual(samples.get_shape(), (n,))
      self.assertEqual(sample_values.shape, (n,))
      self.assertAllClose(
          sample_values.mean(),
          stats.gamma.mean(
              alpha_v, scale=1 / beta_v),
          atol=.01)
      self.assertAllClose(
          sample_values.var(),
          stats.gamma.var(alpha_v, scale=1 / beta_v),
          atol=.15)
      self.assertTrue(self._kstest(alpha_v, beta_v, sample_values))

  def testGammaSampleMultiDimensional(self):
    with session.Session():
      alpha_v = np.array([np.arange(1, 101, dtype=np.float32)])  # 1 x 100
      beta_v = np.array([np.arange(1, 11, dtype=np.float32)]).T  # 10 x 1
      gamma = gamma_lib.Gamma(alpha=alpha_v, beta=beta_v)
      n = 10000
      samples = gamma.sample(n, seed=137)
      sample_values = samples.eval()
      self.assertEqual(samples.get_shape(), (n, 10, 100))
      self.assertEqual(sample_values.shape, (n, 10, 100))
      zeros = np.zeros_like(alpha_v + beta_v)  # 10 x 100
      alpha_bc = alpha_v + zeros
      beta_bc = beta_v + zeros
      self.assertAllClose(
          sample_values.mean(axis=0),
          stats.gamma.mean(
              alpha_bc, scale=1 / beta_bc),
          rtol=.035)
      self.assertAllClose(
          sample_values.var(axis=0),
          stats.gamma.var(alpha_bc, scale=1 / beta_bc),
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
    ks, _ = stats.kstest(samples, stats.gamma(alpha, scale=1 / beta).cdf)
    # Return True when the test passes.
    return ks < 0.02

  def testGammaPdfOfSampleMultiDims(self):
    with session.Session() as sess:
      gamma = gamma_lib.Gamma(alpha=[7., 11.], beta=[[5.], [6.]])
      num = 50000
      samples = gamma.sample(num, seed=137)
      pdfs = gamma.pdf(samples)
      sample_vals, pdf_vals = sess.run([samples, pdfs])
      self.assertEqual(samples.get_shape(), (num, 2, 2))
      self.assertEqual(pdfs.get_shape(), (num, 2, 2))
      self.assertAllClose(
          stats.gamma.mean(
              [[7., 11.], [7., 11.]], scale=1 / np.array([[5., 5.], [6., 6.]])),
          sample_vals.mean(axis=0),
          atol=.1)
      self.assertAllClose(
          stats.gamma.var([[7., 11.], [7., 11.]],
                          scale=1 / np.array([[5., 5.], [6., 6.]])),
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

  def testGammaNonPositiveInitializationParamsRaises(self):
    with self.test_session():
      alpha_v = constant_op.constant(0.0, name="alpha")
      beta_v = constant_op.constant(1.0, name="beta")
      gamma = gamma_lib.Gamma(alpha=alpha_v, beta=beta_v, validate_args=True)
      with self.assertRaisesOpError("alpha"):
        gamma.mean().eval()
      alpha_v = constant_op.constant(1.0, name="alpha")
      beta_v = constant_op.constant(0.0, name="beta")
      gamma = gamma_lib.Gamma(alpha=alpha_v, beta=beta_v, validate_args=True)
      with self.assertRaisesOpError("beta"):
        gamma.mean().eval()

  def testGammaWithSoftplusAlphaBeta(self):
    with self.test_session():
      alpha_v = constant_op.constant([0.0, -2.1], name="alpha")
      beta_v = constant_op.constant([1.0, -3.6], name="beta")
      gamma = gamma_lib.GammaWithSoftplusAlphaBeta(alpha=alpha_v, beta=beta_v)
      self.assertAllEqual(nn_ops.softplus(alpha_v).eval(), gamma.alpha.eval())
      self.assertAllEqual(nn_ops.softplus(beta_v).eval(), gamma.beta.eval())


if __name__ == "__main__":
  test.main()
