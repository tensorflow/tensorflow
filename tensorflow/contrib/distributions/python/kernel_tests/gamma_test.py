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
import tensorflow as tf


class GammaTest(tf.test.TestCase):

  def testGammaShape(self):
    with self.test_session():
      alpha = tf.constant([3.0] * 5)
      beta = tf.constant(11.0)
      gamma = tf.contrib.distributions.Gamma(alpha=alpha, beta=beta)

      self.assertEqual(gamma.batch_shape().eval(), (5,))
      self.assertEqual(gamma.get_batch_shape(), tf.TensorShape([5]))
      self.assertAllEqual(gamma.event_shape().eval(), [])
      self.assertEqual(gamma.get_event_shape(), tf.TensorShape([]))

  def testGammaLogPDF(self):
    with self.test_session():
      batch_size = 6
      alpha = tf.constant([2.0] * batch_size)
      beta = tf.constant([3.0] * batch_size)
      alpha_v = 2.0
      beta_v = 3.0
      x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)
      gamma = tf.contrib.distributions.Gamma(alpha=alpha, beta=beta)
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
      alpha = tf.constant([[2.0, 4.0]] * batch_size)
      beta = tf.constant([[3.0, 4.0]] * batch_size)
      alpha_v = np.array([2.0, 4.0])
      beta_v = np.array([3.0, 4.0])
      x = np.array([[2.5, 2.5, 4.0, 0.1, 1.0, 2.0]], dtype=np.float32).T
      gamma = tf.contrib.distributions.Gamma(alpha=alpha, beta=beta)
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
      alpha = tf.constant([[2.0, 4.0]] * batch_size)
      beta = tf.constant(3.0)
      alpha_v = np.array([2.0, 4.0])
      beta_v = 3.0
      x = np.array([[2.5, 2.5, 4.0, 0.1, 1.0, 2.0]], dtype=np.float32).T
      gamma = tf.contrib.distributions.Gamma(alpha=alpha, beta=beta)
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
      alpha = tf.constant([2.0] * batch_size)
      beta = tf.constant([3.0] * batch_size)
      alpha_v = 2.0
      beta_v = 3.0
      x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)

      gamma = tf.contrib.distributions.Gamma(alpha=alpha, beta=beta)
      expected_cdf = stats.gamma.cdf(x, alpha_v, scale=1 / beta_v)

      cdf = gamma.cdf(x)
      self.assertEqual(cdf.get_shape(), (6,))
      self.assertAllClose(cdf.eval(), expected_cdf)

  def testGammaMean(self):
    with self.test_session():
      alpha_v = np.array([1.0, 3.0, 2.5])
      beta_v = np.array([1.0, 4.0, 5.0])
      gamma = tf.contrib.distributions.Gamma(alpha=alpha_v, beta=beta_v)
      expected_means = stats.gamma.mean(alpha_v, scale=1 / beta_v)
      self.assertEqual(gamma.mean().get_shape(), (3,))
      self.assertAllClose(gamma.mean().eval(), expected_means)

  def testGammaMode(self):
    with self.test_session():
      # Mode will not be defined for the first entry.
      alpha_v = np.array([0.5, 3.0, 2.5])
      beta_v = np.array([1.0, 4.0, 5.0])
      gamma = tf.contrib.distributions.Gamma(alpha=alpha_v, beta=beta_v)
      expected_modes = (alpha_v - 1) / beta_v
      expected_modes[0] = np.nan
      self.assertEqual(gamma.mode().get_shape(), (3,))
      self.assertAllClose(gamma.mode().eval(), expected_modes)

  def testGammaVariance(self):
    with self.test_session():
      alpha_v = np.array([1.0, 3.0, 2.5])
      beta_v = np.array([1.0, 4.0, 5.0])
      gamma = tf.contrib.distributions.Gamma(alpha=alpha_v, beta=beta_v)
      expected_variances = stats.gamma.var(alpha_v, scale=1 / beta_v)
      self.assertEqual(gamma.variance().get_shape(), (3,))
      self.assertAllClose(gamma.variance().eval(), expected_variances)

  def testGammaStd(self):
    with self.test_session():
      alpha_v = np.array([1.0, 3.0, 2.5])
      beta_v = np.array([1.0, 4.0, 5.0])
      gamma = tf.contrib.distributions.Gamma(alpha=alpha_v, beta=beta_v)
      expected_std = stats.gamma.std(alpha_v, scale=1 / beta_v)
      self.assertEqual(gamma.std().get_shape(), (3,))
      self.assertAllClose(gamma.std().eval(), expected_std)

  def testGammaEntropy(self):
    with self.test_session():
      alpha_v = np.array([1.0, 3.0, 2.5])
      beta_v = np.array([1.0, 4.0, 5.0])
      expected_entropy = stats.gamma.entropy(alpha_v, scale=1 / beta_v)
      gamma = tf.contrib.distributions.Gamma(alpha=alpha_v, beta=beta_v)
      self.assertEqual(gamma.entropy().get_shape(), (3,))
      self.assertAllClose(gamma.entropy().eval(), expected_entropy)

  def testGammaNonPositiveInitializationParamsRaises(self):
    with self.test_session():
      alpha_v = tf.constant(0.0, name='alpha')
      beta_v = tf.constant(1.0, name='beta')
      gamma = tf.contrib.distributions.Gamma(alpha=alpha_v, beta=beta_v)
      with self.assertRaisesOpError('alpha'):
        gamma.mean().eval()


if __name__ == '__main__':
  tf.test.main()
