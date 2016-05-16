# Copyright 2015 Google Inc. All Rights Reserved.
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
"""Tests for Uniform distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class UniformTest(tf.test.TestCase):

  def testUniformRange(self):
    with self.test_session():
      a = 3.0
      b = 10.0
      uniform = tf.contrib.distributions.Uniform(a=a, b=b)
      self.assertAllClose(a, uniform.a.eval())
      self.assertAllClose(b, uniform.b.eval())
      self.assertAllClose(b - a, uniform.range.eval())

  def testUniformPDF(self):
    with self.test_session():
      a = tf.constant([-3.0] * 5 + [15.0])
      b = tf.constant([11.0] * 5 + [20.0])
      uniform = tf.contrib.distributions.Uniform(a=a, b=b)

      a_v = -3.0
      b_v = 11.0
      x = np.array([-10.5, 4.0, 0.0, 10.99, 11.3, 17.0], dtype=np.float32)

      def _expected_pdf():
        pdf = np.zeros_like(x) + 1.0 / (b_v - a_v)
        pdf[x > b_v] = 0.0
        pdf[x < a_v] = 0.0
        pdf[5] = 1.0 / (20.0 - 15.0)
        return pdf

      expected_pdf = _expected_pdf()

      pdf = uniform.pdf(x)
      self.assertAllClose(expected_pdf, pdf.eval())

      log_pdf = uniform.log_pdf(x)
      self.assertAllClose(np.log(expected_pdf), log_pdf.eval())

  def testUniformShape(self):
    with self.test_session():
      a = tf.constant([-3.0] * 5)
      b = tf.constant(11.0)
      uniform = tf.contrib.distributions.Uniform(a=a, b=b)

      self.assertEqual(uniform.batch_shape().eval(), (5,))
      self.assertEqual(uniform.get_batch_shape(), tf.TensorShape([5]))
      self.assertEqual(uniform.event_shape().eval(), 1)
      self.assertEqual(uniform.get_event_shape(), tf.TensorShape([]))

  def testUniformPDFWithScalarEndpoint(self):
    with self.test_session():
      a = tf.constant([0.0, 5.0])
      b = tf.constant(10.0)
      uniform = tf.contrib.distributions.Uniform(a=a, b=b)

      x = np.array([0.0, 8.0], dtype=np.float32)
      expected_pdf = np.array([1.0 / (10.0 - 0.0), 1.0 / (10.0 - 5.0)])

      pdf = uniform.pdf(x)
      self.assertAllClose(expected_pdf, pdf.eval())

  def testUniformCDF(self):
    with self.test_session():
      batch_size = 6
      a = tf.constant([1.0] * batch_size)
      b = tf.constant([11.0] * batch_size)
      a_v = 1.0
      b_v = 11.0
      x = np.array([-2.5, 2.5, 4.0, 0.0, 10.99, 12.0], dtype=np.float32)

      uniform = tf.contrib.distributions.Uniform(a=a, b=b)

      def _expected_cdf():
        cdf = (x - a_v) / (b_v - a_v)
        cdf[x >= b_v] = 1
        cdf[x < a_v] = 0
        return cdf

      cdf = uniform.cdf(x)
      self.assertAllClose(_expected_cdf(), cdf.eval())

      log_cdf = uniform.log_cdf(x)
      self.assertAllClose(np.log(_expected_cdf()), log_cdf.eval())

  def testUniformEntropy(self):
    with self.test_session():
      a_v = np.array([1.0, 1.0, 1.0])
      b_v = np.array([[1.5, 2.0, 3.0]])
      uniform = tf.contrib.distributions.Uniform(a=a_v, b=b_v)

      expected_entropy = np.log(b_v - a_v)
      self.assertAllClose(expected_entropy, uniform.entropy().eval())

  def testUniformAssertMaxGtMin(self):
    with self.test_session():
      a_v = np.array([1.0, 1.0, 1.0], dtype=np.float32)
      b_v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
      uniform = tf.contrib.distributions.Uniform(a=a_v, b=b_v)

      with self.assertRaisesWithPredicateMatch(tf.errors.InvalidArgumentError,
                                               "x < y"):
        uniform.a.eval()

  def testUniformSample(self):
    with self.test_session():
      a = tf.constant([3.0, 4.0])
      b = tf.constant(13.0)
      a1_v = 3.0
      a2_v = 4.0
      b_v = 13.0
      n = tf.constant(100000)
      uniform = tf.contrib.distributions.Uniform(a=a, b=b)

      samples = uniform.sample(n, seed=137)
      sample_values = samples.eval()
      self.assertEqual(sample_values.shape, (100000, 2))
      self.assertAllClose(sample_values[::, 0].mean(), (b_v + a1_v) / 2,
                          atol=1e-2)
      self.assertAllClose(sample_values[::, 1].mean(), (b_v + a2_v) / 2,
                          atol=1e-2)
      self.assertFalse(np.any(sample_values[::, 0] < a1_v) or np.any(
          sample_values >= b_v))
      self.assertFalse(np.any(sample_values[::, 1] < a2_v) or np.any(
          sample_values >= b_v))

  def testUniformSampleMultiDimensional(self):
    with self.test_session():
      batch_size = 2
      a_v = [3.0, 22.0]
      b_v = [13.0, 35.0]
      a = tf.constant([a_v] * batch_size)
      b = tf.constant([b_v] * batch_size)

      uniform = tf.contrib.distributions.Uniform(a=a, b=b)

      n_v = 100000
      n = tf.constant(n_v)
      samples = uniform.sample(n, seed=138)
      self.assertEqual(samples.get_shape(), (n_v, batch_size, 2))

      sample_values = samples.eval()

      self.assertFalse(np.any(sample_values[:, 0, 0] < a_v[0]) or np.any(
          sample_values[:, 0, 0] >= b_v[0]))
      self.assertFalse(np.any(sample_values[:, 0, 1] < a_v[1]) or np.any(
          sample_values[:, 0, 1] >= b_v[1]))

      self.assertAllClose(sample_values[:, 0, 0].mean(), (a_v[0] + b_v[0]) / 2,
                          atol=1e-2)
      self.assertAllClose(sample_values[:, 0, 1].mean(), (a_v[1] + b_v[1]) / 2,
                          atol=1e-2)

  def testUniformMeanAndVariance(self):
    with self.test_session():
      a = 10.0
      b = 100.0
      uniform = tf.contrib.distributions.Uniform(a=a, b=b)
      self.assertAllClose(uniform.variance.eval(), (b - a)**2 / 12)
      self.assertAllClose(uniform.mean.eval(), (b + a) / 2)

  def testUniformNans(self):
    with self.test_session():
      a = 10.0
      b = [11.0, 100.0]
      uniform = tf.contrib.distributions.Uniform(a=a, b=b)

      no_nans = tf.constant(1.0)
      nans = tf.constant(0.0) / tf.constant(0.0)
      self.assertTrue(tf.is_nan(nans).eval())
      with_nans = tf.pack([no_nans, nans])

      pdf = uniform.pdf(with_nans)

      is_nan = tf.is_nan(pdf).eval()
      print(pdf.eval())
      self.assertFalse(is_nan[0])
      self.assertTrue(is_nan[1])

  def testUniformSamplePdf(self):
    with self.test_session():
      a = 10.0
      b = [11.0, 100.0]
      uniform = tf.contrib.distributions.Uniform(a, b)
      self.assertTrue(tf.reduce_all(uniform.pdf(uniform.sample(10)) > 0).eval())

  def testUniformBroadcasting(self):
    with self.test_session():
      a = 10.0
      b = [11.0, 20.0]
      uniform = tf.contrib.distributions.Uniform(a, b)

      pdf = uniform.pdf([[10.5, 11.5], [9.0, 19.0], [10.5, 21.0]])
      expected_pdf = np.array([[1.0, 0.1], [0.0, 0.1], [1.0, 0.0]])
      self.assertAllClose(expected_pdf, pdf.eval())


if __name__ == "__main__":
  tf.test.main()
