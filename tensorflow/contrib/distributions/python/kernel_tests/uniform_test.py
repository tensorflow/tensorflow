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
"""Tests for Uniform distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import stats
from tensorflow.contrib.distributions.python.ops import uniform as uniform_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class UniformTest(test.TestCase):

  def testUniformRange(self):
    with self.test_session():
      a = 3.0
      b = 10.0
      uniform = uniform_lib.Uniform(a=a, b=b)
      self.assertAllClose(a, uniform.a.eval())
      self.assertAllClose(b, uniform.b.eval())
      self.assertAllClose(b - a, uniform.range().eval())

  def testUniformPDF(self):
    with self.test_session():
      a = constant_op.constant([-3.0] * 5 + [15.0])
      b = constant_op.constant([11.0] * 5 + [20.0])
      uniform = uniform_lib.Uniform(a=a, b=b)

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
      a = constant_op.constant([-3.0] * 5)
      b = constant_op.constant(11.0)
      uniform = uniform_lib.Uniform(a=a, b=b)

      self.assertEqual(uniform.batch_shape().eval(), (5,))
      self.assertEqual(uniform.get_batch_shape(), tensor_shape.TensorShape([5]))
      self.assertAllEqual(uniform.event_shape().eval(), [])
      self.assertEqual(uniform.get_event_shape(), tensor_shape.TensorShape([]))

  def testUniformPDFWithScalarEndpoint(self):
    with self.test_session():
      a = constant_op.constant([0.0, 5.0])
      b = constant_op.constant(10.0)
      uniform = uniform_lib.Uniform(a=a, b=b)

      x = np.array([0.0, 8.0], dtype=np.float32)
      expected_pdf = np.array([1.0 / (10.0 - 0.0), 1.0 / (10.0 - 5.0)])

      pdf = uniform.pdf(x)
      self.assertAllClose(expected_pdf, pdf.eval())

  def testUniformCDF(self):
    with self.test_session():
      batch_size = 6
      a = constant_op.constant([1.0] * batch_size)
      b = constant_op.constant([11.0] * batch_size)
      a_v = 1.0
      b_v = 11.0
      x = np.array([-2.5, 2.5, 4.0, 0.0, 10.99, 12.0], dtype=np.float32)

      uniform = uniform_lib.Uniform(a=a, b=b)

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
      uniform = uniform_lib.Uniform(a=a_v, b=b_v)

      expected_entropy = np.log(b_v - a_v)
      self.assertAllClose(expected_entropy, uniform.entropy().eval())

  def testUniformAssertMaxGtMin(self):
    with self.test_session():
      a_v = np.array([1.0, 1.0, 1.0], dtype=np.float32)
      b_v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
      uniform = uniform_lib.Uniform(a=a_v, b=b_v, validate_args=True)

      with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError,
                                               "x < y"):
        uniform.a.eval()

  def testUniformSample(self):
    with self.test_session():
      a = constant_op.constant([3.0, 4.0])
      b = constant_op.constant(13.0)
      a1_v = 3.0
      a2_v = 4.0
      b_v = 13.0
      n = constant_op.constant(100000)
      uniform = uniform_lib.Uniform(a=a, b=b)

      samples = uniform.sample(n, seed=137)
      sample_values = samples.eval()
      self.assertEqual(sample_values.shape, (100000, 2))
      self.assertAllClose(
          sample_values[::, 0].mean(), (b_v + a1_v) / 2, atol=1e-2)
      self.assertAllClose(
          sample_values[::, 1].mean(), (b_v + a2_v) / 2, atol=1e-2)
      self.assertFalse(
          np.any(sample_values[::, 0] < a1_v) or np.any(sample_values >= b_v))
      self.assertFalse(
          np.any(sample_values[::, 1] < a2_v) or np.any(sample_values >= b_v))

  def _testUniformSampleMultiDimensional(self):
    # DISABLED: Please enable this test once b/issues/30149644 is resolved.
    with self.test_session():
      batch_size = 2
      a_v = [3.0, 22.0]
      b_v = [13.0, 35.0]
      a = constant_op.constant([a_v] * batch_size)
      b = constant_op.constant([b_v] * batch_size)

      uniform = uniform_lib.Uniform(a=a, b=b)

      n_v = 100000
      n = constant_op.constant(n_v)
      samples = uniform.sample(n)
      self.assertEqual(samples.get_shape(), (n_v, batch_size, 2))

      sample_values = samples.eval()

      self.assertFalse(
          np.any(sample_values[:, 0, 0] < a_v[0]) or
          np.any(sample_values[:, 0, 0] >= b_v[0]))
      self.assertFalse(
          np.any(sample_values[:, 0, 1] < a_v[1]) or
          np.any(sample_values[:, 0, 1] >= b_v[1]))

      self.assertAllClose(
          sample_values[:, 0, 0].mean(), (a_v[0] + b_v[0]) / 2, atol=1e-2)
      self.assertAllClose(
          sample_values[:, 0, 1].mean(), (a_v[1] + b_v[1]) / 2, atol=1e-2)

  def testUniformMean(self):
    with self.test_session():
      a = 10.0
      b = 100.0
      uniform = uniform_lib.Uniform(a=a, b=b)
      s_uniform = stats.uniform(loc=a, scale=b - a)
      self.assertAllClose(uniform.mean().eval(), s_uniform.mean())

  def testUniformVariance(self):
    with self.test_session():
      a = 10.0
      b = 100.0
      uniform = uniform_lib.Uniform(a=a, b=b)
      s_uniform = stats.uniform(loc=a, scale=b - a)
      self.assertAllClose(uniform.variance().eval(), s_uniform.var())

  def testUniformStd(self):
    with self.test_session():
      a = 10.0
      b = 100.0
      uniform = uniform_lib.Uniform(a=a, b=b)
      s_uniform = stats.uniform(loc=a, scale=b - a)
      self.assertAllClose(uniform.std().eval(), s_uniform.std())

  def testUniformNans(self):
    with self.test_session():
      a = 10.0
      b = [11.0, 100.0]
      uniform = uniform_lib.Uniform(a=a, b=b)

      no_nans = constant_op.constant(1.0)
      nans = constant_op.constant(0.0) / constant_op.constant(0.0)
      self.assertTrue(math_ops.is_nan(nans).eval())
      with_nans = array_ops.stack([no_nans, nans])

      pdf = uniform.pdf(with_nans)

      is_nan = math_ops.is_nan(pdf).eval()
      self.assertFalse(is_nan[0])
      self.assertTrue(is_nan[1])

  def testUniformSamplePdf(self):
    with self.test_session():
      a = 10.0
      b = [11.0, 100.0]
      uniform = uniform_lib.Uniform(a, b)
      self.assertTrue(
          math_ops.reduce_all(uniform.pdf(uniform.sample(10)) > 0).eval())

  def testUniformBroadcasting(self):
    with self.test_session():
      a = 10.0
      b = [11.0, 20.0]
      uniform = uniform_lib.Uniform(a, b)

      pdf = uniform.pdf([[10.5, 11.5], [9.0, 19.0], [10.5, 21.0]])
      expected_pdf = np.array([[1.0, 0.1], [0.0, 0.1], [1.0, 0.0]])
      self.assertAllClose(expected_pdf, pdf.eval())

  def testUniformSampleWithShape(self):
    with self.test_session():
      a = 10.0
      b = [11.0, 20.0]
      uniform = uniform_lib.Uniform(a, b)

      pdf = uniform.pdf(uniform.sample((2, 3)))
      # pylint: disable=bad-continuation
      expected_pdf = [
          [[1.0, 0.1], [1.0, 0.1], [1.0, 0.1]],
          [[1.0, 0.1], [1.0, 0.1], [1.0, 0.1]],
      ]
      # pylint: enable=bad-continuation
      self.assertAllClose(expected_pdf, pdf.eval())

      pdf = uniform.pdf(uniform.sample())
      expected_pdf = [1.0, 0.1]
      self.assertAllClose(expected_pdf, pdf.eval())


if __name__ == "__main__":
  test.main()
