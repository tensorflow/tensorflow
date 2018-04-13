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

import importlib

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import uniform as uniform_lib
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


class UniformTest(test.TestCase):

  def testUniformRange(self):
    with self.test_session():
      a = 3.0
      b = 10.0
      uniform = uniform_lib.Uniform(low=a, high=b)
      self.assertAllClose(a, uniform.low.eval())
      self.assertAllClose(b, uniform.high.eval())
      self.assertAllClose(b - a, uniform.range().eval())

  def testUniformPDF(self):
    with self.test_session():
      a = constant_op.constant([-3.0] * 5 + [15.0])
      b = constant_op.constant([11.0] * 5 + [20.0])
      uniform = uniform_lib.Uniform(low=a, high=b)

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

      pdf = uniform.prob(x)
      self.assertAllClose(expected_pdf, pdf.eval())

      log_pdf = uniform.log_prob(x)
      self.assertAllClose(np.log(expected_pdf), log_pdf.eval())

  def testUniformShape(self):
    with self.test_session():
      a = constant_op.constant([-3.0] * 5)
      b = constant_op.constant(11.0)
      uniform = uniform_lib.Uniform(low=a, high=b)

      self.assertEqual(uniform.batch_shape_tensor().eval(), (5,))
      self.assertEqual(uniform.batch_shape, tensor_shape.TensorShape([5]))
      self.assertAllEqual(uniform.event_shape_tensor().eval(), [])
      self.assertEqual(uniform.event_shape, tensor_shape.TensorShape([]))

  def testUniformPDFWithScalarEndpoint(self):
    with self.test_session():
      a = constant_op.constant([0.0, 5.0])
      b = constant_op.constant(10.0)
      uniform = uniform_lib.Uniform(low=a, high=b)

      x = np.array([0.0, 8.0], dtype=np.float32)
      expected_pdf = np.array([1.0 / (10.0 - 0.0), 1.0 / (10.0 - 5.0)])

      pdf = uniform.prob(x)
      self.assertAllClose(expected_pdf, pdf.eval())

  def testUniformCDF(self):
    with self.test_session():
      batch_size = 6
      a = constant_op.constant([1.0] * batch_size)
      b = constant_op.constant([11.0] * batch_size)
      a_v = 1.0
      b_v = 11.0
      x = np.array([-2.5, 2.5, 4.0, 0.0, 10.99, 12.0], dtype=np.float32)

      uniform = uniform_lib.Uniform(low=a, high=b)

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
      uniform = uniform_lib.Uniform(low=a_v, high=b_v)

      expected_entropy = np.log(b_v - a_v)
      self.assertAllClose(expected_entropy, uniform.entropy().eval())

  def testUniformAssertMaxGtMin(self):
    with self.test_session():
      a_v = np.array([1.0, 1.0, 1.0], dtype=np.float32)
      b_v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
      uniform = uniform_lib.Uniform(low=a_v, high=b_v, validate_args=True)

      with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError,
                                               "x < y"):
        uniform.low.eval()

  def testUniformSample(self):
    with self.test_session():
      a = constant_op.constant([3.0, 4.0])
      b = constant_op.constant(13.0)
      a1_v = 3.0
      a2_v = 4.0
      b_v = 13.0
      n = constant_op.constant(100000)
      uniform = uniform_lib.Uniform(low=a, high=b)

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

      uniform = uniform_lib.Uniform(low=a, high=b)

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
      uniform = uniform_lib.Uniform(low=a, high=b)
      if not stats:
        return
      s_uniform = stats.uniform(loc=a, scale=b - a)
      self.assertAllClose(uniform.mean().eval(), s_uniform.mean())

  def testUniformVariance(self):
    with self.test_session():
      a = 10.0
      b = 100.0
      uniform = uniform_lib.Uniform(low=a, high=b)
      if not stats:
        return
      s_uniform = stats.uniform(loc=a, scale=b - a)
      self.assertAllClose(uniform.variance().eval(), s_uniform.var())

  def testUniformStd(self):
    with self.test_session():
      a = 10.0
      b = 100.0
      uniform = uniform_lib.Uniform(low=a, high=b)
      if not stats:
        return
      s_uniform = stats.uniform(loc=a, scale=b - a)
      self.assertAllClose(uniform.stddev().eval(), s_uniform.std())

  def testUniformNans(self):
    with self.test_session():
      a = 10.0
      b = [11.0, 100.0]
      uniform = uniform_lib.Uniform(low=a, high=b)

      no_nans = constant_op.constant(1.0)
      nans = constant_op.constant(0.0) / constant_op.constant(0.0)
      self.assertTrue(math_ops.is_nan(nans).eval())
      with_nans = array_ops.stack([no_nans, nans])

      pdf = uniform.prob(with_nans)

      is_nan = math_ops.is_nan(pdf).eval()
      self.assertFalse(is_nan[0])
      self.assertTrue(is_nan[1])

  def testUniformSamplePdf(self):
    with self.test_session():
      a = 10.0
      b = [11.0, 100.0]
      uniform = uniform_lib.Uniform(a, b)
      self.assertTrue(
          math_ops.reduce_all(uniform.prob(uniform.sample(10)) > 0).eval())

  def testUniformBroadcasting(self):
    with self.test_session():
      a = 10.0
      b = [11.0, 20.0]
      uniform = uniform_lib.Uniform(a, b)

      pdf = uniform.prob([[10.5, 11.5], [9.0, 19.0], [10.5, 21.0]])
      expected_pdf = np.array([[1.0, 0.1], [0.0, 0.1], [1.0, 0.0]])
      self.assertAllClose(expected_pdf, pdf.eval())

  def testUniformSampleWithShape(self):
    with self.test_session():
      a = 10.0
      b = [11.0, 20.0]
      uniform = uniform_lib.Uniform(a, b)

      pdf = uniform.prob(uniform.sample((2, 3)))
      # pylint: disable=bad-continuation
      expected_pdf = [
          [[1.0, 0.1], [1.0, 0.1], [1.0, 0.1]],
          [[1.0, 0.1], [1.0, 0.1], [1.0, 0.1]],
      ]
      # pylint: enable=bad-continuation
      self.assertAllClose(expected_pdf, pdf.eval())

      pdf = uniform.prob(uniform.sample())
      expected_pdf = [1.0, 0.1]
      self.assertAllClose(expected_pdf, pdf.eval())

  def testUniformFloat64(self):
    uniform = uniform_lib.Uniform(
        low=np.float64(0.), high=np.float64(1.))

    self.assertAllClose(
        [1., 1.],
        self.evaluate(uniform.prob(np.array([0.5, 0.6], dtype=np.float64))))

    self.assertAllClose(
        [0.5, 0.6],
        self.evaluate(uniform.cdf(np.array([0.5, 0.6], dtype=np.float64))))

    self.assertAllClose(0.5, self.evaluate(uniform.mean()))
    self.assertAllClose(1 / 12., self.evaluate(uniform.variance()))
    self.assertAllClose(0., self.evaluate(uniform.entropy()))


if __name__ == "__main__":
  test.main()
