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

from tensorflow.contrib.distributions.python.ops import kumaraswamy as kumaraswamy_lib
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import math_ops
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


def _kumaraswamy_mode(a, b):
  a = np.asarray(a)
  b = np.asarray(b)
  return ((a - 1) / (a * b - 1))**(1 / a)


def _kumaraswamy_moment(a, b, n):
  a = np.asarray(a)
  b = np.asarray(b)
  return b * special.beta(1.0 + n / a, b)


def _harmonic_number(b):
  b = np.asarray(b)
  return special.psi(b + 1) - special.psi(1)


def _kumaraswamy_cdf(a, b, x):
  a = np.asarray(a)
  b = np.asarray(b)
  x = np.asarray(x)
  return 1 - (1 - x**a)**b


def _kumaraswamy_pdf(a, b, x):
  a = np.asarray(a)
  b = np.asarray(b)
  x = np.asarray(x)
  return a * b * x ** (a - 1) * (1 - x ** a) ** (b - 1)


class KumaraswamyTest(test.TestCase):

  def testSimpleShapes(self):
    with self.test_session():
      a = np.random.rand(3)
      b = np.random.rand(3)
      dist = kumaraswamy_lib.Kumaraswamy(a, b)
      self.assertAllEqual([], dist.event_shape_tensor().eval())
      self.assertAllEqual([3], dist.batch_shape_tensor().eval())
      self.assertEqual(tensor_shape.TensorShape([]), dist.event_shape)
      self.assertEqual(tensor_shape.TensorShape([3]), dist.batch_shape)

  def testComplexShapes(self):
    with self.test_session():
      a = np.random.rand(3, 2, 2)
      b = np.random.rand(3, 2, 2)
      dist = kumaraswamy_lib.Kumaraswamy(a, b)
      self.assertAllEqual([], dist.event_shape_tensor().eval())
      self.assertAllEqual([3, 2, 2], dist.batch_shape_tensor().eval())
      self.assertEqual(tensor_shape.TensorShape([]), dist.event_shape)
      self.assertEqual(tensor_shape.TensorShape([3, 2, 2]), dist.batch_shape)

  def testComplexShapesBroadcast(self):
    with self.test_session():
      a = np.random.rand(3, 2, 2)
      b = np.random.rand(2, 2)
      dist = kumaraswamy_lib.Kumaraswamy(a, b)
      self.assertAllEqual([], dist.event_shape_tensor().eval())
      self.assertAllEqual([3, 2, 2], dist.batch_shape_tensor().eval())
      self.assertEqual(tensor_shape.TensorShape([]), dist.event_shape)
      self.assertEqual(tensor_shape.TensorShape([3, 2, 2]), dist.batch_shape)

  def testAProperty(self):
    a = [[1., 2, 3]]
    b = [[2., 4, 3]]
    with self.test_session():
      dist = kumaraswamy_lib.Kumaraswamy(a, b)
      self.assertEqual([1, 3], dist.concentration1.get_shape())
      self.assertAllClose(a, dist.concentration1.eval())

  def testBProperty(self):
    a = [[1., 2, 3]]
    b = [[2., 4, 3]]
    with self.test_session():
      dist = kumaraswamy_lib.Kumaraswamy(a, b)
      self.assertEqual([1, 3], dist.concentration0.get_shape())
      self.assertAllClose(b, dist.concentration0.eval())

  def testPdfXProper(self):
    a = [[1., 2, 3]]
    b = [[2., 4, 3]]
    with self.test_session():
      dist = kumaraswamy_lib.Kumaraswamy(a, b, validate_args=True)
      dist.prob([.1, .3, .6]).eval()
      dist.prob([.2, .3, .5]).eval()
      # Either condition can trigger.
      with self.assertRaisesOpError("sample must be non-negative"):
        dist.prob([-1., 0.1, 0.5]).eval()
      with self.assertRaisesOpError("sample must be no larger than `1`"):
        dist.prob([.1, .2, 1.2]).eval()

  def testPdfTwoBatches(self):
    with self.test_session():
      a = [1., 2]
      b = [1., 2]
      x = [.5, .5]
      dist = kumaraswamy_lib.Kumaraswamy(a, b)
      pdf = dist.prob(x)
      expected_pdf = _kumaraswamy_pdf(a, b, x)
      self.assertAllClose(expected_pdf, pdf.eval())
      self.assertEqual((2,), pdf.get_shape())

  def testPdfTwoBatchesNontrivialX(self):
    with self.test_session():
      a = [1., 2]
      b = [1., 2]
      x = [.3, .7]
      dist = kumaraswamy_lib.Kumaraswamy(a, b)
      pdf = dist.prob(x)
      expected_pdf = _kumaraswamy_pdf(a, b, x)
      self.assertAllClose(expected_pdf, pdf.eval())
      self.assertEqual((2,), pdf.get_shape())

  def testPdfUniformZeroBatch(self):
    with self.test_session():
      # This is equivalent to a uniform distribution
      a = 1.
      b = 1.
      x = np.array([.1, .2, .3, .5, .8], dtype=np.float32)
      dist = kumaraswamy_lib.Kumaraswamy(a, b)
      pdf = dist.prob(x)
      expected_pdf = _kumaraswamy_pdf(a, b, x)
      self.assertAllClose(expected_pdf, pdf.eval())
      self.assertEqual((5,), pdf.get_shape())

  def testPdfAStretchedInBroadcastWhenSameRank(self):
    with self.test_session():
      a = [[1., 2]]
      b = [[1., 2]]
      x = [[.5, .5], [.3, .7]]
      dist = kumaraswamy_lib.Kumaraswamy(a, b)
      pdf = dist.prob(x)
      expected_pdf = _kumaraswamy_pdf(a, b, x)
      self.assertAllClose(expected_pdf, pdf.eval())
      self.assertEqual((2, 2), pdf.get_shape())

  def testPdfAStretchedInBroadcastWhenLowerRank(self):
    with self.test_session():
      a = [1., 2]
      b = [1., 2]
      x = [[.5, .5], [.2, .8]]
      pdf = kumaraswamy_lib.Kumaraswamy(a, b).prob(x)
      expected_pdf = _kumaraswamy_pdf(a, b, x)
      self.assertAllClose(expected_pdf, pdf.eval())
      self.assertEqual((2, 2), pdf.get_shape())

  def testPdfXStretchedInBroadcastWhenSameRank(self):
    with self.test_session():
      a = [[1., 2], [2., 3]]
      b = [[1., 2], [2., 3]]
      x = [[.5, .5]]
      pdf = kumaraswamy_lib.Kumaraswamy(a, b).prob(x)
      expected_pdf = _kumaraswamy_pdf(a, b, x)
      self.assertAllClose(expected_pdf, pdf.eval())
      self.assertEqual((2, 2), pdf.get_shape())

  def testPdfXStretchedInBroadcastWhenLowerRank(self):
    with self.test_session():
      a = [[1., 2], [2., 3]]
      b = [[1., 2], [2., 3]]
      x = [.5, .5]
      pdf = kumaraswamy_lib.Kumaraswamy(a, b).prob(x)
      expected_pdf = _kumaraswamy_pdf(a, b, x)
      self.assertAllClose(expected_pdf, pdf.eval())
      self.assertEqual((2, 2), pdf.get_shape())

  def testKumaraswamyMean(self):
    with session.Session():
      a = [1., 2, 3]
      b = [2., 4, 1.2]
      dist = kumaraswamy_lib.Kumaraswamy(a, b)
      self.assertEqual(dist.mean().get_shape(), (3,))
      if not stats:
        return
      expected_mean = _kumaraswamy_moment(a, b, 1)
      self.assertAllClose(expected_mean, dist.mean().eval())

  def testKumaraswamyVariance(self):
    with session.Session():
      a = [1., 2, 3]
      b = [2., 4, 1.2]
      dist = kumaraswamy_lib.Kumaraswamy(a, b)
      self.assertEqual(dist.variance().get_shape(), (3,))
      if not stats:
        return
      expected_variance = _kumaraswamy_moment(a, b, 2) - _kumaraswamy_moment(
          a, b, 1)**2
      self.assertAllClose(expected_variance, dist.variance().eval())

  def testKumaraswamyMode(self):
    with session.Session():
      a = np.array([1.1, 2, 3])
      b = np.array([2., 4, 1.2])
      expected_mode = _kumaraswamy_mode(a, b)
      dist = kumaraswamy_lib.Kumaraswamy(a, b)
      self.assertEqual(dist.mode().get_shape(), (3,))
      self.assertAllClose(expected_mode, dist.mode().eval())

  def testKumaraswamyModeInvalid(self):
    with session.Session():
      a = np.array([1., 2, 3])
      b = np.array([2., 4, 1.2])
      dist = kumaraswamy_lib.Kumaraswamy(a, b, allow_nan_stats=False)
      with self.assertRaisesOpError("Mode undefined for concentration1 <= 1."):
        dist.mode().eval()

      a = np.array([2., 2, 3])
      b = np.array([1., 4, 1.2])
      dist = kumaraswamy_lib.Kumaraswamy(a, b, allow_nan_stats=False)
      with self.assertRaisesOpError("Mode undefined for concentration0 <= 1."):
        dist.mode().eval()

  def testKumaraswamyModeEnableAllowNanStats(self):
    with session.Session():
      a = np.array([1., 2, 3])
      b = np.array([2., 4, 1.2])
      dist = kumaraswamy_lib.Kumaraswamy(a, b, allow_nan_stats=True)

      expected_mode = _kumaraswamy_mode(a, b)
      expected_mode[0] = np.nan
      self.assertEqual((3,), dist.mode().get_shape())
      self.assertAllClose(expected_mode, dist.mode().eval())

      a = np.array([2., 2, 3])
      b = np.array([1., 4, 1.2])
      dist = kumaraswamy_lib.Kumaraswamy(a, b, allow_nan_stats=True)

      expected_mode = _kumaraswamy_mode(a, b)
      expected_mode[0] = np.nan
      self.assertEqual((3,), dist.mode().get_shape())
      self.assertAllClose(expected_mode, dist.mode().eval())

  def testKumaraswamyEntropy(self):
    with session.Session():
      a = np.array([1., 2, 3])
      b = np.array([2., 4, 1.2])
      dist = kumaraswamy_lib.Kumaraswamy(a, b)
      self.assertEqual(dist.entropy().get_shape(), (3,))
      if not stats:
        return
      expected_entropy = (1 - 1. / a) + (
          1 - 1. / b) * _harmonic_number(b) + np.log(a * b)
      self.assertAllClose(expected_entropy, dist.entropy().eval())

  def testKumaraswamySample(self):
    with self.test_session():
      a = 1.
      b = 2.
      kumaraswamy = kumaraswamy_lib.Kumaraswamy(a, b)
      n = constant_op.constant(100000)
      samples = kumaraswamy.sample(n)
      sample_values = samples.eval()
      self.assertEqual(sample_values.shape, (100000,))
      self.assertFalse(np.any(sample_values < 0.0))
      if not stats:
        return
      self.assertLess(
          stats.kstest(
              # Kumaraswamy is a univariate distribution.
              sample_values,
              lambda x: _kumaraswamy_cdf(1., 2., x))[0],
          0.01)
      # The standard error of the sample mean is 1 / (sqrt(18 * n))
      expected_mean = _kumaraswamy_moment(a, b, 1)
      self.assertAllClose(sample_values.mean(axis=0), expected_mean, atol=1e-2)
      expected_variance = _kumaraswamy_moment(a, b, 2) - _kumaraswamy_moment(
          a, b, 1)**2
      self.assertAllClose(
          np.cov(sample_values, rowvar=0), expected_variance, atol=1e-1)

  # Test that sampling with the same seed twice gives the same results.
  def testKumaraswamySampleMultipleTimes(self):
    with self.test_session():
      a_val = 1.
      b_val = 2.
      n_val = 100

      random_seed.set_random_seed(654321)
      kumaraswamy1 = kumaraswamy_lib.Kumaraswamy(
          concentration1=a_val, concentration0=b_val, name="kumaraswamy1")
      samples1 = kumaraswamy1.sample(n_val, seed=123456).eval()

      random_seed.set_random_seed(654321)
      kumaraswamy2 = kumaraswamy_lib.Kumaraswamy(
          concentration1=a_val, concentration0=b_val, name="kumaraswamy2")
      samples2 = kumaraswamy2.sample(n_val, seed=123456).eval()

      self.assertAllClose(samples1, samples2)

  def testKumaraswamySampleMultidimensional(self):
    with self.test_session():
      a = np.random.rand(3, 2, 2).astype(np.float32)
      b = np.random.rand(3, 2, 2).astype(np.float32)
      kumaraswamy = kumaraswamy_lib.Kumaraswamy(a, b)
      n = constant_op.constant(100000)
      samples = kumaraswamy.sample(n)
      sample_values = samples.eval()
      self.assertEqual(sample_values.shape, (100000, 3, 2, 2))
      self.assertFalse(np.any(sample_values < 0.0))
      if not stats:
        return
      self.assertAllClose(
          sample_values[:, 1, :].mean(axis=0),
          _kumaraswamy_moment(a, b, 1)[1, :],
          atol=1e-1)

  def testKumaraswamyCdf(self):
    with self.test_session():
      shape = (30, 40, 50)
      for dt in (np.float32, np.float64):
        a = 10. * np.random.random(shape).astype(dt)
        b = 10. * np.random.random(shape).astype(dt)
        x = np.random.random(shape).astype(dt)
        actual = kumaraswamy_lib.Kumaraswamy(a, b).cdf(x).eval()
        self.assertAllEqual(np.ones(shape, dtype=np.bool), 0. <= x)
        self.assertAllEqual(np.ones(shape, dtype=np.bool), 1. >= x)
        if not stats:
          return
        self.assertAllClose(
            _kumaraswamy_cdf(a, b, x), actual, rtol=1e-4, atol=0)

  def testKumaraswamyLogCdf(self):
    with self.test_session():
      shape = (30, 40, 50)
      for dt in (np.float32, np.float64):
        a = 10. * np.random.random(shape).astype(dt)
        b = 10. * np.random.random(shape).astype(dt)
        x = np.random.random(shape).astype(dt)
        actual = math_ops.exp(kumaraswamy_lib.Kumaraswamy(a,
                                                          b).log_cdf(x)).eval()
        self.assertAllEqual(np.ones(shape, dtype=np.bool), 0. <= x)
        self.assertAllEqual(np.ones(shape, dtype=np.bool), 1. >= x)
        if not stats:
          return
        self.assertAllClose(
            _kumaraswamy_cdf(a, b, x), actual, rtol=1e-4, atol=0)


if __name__ == "__main__":
  test.main()
