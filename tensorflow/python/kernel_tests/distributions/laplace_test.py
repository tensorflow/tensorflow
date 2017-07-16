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

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.distributions import laplace as laplace_lib
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


class LaplaceTest(test.TestCase):

  def testLaplaceShape(self):
    with self.test_session():
      loc = constant_op.constant([3.0] * 5)
      scale = constant_op.constant(11.0)
      laplace = laplace_lib.Laplace(loc=loc, scale=scale)

      self.assertEqual(laplace.batch_shape_tensor().eval(), (5,))
      self.assertEqual(laplace.batch_shape, tensor_shape.TensorShape([5]))
      self.assertAllEqual(laplace.event_shape_tensor().eval(), [])
      self.assertEqual(laplace.event_shape, tensor_shape.TensorShape([]))

  def testLaplaceLogPDF(self):
    with self.test_session():
      batch_size = 6
      loc = constant_op.constant([2.0] * batch_size)
      scale = constant_op.constant([3.0] * batch_size)
      loc_v = 2.0
      scale_v = 3.0
      x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)
      laplace = laplace_lib.Laplace(loc=loc, scale=scale)
      log_pdf = laplace.log_prob(x)
      self.assertEqual(log_pdf.get_shape(), (6,))
      if not stats:
        return
      expected_log_pdf = stats.laplace.logpdf(x, loc_v, scale=scale_v)
      self.assertAllClose(log_pdf.eval(), expected_log_pdf)

      pdf = laplace.prob(x)
      self.assertEqual(pdf.get_shape(), (6,))
      self.assertAllClose(pdf.eval(), np.exp(expected_log_pdf))

  def testLaplaceLogPDFMultidimensional(self):
    with self.test_session():
      batch_size = 6
      loc = constant_op.constant([[2.0, 4.0]] * batch_size)
      scale = constant_op.constant([[3.0, 4.0]] * batch_size)
      loc_v = np.array([2.0, 4.0])
      scale_v = np.array([3.0, 4.0])
      x = np.array([[2.5, 2.5, 4.0, 0.1, 1.0, 2.0]], dtype=np.float32).T
      laplace = laplace_lib.Laplace(loc=loc, scale=scale)
      log_pdf = laplace.log_prob(x)
      log_pdf_values = log_pdf.eval()
      self.assertEqual(log_pdf.get_shape(), (6, 2))

      pdf = laplace.prob(x)
      pdf_values = pdf.eval()
      self.assertEqual(pdf.get_shape(), (6, 2))
      if not stats:
        return
      expected_log_pdf = stats.laplace.logpdf(x, loc_v, scale=scale_v)
      self.assertAllClose(log_pdf_values, expected_log_pdf)
      self.assertAllClose(pdf_values, np.exp(expected_log_pdf))

  def testLaplaceLogPDFMultidimensionalBroadcasting(self):
    with self.test_session():
      batch_size = 6
      loc = constant_op.constant([[2.0, 4.0]] * batch_size)
      scale = constant_op.constant(3.0)
      loc_v = np.array([2.0, 4.0])
      scale_v = 3.0
      x = np.array([[2.5, 2.5, 4.0, 0.1, 1.0, 2.0]], dtype=np.float32).T
      laplace = laplace_lib.Laplace(loc=loc, scale=scale)
      log_pdf = laplace.log_prob(x)
      log_pdf_values = log_pdf.eval()
      self.assertEqual(log_pdf.get_shape(), (6, 2))

      pdf = laplace.prob(x)
      pdf_values = pdf.eval()
      self.assertEqual(pdf.get_shape(), (6, 2))
      if not stats:
        return
      expected_log_pdf = stats.laplace.logpdf(x, loc_v, scale=scale_v)
      self.assertAllClose(log_pdf_values, expected_log_pdf)
      self.assertAllClose(pdf_values, np.exp(expected_log_pdf))

  def testLaplaceCDF(self):
    with self.test_session():
      batch_size = 6
      loc = constant_op.constant([2.0] * batch_size)
      scale = constant_op.constant([3.0] * batch_size)
      loc_v = 2.0
      scale_v = 3.0
      x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)

      laplace = laplace_lib.Laplace(loc=loc, scale=scale)

      cdf = laplace.cdf(x)
      self.assertEqual(cdf.get_shape(), (6,))
      if not stats:
        return
      expected_cdf = stats.laplace.cdf(x, loc_v, scale=scale_v)
      self.assertAllClose(cdf.eval(), expected_cdf)

  def testLaplaceLogCDF(self):
    with self.test_session():
      batch_size = 6
      loc = constant_op.constant([2.0] * batch_size)
      scale = constant_op.constant([3.0] * batch_size)
      loc_v = 2.0
      scale_v = 3.0
      x = np.array([-2.5, 2.5, -4.0, 0.1, 1.0, 2.0], dtype=np.float32)

      laplace = laplace_lib.Laplace(loc=loc, scale=scale)

      cdf = laplace.log_cdf(x)
      self.assertEqual(cdf.get_shape(), (6,))
      if not stats:
        return
      expected_cdf = stats.laplace.logcdf(x, loc_v, scale=scale_v)
      self.assertAllClose(cdf.eval(), expected_cdf)

  def testLaplaceLogSurvivalFunction(self):
    with self.test_session():
      batch_size = 6
      loc = constant_op.constant([2.0] * batch_size)
      scale = constant_op.constant([3.0] * batch_size)
      loc_v = 2.0
      scale_v = 3.0
      x = np.array([-2.5, 2.5, -4.0, 0.1, 1.0, 2.0], dtype=np.float32)

      laplace = laplace_lib.Laplace(loc=loc, scale=scale)

      sf = laplace.log_survival_function(x)
      self.assertEqual(sf.get_shape(), (6,))
      if not stats:
        return
      expected_sf = stats.laplace.logsf(x, loc_v, scale=scale_v)
      self.assertAllClose(sf.eval(), expected_sf)

  def testLaplaceMean(self):
    with self.test_session():
      loc_v = np.array([1.0, 3.0, 2.5])
      scale_v = np.array([1.0, 4.0, 5.0])
      laplace = laplace_lib.Laplace(loc=loc_v, scale=scale_v)
      self.assertEqual(laplace.mean().get_shape(), (3,))
      if not stats:
        return
      expected_means = stats.laplace.mean(loc_v, scale=scale_v)
      self.assertAllClose(laplace.mean().eval(), expected_means)

  def testLaplaceMode(self):
    with self.test_session():
      loc_v = np.array([0.5, 3.0, 2.5])
      scale_v = np.array([1.0, 4.0, 5.0])
      laplace = laplace_lib.Laplace(loc=loc_v, scale=scale_v)
      self.assertEqual(laplace.mode().get_shape(), (3,))
      self.assertAllClose(laplace.mode().eval(), loc_v)

  def testLaplaceVariance(self):
    with self.test_session():
      loc_v = np.array([1.0, 3.0, 2.5])
      scale_v = np.array([1.0, 4.0, 5.0])
      laplace = laplace_lib.Laplace(loc=loc_v, scale=scale_v)
      self.assertEqual(laplace.variance().get_shape(), (3,))
      if not stats:
        return
      expected_variances = stats.laplace.var(loc_v, scale=scale_v)
      self.assertAllClose(laplace.variance().eval(), expected_variances)

  def testLaplaceStd(self):
    with self.test_session():
      loc_v = np.array([1.0, 3.0, 2.5])
      scale_v = np.array([1.0, 4.0, 5.0])
      laplace = laplace_lib.Laplace(loc=loc_v, scale=scale_v)
      self.assertEqual(laplace.stddev().get_shape(), (3,))
      if not stats:
        return
      expected_stddev = stats.laplace.std(loc_v, scale=scale_v)
      self.assertAllClose(laplace.stddev().eval(), expected_stddev)

  def testLaplaceEntropy(self):
    with self.test_session():
      loc_v = np.array([1.0, 3.0, 2.5])
      scale_v = np.array([1.0, 4.0, 5.0])
      laplace = laplace_lib.Laplace(loc=loc_v, scale=scale_v)
      self.assertEqual(laplace.entropy().get_shape(), (3,))
      if not stats:
        return
      expected_entropy = stats.laplace.entropy(loc_v, scale=scale_v)
      self.assertAllClose(laplace.entropy().eval(), expected_entropy)

  def testLaplaceSample(self):
    with session.Session():
      loc_v = 4.0
      scale_v = 3.0
      loc = constant_op.constant(loc_v)
      scale = constant_op.constant(scale_v)
      n = 100000
      laplace = laplace_lib.Laplace(loc=loc, scale=scale)
      samples = laplace.sample(n, seed=137)
      sample_values = samples.eval()
      self.assertEqual(samples.get_shape(), (n,))
      self.assertEqual(sample_values.shape, (n,))
      if not stats:
        return
      self.assertAllClose(
          sample_values.mean(),
          stats.laplace.mean(
              loc_v, scale=scale_v),
          rtol=0.05,
          atol=0.)
      self.assertAllClose(
          sample_values.var(),
          stats.laplace.var(loc_v, scale=scale_v),
          rtol=0.05,
          atol=0.)
      self.assertTrue(self._kstest(loc_v, scale_v, sample_values))

  def testLaplaceSampleMultiDimensional(self):
    with session.Session():
      loc_v = np.array([np.arange(1, 101, dtype=np.float32)])  # 1 x 100
      scale_v = np.array([np.arange(1, 11, dtype=np.float32)]).T  # 10 x 1
      laplace = laplace_lib.Laplace(loc=loc_v, scale=scale_v)
      n = 10000
      samples = laplace.sample(n, seed=137)
      sample_values = samples.eval()
      self.assertEqual(samples.get_shape(), (n, 10, 100))
      self.assertEqual(sample_values.shape, (n, 10, 100))
      zeros = np.zeros_like(loc_v + scale_v)  # 10 x 100
      loc_bc = loc_v + zeros
      scale_bc = scale_v + zeros
      if not stats:
        return
      self.assertAllClose(
          sample_values.mean(axis=0),
          stats.laplace.mean(
              loc_bc, scale=scale_bc),
          rtol=0.35,
          atol=0.)
      self.assertAllClose(
          sample_values.var(axis=0),
          stats.laplace.var(loc_bc, scale=scale_bc),
          rtol=0.10,
          atol=0.)
      fails = 0
      trials = 0
      for ai, a in enumerate(np.reshape(loc_v, [-1])):
        for bi, b in enumerate(np.reshape(scale_v, [-1])):
          s = sample_values[:, bi, ai]
          trials += 1
          fails += 0 if self._kstest(a, b, s) else 1
      self.assertLess(fails, trials * 0.03)

  def _kstest(self, loc, scale, samples):
    # Uses the Kolmogorov-Smirnov test for goodness of fit.
    if not stats:
      return True  # If scipy isn't available, return "True" for passing
    ks, _ = stats.kstest(samples, stats.laplace(loc, scale=scale).cdf)
    # Return True when the test passes.
    return ks < 0.02

  def testLaplacePdfOfSampleMultiDims(self):
    with session.Session() as sess:
      laplace = laplace_lib.Laplace(loc=[7., 11.], scale=[[5.], [6.]])
      num = 50000
      samples = laplace.sample(num, seed=137)
      pdfs = laplace.prob(samples)
      sample_vals, pdf_vals = sess.run([samples, pdfs])
      self.assertEqual(samples.get_shape(), (num, 2, 2))
      self.assertEqual(pdfs.get_shape(), (num, 2, 2))
      self._assertIntegral(sample_vals[:, 0, 0], pdf_vals[:, 0, 0], err=0.02)
      self._assertIntegral(sample_vals[:, 0, 1], pdf_vals[:, 0, 1], err=0.02)
      self._assertIntegral(sample_vals[:, 1, 0], pdf_vals[:, 1, 0], err=0.02)
      self._assertIntegral(sample_vals[:, 1, 1], pdf_vals[:, 1, 1], err=0.02)
      if not stats:
        return
      self.assertAllClose(
          stats.laplace.mean(
              [[7., 11.], [7., 11.]], scale=np.array([[5., 5.], [6., 6.]])),
          sample_vals.mean(axis=0),
          rtol=0.05,
          atol=0.)
      self.assertAllClose(
          stats.laplace.var([[7., 11.], [7., 11.]],
                            scale=np.array([[5., 5.], [6., 6.]])),
          sample_vals.var(axis=0),
          rtol=0.05,
          atol=0.)

  def _assertIntegral(self, sample_vals, pdf_vals, err=1e-3):
    s_p = zip(sample_vals, pdf_vals)
    prev = (0, 0)
    total = 0
    for k in sorted(s_p, key=lambda x: x[0]):
      pair_pdf = (k[1] + prev[1]) / 2
      total += (k[0] - prev[0]) * pair_pdf
      prev = k
    self.assertNear(1., total, err=err)

  def testLaplaceNonPositiveInitializationParamsRaises(self):
    with self.test_session():
      loc_v = constant_op.constant(0.0, name="loc")
      scale_v = constant_op.constant(-1.0, name="scale")
      laplace = laplace_lib.Laplace(
          loc=loc_v, scale=scale_v, validate_args=True)
      with self.assertRaisesOpError("scale"):
        laplace.mean().eval()
      loc_v = constant_op.constant(1.0, name="loc")
      scale_v = constant_op.constant(0.0, name="scale")
      laplace = laplace_lib.Laplace(
          loc=loc_v, scale=scale_v, validate_args=True)
      with self.assertRaisesOpError("scale"):
        laplace.mean().eval()

  def testLaplaceWithSoftplusScale(self):
    with self.test_session():
      loc_v = constant_op.constant([0.0, 1.0], name="loc")
      scale_v = constant_op.constant([-1.0, 2.0], name="scale")
      laplace = laplace_lib.LaplaceWithSoftplusScale(loc=loc_v, scale=scale_v)
      self.assertAllClose(nn_ops.softplus(scale_v).eval(), laplace.scale.eval())
      self.assertAllClose(loc_v.eval(), laplace.loc.eval())


if __name__ == "__main__":
  test.main()
