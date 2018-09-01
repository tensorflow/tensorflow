# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Cauchy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import numpy as np

from tensorflow.contrib.distributions.python.ops import cauchy as cauchy_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import variables
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


class CauchyTest(test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(123)

  def assertAllFinite(self, tensor):
    is_finite = np.isfinite(tensor.eval())
    all_true = np.ones_like(is_finite, dtype=np.bool)
    self.assertAllEqual(all_true, is_finite)

  def _testParamShapes(self, sample_shape, expected):
    with self.cached_session():
      param_shapes = cauchy_lib.Cauchy.param_shapes(sample_shape)
      loc_shape, scale_shape = param_shapes["loc"], param_shapes["scale"]
      self.assertAllEqual(expected, loc_shape.eval())
      self.assertAllEqual(expected, scale_shape.eval())
      loc = array_ops.zeros(loc_shape)
      scale = array_ops.ones(scale_shape)
      self.assertAllEqual(expected,
                          array_ops.shape(
                              cauchy_lib.Cauchy(loc, scale).sample()).eval())

  def _testParamStaticShapes(self, sample_shape, expected):
    param_shapes = cauchy_lib.Cauchy.param_static_shapes(sample_shape)
    loc_shape, scale_shape = param_shapes["loc"], param_shapes["scale"]
    self.assertEqual(expected, loc_shape)
    self.assertEqual(expected, scale_shape)

  def testParamShapes(self):
    sample_shape = [10, 3, 4]
    self._testParamShapes(sample_shape, sample_shape)
    self._testParamShapes(constant_op.constant(sample_shape), sample_shape)

  def testParamStaticShapes(self):
    sample_shape = [10, 3, 4]
    self._testParamStaticShapes(sample_shape, sample_shape)
    self._testParamStaticShapes(
        tensor_shape.TensorShape(sample_shape), sample_shape)

  def testCauchyLogPDF(self):
    with self.cached_session():
      batch_size = 6
      loc = constant_op.constant([3.0] * batch_size)
      scale = constant_op.constant([np.sqrt(10.0)] * batch_size)
      x = np.array([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0], dtype=np.float32)
      cauchy = cauchy_lib.Cauchy(loc=loc, scale=scale)

      log_pdf = cauchy.log_prob(x)
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(), log_pdf.shape)
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(),
                          log_pdf.eval().shape)
      self.assertAllEqual(cauchy.batch_shape, log_pdf.shape)
      self.assertAllEqual(cauchy.batch_shape, log_pdf.eval().shape)

      pdf = cauchy.prob(x)
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(), pdf.shape)
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(), pdf.eval().shape)
      self.assertAllEqual(cauchy.batch_shape, pdf.shape)
      self.assertAllEqual(cauchy.batch_shape, pdf.eval().shape)

      if not stats:
        return
      expected_log_pdf = stats.cauchy(loc.eval(), scale.eval()).logpdf(x)
      self.assertAllClose(expected_log_pdf, log_pdf.eval())
      self.assertAllClose(np.exp(expected_log_pdf), pdf.eval())

  def testCauchyLogPDFMultidimensional(self):
    with self.cached_session():
      batch_size = 6
      loc = constant_op.constant([[3.0, -3.0]] * batch_size)
      scale = constant_op.constant(
          [[np.sqrt(10.0), np.sqrt(15.0)]] * batch_size)
      x = np.array([[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0]], dtype=np.float32).T
      cauchy = cauchy_lib.Cauchy(loc=loc, scale=scale)

      log_pdf = cauchy.log_prob(x)
      log_pdf_values = log_pdf.eval()
      self.assertEqual(log_pdf.shape, (6, 2))
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(), log_pdf.shape)
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(),
                          log_pdf.eval().shape)
      self.assertAllEqual(cauchy.batch_shape, log_pdf.shape)
      self.assertAllEqual(cauchy.batch_shape, log_pdf.eval().shape)

      pdf = cauchy.prob(x)
      pdf_values = pdf.eval()
      self.assertEqual(pdf.shape, (6, 2))
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(), pdf.shape)
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(), pdf_values.shape)
      self.assertAllEqual(cauchy.batch_shape, pdf.shape)
      self.assertAllEqual(cauchy.batch_shape, pdf_values.shape)

      if not stats:
        return
      expected_log_pdf = stats.cauchy(loc.eval(), scale.eval()).logpdf(x)
      self.assertAllClose(expected_log_pdf, log_pdf_values)
      self.assertAllClose(np.exp(expected_log_pdf), pdf_values)

  def testCauchyCDF(self):
    with self.cached_session():
      batch_size = 50
      loc = self._rng.randn(batch_size)
      scale = self._rng.rand(batch_size) + 1.0
      x = np.linspace(-8.0, 8.0, batch_size).astype(np.float64)

      cauchy = cauchy_lib.Cauchy(loc=loc, scale=scale)
      cdf = cauchy.cdf(x)
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(), cdf.shape)
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(), cdf.eval().shape)
      self.assertAllEqual(cauchy.batch_shape, cdf.shape)
      self.assertAllEqual(cauchy.batch_shape, cdf.eval().shape)
      if not stats:
        return
      expected_cdf = stats.cauchy(loc, scale).cdf(x)
      self.assertAllClose(expected_cdf, cdf.eval(), atol=0)

  def testCauchySurvivalFunction(self):
    with self.cached_session():
      batch_size = 50
      loc = self._rng.randn(batch_size)
      scale = self._rng.rand(batch_size) + 1.0
      x = np.linspace(-8.0, 8.0, batch_size).astype(np.float64)

      cauchy = cauchy_lib.Cauchy(loc=loc, scale=scale)

      sf = cauchy.survival_function(x)
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(), sf.shape)
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(), sf.eval().shape)
      self.assertAllEqual(cauchy.batch_shape, sf.shape)
      self.assertAllEqual(cauchy.batch_shape, sf.eval().shape)
      if not stats:
        return
      expected_sf = stats.cauchy(loc, scale).sf(x)
      self.assertAllClose(expected_sf, sf.eval(), atol=0)

  def testCauchyLogCDF(self):
    with self.cached_session():
      batch_size = 50
      loc = self._rng.randn(batch_size)
      scale = self._rng.rand(batch_size) + 1.0
      x = np.linspace(-100.0, 10.0, batch_size).astype(np.float64)

      cauchy = cauchy_lib.Cauchy(loc=loc, scale=scale)

      cdf = cauchy.log_cdf(x)
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(), cdf.shape)
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(), cdf.eval().shape)
      self.assertAllEqual(cauchy.batch_shape, cdf.shape)
      self.assertAllEqual(cauchy.batch_shape, cdf.eval().shape)

      if not stats:
        return
      expected_cdf = stats.cauchy(loc, scale).logcdf(x)
      self.assertAllClose(expected_cdf, cdf.eval(), atol=0, rtol=1e-5)

  def testFiniteGradientAtDifficultPoints(self):
    for dtype in [np.float32, np.float64]:
      g = ops.Graph()
      with g.as_default():
        loc = variables.Variable(dtype(0.0))
        scale = variables.Variable(dtype(1.0))
        dist = cauchy_lib.Cauchy(loc=loc, scale=scale)
        x = np.array([-100., -20., -5., 0., 5., 20., 100.]).astype(dtype)
        for func in [
            dist.cdf, dist.log_cdf, dist.survival_function,
            dist.log_survival_function, dist.log_prob, dist.prob
        ]:
          value = func(x)
          grads = gradients_impl.gradients(value, [loc, scale])
          with self.session(graph=g):
            variables.global_variables_initializer().run()
            self.assertAllFinite(value)
            self.assertAllFinite(grads[0])
            self.assertAllFinite(grads[1])

  def testCauchyLogSurvivalFunction(self):
    with self.cached_session():
      batch_size = 50
      loc = self._rng.randn(batch_size)
      scale = self._rng.rand(batch_size) + 1.0
      x = np.linspace(-10.0, 100.0, batch_size).astype(np.float64)

      cauchy = cauchy_lib.Cauchy(loc=loc, scale=scale)

      sf = cauchy.log_survival_function(x)
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(), sf.shape)
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(), sf.eval().shape)
      self.assertAllEqual(cauchy.batch_shape, sf.shape)
      self.assertAllEqual(cauchy.batch_shape, sf.eval().shape)

      if not stats:
        return
      expected_sf = stats.cauchy(loc, scale).logsf(x)
      self.assertAllClose(expected_sf, sf.eval(), atol=0, rtol=1e-5)

  def testCauchyEntropy(self):
    with self.cached_session():
      loc = np.array([1.0, 1.0, 1.0])
      scale = np.array([[1.0, 2.0, 3.0]])
      cauchy = cauchy_lib.Cauchy(loc=loc, scale=scale)

      entropy = cauchy.entropy()
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(), entropy.shape)
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(),
                          entropy.eval().shape)
      self.assertAllEqual(cauchy.batch_shape, entropy.shape)
      self.assertAllEqual(cauchy.batch_shape, entropy.eval().shape)

      if not stats:
        return
      expected_entropy = stats.cauchy(loc, scale[0]).entropy().reshape((1, 3))
      self.assertAllClose(expected_entropy, entropy.eval())

  def testCauchyMode(self):
    with self.cached_session():
      # Mu will be broadcast to [7, 7, 7].
      loc = [7.]
      scale = [11., 12., 13.]

      cauchy = cauchy_lib.Cauchy(loc=loc, scale=scale)

      self.assertAllEqual((3,), cauchy.mode().shape)
      self.assertAllEqual([7., 7, 7], cauchy.mode().eval())

  def testCauchyMean(self):
    with self.cached_session():
      loc = [1., 2., 3.]
      scale = [7.]
      cauchy = cauchy_lib.Cauchy(loc=loc, scale=scale)

      self.assertAllEqual((3,), cauchy.mean().shape)
      self.assertAllEqual([np.nan] * 3, cauchy.mean().eval())

  def testCauchyNanMean(self):
    with self.cached_session():
      loc = [1., 2., 3.]
      scale = [7.]
      cauchy = cauchy_lib.Cauchy(loc=loc, scale=scale, allow_nan_stats=False)

      with self.assertRaises(ValueError):
        cauchy.mean().eval()

  def testCauchyQuantile(self):
    with self.cached_session():
      batch_size = 50
      loc = self._rng.randn(batch_size)
      scale = self._rng.rand(batch_size) + 1.0
      p = np.linspace(0.000001, 0.999999, batch_size).astype(np.float64)

      cauchy = cauchy_lib.Cauchy(loc=loc, scale=scale)
      x = cauchy.quantile(p)

      self.assertAllEqual(cauchy.batch_shape_tensor().eval(), x.shape)
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(), x.eval().shape)
      self.assertAllEqual(cauchy.batch_shape, x.shape)
      self.assertAllEqual(cauchy.batch_shape, x.eval().shape)

      if not stats:
        return
      expected_x = stats.cauchy(loc, scale).ppf(p)
      self.assertAllClose(expected_x, x.eval(), atol=0.)

  def testCauchyVariance(self):
    with self.cached_session():
      # scale will be broadcast to [7, 7, 7]
      loc = [1., 2., 3.]
      scale = [7.]
      cauchy = cauchy_lib.Cauchy(loc=loc, scale=scale)

      self.assertAllEqual((3,), cauchy.variance().shape)
      self.assertAllEqual([np.nan] * 3, cauchy.variance().eval())

  def testCauchyNanVariance(self):
    with self.cached_session():
      # scale will be broadcast to [7, 7, 7]
      loc = [1., 2., 3.]
      scale = [7.]
      cauchy = cauchy_lib.Cauchy(loc=loc, scale=scale, allow_nan_stats=False)

      with self.assertRaises(ValueError):
        cauchy.variance().eval()

  def testCauchyStandardDeviation(self):
    with self.cached_session():
      # scale will be broadcast to [7, 7, 7]
      loc = [1., 2., 3.]
      scale = [7.]
      cauchy = cauchy_lib.Cauchy(loc=loc, scale=scale)

      self.assertAllEqual((3,), cauchy.stddev().shape)
      self.assertAllEqual([np.nan] * 3, cauchy.stddev().eval())

  def testCauchyNanStandardDeviation(self):
    with self.cached_session():
      # scale will be broadcast to [7, 7, 7]
      loc = [1., 2., 3.]
      scale = [7.]
      cauchy = cauchy_lib.Cauchy(loc=loc, scale=scale, allow_nan_stats=False)

      with self.assertRaises(ValueError):
        cauchy.stddev().eval()

  def testCauchySample(self):
    with self.cached_session():
      loc = constant_op.constant(3.0)
      scale = constant_op.constant(1.0)
      loc_v = 3.0
      n = constant_op.constant(100000)
      cauchy = cauchy_lib.Cauchy(loc=loc, scale=scale)
      samples = cauchy.sample(n)
      sample_values = samples.eval()

      self.assertEqual(sample_values.shape, (100000,))
      self.assertAllClose(np.median(sample_values), loc_v, atol=1e-1)

      expected_shape = tensor_shape.TensorShape([n.eval()]).concatenate(
          tensor_shape.TensorShape(cauchy.batch_shape_tensor().eval()))

      self.assertAllEqual(expected_shape, samples.shape)
      self.assertAllEqual(expected_shape, sample_values.shape)

      expected_shape = (
          tensor_shape.TensorShape([n.eval()]).concatenate(cauchy.batch_shape))

      self.assertAllEqual(expected_shape, samples.shape)
      self.assertAllEqual(expected_shape, sample_values.shape)

  def testCauchySampleMultiDimensional(self):
    with self.cached_session():
      batch_size = 2
      loc = constant_op.constant([[3.0, -3.0]] * batch_size)
      scale = constant_op.constant([[0.5, 1.0]] * batch_size)
      loc_v = [3.0, -3.0]
      n = constant_op.constant(100000)
      cauchy = cauchy_lib.Cauchy(loc=loc, scale=scale)
      samples = cauchy.sample(n)
      sample_values = samples.eval()
      self.assertEqual(samples.shape, (100000, batch_size, 2))
      self.assertAllClose(
          np.median(sample_values[:, 0, 0]), loc_v[0], atol=1e-1)
      self.assertAllClose(
          np.median(sample_values[:, 0, 1]), loc_v[1], atol=1e-1)

      expected_shape = tensor_shape.TensorShape([n.eval()]).concatenate(
          tensor_shape.TensorShape(cauchy.batch_shape_tensor().eval()))
      self.assertAllEqual(expected_shape, samples.shape)
      self.assertAllEqual(expected_shape, sample_values.shape)

      expected_shape = (
          tensor_shape.TensorShape([n.eval()]).concatenate(cauchy.batch_shape))
      self.assertAllEqual(expected_shape, samples.shape)
      self.assertAllEqual(expected_shape, sample_values.shape)

  def testCauchyNegativeLocFails(self):
    with self.cached_session():
      cauchy = cauchy_lib.Cauchy(loc=[1.], scale=[-5.], validate_args=True)
      with self.assertRaisesOpError("Condition x > 0 did not hold"):
        cauchy.mode().eval()

  def testCauchyShape(self):
    with self.cached_session():
      loc = constant_op.constant([-3.0] * 5)
      scale = constant_op.constant(11.0)
      cauchy = cauchy_lib.Cauchy(loc=loc, scale=scale)

      self.assertEqual(cauchy.batch_shape_tensor().eval(), [5])
      self.assertEqual(cauchy.batch_shape, tensor_shape.TensorShape([5]))
      self.assertAllEqual(cauchy.event_shape_tensor().eval(), [])
      self.assertEqual(cauchy.event_shape, tensor_shape.TensorShape([]))

  def testCauchyShapeWithPlaceholders(self):
    loc = array_ops.placeholder(dtype=dtypes.float32)
    scale = array_ops.placeholder(dtype=dtypes.float32)
    cauchy = cauchy_lib.Cauchy(loc=loc, scale=scale)

    with self.cached_session() as sess:
      # get_batch_shape should return an "<unknown>" tensor.
      self.assertEqual(cauchy.batch_shape, tensor_shape.TensorShape(None))
      self.assertEqual(cauchy.event_shape, ())
      self.assertAllEqual(cauchy.event_shape_tensor().eval(), [])
      self.assertAllEqual(
          sess.run(
              cauchy.batch_shape_tensor(),
              feed_dict={
                  loc: 5.0,
                  scale: [1.0, 2.0]
              }), [2])


if __name__ == "__main__":
  test.main()
