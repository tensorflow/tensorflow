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
"""Tests for initializers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import numpy as np

from tensorflow.contrib.distributions.python.ops import half_normal as hn_lib
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


class HalfNormalTest(test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(123)

  def assertAllFinite(self, tensor):
    is_finite = np.isfinite(tensor.eval())
    all_true = np.ones_like(is_finite, dtype=np.bool)
    self.assertAllEqual(all_true, is_finite)

  def _testParamShapes(self, sample_shape, expected):
    with self.test_session():
      param_shapes = hn_lib.HalfNormal.param_shapes(sample_shape)
      scale_shape = param_shapes["scale"]
      self.assertAllEqual(expected, scale_shape.eval())
      scale = array_ops.ones(scale_shape)
      self.assertAllEqual(
          expected,
          array_ops.shape(hn_lib.HalfNormal(scale).sample()).eval())

  def _testParamStaticShapes(self, sample_shape, expected):
    param_shapes = hn_lib.HalfNormal.param_static_shapes(sample_shape)
    scale_shape = param_shapes["scale"]
    self.assertEqual(expected, scale_shape)

  def _testBatchShapes(self, dist, tensor):
    self.assertAllEqual(dist.batch_shape_tensor().eval(), tensor.shape)
    self.assertAllEqual(dist.batch_shape_tensor().eval(), tensor.eval().shape)
    self.assertAllEqual(dist.batch_shape, tensor.shape)
    self.assertAllEqual(dist.batch_shape, tensor.eval().shape)

  def testParamShapes(self):
    sample_shape = [10, 3, 4]
    self._testParamShapes(sample_shape, sample_shape)
    self._testParamShapes(constant_op.constant(sample_shape), sample_shape)

  def testParamStaticShapes(self):
    sample_shape = [10, 3, 4]
    self._testParamStaticShapes(sample_shape, sample_shape)
    self._testParamStaticShapes(
        tensor_shape.TensorShape(sample_shape), sample_shape)

  def testHalfNormalLogPDF(self):
    with self.test_session():
      batch_size = 6
      scale = constant_op.constant([3.0] * batch_size)
      x = np.array([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0], dtype=np.float32)
      halfnorm = hn_lib.HalfNormal(scale=scale)

      log_pdf = halfnorm.log_prob(x)
      self._testBatchShapes(halfnorm, log_pdf)

      pdf = halfnorm.prob(x)
      self._testBatchShapes(halfnorm, pdf)

      if not stats:
        return
      expected_log_pdf = stats.halfnorm(scale=scale.eval()).logpdf(x)
      self.assertAllClose(expected_log_pdf, log_pdf.eval())
      self.assertAllClose(np.exp(expected_log_pdf), pdf.eval())

  def testHalfNormalLogPDFMultidimensional(self):
    with self.test_session():
      batch_size = 6
      scale = constant_op.constant([[3.0, 1.0]] * batch_size)
      x = np.array([[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0]], dtype=np.float32).T
      halfnorm = hn_lib.HalfNormal(scale=scale)

      log_pdf = halfnorm.log_prob(x)
      self._testBatchShapes(halfnorm, log_pdf)

      pdf = halfnorm.prob(x)
      self._testBatchShapes(halfnorm, pdf)

      if not stats:
        return
      expected_log_pdf = stats.halfnorm(scale=scale.eval()).logpdf(x)
      self.assertAllClose(expected_log_pdf, log_pdf.eval())
      self.assertAllClose(np.exp(expected_log_pdf), pdf.eval())

  def testHalfNormalCDF(self):
    with self.test_session():
      batch_size = 50
      scale = self._rng.rand(batch_size) + 1.0
      x = np.linspace(-8.0, 8.0, batch_size).astype(np.float64)
      halfnorm = hn_lib.HalfNormal(scale=scale)

      cdf = halfnorm.cdf(x)
      self._testBatchShapes(halfnorm, cdf)

      log_cdf = halfnorm.log_cdf(x)
      self._testBatchShapes(halfnorm, log_cdf)

      if not stats:
        return
      expected_logcdf = stats.halfnorm(scale=scale).logcdf(x)
      self.assertAllClose(expected_logcdf, log_cdf.eval(), atol=0)
      self.assertAllClose(np.exp(expected_logcdf), cdf.eval(), atol=0)

  def testHalfNormalSurvivalFunction(self):
    with self.test_session():
      batch_size = 50
      scale = self._rng.rand(batch_size) + 1.0
      x = np.linspace(-8.0, 8.0, batch_size).astype(np.float64)
      halfnorm = hn_lib.HalfNormal(scale=scale)

      sf = halfnorm.survival_function(x)
      self._testBatchShapes(halfnorm, sf)

      log_sf = halfnorm.log_survival_function(x)
      self._testBatchShapes(halfnorm, log_sf)

      if not stats:
        return
      expected_logsf = stats.halfnorm(scale=scale).logsf(x)
      self.assertAllClose(expected_logsf, log_sf.eval(), atol=0)
      self.assertAllClose(np.exp(expected_logsf), sf.eval(), atol=0)

  def testHalfNormalQuantile(self):
    with self.test_session():
      batch_size = 50
      scale = self._rng.rand(batch_size) + 1.0
      p = np.linspace(0., 1.0, batch_size).astype(np.float64)

      halfnorm = hn_lib.HalfNormal(scale=scale)
      x = halfnorm.quantile(p)
      self._testBatchShapes(halfnorm, x)

      if not stats:
        return
      expected_x = stats.halfnorm(scale=scale).ppf(p)
      self.assertAllClose(expected_x, x.eval(), atol=0)

  def testFiniteGradients(self):
    for dtype in [np.float32, np.float64]:
      g = ops.Graph()
      with g.as_default():
        scale = variables.Variable(dtype(3.0))
        dist = hn_lib.HalfNormal(scale=scale)
        x = np.array([0.01, 0.1, 1., 5., 10.]).astype(dtype)
        for func in [
            dist.cdf, dist.log_cdf, dist.survival_function,
            dist.log_prob, dist.prob, dist.log_survival_function,
        ]:
          print(func.__name__)
          value = func(x)
          grads = gradients_impl.gradients(value, [scale])
          with self.test_session(graph=g):
            variables.global_variables_initializer().run()
            self.assertAllFinite(value)
            self.assertAllFinite(grads[0])

  def testHalfNormalEntropy(self):
    with self.test_session():
      scale = np.array([[1.0, 2.0, 3.0]])
      halfnorm = hn_lib.HalfNormal(scale=scale)

      # See https://en.wikipedia.org/wiki/Half-normal_distribution for the
      # entropy formula used here.
      expected_entropy = 0.5 * np.log(np.pi * scale ** 2.0 / 2.0) + 0.5

      entropy = halfnorm.entropy()
      self._testBatchShapes(halfnorm, entropy)
      self.assertAllClose(expected_entropy, entropy.eval())

  def testHalfNormalMeanAndMode(self):
    with self.test_session():
      scale = np.array([11., 12., 13.])

      halfnorm = hn_lib.HalfNormal(scale=scale)
      expected_mean = scale * np.sqrt(2.0) / np.sqrt(np.pi)

      self.assertAllEqual((3,), halfnorm.mean().eval().shape)
      self.assertAllEqual(expected_mean, halfnorm.mean().eval())

      self.assertAllEqual((3,), halfnorm.mode().eval().shape)
      self.assertAllEqual([0., 0., 0.], halfnorm.mode().eval())

  def testHalfNormalVariance(self):
    with self.test_session():
      scale = np.array([7., 7., 7.])
      halfnorm = hn_lib.HalfNormal(scale=scale)
      expected_variance = scale ** 2.0 * (1.0 - 2.0 / np.pi)

      self.assertAllEqual((3,), halfnorm.variance().eval().shape)
      self.assertAllEqual(expected_variance, halfnorm.variance().eval())

  def testHalfNormalStandardDeviation(self):
    with self.test_session():
      scale = np.array([7., 7., 7.])
      halfnorm = hn_lib.HalfNormal(scale=scale)
      expected_variance = scale ** 2.0 * (1.0 - 2.0 / np.pi)

      self.assertAllEqual((3,), halfnorm.stddev().shape)
      self.assertAllEqual(np.sqrt(expected_variance), halfnorm.stddev().eval())

  def testHalfNormalSample(self):
    with self.test_session():
      scale = constant_op.constant(3.0)
      n = constant_op.constant(100000)
      halfnorm = hn_lib.HalfNormal(scale=scale)

      sample = halfnorm.sample(n)

      self.assertEqual(sample.eval().shape, (100000,))
      self.assertAllClose(sample.eval().mean(),
                          3.0 * np.sqrt(2.0) / np.sqrt(np.pi), atol=1e-1)

      expected_shape = tensor_shape.TensorShape([n.eval()]).concatenate(
          tensor_shape.TensorShape(halfnorm.batch_shape_tensor().eval()))
      self.assertAllEqual(expected_shape, sample.shape)
      self.assertAllEqual(expected_shape, sample.eval().shape)

      expected_shape_static = (tensor_shape.TensorShape(
          [n.eval()]).concatenate(halfnorm.batch_shape))
      self.assertAllEqual(expected_shape_static, sample.shape)
      self.assertAllEqual(expected_shape_static, sample.eval().shape)

  def testHalfNormalSampleMultiDimensional(self):
    with self.test_session():
      batch_size = 2
      scale = constant_op.constant([[2.0, 3.0]] * batch_size)
      n = constant_op.constant(100000)
      halfnorm = hn_lib.HalfNormal(scale=scale)

      sample = halfnorm.sample(n)
      self.assertEqual(sample.shape, (100000, batch_size, 2))
      self.assertAllClose(sample.eval()[:, 0, 0].mean(),
                          2.0 * np.sqrt(2.0) / np.sqrt(np.pi), atol=1e-1)
      self.assertAllClose(sample.eval()[:, 0, 1].mean(),
                          3.0 * np.sqrt(2.0) / np.sqrt(np.pi), atol=1e-1)

      expected_shape = tensor_shape.TensorShape([n.eval()]).concatenate(
          tensor_shape.TensorShape(halfnorm.batch_shape_tensor().eval()))
      self.assertAllEqual(expected_shape, sample.shape)
      self.assertAllEqual(expected_shape, sample.eval().shape)

      expected_shape_static = (tensor_shape.TensorShape(
          [n.eval()]).concatenate(halfnorm.batch_shape))
      self.assertAllEqual(expected_shape_static, sample.shape)
      self.assertAllEqual(expected_shape_static, sample.eval().shape)

  def testNegativeSigmaFails(self):
    with self.test_session():
      halfnorm = hn_lib.HalfNormal(scale=[-5.], validate_args=True, name="G")
      with self.assertRaisesOpError("Condition x > 0 did not hold"):
        halfnorm.mean().eval()

  def testHalfNormalShape(self):
    with self.test_session():
      scale = constant_op.constant([6.0] * 5)
      halfnorm = hn_lib.HalfNormal(scale=scale)

      self.assertEqual(halfnorm.batch_shape_tensor().eval(), [5])
      self.assertEqual(halfnorm.batch_shape, tensor_shape.TensorShape([5]))
      self.assertAllEqual(halfnorm.event_shape_tensor().eval(), [])
      self.assertEqual(halfnorm.event_shape, tensor_shape.TensorShape([]))

  def testHalfNormalShapeWithPlaceholders(self):
    scale = array_ops.placeholder(dtype=dtypes.float32)
    halfnorm = hn_lib.HalfNormal(scale=scale)

    with self.test_session() as sess:
      # get_batch_shape should return an "<unknown>" tensor.
      self.assertEqual(halfnorm.batch_shape, tensor_shape.TensorShape(None))
      self.assertEqual(halfnorm.event_shape, ())
      self.assertAllEqual(halfnorm.event_shape_tensor().eval(), [])
      self.assertAllEqual(
          sess.run(halfnorm.batch_shape_tensor(),
                   feed_dict={scale: [1.0, 2.0]}), [2])


if __name__ == "__main__":
  test.main()
