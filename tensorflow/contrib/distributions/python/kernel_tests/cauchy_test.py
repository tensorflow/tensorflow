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
"""Tests for Cauchy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import math

import numpy as np

from tensorflow.contrib.distributions.python.ops import cauchy as cauchy_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.distributions import kullback_leibler
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
    with self.test_session():
      param_shapes = cauchy_lib.Cauchy.param_shapes(sample_shape)
      loc_shape, scale_shape = param_shapes["loc"], param_shapes["scale"]
      self.assertAllEqual(expected, loc_shape.eval())
      self.assertAllEqual(expected, scale_shape.eval())
      loc = array_ops.zeros(loc_shape)
      scale = array_ops.ones(scale_shape)
      self.assertAllEqual(
          expected,
          array_ops.shape(cauchy_lib.Cauchy(loc, scale).sample()).eval())

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
    with self.test_session():
      batch_size = 6
      loc = constant_op.constant([3.0] * batch_size)
      scale = constant_op.constant([math.sqrt(10.0)] * batch_size)
      x = np.array([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0], dtype=np.float32)
      cauchy = cauchy_lib.Cauchy(loc=loc, scale=scale)

      log_pdf = cauchy.log_prob(x)
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(),
                          log_pdf.get_shape())
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(),
                          log_pdf.eval().shape)
      self.assertAllEqual(cauchy.batch_shape, log_pdf.get_shape())
      self.assertAllEqual(cauchy.batch_shape, log_pdf.eval().shape)

      pdf = cauchy.prob(x)
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(), pdf.get_shape())
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(), pdf.eval().shape)
      self.assertAllEqual(cauchy.batch_shape, pdf.get_shape())
      self.assertAllEqual(cauchy.batch_shape, pdf.eval().shape)

      if not stats:
        return
      expected_log_pdf = stats.cauchy(loc.eval(), scale.eval()).logpdf(x)
      self.assertAllClose(expected_log_pdf, log_pdf.eval())
      self.assertAllClose(np.exp(expected_log_pdf), pdf.eval())

  def testCauchyLogPDFMultidimensional(self):
    with self.test_session():
      batch_size = 6
      loc = constant_op.constant([[3.0, -3.0]] * batch_size)
      scale = constant_op.constant([[math.sqrt(10.0), math.sqrt(15.0)]] *
                                   batch_size)
      x = np.array([[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0]], dtype=np.float32).T
      cauchy = cauchy_lib.Cauchy(loc=loc, scale=scale)

      log_pdf = cauchy.log_prob(x)
      log_pdf_values = log_pdf.eval()
      self.assertEqual(log_pdf.get_shape(), (6, 2))
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(),
                          log_pdf.get_shape())
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(),
                          log_pdf.eval().shape)
      self.assertAllEqual(cauchy.batch_shape, log_pdf.get_shape())
      self.assertAllEqual(cauchy.batch_shape, log_pdf.eval().shape)

      pdf = cauchy.prob(x)
      pdf_values = pdf.eval()
      self.assertEqual(pdf.get_shape(), (6, 2))
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(), pdf.get_shape())
      self.assertAllEqual(cauchy.batch_shape_tensor().eval(), pdf_values.shape)
      self.assertAllEqual(cauchy.batch_shape, pdf.get_shape())
      self.assertAllEqual(cauchy.batch_shape, pdf_values.shape)

      if not stats:
        return
      expected_log_pdf = stats.cauchy(loc.eval(), scale.eval()).logpdf(x)
      self.assertAllClose(expected_log_pdf, log_pdf_values)
      self.assertAllClose(np.exp(expected_log_pdf), pdf_values)





if __name__ == "__main__":
  test.main()
