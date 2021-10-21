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
"""Functional tests for 3d convolutional operations."""

import itertools

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


class BetaincTest(test.TestCase):

  def _testBetaInc(self, a_s, b_s, x_s, dtype):
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top
      np_dt = dtype.as_numpy_dtype

      # Test random values
      a_s = a_s.astype(np_dt)  # in (0, infty)
      b_s = b_s.astype(np_dt)  # in (0, infty)
      x_s = x_s.astype(np_dt)  # in (0, 1)
      tf_a_s = constant_op.constant(a_s, dtype=dtype)
      tf_b_s = constant_op.constant(b_s, dtype=dtype)
      tf_x_s = constant_op.constant(x_s, dtype=dtype)
      tf_out_t = math_ops.betainc(tf_a_s, tf_b_s, tf_x_s)
      with self.cached_session():
        tf_out = self.evaluate(tf_out_t)
      scipy_out = special.betainc(a_s, b_s, x_s, dtype=np_dt)

      # the scipy version of betainc uses a double-only implementation.
      # TODO(ebrevdo): identify reasons for (sometime) precision loss
      # with doubles
      rtol = 1e-4
      atol = 1e-5
      self.assertAllCloseAccordingToType(
          scipy_out, tf_out, rtol=rtol, atol=atol)

      # Test out-of-range values (most should return nan output)
      combinations = list(itertools.product([-1, 0, 0.5, 1.0, 1.5], repeat=3))
      a_comb, b_comb, x_comb = np.asarray(list(zip(*combinations)), dtype=np_dt)
      with self.cached_session():
        tf_comb = math_ops.betainc(a_comb, b_comb, x_comb).eval()
      scipy_comb = special.betainc(a_comb, b_comb, x_comb, dtype=np_dt)
      self.assertAllCloseAccordingToType(
          scipy_comb, tf_comb, rtol=rtol, atol=atol)

      # Test broadcasting between scalars and other shapes
      with self.cached_session():
        self.assertAllCloseAccordingToType(
            special.betainc(0.1, b_s, x_s, dtype=np_dt),
            math_ops.betainc(0.1, b_s, x_s).eval(),
            rtol=rtol,
            atol=atol)
        self.assertAllCloseAccordingToType(
            special.betainc(a_s, 0.1, x_s, dtype=np_dt),
            math_ops.betainc(a_s, 0.1, x_s).eval(),
            rtol=rtol,
            atol=atol)
        self.assertAllCloseAccordingToType(
            special.betainc(a_s, b_s, 0.1, dtype=np_dt),
            math_ops.betainc(a_s, b_s, 0.1).eval(),
            rtol=rtol,
            atol=atol)
        self.assertAllCloseAccordingToType(
            special.betainc(0.1, b_s, 0.1, dtype=np_dt),
            math_ops.betainc(0.1, b_s, 0.1).eval(),
            rtol=rtol,
            atol=atol)
        self.assertAllCloseAccordingToType(
            special.betainc(0.1, 0.1, 0.1, dtype=np_dt),
            math_ops.betainc(0.1, 0.1, 0.1).eval(),
            rtol=rtol,
            atol=atol)

      with self.assertRaisesRegex(ValueError, "must be equal"):
        math_ops.betainc(0.5, [0.5], [[0.5]])

      with self.cached_session():
        with self.assertRaisesOpError("Shapes of .* are inconsistent"):
          a_p = array_ops.placeholder(dtype)
          b_p = array_ops.placeholder(dtype)
          x_p = array_ops.placeholder(dtype)
          math_ops.betainc(a_p, b_p, x_p).eval(
              feed_dict={a_p: 0.5,
                         b_p: [0.5],
                         x_p: [[0.5]]})

    except ImportError as e:
      tf_logging.warn("Cannot test special functions: %s" % str(e))

  @test_util.run_deprecated_v1
  def testBetaIncFloat(self):
    a_s = np.abs(np.random.randn(10, 10) * 30)  # in (0, infty)
    b_s = np.abs(np.random.randn(10, 10) * 30)  # in (0, infty)
    x_s = np.random.rand(10, 10)  # in (0, 1)
    self._testBetaInc(a_s, b_s, x_s, dtypes.float32)

  @test_util.run_deprecated_v1
  def testBetaIncDouble(self):
    a_s = np.abs(np.random.randn(10, 10) * 30)  # in (0, infty)
    b_s = np.abs(np.random.randn(10, 10) * 30)  # in (0, infty)
    x_s = np.random.rand(10, 10)  # in (0, 1)
    self._testBetaInc(a_s, b_s, x_s, dtypes.float64)

  @test_util.run_deprecated_v1
  def testBetaIncDoubleVeryLargeValues(self):
    a_s = np.abs(np.random.randn(10, 10) * 1e15)  # in (0, infty)
    b_s = np.abs(np.random.randn(10, 10) * 1e15)  # in (0, infty)
    x_s = np.random.rand(10, 10)  # in (0, 1)
    self._testBetaInc(a_s, b_s, x_s, dtypes.float64)

  @test_util.run_deprecated_v1
  @test_util.disable_xla("b/178338235")
  def testBetaIncDoubleVerySmallValues(self):
    a_s = np.abs(np.random.randn(10, 10) * 1e-16)  # in (0, infty)
    b_s = np.abs(np.random.randn(10, 10) * 1e-16)  # in (0, infty)
    x_s = np.random.rand(10, 10)  # in (0, 1)
    self._testBetaInc(a_s, b_s, x_s, dtypes.float64)

  @test_util.run_deprecated_v1
  @test_util.disable_xla("b/178338235")
  def testBetaIncFloatVerySmallValues(self):
    a_s = np.abs(np.random.randn(10, 10) * 1e-8)  # in (0, infty)
    b_s = np.abs(np.random.randn(10, 10) * 1e-8)  # in (0, infty)
    x_s = np.random.rand(10, 10)  # in (0, 1)
    self._testBetaInc(a_s, b_s, x_s, dtypes.float32)

  @test_util.run_deprecated_v1
  def testBetaIncFpropAndBpropAreNeverNAN(self):
    with self.cached_session() as sess:
      space = np.logspace(-8, 5).tolist()
      space_x = np.linspace(1e-16, 1 - 1e-16).tolist()
      ga_s, gb_s, gx_s = zip(*list(itertools.product(space, space, space_x)))
      # Test grads are never nan
      ga_s_t = constant_op.constant(ga_s, dtype=dtypes.float32)
      gb_s_t = constant_op.constant(gb_s, dtype=dtypes.float32)
      gx_s_t = constant_op.constant(gx_s, dtype=dtypes.float32)
      tf_gout_t = math_ops.betainc(ga_s_t, gb_s_t, gx_s_t)
      tf_gout, grads_x = sess.run(
          [tf_gout_t,
           gradients_impl.gradients(tf_gout_t, [ga_s_t, gb_s_t, gx_s_t])[2]])

      # Equivalent to `assertAllFalse` (if it existed).
      self.assertAllEqual(
          np.zeros_like(grads_x).astype(np.bool_), np.isnan(tf_gout))
      self.assertAllEqual(
          np.zeros_like(grads_x).astype(np.bool_), np.isnan(grads_x))

  @test_util.run_deprecated_v1
  def testBetaIncGrads(self):
    err_tolerance = 1e-3
    with self.cached_session():
      # Test gradient
      ga_s = np.abs(np.random.randn(2, 2) * 30)  # in (0, infty)
      gb_s = np.abs(np.random.randn(2, 2) * 30)  # in (0, infty)
      gx_s = np.random.rand(2, 2)  # in (0, 1)
      tf_ga_s = constant_op.constant(ga_s, dtype=dtypes.float64)
      tf_gb_s = constant_op.constant(gb_s, dtype=dtypes.float64)
      tf_gx_s = constant_op.constant(gx_s, dtype=dtypes.float64)
      tf_gout_t = math_ops.betainc(tf_ga_s, tf_gb_s, tf_gx_s)
      err = gradient_checker.compute_gradient_error(
          [tf_gx_s], [gx_s.shape], tf_gout_t, gx_s.shape)
      tf_logging.info("betainc gradient err = %g " % err)
      self.assertLess(err, err_tolerance)

      # Test broadcast gradient
      gx_s = np.random.rand()  # in (0, 1)
      tf_gx_s = constant_op.constant(gx_s, dtype=dtypes.float64)
      tf_gout_t = math_ops.betainc(tf_ga_s, tf_gb_s, tf_gx_s)
      err = gradient_checker.compute_gradient_error(
          [tf_gx_s], [()], tf_gout_t, ga_s.shape)
      tf_logging.info("betainc gradient err = %g " % err)
      self.assertLess(err, err_tolerance)


if __name__ == "__main__":
  test.main()
