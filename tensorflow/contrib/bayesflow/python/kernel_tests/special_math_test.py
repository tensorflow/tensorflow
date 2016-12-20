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
"""Tests for Special Math Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
from scipy import special
from scipy import stats
import tensorflow as tf

sm = tf.contrib.bayesflow.special_math


def _check_strictly_increasing(array_1d):
  diff = np.diff(array_1d)
  np.testing.assert_array_less(0, diff)


def _make_grid(dtype, grid_spec):
  """Returns a uniform grid + noise, reshaped to shape argument."""
  rng = np.random.RandomState(0)
  num_points = np.prod(grid_spec.shape)
  grid = np.linspace(
      grid_spec.min, grid_spec.max, num=num_points).astype(dtype)
  grid_spacing = (grid_spec.max - grid_spec.min) / num_points
  grid += 0.1 * grid_spacing * rng.randn(*grid.shape)
  # More useful if it's sorted (e.g. for testing monotonicity, or debugging).
  grid = np.sort(grid)
  return np.reshape(grid, grid_spec.shape)


GridSpec = collections.namedtuple("GridSpec", ["min", "max", "shape"])


ErrorSpec = collections.namedtuple("ErrorSpec", ["rtol", "atol"])


class NdtrTest(tf.test.TestCase):
  _use_log = False
  # Grid min/max chosen to ensure 0 < cdf(x) < 1.
  _grid32 = GridSpec(min=-12.9, max=5., shape=[100])
  _grid64 = GridSpec(min=-37.5, max=8., shape=[100])
  _error32 = ErrorSpec(rtol=1e-4, atol=0.)
  _error64 = ErrorSpec(rtol=1e-6, atol=0.)

  def _test_grid(self, dtype, grid_spec, error_spec):
    if self._use_log:
      self._test_grid_log(dtype, grid_spec, error_spec)
    else:
      self._test_grid_no_log(dtype, grid_spec, error_spec)

  def _test_grid_log(self, dtype, grid_spec, error_spec):
    with self.test_session():
      grid = _make_grid(dtype, grid_spec)
      actual = sm.log_ndtr(grid).eval()

      # Basic tests.
      self.assertTrue(np.isfinite(actual).all())
      # On the grid, -inf < log_cdf(x) < 0.  In this case, we should be able
      # to use a huge grid because we have used tricks to escape numerical
      # difficulties.
      self.assertTrue((actual < 0).all())
      _check_strictly_increasing(actual)

      # Versus scipy.
      expected = special.log_ndtr(grid)
      # Scipy prematurely goes to zero at some places that we don't.  So don't
      # include these in the comparison.
      self.assertAllClose(expected.astype(np.float64)[expected < 0],
                          actual.astype(np.float64)[expected < 0],
                          rtol=error_spec.rtol, atol=error_spec.atol)

  def _test_grid_no_log(self, dtype, grid_spec, error_spec):
    with self.test_session():
      grid = _make_grid(dtype, grid_spec)
      actual = sm.ndtr(grid).eval()

      # Basic tests.
      self.assertTrue(np.isfinite(actual).all())
      # On the grid, 0 < cdf(x) < 1.  The grid cannot contain everything due
      # to numerical limitations of cdf.
      self.assertTrue((actual > 0).all())
      self.assertTrue((actual < 1).all())
      _check_strictly_increasing(actual)

      # Versus scipy.
      expected = special.ndtr(grid)
      # Scipy prematurely goes to zero at some places that we don't.  So don't
      # include these in the comparison.
      self.assertAllClose(expected.astype(np.float64)[expected < 0],
                          actual.astype(np.float64)[expected < 0],
                          rtol=error_spec.rtol, atol=error_spec.atol)

  def test_float32(self):
    self._test_grid(np.float32, self._grid32, self._error32)

  def test_float64(self):
    self._test_grid(np.float64, self._grid64, self._error64)


class LogNdtrTestLower(NdtrTest):
  _use_log = True
  _grid32 = GridSpec(min=-100., max=sm.LOGNDTR_FLOAT32_LOWER, shape=[100])
  _grid64 = GridSpec(min=-100., max=sm.LOGNDTR_FLOAT64_LOWER, shape=[100])
  _error32 = ErrorSpec(rtol=1e-4, atol=0.)
  _error64 = ErrorSpec(rtol=1e-4, atol=0.)


# The errors are quite large when the input is > 6 or so.  Also,
# scipy.special.log_ndtr becomes zero very early, before 10,
# (due to ndtr becoming 1).  We approximate Log[1 + epsilon] as epsilon, and
# avoid this issue.
class LogNdtrTestMid(NdtrTest):
  _use_log = True
  _grid32 = GridSpec(
      min=sm.LOGNDTR_FLOAT32_LOWER,
      max=sm.LOGNDTR_FLOAT32_UPPER,
      shape=[100])
  _grid64 = GridSpec(
      min=sm.LOGNDTR_FLOAT64_LOWER,
      max=sm.LOGNDTR_FLOAT64_UPPER,
      shape=[100])
  # Differences show up as soon as we're in the tail, so add some atol.
  _error32 = ErrorSpec(rtol=0.1, atol=1e-7)
  _error64 = ErrorSpec(rtol=0.1, atol=1e-7)


class LogNdtrTestUpper(NdtrTest):
  _use_log = True
  _grid32 = GridSpec(
      min=sm.LOGNDTR_FLOAT32_UPPER,
      max=12.,  # Beyond this, log_cdf(x) may be zero.
      shape=[100])
  _grid64 = GridSpec(
      min=sm.LOGNDTR_FLOAT64_UPPER,
      max=35.,  # Beyond this, log_cdf(x) may be zero.
      shape=[100])
  _error32 = ErrorSpec(rtol=1e-6, atol=1e-14)
  _error64 = ErrorSpec(rtol=1e-6, atol=1e-14)


class NdtrGradientTest(tf.test.TestCase):
  _use_log = False
  _grid = GridSpec(min=-100., max=100., shape=[1, 2, 3, 8])
  _error32 = ErrorSpec(rtol=1e-4, atol=0)
  _error64 = ErrorSpec(rtol=1e-7, atol=0)

  def assert_all_true(self, v):
    self.assertAllEqual(np.ones_like(v, dtype=np.bool), v)

  def assert_all_false(self, v):
    self.assertAllEqual(np.zeros_like(v, dtype=np.bool), v)

  def _test_grad_finite(self, dtype):
    with self.test_session():
      x = tf.Variable([-100., 0., 100.], dtype=dtype)
      output = (sm.log_ndtr(x) if self._use_log else sm.ndtr(x))
      grad_output = tf.gradients(output, x)
      tf.global_variables_initializer().run()
      self.assert_all_true(np.isfinite(output.eval()))
      self.assert_all_true(np.isfinite(grad_output[0].eval()))

  def _test_grad_accuracy(self, dtype, grid_spec, error_spec):
    raw_grid = _make_grid(dtype, grid_spec)
    grid = tf.convert_to_tensor(raw_grid)
    with self.test_session():
      fn = sm.log_ndtr if self._use_log else sm.ndtr

      # If there are N points in the grid,
      # grad_eval.shape = (N, N), with grad_eval[i, j] the partial derivative of
      # the ith output point w.r.t. the jth grid point.  We only expect the
      # diagonal to be nonzero.
      # TODO(b/31131137): Replace tf.test.compute_gradient with our own custom
      # gradient evaluation to ensure we correctly handle small function delta.
      grad_eval, _ = tf.test.compute_gradient(
          grid, grid_spec.shape, fn(grid), grid_spec.shape)
      grad_eval = np.diag(grad_eval)

      # Check for NaN separately in order to get informative failures.
      self.assert_all_false(np.isnan(grad_eval))
      self.assert_all_true(grad_eval > 0.)
      self.assert_all_true(np.isfinite(grad_eval))

      # Do the same checks but explicitly compute the gradient.
      # (We did this because we're not sure if we trust
      # tf.test.compute_gradient.)
      grad_eval = tf.gradients(fn(grid), grid)[0].eval()
      self.assert_all_false(np.isnan(grad_eval))
      if self._use_log:
        g = np.reshape(grad_eval, [-1])
        half = np.ceil(len(g)/2)
        self.assert_all_true(g[:half] > 0.)
        self.assert_all_true(g[half:] >= 0.)
      else:
        # The ndtr gradient will only be non-zero in the range [-14, 14] for
        # float32 and [-38, 38] for float64.
        self.assert_all_true(grad_eval >= 0.)
      self.assert_all_true(np.isfinite(grad_eval))

      # Versus scipy.
      expected = stats.norm.pdf(raw_grid)
      if self._use_log:
        expected /= special.ndtr(raw_grid)
        expected[np.isnan(expected)] = 0.
      # Scipy prematurely goes to zero at some places that we don't.  So don't
      # include these in the comparison.
      self.assertAllClose(expected.astype(np.float64)[expected < 0],
                          grad_eval.astype(np.float64)[expected < 0],
                          rtol=error_spec.rtol, atol=error_spec.atol)

  def test_float32(self):
    self._test_grad_accuracy(np.float32, self._grid, self._error32)
    self._test_grad_finite(np.float32)

  def test_float64(self):
    self._test_grad_accuracy(np.float64, self._grid, self._error64)
    self._test_grad_finite(np.float64)


class LogNdtrGradientTest(NdtrGradientTest):
  _use_log = True


if __name__ == "__main__":
  tf.test.main()
