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
"""Tests for Sample Stats Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import sample_stats
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import spectral_ops_test_util
from tensorflow.python.platform import test

rng = np.random.RandomState(0)


class _AutoCorrelationTest(object):

  @property
  def use_static_shape(self):
    raise NotImplementedError("Subclass failed to implement `use_static_shape`")

  @property
  def dtype(self):
    raise NotImplementedError("Subclass failed to implement `dtype`.")

  def test_constant_sequence_axis_0_max_lags_none_center_false(self):
    x_ = np.array([[0., 0., 0.],
                   [1., 1., 1.]]).astype(self.dtype)
    x_ph = array_ops.placeholder_with_default(
        input=x_,
        shape=x_.shape if self.use_static_shape else None)
    with spectral_ops_test_util.fft_kernel_label_map():
      with self.test_session() as sess:
        # Setting normalize = True means we divide by zero.
        auto_corr = sample_stats.auto_correlation(
            x_ph, axis=1, center=False, normalize=False)
        if self.use_static_shape:
          self.assertEqual((2, 3), auto_corr.shape)
        auto_corr_ = sess.run(auto_corr)
        self.assertAllClose(
            [[0., 0., 0.],
             [1., 1., 1.]], auto_corr_)

  def test_constant_sequence_axis_0_max_lags_none_center_true(self):
    x_ = np.array([[0., 0., 0.],
                   [1., 1., 1.]]).astype(self.dtype)
    x_ph = array_ops.placeholder_with_default(
        input=x_,
        shape=x_.shape if self.use_static_shape else None)
    with spectral_ops_test_util.fft_kernel_label_map():
      with self.test_session() as sess:
        # Setting normalize = True means we divide by zero.
        auto_corr = sample_stats.auto_correlation(
            x_ph, axis=1, normalize=False, center=True)
        if self.use_static_shape:
          self.assertEqual((2, 3), auto_corr.shape)
        auto_corr_ = sess.run(auto_corr)
        self.assertAllClose(
            [[0., 0., 0.],
             [0., 0., 0.]], auto_corr_)

  def check_results_versus_brute_force(
      self, x, axis, max_lags, center, normalize):
    """Compute auto-correlation by brute force, then compare to tf result."""
    # Brute for auto-corr -- avoiding fft and transpositions.
    axis_len = x.shape[axis]
    if max_lags is None:
      max_lags = axis_len - 1
    else:
      max_lags = min(axis_len - 1, max_lags)
    auto_corr_at_lag = []
    if center:
      x -= x.mean(axis=axis, keepdims=True)
    for m in range(max_lags + 1):
      auto_corr_at_lag.append((
          np.take(x, indices=range(0, axis_len - m), axis=axis) *
          np.conj(np.take(x, indices=range(m, axis_len), axis=axis))
      ).mean(axis=axis, keepdims=True))
    rxx = np.concatenate(auto_corr_at_lag, axis=axis)
    if normalize:
      rxx /= np.take(rxx, [0], axis=axis)

    x_ph = array_ops.placeholder_with_default(
        x, shape=x.shape if self.use_static_shape else None)
    with spectral_ops_test_util.fft_kernel_label_map():
      with self.test_session():
        auto_corr = sample_stats.auto_correlation(
            x_ph, axis=axis, max_lags=max_lags, center=center,
            normalize=normalize)
        if self.use_static_shape:
          output_shape = list(x.shape)
          output_shape[axis] = max_lags + 1
          self.assertAllEqual(output_shape, auto_corr.shape)
        self.assertAllClose(rxx, auto_corr.eval(), rtol=1e-5, atol=1e-5)

  def test_axis_n1_center_false_max_lags_none(self):
    x = rng.randn(2, 3, 4).astype(self.dtype)
    if self.dtype in [np.complex64]:
      x = 1j * rng.randn(2, 3, 4).astype(self.dtype)
    self.check_results_versus_brute_force(
        x, axis=-1, max_lags=None, center=False, normalize=False)

  def test_axis_n2_center_false_max_lags_none(self):
    x = rng.randn(3, 4, 5).astype(self.dtype)
    if self.dtype in [np.complex64]:
      x = 1j * rng.randn(3, 4, 5).astype(self.dtype)
    self.check_results_versus_brute_force(
        x, axis=-2, max_lags=None, center=False, normalize=False)

  def test_axis_n1_center_false_max_lags_none_normalize_true(self):
    x = rng.randn(2, 3, 4).astype(self.dtype)
    if self.dtype in [np.complex64]:
      x = 1j * rng.randn(2, 3, 4).astype(self.dtype)
    self.check_results_versus_brute_force(
        x, axis=-1, max_lags=None, center=False, normalize=True)

  def test_axis_n2_center_false_max_lags_none_normalize_true(self):
    x = rng.randn(3, 4, 5).astype(self.dtype)
    if self.dtype in [np.complex64]:
      x = 1j * rng.randn(3, 4, 5).astype(self.dtype)
    self.check_results_versus_brute_force(
        x, axis=-2, max_lags=None, center=False, normalize=True)

  def test_axis_0_center_true_max_lags_none(self):
    x = rng.randn(3, 4, 5).astype(self.dtype)
    if self.dtype in [np.complex64]:
      x = 1j * rng.randn(3, 4, 5).astype(self.dtype)
    self.check_results_versus_brute_force(
        x, axis=0, max_lags=None, center=True, normalize=False)

  def test_axis_2_center_true_max_lags_1(self):
    x = rng.randn(3, 4, 5).astype(self.dtype)
    if self.dtype in [np.complex64]:
      x = 1j * rng.randn(3, 4, 5).astype(self.dtype)
    self.check_results_versus_brute_force(
        x, axis=2, max_lags=1, center=True, normalize=False)

  def test_axis_2_center_true_max_lags_100(self):
    # There are less than 100 elements in axis 2, so expect we get back an array
    # the same size as x, despite having asked for 100 lags.
    x = rng.randn(3, 4, 5).astype(self.dtype)
    if self.dtype in [np.complex64]:
      x = 1j * rng.randn(3, 4, 5).astype(self.dtype)
    self.check_results_versus_brute_force(
        x, axis=2, max_lags=100, center=True, normalize=False)

  def test_long_orthonormal_sequence_has_corr_length_0(self):
    l = 10000
    x = rng.randn(l).astype(self.dtype)
    x_ph = array_ops.placeholder_with_default(
        x, shape=(l,) if self.use_static_shape else None)
    with spectral_ops_test_util.fft_kernel_label_map():
      with self.test_session():
        rxx = sample_stats.auto_correlation(
            x_ph, max_lags=l // 2, center=True, normalize=False)
        if self.use_static_shape:
          self.assertAllEqual((l // 2 + 1,), rxx.shape)
        rxx_ = rxx.eval()
        # OSS CPU FFT has some accuracy issues is not the most accurate.
        # So this tolerance is a bit bad.
        self.assertAllClose(1., rxx_[0], rtol=0.05)
        # The maximal error in the rest of the sequence is not great.
        self.assertAllClose(np.zeros(l // 2), rxx_[1:], atol=0.1)
        # The mean error in the rest is ok, actually 0.008 when I tested it.
        self.assertLess(np.abs(rxx_[1:]).mean(), 0.02)

  def test_step_function_sequence(self):
    # x jumps to new random value every 10 steps.  So correlation length = 10.
    x = (rng.randint(-10, 10, size=(1000, 1))
         * np.ones((1, 10))).ravel().astype(self.dtype)
    x_ph = array_ops.placeholder_with_default(
        x, shape=(1000 * 10,) if self.use_static_shape else None)
    with spectral_ops_test_util.fft_kernel_label_map():
      with self.test_session():
        rxx = sample_stats.auto_correlation(
            x_ph, max_lags=1000 * 10 // 2, center=True, normalize=False)
        if self.use_static_shape:
          self.assertAllEqual((1000 * 10 // 2 + 1,), rxx.shape)
        rxx_ = rxx.eval()
        rxx_ /= rxx_[0]
        # Expect positive correlation for the first 10 lags, then significantly
        # smaller negative.
        self.assertGreater(rxx_[:10].min(), 0)
        self.assertGreater(rxx_[9], 5 * rxx_[10:20].mean())
        # RXX should be decreasing for the first 10 lags.
        diff = np.diff(rxx_)
        self.assertLess(diff[:10].max(), 0)

  def test_normalization(self):
    l = 10000
    x = 3 * rng.randn(l).astype(self.dtype)
    x_ph = array_ops.placeholder_with_default(
        x, shape=(l,) if self.use_static_shape else None)
    with spectral_ops_test_util.fft_kernel_label_map():
      with self.test_session():
        rxx = sample_stats.auto_correlation(
            x_ph, max_lags=l // 2, center=True, normalize=True)
        if self.use_static_shape:
          self.assertAllEqual((l // 2 + 1,), rxx.shape)
        rxx_ = rxx.eval()
        # Note that RXX[0] = 1, despite the fact that E[X^2] = 9, and this is
        # due to normalize=True.
        # OSS CPU FFT has some accuracy issues is not the most accurate.
        # So this tolerance is a bit bad.
        self.assertAllClose(1., rxx_[0], rtol=0.05)
        # The maximal error in the rest of the sequence is not great.
        self.assertAllClose(np.zeros(l // 2), rxx_[1:], atol=0.1)
        # The mean error in the rest is ok, actually 0.008 when I tested it.
        self.assertLess(np.abs(rxx_[1:]).mean(), 0.02)


class AutoCorrelationTestStaticShapeFloat32(test.TestCase,
                                            _AutoCorrelationTest):

  @property
  def dtype(self):
    return np.float32

  @property
  def use_static_shape(self):
    return True


class AutoCorrelationTestStaticShapeComplex64(test.TestCase,
                                              _AutoCorrelationTest):

  @property
  def dtype(self):
    return np.complex64

  @property
  def use_static_shape(self):
    return True


class AutoCorrelationTestDynamicShapeFloat32(test.TestCase,
                                             _AutoCorrelationTest):

  @property
  def dtype(self):
    return np.float32

  @property
  def use_static_shape(self):
    return False


class PercentileTestWithLowerInterpolation(test.TestCase):

  _interpolation = "lower"

  def test_one_dim_odd_input(self):
    x = [1., 5., 3., 2., 4.]
    for q in [0, 10, 25, 49.9, 50, 50.01, 90, 95, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation, axis=0)
      with self.test_session():
        pct = sample_stats.percentile(
            x, q=q, interpolation=self._interpolation, axis=[0])
        self.assertAllEqual((), pct.get_shape())
        self.assertAllClose(expected_percentile, pct.eval())

  def test_one_dim_even_input(self):
    x = [1., 5., 3., 2., 4., 5.]
    for q in [0, 10, 25, 49.9, 50, 50.01, 90, 95, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation)
      with self.test_session():
        pct = sample_stats.percentile(x, q=q, interpolation=self._interpolation)
        self.assertAllEqual((), pct.get_shape())
        self.assertAllClose(expected_percentile, pct.eval())

  def test_two_dim_odd_input_axis_0(self):
    x = np.array([[-1., 50., -3.5, 2., -1], [0., 0., 3., 2., 4.]]).T
    for q in [0, 10, 25, 49.9, 50, 50.01, 90, 95, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation, axis=0)
      with self.test_session():
        # Get dim 1 with negative and positive indices.
        pct_neg_index = sample_stats.percentile(
            x, q=q, interpolation=self._interpolation, axis=[0])
        pct_pos_index = sample_stats.percentile(
            x, q=q, interpolation=self._interpolation, axis=[0])
        self.assertAllEqual((2,), pct_neg_index.get_shape())
        self.assertAllEqual((2,), pct_pos_index.get_shape())
        self.assertAllClose(expected_percentile, pct_neg_index.eval())
        self.assertAllClose(expected_percentile, pct_pos_index.eval())

  def test_two_dim_even_axis_0(self):
    x = np.array([[1., 2., 4., 50.], [1., 2., -4., 5.]]).T
    for q in [0, 10, 25, 49.9, 50, 50.01, 90, 95, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation, axis=0)
      with self.test_session():
        pct = sample_stats.percentile(
            x, q=q, interpolation=self._interpolation, axis=[0])
        self.assertAllEqual((2,), pct.get_shape())
        self.assertAllClose(expected_percentile, pct.eval())

  def test_two_dim_even_input_and_keep_dims_true(self):
    x = np.array([[1., 2., 4., 50.], [1., 2., -4., 5.]]).T
    for q in [0, 10, 25, 49.9, 50, 50.01, 90, 95, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation, keepdims=True, axis=0)
      with self.test_session():
        pct = sample_stats.percentile(
            x,
            q=q,
            interpolation=self._interpolation,
            keep_dims=True,
            axis=[0])
        self.assertAllEqual((1, 2), pct.get_shape())
        self.assertAllClose(expected_percentile, pct.eval())

  def test_four_dimensional_input(self):
    x = rng.rand(2, 3, 4, 5)
    for axis in [None, 0, 1, -2, (0,), (-1,), (-1, 1), (3, 1), (-3, 0)]:
      expected_percentile = np.percentile(
          x, q=0.77, interpolation=self._interpolation, axis=axis)
      with self.test_session():
        pct = sample_stats.percentile(
            x,
            q=0.77,
            interpolation=self._interpolation,
            axis=axis)
        self.assertAllEqual(expected_percentile.shape, pct.get_shape())
        self.assertAllClose(expected_percentile, pct.eval())

  def test_four_dimensional_input_and_keepdims(self):
    x = rng.rand(2, 3, 4, 5)
    for axis in [None, 0, 1, -2, (0,), (-1,), (-1, 1), (3, 1), (-3, 0)]:
      expected_percentile = np.percentile(
          x,
          q=0.77,
          interpolation=self._interpolation,
          axis=axis,
          keepdims=True)
      with self.test_session():
        pct = sample_stats.percentile(
            x,
            q=0.77,
            interpolation=self._interpolation,
            axis=axis,
            keep_dims=True)
        self.assertAllEqual(expected_percentile.shape, pct.get_shape())
        self.assertAllClose(expected_percentile, pct.eval())

  def test_four_dimensional_input_x_static_ndims_but_dynamic_sizes(self):
    x = rng.rand(2, 3, 4, 5)
    x_ph = array_ops.placeholder(dtypes.float64, shape=[None, None, None, None])
    for axis in [None, 0, 1, -2, (0,), (-1,), (-1, 1), (3, 1), (-3, 0)]:
      expected_percentile = np.percentile(
          x, q=0.77, interpolation=self._interpolation, axis=axis)
      with self.test_session():
        pct = sample_stats.percentile(
            x_ph,
            q=0.77,
            interpolation=self._interpolation,
            axis=axis)
        self.assertAllClose(expected_percentile, pct.eval(feed_dict={x_ph: x}))

  def test_four_dimensional_input_and_keepdims_x_static_ndims_dynamic_sz(self):
    x = rng.rand(2, 3, 4, 5)
    x_ph = array_ops.placeholder(dtypes.float64, shape=[None, None, None, None])
    for axis in [None, 0, 1, -2, (0,), (-1,), (-1, 1), (3, 1), (-3, 0)]:
      expected_percentile = np.percentile(
          x,
          q=0.77,
          interpolation=self._interpolation,
          axis=axis,
          keepdims=True)
      with self.test_session():
        pct = sample_stats.percentile(
            x_ph,
            q=0.77,
            interpolation=self._interpolation,
            axis=axis,
            keep_dims=True)
        self.assertAllClose(expected_percentile, pct.eval(feed_dict={x_ph: x}))

  def test_with_integer_dtype(self):
    x = [1, 5, 3, 2, 4]
    for q in [0, 10, 25, 49.9, 50, 50.01, 90, 95, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation)
      with self.test_session():
        pct = sample_stats.percentile(x, q=q, interpolation=self._interpolation)
        self.assertEqual(dtypes.int32, pct.dtype)
        self.assertAllEqual((), pct.get_shape())
        self.assertAllClose(expected_percentile, pct.eval())


class PercentileTestWithHigherInterpolation(
    PercentileTestWithLowerInterpolation):

  _interpolation = "higher"


class PercentileTestWithNearestInterpolation(test.TestCase):
  """Test separately because np.round and tf.round make different choices."""

  _interpolation = "nearest"

  def test_one_dim_odd_input(self):
    x = [1., 5., 3., 2., 4.]
    for q in [0, 10.1, 25.1, 49.9, 50.1, 50.01, 89, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation)
      with self.test_session():
        pct = sample_stats.percentile(x, q=q, interpolation=self._interpolation)
        self.assertAllEqual((), pct.get_shape())
        self.assertAllClose(expected_percentile, pct.eval())

  def test_one_dim_even_input(self):
    x = [1., 5., 3., 2., 4., 5.]
    for q in [0, 10.1, 25.1, 49.9, 50.1, 50.01, 89, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation)
      with self.test_session():
        pct = sample_stats.percentile(x, q=q, interpolation=self._interpolation)
        self.assertAllEqual((), pct.get_shape())
        self.assertAllClose(expected_percentile, pct.eval())

  def test_invalid_interpolation_raises(self):
    x = [1., 5., 3., 2., 4.]
    with self.assertRaisesRegexp(ValueError, "interpolation"):
      sample_stats.percentile(x, q=0.5, interpolation="bad")

  def test_vector_q_raises_static(self):
    x = [1., 5., 3., 2., 4.]
    with self.assertRaisesRegexp(ValueError, "Expected.*ndims"):
      sample_stats.percentile(x, q=[0.5])

  def test_vector_q_raises_dynamic(self):
    x = [1., 5., 3., 2., 4.]
    q_ph = array_ops.placeholder(dtypes.float32)
    pct = sample_stats.percentile(x, q=q_ph, validate_args=True)
    with self.test_session():
      with self.assertRaisesOpError("rank"):
        pct.eval(feed_dict={q_ph: [0.5]})

  def test_finds_max_of_long_array(self):
    # d - 1 == d in float32 and d = 3e7.
    # So this test only passes if we use double for the percentile indices.
    # If float is used, it fails with InvalidArgumentError about an index out of
    # bounds.
    x = math_ops.linspace(0., 3e7, num=int(3e7))
    with self.test_session():
      minval = sample_stats.percentile(x, q=0, validate_args=True)
      self.assertAllEqual(0, minval.eval())


if __name__ == "__main__":
  test.main()
