# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for signal processing functions in np_signal_ops."""

import numpy as np

from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_signal_ops
from tensorflow.python.platform import test


class ConvolveTest(test.TestCase):
  """Tests for convolve function."""

  def test_convolve_full_mode(self):
    """Test convolution with 'full' mode."""
    a = np.array([1.0, 2.0, 3.0])
    v = np.array([0.0, 1.0, 0.5])
    np_result = np.convolve(a, v, mode='full')
    tf_result = np_signal_ops.convolve(a, v, mode='full')

    self.assertAllClose(np_result, tf_result, rtol=1e-5)

  def test_convolve_same_mode(self):
    """Test convolution with 'same' mode."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    v = np.array([1.0, 2.0, 1.0])
    np_result = np.convolve(a, v, mode='same')
    tf_result = np_signal_ops.convolve(a, v, mode='same')

    self.assertEqual(len(np_result), len(tf_result))
    self.assertAllClose(np_result, tf_result, rtol=1e-5)

  def test_convolve_valid_mode(self):
    """Test convolution with 'valid' mode."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    v = np.array([1.0, 2.0, 1.0])
    np_result = np.convolve(a, v, mode='valid')
    tf_result = np_signal_ops.convolve(a, v, mode='valid')

    self.assertAllClose(np_result, tf_result, rtol=1e-5)

  def test_convolve_identity(self):
    """Test convolution with identity kernel."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    v = np.array([1.0])
    tf_result = np_signal_ops.convolve(a, v, mode='same')

    self.assertAllClose(a, tf_result, rtol=1e-5)

  def test_convolve_invalid_mode(self):
    """Test that invalid mode raises error."""
    a = np.array([1.0, 2.0])
    v = np.array([1.0])
    with self.assertRaises(ValueError):
      np_signal_ops.convolve(a, v, mode='invalid')


class CorrelateTest(test.TestCase):
  """Tests for correlate function."""

  def test_correlate_valid_mode(self):
    """Test correlation with 'valid' mode."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    v = np.array([1.0, 2.0, 1.0])
    np_result = np.correlate(a, v, mode='valid')
    tf_result = np_signal_ops.correlate(a, v, mode='valid')

    self.assertAllClose(np_result, tf_result, rtol=1e-5)

  def test_correlate_full_mode(self):
    """Test correlation with 'full' mode."""
    a = np.array([1.0, 2.0, 3.0])
    v = np.array([0.5, 1.0])
    np_result = np.correlate(a, v, mode='full')
    tf_result = np_signal_ops.correlate(a, v, mode='full')

    self.assertAllClose(np_result, tf_result, rtol=1e-5)


class SearchsortedTest(test.TestCase):
  """Tests for searchsorted function."""

  def test_searchsorted_left(self):
    """Test searchsorted with left side."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    v = np.array([2.5, 3.5, 0.5, 5.5])
    np_result = np.searchsorted(a, v, side='left')
    tf_result = np_signal_ops.searchsorted(a, v, side='left')

    self.assertAllEqual(np_result, tf_result)

  def test_searchsorted_right(self):
    """Test searchsorted with right side."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    v = np.array([2.0, 3.0, 4.0])
    np_result = np.searchsorted(a, v, side='right')
    tf_result = np_signal_ops.searchsorted(a, v, side='right')

    self.assertAllEqual(np_result, tf_result)

  def test_searchsorted_scalar(self):
    """Test searchsorted with scalar value."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    v = 2.5
    np_result = np.searchsorted(a, v)
    tf_result = np_signal_ops.searchsorted(a, v)

    self.assertAllEqual(np_result, tf_result)

  def test_searchsorted_invalid_side(self):
    """Test that invalid side raises error."""
    a = np.array([1.0, 2.0, 3.0])
    with self.assertRaises(ValueError):
      np_signal_ops.searchsorted(a, 1.5, side='middle')


class InterpTest(test.TestCase):
  """Tests for interp function."""

  def test_interp_basic(self):
    """Test basic interpolation."""
    xp = np.array([0.0, 1.0, 2.0, 3.0])
    fp = np.array([0.0, 1.0, 4.0, 9.0])
    x = np.array([0.5, 1.5, 2.5])
    np_result = np.interp(x, xp, fp)
    tf_result = np_signal_ops.interp(x, xp, fp)

    self.assertAllClose(np_result, tf_result, rtol=1e-5)

  def test_interp_out_of_bounds(self):
    """Test interpolation at boundaries."""
    xp = np.array([1.0, 2.0, 3.0])
    fp = np.array([10.0, 20.0, 30.0])
    x = np.array([0.0, 4.0])
    np_result = np.interp(x, xp, fp)
    tf_result = np_signal_ops.interp(x, xp, fp)

    # Out of bounds should return edge values
    self.assertAllClose(np_result, tf_result, rtol=1e-5)

  def test_interp_with_left_right(self):
    """Test interpolation with custom left/right values."""
    xp = np.array([1.0, 2.0, 3.0])
    fp = np.array([10.0, 20.0, 30.0])
    x = np.array([0.0, 4.0])
    np_result = np.interp(x, xp, fp, left=-1.0, right=-1.0)
    tf_result = np_signal_ops.interp(x, xp, fp, left=-1.0, right=-1.0)

    self.assertAllClose(np_result, tf_result, rtol=1e-5)

  def test_interp_exact_points(self):
    """Test interpolation at exact data points."""
    xp = np.array([0.0, 1.0, 2.0])
    fp = np.array([0.0, 10.0, 20.0])
    x = np.array([0.0, 1.0, 2.0])
    tf_result = np_signal_ops.interp(x, xp, fp)

    self.assertAllClose(fp, tf_result, rtol=1e-5)


class PiecewiseTest(test.TestCase):
  """Tests for piecewise function."""

  def test_piecewise_basic(self):
    """Test basic piecewise function."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    condlist = [x < 2.5, x >= 2.5]
    funclist = [lambda x: x, lambda x: x * 2]

    np_result = np.piecewise(x, condlist, funclist)
    tf_result = np_signal_ops.piecewise(x, condlist, funclist)

    self.assertAllClose(np_result, tf_result, rtol=1e-5)

  def test_piecewise_with_default(self):
    """Test piecewise with default value."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    condlist = [x < 2]
    funclist = [10.0, 0.0]  # Last value is default

    np_result = np.piecewise(x, condlist, funclist)
    tf_result = np_signal_ops.piecewise(x, condlist, funclist)

    self.assertAllClose(np_result, tf_result, rtol=1e-5)

  def test_piecewise_scalar_funcs(self):
    """Test piecewise with scalar function values."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    condlist = [x < 2.5, x >= 2.5]
    funclist = [1.0, 2.0]

    np_result = np.piecewise(x, condlist, funclist)
    tf_result = np_signal_ops.piecewise(x, condlist, funclist)

    self.assertAllClose(np_result, tf_result, rtol=1e-5)


class DigitizeTest(test.TestCase):
  """Tests for digitize function."""

  def test_digitize_basic(self):
    """Test basic digitize."""
    x = np.array([0.5, 1.5, 2.5, 3.5])
    bins = np.array([1.0, 2.0, 3.0])
    np_result = np.digitize(x, bins)
    tf_result = np_signal_ops.digitize(x, bins)

    self.assertAllEqual(np_result, tf_result)

  def test_digitize_right(self):
    """Test digitize with right=True."""
    x = np.array([1.0, 2.0, 3.0])
    bins = np.array([1.0, 2.0, 3.0])
    np_result = np.digitize(x, bins, right=True)
    tf_result = np_signal_ops.digitize(x, bins, right=True)

    self.assertAllEqual(np_result, tf_result)


if __name__ == '__main__':
  test.main()
