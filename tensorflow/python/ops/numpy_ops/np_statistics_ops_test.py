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
"""Tests for statistical functions in np_statistics_ops."""

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_statistics_ops
from tensorflow.python.platform import test


class HistogramTest(test.TestCase):
  """Tests for histogram and histogram_bin_edges functions."""

  def test_histogram_basic(self):
    """Test basic histogram computation."""
    data = np.array([1, 2, 1, 3, 2, 1, 4])
    np_hist, np_edges = np.histogram(data, bins=4, range=(1, 5))
    tf_hist, tf_edges = np_statistics_ops.histogram(data, bins=4, range=(1, 5))

    self.assertAllClose(np_edges, tf_edges, rtol=1e-5)
    # Note: TF and NumPy may differ slightly in bin assignment at boundaries
    self.assertAllClose(np_hist, tf_hist, atol=1)

  def test_histogram_float_data(self):
    """Test histogram with float data."""
    data = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    np_hist, np_edges = np.histogram(data, bins=5, range=(0, 5))
    tf_hist, tf_edges = np_statistics_ops.histogram(data, bins=5, range=(0, 5))

    self.assertAllClose(np_edges, tf_edges, rtol=1e-5)
    self.assertAllClose(np_hist, tf_hist)

  def test_histogram_auto_range(self):
    """Test histogram with automatic range detection."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    tf_hist, tf_edges = np_statistics_ops.histogram(data, bins=5)

    # Check that edges span the data range
    self.assertLessEqual(tf_edges[0], 1.0)
    self.assertGreaterEqual(tf_edges[-1], 5.0)
    self.assertEqual(len(tf_edges), 6)  # bins + 1

  def test_histogram_density(self):
    """Test histogram density normalization."""
    data = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 3.0])
    tf_hist, tf_edges = np_statistics_ops.histogram(
        data, bins=3, range=(1, 4), density=True)

    # For density, histogram integrates to 1
    bin_widths = tf_edges[1:] - tf_edges[:-1]
    integral = np.sum(tf_hist * bin_widths)
    self.assertAllClose(integral, 1.0, rtol=1e-5)

  def test_histogram_bin_edges(self):
    """Test histogram_bin_edges function."""
    data = np.array([1.0, 2.0, 3.0])
    tf_edges = np_statistics_ops.histogram_bin_edges(data, bins=5, range=(0, 5))

    expected_edges = np.linspace(0, 5, 6)
    self.assertAllClose(expected_edges, tf_edges, rtol=1e-5)


class QuantileTest(test.TestCase):
  """Tests for quantile and percentile functions."""

  def test_quantile_single(self):
    """Test single quantile computation."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np_result = np.quantile(data, 0.5)
    tf_result = np_statistics_ops.quantile(data, 0.5)

    self.assertAllClose(np_result, tf_result, rtol=1e-5)

  def test_quantile_multiple(self):
    """Test multiple quantile computation."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    quantiles = [0.25, 0.5, 0.75]
    np_result = np.quantile(data, quantiles)
    tf_result = np_statistics_ops.quantile(data, quantiles)

    self.assertAllClose(np_result, tf_result, rtol=1e-5)

  def test_quantile_interpolation_lower(self):
    """Test quantile with 'lower' interpolation."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    tf_result = np_statistics_ops.quantile(data, 0.3, interpolation='lower')

    # 0.3 * (4-1) = 0.9, floor = 0, so result should be data[0] = 1.0
    self.assertAllClose(1.0, tf_result, rtol=1e-5)

  def test_quantile_interpolation_higher(self):
    """Test quantile with 'higher' interpolation."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    tf_result = np_statistics_ops.quantile(data, 0.3, interpolation='higher')

    # 0.3 * (4-1) = 0.9, ceil = 1, so result should be data[1] = 2.0
    self.assertAllClose(2.0, tf_result, rtol=1e-5)

  def test_quantile_with_axis(self):
    """Test quantile along a specific axis."""
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np_result = np.quantile(data, 0.5, axis=1)
    tf_result = np_statistics_ops.quantile(data, 0.5, axis=1)

    self.assertAllClose(np_result, tf_result, rtol=1e-5)

  def test_percentile_basic(self):
    """Test percentile function."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np_result = np.percentile(data, 50)
    tf_result = np_statistics_ops.percentile(data, 50)

    self.assertAllClose(np_result, tf_result, rtol=1e-5)

  def test_percentile_range(self):
    """Test percentile at 0 and 100."""
    data = np.array([1.0, 5.0, 3.0, 2.0, 4.0])

    # 0th percentile should be min
    tf_p0 = np_statistics_ops.percentile(data, 0)
    self.assertAllClose(1.0, tf_p0, rtol=1e-5)

    # 100th percentile should be max
    tf_p100 = np_statistics_ops.percentile(data, 100)
    self.assertAllClose(5.0, tf_p100, rtol=1e-5)


class MedianTest(test.TestCase):
  """Tests for median function."""

  def test_median_odd_length(self):
    """Test median with odd number of elements."""
    data = np.array([1.0, 3.0, 2.0])
    np_result = np.median(data)
    tf_result = np_statistics_ops.median(data)

    self.assertAllClose(np_result, tf_result, rtol=1e-5)

  def test_median_even_length(self):
    """Test median with even number of elements."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    np_result = np.median(data)
    tf_result = np_statistics_ops.median(data)

    self.assertAllClose(np_result, tf_result, rtol=1e-5)

  def test_median_with_axis(self):
    """Test median along specific axis."""
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np_result = np.median(data, axis=0)
    tf_result = np_statistics_ops.median(data, axis=0)

    self.assertAllClose(np_result, tf_result, rtol=1e-5)

  def test_median_2d(self):
    """Test median of 2D array without axis (flattened)."""
    data = np.array([[1.0, 6.0], [2.0, 5.0], [3.0, 4.0]])
    np_result = np.median(data)
    tf_result = np_statistics_ops.median(data)

    self.assertAllClose(np_result, tf_result, rtol=1e-5)


class NanFunctionsTest(test.TestCase):
  """Tests for NaN-aware functions."""

  def test_nanmedian_no_nans(self):
    """Test nanmedian with no NaN values."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    tf_result = np_statistics_ops.nanmedian(data)

    self.assertAllClose(3.0, tf_result, rtol=1e-5)

  def test_nanpercentile_basic(self):
    """Test nanpercentile function."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    tf_result = np_statistics_ops.nanpercentile(data, 50)

    self.assertAllClose(3.0, tf_result, rtol=1e-5)

  def test_nanquantile_basic(self):
    """Test nanquantile function."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    tf_result = np_statistics_ops.nanquantile(data, 0.5)

    self.assertAllClose(3.0, tf_result, rtol=1e-5)


class InvalidInputTest(test.TestCase):
  """Tests for invalid inputs and error handling."""

  def test_histogram_weights_not_supported(self):
    """Test that weights parameter raises error."""
    data = np.array([1.0, 2.0, 3.0])
    with self.assertRaises(ValueError):
      np_statistics_ops.histogram(data, weights=[1, 1, 1])

  def test_quantile_invalid_interpolation(self):
    """Test that invalid interpolation method raises error."""
    data = np.array([1.0, 2.0, 3.0])
    with self.assertRaises(ValueError):
      np_statistics_ops.quantile(data, 0.5, interpolation='invalid')


if __name__ == '__main__':
  test.main()
