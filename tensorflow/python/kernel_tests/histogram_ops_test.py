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
"""Tests for tensorflow.ops.histogram_ops."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import histogram_ops
from tensorflow.python.platform import test


class BinValuesFixedWidth(test.TestCase, parameterized.TestCase):

  def test_empty_input_gives_all_zero_counts(self):
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    value_range = [0.0, 5.0]
    values = []
    expected_bins = []
    with self.cached_session():
      bins = histogram_ops.histogram_fixed_width_bins(
          values, value_range, nbins=5)
      self.assertEqual(dtypes.int32, bins.dtype)
      self.assertAllClose(expected_bins, self.evaluate(bins))

  @parameterized.parameters(
      np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype
  )
  def test_1d_values_int32_output(self, dtype):
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    value_range = np.array([0.0, 5.0]).astype(dtype)
    values = np.array([-1.0, 0.0, 1.5, 2.0, 5.0, 15]).astype(dtype)
    expected_bins = [0, 0, 1, 2, 4, 4]
    with self.cached_session():
      bins = histogram_ops.histogram_fixed_width_bins(
          values, value_range, nbins=5)
      self.assertEqual(dtypes.int32, bins.dtype)
      self.assertAllClose(expected_bins, self.evaluate(bins))

  def test_2d_values(self):
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    value_range = [0.0, 5.0]
    values = constant_op.constant(
        [[-1.0, 0.0, 1.5], [2.0, 5.0, 15]], shape=(2, 3))
    expected_bins = [[0, 0, 1], [2, 4, 4]]
    with self.cached_session():
      bins = histogram_ops.histogram_fixed_width_bins(
          values, value_range, nbins=5)
      self.assertEqual(dtypes.int32, bins.dtype)
      self.assertAllClose(expected_bins, self.evaluate(bins))

  def test_negative_nbins(self):
    value_range = [0.0, 5.0]
    values = []
    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                "must > 0"):
      with self.session():
        bins = histogram_ops.histogram_fixed_width_bins(
            values, value_range, nbins=-1)
        self.evaluate(bins)


class HistogramFixedWidthTest(test.TestCase):

  def setUp(self):
    self.rng = np.random.RandomState(0)

  def test_with_invalid_value_range(self):
    values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
    with self.assertRaisesRegex(
        (errors.InvalidArgumentError, ValueError),
        "Shape must be rank 1 but is rank 0|should be a vector"):
      self.evaluate(histogram_ops.histogram_fixed_width(values, 1.0))
    with self.assertRaisesRegex(
        (errors.InvalidArgumentError, ValueError),
        "Dimension must be 2 but is 3|should be a vector of 2 elements"):
      self.evaluate(
          histogram_ops.histogram_fixed_width(values, [1.0, 2.0, 3.0]))

  def test_with_invalid_nbins(self):
    values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
    with self.assertRaisesRegex(
        (errors.InvalidArgumentError, ValueError),
        "Shape must be rank 0 but is rank 1|should be a scalar"):
      self.evaluate(
          histogram_ops.histogram_fixed_width(values, [1.0, 5.0], nbins=[1, 2]))
    with self.assertRaisesRegex(
        (errors.InvalidArgumentError, ValueError),
        "Requires nbins > 0|should be a positive number"):
      self.evaluate(
          histogram_ops.histogram_fixed_width(values, [1.0, 5.0], nbins=-5))

  def test_empty_input_gives_all_zero_counts(self):
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    value_range = [0.0, 5.0]
    values = []
    expected_bin_counts = [0, 0, 0, 0, 0]
    hist = histogram_ops.histogram_fixed_width(values, value_range, nbins=5)
    self.assertEqual(dtypes.int32, hist.dtype)
    self.assertAllClose(expected_bin_counts, self.evaluate(hist))

  def test_1d_values_int64_output(self):
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    value_range = [0.0, 5.0]
    values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
    expected_bin_counts = [2, 1, 1, 0, 2]
    hist = histogram_ops.histogram_fixed_width(
        values, value_range, nbins=5, dtype=dtypes.int64)
    self.assertEqual(dtypes.int64, hist.dtype)
    self.assertAllClose(expected_bin_counts, self.evaluate(hist))

  def test_1d_float64_values(self):
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    value_range = np.float64([0.0, 5.0])
    values = np.float64([-1.0, 0.0, 1.5, 2.0, 5.0, 15])
    expected_bin_counts = [2, 1, 1, 0, 2]
    hist = histogram_ops.histogram_fixed_width(values, value_range, nbins=5)
    self.assertEqual(dtypes.int32, hist.dtype)
    self.assertAllClose(expected_bin_counts, self.evaluate(hist))

  def test_2d_values(self):
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    value_range = [0.0, 5.0]
    values = [[-1.0, 0.0, 1.5], [2.0, 5.0, 15]]
    expected_bin_counts = [2, 1, 1, 0, 2]
    hist = histogram_ops.histogram_fixed_width(values, value_range, nbins=5)
    self.assertEqual(dtypes.int32, hist.dtype)
    self.assertAllClose(expected_bin_counts, self.evaluate(hist))

  @test_util.run_deprecated_v1
  def test_shape_inference(self):
    value_range = [0.0, 5.0]
    values = [[-1.0, 0.0, 1.5], [2.0, 5.0, 15]]
    expected_bin_counts = [2, 1, 1, 0, 2]
    placeholder = array_ops.placeholder(dtypes.int32)
    with self.session():
      hist = histogram_ops.histogram_fixed_width(values, value_range, nbins=5)
      self.assertAllEqual(hist.shape.as_list(), (5,))
      self.assertEqual(dtypes.int32, hist.dtype)
      self.assertAllClose(expected_bin_counts, self.evaluate(hist))

      hist = histogram_ops.histogram_fixed_width(
          values, value_range, nbins=placeholder)
      self.assertEqual(hist.shape.ndims, 1)
      self.assertIs(hist.shape.dims[0].value, None)
      self.assertEqual(dtypes.int32, hist.dtype)
      self.assertAllClose(expected_bin_counts, hist.eval({placeholder: 5}))

  def test_single_bin(self):
    hist = histogram_ops.histogram_fixed_width(
        values=constant_op.constant([3e+38, 100], dtype=dtypes.float32),
        value_range=constant_op.constant([-1e+38, 3e+38]),
        nbins=1)
    self.assertAllEqual(hist, [2])

  def test_range_overflow(self):
    hist = histogram_ops.histogram_fixed_width(
        values=constant_op.constant([3e+38, 100], dtype=dtypes.float32),
        value_range=constant_op.constant([-1e+38, 3e+38]),
        nbins=2)
    self.assertAllEqual(hist, [1, 1])

  def test_large_range(self):
    hist = histogram_ops.histogram_fixed_width(
        values=constant_op.constant(
            [-(2**31), 2**31 - 1], dtype=dtypes.int32
        ),
        value_range=constant_op.constant(
            [-(2**31), 2**31 - 1], dtype=dtypes.int32
        ),
        nbins=2,
    )
    self.assertAllEqual(hist, [1, 1])


if __name__ == '__main__':
  test.main()
