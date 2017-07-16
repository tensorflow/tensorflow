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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import histogram_ops
from tensorflow.python.platform import test


class HistogramFixedWidthTest(test.TestCase):

  def setUp(self):
    self.rng = np.random.RandomState(0)

  def test_empty_input_gives_all_zero_counts(self):
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    value_range = [0.0, 5.0]
    values = []
    expected_bin_counts = [0, 0, 0, 0, 0]
    with self.test_session():
      hist = histogram_ops.histogram_fixed_width(values, value_range, nbins=5)
      self.assertEqual(dtypes.int32, hist.dtype)
      self.assertAllClose(expected_bin_counts, hist.eval())

  def test_1d_values_int64_output(self):
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    value_range = [0.0, 5.0]
    values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
    expected_bin_counts = [2, 1, 1, 0, 2]
    with self.test_session():
      hist = histogram_ops.histogram_fixed_width(
          values, value_range, nbins=5, dtype=dtypes.int64)
      self.assertEqual(dtypes.int64, hist.dtype)
      self.assertAllClose(expected_bin_counts, hist.eval())

  def test_1d_float64_values(self):
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    value_range = np.float64([0.0, 5.0])
    values = np.float64([-1.0, 0.0, 1.5, 2.0, 5.0, 15])
    expected_bin_counts = [2, 1, 1, 0, 2]
    with self.test_session():
      hist = histogram_ops.histogram_fixed_width(values, value_range, nbins=5)
      self.assertEqual(dtypes.int32, hist.dtype)
      self.assertAllClose(expected_bin_counts, hist.eval())

  def test_2d_values(self):
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    value_range = [0.0, 5.0]
    values = [[-1.0, 0.0, 1.5], [2.0, 5.0, 15]]
    expected_bin_counts = [2, 1, 1, 0, 2]
    with self.test_session():
      hist = histogram_ops.histogram_fixed_width(values, value_range, nbins=5)
      self.assertEqual(dtypes.int32, hist.dtype)
      self.assertAllClose(expected_bin_counts, hist.eval())


if __name__ == '__main__':
  test.main()
