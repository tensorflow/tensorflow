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
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import histogram_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variables
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

      # Hist should start "fresh" with every eval.
      self.assertAllClose(expected_bin_counts, hist.eval())
      self.assertAllClose(expected_bin_counts, hist.eval())

  def test_one_update_on_constant_input(self):
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    value_range = [0.0, 5.0]
    values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
    expected_bin_counts = [2, 1, 1, 0, 2]
    with self.test_session():
      hist = histogram_ops.histogram_fixed_width(values, value_range, nbins=5)

      # Hist should start "fresh" with every eval.
      self.assertAllClose(expected_bin_counts, hist.eval())
      self.assertAllClose(expected_bin_counts, hist.eval())

  def test_one_update_on_constant_2d_input(self):
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    value_range = [0.0, 5.0]
    values = [[-1.0, 0.0, 1.5], [2.0, 5.0, 15]]
    expected_bin_counts = [2, 1, 1, 0, 2]
    with self.test_session():
      hist = histogram_ops.histogram_fixed_width(values, value_range, nbins=5)

      # Hist should start "fresh" with every eval.
      self.assertAllClose(expected_bin_counts, hist.eval())
      self.assertAllClose(expected_bin_counts, hist.eval())

  def test_two_updates_on_constant_input(self):
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    value_range = [0.0, 5.0]
    values_1 = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
    values_2 = [1.5, 4.5, 4.5, 4.5, 0.0, 0.0]
    expected_bin_counts_1 = [2, 1, 1, 0, 2]
    expected_bin_counts_2 = [2, 1, 0, 0, 3]
    with self.test_session():
      values = array_ops.placeholder(dtypes.float32, shape=[6])
      hist = histogram_ops.histogram_fixed_width(values, value_range, nbins=5)

      # The values in hist should depend on the current feed and nothing else.
      self.assertAllClose(
          expected_bin_counts_1, hist.eval(feed_dict={values: values_1}))
      self.assertAllClose(
          expected_bin_counts_2, hist.eval(feed_dict={values: values_2}))
      self.assertAllClose(
          expected_bin_counts_1, hist.eval(feed_dict={values: values_1}))
      self.assertAllClose(
          expected_bin_counts_1, hist.eval(feed_dict={values: values_1}))

  def test_two_updates_on_scalar_input(self):
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    value_range = [0.0, 5.0]
    values_1 = 1.5
    values_2 = 2.5
    expected_bin_counts_1 = [0, 1, 0, 0, 0]
    expected_bin_counts_2 = [0, 0, 1, 0, 0]
    with self.test_session():
      values = array_ops.placeholder(dtypes.float32, shape=[])
      hist = histogram_ops.histogram_fixed_width(values, value_range, nbins=5)

      # The values in hist should depend on the current feed and nothing else.
      self.assertAllClose(
          expected_bin_counts_2, hist.eval(feed_dict={values: values_2}))
      self.assertAllClose(
          expected_bin_counts_1, hist.eval(feed_dict={values: values_1}))
      self.assertAllClose(
          expected_bin_counts_1, hist.eval(feed_dict={values: values_1}))
      self.assertAllClose(
          expected_bin_counts_2, hist.eval(feed_dict={values: values_2}))

  def test_multiple_random_accumulating_updates_results_in_right_dist(self):
    # Accumulate the updates in a new variable.  Resultant
    # histogram should be uniform.  Use only 3 bins because with many bins it
    # would be unlikely that all would be close to 1/n.  If someone ever wants
    # to test that, it would be better to check that the cdf was linear.
    value_range = [1.0, 4.14159]
    with self.test_session() as sess:
      values = array_ops.placeholder(dtypes.float32, shape=[4, 4, 4])
      hist = histogram_ops.histogram_fixed_width(
          values, value_range, nbins=3, dtype=dtypes.int64)

      hist_accum = variables.Variable(init_ops.zeros_initializer()(
          [3], dtype=dtypes.int64))
      hist_accum = hist_accum.assign_add(hist)

      variables.global_variables_initializer().run()

      for _ in range(100):
        # Map the rv: U[0, 1] --> U[value_range[0], value_range[1]].
        values_arr = (
            value_range[0] +
            (value_range[1] - value_range[0]) * self.rng.rand(4, 4, 4))

        hist_accum_arr = sess.run(hist_accum, feed_dict={values: values_arr})

    pmf = hist_accum_arr / float(hist_accum_arr.sum())
    np.testing.assert_allclose(1 / 3, pmf, atol=0.02)


if __name__ == '__main__':
  test.main()
