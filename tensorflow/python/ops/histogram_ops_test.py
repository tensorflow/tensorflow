# Copyright 2015 Google Inc. All Rights Reserved.
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

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import histogram_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest

import numpy as np


class HistogramFixedWidthTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.rng = np.random.RandomState(0)

  def test_one_update_on_constant_input(self):
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    nbins = [5]
    value_range = [0.0, 5.0]
    new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
    expected_bin_counts = [2, 1, 1, 0, 2]
    with self.test_session() as sess:
      hist = variables.Variable(array_ops.zeros(nbins, dtype=dtypes.int32))
      hist_update = histogram_ops.histogram_fixed_width(hist, new_values,
                                                        value_range)
      variables.initialize_all_variables().run()
      self.assertTrue(hist.dtype.is_compatible_with(hist_update.dtype))
      updated_hist_array = sess.run(hist_update)

      # The new updated_hist_array is returned by the updating op.
      self.assertAllClose(expected_bin_counts, updated_hist_array)

      # hist should contain updated values, but eval() should not change it.
      self.assertAllClose(expected_bin_counts, hist.eval())
      self.assertAllClose(expected_bin_counts, hist.eval())

  def test_one_update_on_constant_2d_input(self):
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    nbins = [5]
    value_range = [0.0, 5.0]
    new_values = [[-1.0, 0.0, 1.5], [2.0, 5.0, 15]]
    expected_bin_counts = [2, 1, 1, 0, 2]
    with self.test_session() as sess:
      hist = variables.Variable(array_ops.zeros(nbins, dtype=dtypes.int32))
      hist_update = histogram_ops.histogram_fixed_width(hist, new_values,
                                                        value_range)
      variables.initialize_all_variables().run()
      self.assertTrue(hist.dtype.is_compatible_with(hist_update.dtype))
      updated_hist_array = sess.run(hist_update)

      # The new updated_hist_array is returned by the updating op.
      self.assertAllClose(expected_bin_counts, updated_hist_array)

      # hist should contain updated values, but eval() should not change it.
      self.assertAllClose(expected_bin_counts, hist.eval())
      self.assertAllClose(expected_bin_counts, hist.eval())

  def test_two_updates_on_constant_input(self):
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    nbins = [5]
    value_range = [0.0, 5.0]
    new_values_1 = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
    new_values_2 = [1.5, 4.5, 4.5, 4.5, 0.0, 0.0]
    expected_bin_counts_1 = [2, 1, 1, 0, 2]
    expected_bin_counts_2 = [4, 2, 1, 0, 5]
    with self.test_session() as sess:
      hist = variables.Variable(array_ops.zeros(nbins, dtype=dtypes.int32))
      new_values = array_ops.placeholder(dtypes.float32, shape=[6])
      hist_update = histogram_ops.histogram_fixed_width(hist, new_values,
                                                        value_range)
      variables.initialize_all_variables().run()
      updated_hist_array = sess.run(hist_update,
                                    feed_dict={new_values: new_values_1})

      # The new updated_hist_array is returned by the updating op.
      # hist should contain the updated values.
      self.assertAllClose(expected_bin_counts_1, updated_hist_array)
      self.assertAllClose(expected_bin_counts_1, hist.eval())

      updated_hist_array = sess.run(hist_update,
                                    feed_dict={new_values: new_values_2})
      self.assertAllClose(expected_bin_counts_2, updated_hist_array)
      self.assertAllClose(expected_bin_counts_2, hist.eval())

  def test_two_updates_on_scalar_input(self):
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    nbins = [5]
    value_range = [0.0, 5.0]
    new_values_1 = 1.5
    new_values_2 = 2.5
    expected_bin_counts_1 = [0, 1, 0, 0, 0]
    expected_bin_counts_2 = [0, 1, 1, 0, 0]
    with self.test_session() as sess:
      hist = variables.Variable(array_ops.zeros(nbins, dtype=dtypes.int32))
      new_values = array_ops.placeholder(dtypes.float32, shape=[])
      hist_update = histogram_ops.histogram_fixed_width(hist, new_values,
                                                        value_range)
      variables.initialize_all_variables().run()

      # The new updated_hist_array is returned by the updating op.
      # hist should contain the updated values.
      updated_hist_array = sess.run(hist_update,
                                    feed_dict={new_values: new_values_1})
      self.assertAllClose(expected_bin_counts_1, updated_hist_array)
      self.assertAllClose(expected_bin_counts_1, hist.eval())

      updated_hist_array = sess.run(hist_update,
                                    feed_dict={new_values: new_values_2})
      self.assertAllClose(expected_bin_counts_2, updated_hist_array)
      self.assertAllClose(expected_bin_counts_2, hist.eval())

  def test_multiple_random_3d_updates_results_in_right_dist(self):
    # Update with uniform 3-D rvs.  Resultant
    # histogram should be uniform.  Use only 3 bins because with many bins it
    # would be unlikely that all would be close to 1/n.  If someone ever wants
    # to test that, it would be better to check that the cdf was linear.
    nbins = [3]
    value_range = [1.0, 4.14159]
    with self.test_session() as sess:
      hist = variables.Variable(array_ops.zeros(nbins, dtype=dtypes.int32))
      new_values = array_ops.placeholder(dtypes.float32, shape=[4, 4, 4])
      hist_update = histogram_ops.histogram_fixed_width(hist, new_values,
                                                        value_range)
      variables.initialize_all_variables().run()

      for _ in range(100):
        # Map the rv: U[0, 1] --> U[value_range[0], value_range[1]].
        new_values_arr = (
            value_range[0] +
            (value_range[1] - value_range[0]) * self.rng.rand(4, 4, 4))

        # The new updated_hist_array is returned by the updating op.
        # hist should contain the updated values.
        updated_hist_array = sess.run(hist_update,
                                      feed_dict={new_values: new_values_arr})

    pmf = updated_hist_array / float(updated_hist_array.sum())
    np.testing.assert_allclose(1 / 3, pmf, atol=0.02)


if __name__ == '__main__':
  googletest.main()
