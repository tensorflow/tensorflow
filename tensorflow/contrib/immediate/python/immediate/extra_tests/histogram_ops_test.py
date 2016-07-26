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

import numpy as np
import tensorflow as tf

# tf-immediate replacements
from tensorflow.contrib.immediate.python.immediate import test_util
import tensorflow.contrib.immediate as immediate
env = immediate.Env(tf)
tf = env.tf


class HistogramFixedWidthTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.rng = np.random.RandomState(0)

  def test_empty_input_gives_all_zero_counts(self):
    # Bins will be:
    #   (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    value_range = [0.0, 5.0]
    values = []
    expected_bin_counts = [0, 0, 0, 0, 0]
    with self.test_session():
      hist = tf.histogram_fixed_width(values, value_range, nbins=5)

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
      hist = tf.histogram_fixed_width(values, value_range, nbins=5)

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
      hist = tf.histogram_fixed_width(values, value_range, nbins=5)

      # Hist should start "fresh" with every eval.
      self.assertAllClose(expected_bin_counts, hist.eval())
      self.assertAllClose(expected_bin_counts, hist.eval())


if __name__ == '__main__':
  tf.test.main()
