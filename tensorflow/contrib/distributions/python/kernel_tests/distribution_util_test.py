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
"""Tests for utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.distributions.python.ops import distribution_util


class DistributionUtilTest(tf.test.TestCase):

  def _np_rotate_transpose(self, x, shift):
    if not isinstance(x, np.ndarray):
      x = np.array(x)
    return np.transpose(x, np.roll(np.arange(len(x.shape)), shift))

  def testRollStatic(self):
    with self.test_session():
      with self.assertRaisesRegexp(
          ValueError, "None values not supported."):
        distribution_util.rotate_transpose(None, 1)
      for x in (np.ones(1), np.ones((2, 1)), np.ones((3, 2, 1))):
        for shift in np.arange(-5, 5):
          y = distribution_util.rotate_transpose(x, shift)
          self.assertAllEqual(self._np_rotate_transpose(x, shift),
                              y.eval())
          self.assertAllEqual(np.roll(x.shape, shift),
                              y.get_shape().as_list())

  def testRollDynamic(self):
    with self.test_session() as sess:
      x = tf.placeholder(tf.float32)
      shift = tf.placeholder(tf.int32)
      for x_value in (np.ones(1, dtype=x.dtype.as_numpy_dtype()),
                      np.ones((2, 1), dtype=x.dtype.as_numpy_dtype()),
                      np.ones((3, 2, 1), dtype=x.dtype.as_numpy_dtype())):
        for shift_value in np.arange(-5, 5):
          self.assertAllEqual(
              self._np_rotate_transpose(x_value, shift_value),
              sess.run(distribution_util.rotate_transpose(x, shift),
                       feed_dict={x: x_value, shift: shift_value}))

  def testChooseVector(self):
    with self.test_session():
      x = np.arange(10, 12)
      y = np.arange(15, 18)
      self.assertAllEqual(
          x, distribution_util.pick_vector(
              tf.less(0, 5), x, y).eval())
      self.assertAllEqual(
          y, distribution_util.pick_vector(
              tf.less(5, 0), x, y).eval())
      self.assertAllEqual(
          x, distribution_util.pick_vector(
              tf.constant(True), x, y))  # No eval.
      self.assertAllEqual(
          y, distribution_util.pick_vector(
              tf.constant(False), x, y))  # No eval.


if __name__ == "__main__":
  tf.test.main()
