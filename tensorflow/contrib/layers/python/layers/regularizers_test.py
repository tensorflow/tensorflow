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
"""Tests for regularizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class RegularizerTest(tf.test.TestCase):

  def test_l1(self):
    with self.assertRaises(ValueError):
      tf.contrib.layers.l1_regularizer(2.)
    with self.assertRaises(ValueError):
      tf.contrib.layers.l1_regularizer(-1.)
    with self.assertRaises(ValueError):
      tf.contrib.layers.l1_regularizer(0)

    self.assertIsNone(tf.contrib.layers.l1_regularizer(0.)(None))

    values = np.array([1., -1., 4., 2.])
    weights = tf.constant(values)
    with tf.Session() as sess:
      result = sess.run(tf.contrib.layers.l1_regularizer(.5)(weights))

    self.assertAllClose(np.abs(values).sum() * .5, result)

  def test_l2(self):
    with self.assertRaises(ValueError):
      tf.contrib.layers.l2_regularizer(2.)
    with self.assertRaises(ValueError):
      tf.contrib.layers.l2_regularizer(-1.)
    with self.assertRaises(ValueError):
      tf.contrib.layers.l2_regularizer(0)

    self.assertIsNone(tf.contrib.layers.l2_regularizer(0.)(None))

    values = np.array([1., -1., 4., 2.])
    weights = tf.constant(values)
    with tf.Session() as sess:
      result = sess.run(tf.contrib.layers.l2_regularizer(.42)(weights))

    self.assertAllClose(np.power(values, 2).sum() / 2.0 * .42, result)


if __name__ == '__main__':
  tf.test.main()
