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

"""Tests for tensorflow.kernels.unique_op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class UniqueTest(tf.test.TestCase):

  def testInt32(self):
    x = np.random.randint(2, high=10, size=7000)
    with self.test_session() as sess:
      y, idx = tf.unique(x)
      tf_y, tf_idx = sess.run([y, idx])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])

  def testString(self):
    indx = np.random.randint(65, high=122, size=7000)
    x = [chr(i) for i in indx]
    with self.test_session() as sess:
      y, idx = tf.unique(x)
      tf_y, tf_idx = sess.run([y, idx])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]].decode('ascii'))


class UniqueWithCountsTest(tf.test.TestCase):

  def testInt32(self):
    x = np.random.randint(2, high=10, size=7000)
    with self.test_session() as sess:
      y, idx, count = tf.unique_with_counts(x)
      tf_y, tf_idx, tf_count = sess.run([y, idx, count])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])
    for value, count in zip(tf_y, tf_count):
      self.assertEqual(count, np.sum(x == value))

  def testString(self):
    indx = np.random.randint(65, high=122, size=7000)
    x = [chr(i) for i in indx]

    with self.test_session() as sess:
      y, idx, count = tf.unique_with_counts(x)
      tf_y, tf_idx, tf_count = sess.run([y, idx, count])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]].decode('ascii'))
    for value, count in zip(tf_y, tf_count):
      v = [1 if x[i] == value.decode('ascii') else 0 for i in range(7000)]
      self.assertEqual(count, sum(v))


if __name__ == '__main__':
  tf.test.main()
