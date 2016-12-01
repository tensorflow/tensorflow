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
"""Tests for metrics.classification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.metrics.python.metrics import classification


class ClassificationTest(tf.test.TestCase):

  def testAccuracy1D(self):
    with self.test_session() as session:
      pred = tf.placeholder(tf.int32, shape=[None])
      labels = tf.placeholder(tf.int32, shape=[None])
      acc = classification.accuracy(pred, labels)
      result = session.run(acc,
                           feed_dict={
                               pred: [1, 0, 1, 0],
                               labels: [1, 1, 0, 0]
                           })
      self.assertEqual(result, 0.5)

  def testAccuracy1DBool(self):
    with self.test_session() as session:
      pred = tf.placeholder(tf.bool, shape=[None])
      labels = tf.placeholder(tf.bool, shape=[None])
      acc = classification.accuracy(pred, labels)
      result = session.run(acc,
                           feed_dict={
                               pred: [1, 0, 1, 0],
                               labels: [1, 1, 0, 0]
                           })
      self.assertEqual(result, 0.5)

  def testAccuracy1DInt64(self):
    with self.test_session() as session:
      pred = tf.placeholder(tf.int64, shape=[None])
      labels = tf.placeholder(tf.int64, shape=[None])
      acc = classification.accuracy(pred, labels)
      result = session.run(acc,
                           feed_dict={
                               pred: [1, 0, 1, 0],
                               labels: [1, 1, 0, 0]
                           })
      self.assertEqual(result, 0.5)

  def testAccuracy1DString(self):
    with self.test_session() as session:
      pred = tf.placeholder(tf.string, shape=[None])
      labels = tf.placeholder(tf.string, shape=[None])
      acc = classification.accuracy(pred, labels)
      result = session.run(acc,
                           feed_dict={
                               pred: ['a', 'b', 'a', 'c'],
                               labels: ['a', 'c', 'b', 'c']
                           })
      self.assertEqual(result, 0.5)

  def testAccuracyDtypeMismatch(self):
    with self.assertRaises(ValueError):
      pred = tf.placeholder(tf.int32, shape=[None])
      labels = tf.placeholder(tf.int64, shape=[None])
      classification.accuracy(pred, labels)

  def testAccuracyFloatLabels(self):
    with self.assertRaises(ValueError):
      pred = tf.placeholder(tf.int32, shape=[None])
      labels = tf.placeholder(tf.float32, shape=[None])
      classification.accuracy(pred, labels)

  def testAccuracy1DWeighted(self):
    with self.test_session() as session:
      pred = tf.placeholder(tf.int32, shape=[None])
      labels = tf.placeholder(tf.int32, shape=[None])
      weights = tf.placeholder(tf.float32, shape=[None])
      acc = classification.accuracy(pred, labels)
      result = session.run(acc,
                           feed_dict={
                               pred: [1, 0, 1, 1],
                               labels: [1, 1, 0, 1],
                               weights: [3.0, 1.0, 2.0, 0.0]
                           })
      self.assertEqual(result, 0.5)

  def testAccuracy1DWeightedBroadcast(self):
    with self.test_session() as session:
      pred = tf.placeholder(tf.int32, shape=[None])
      labels = tf.placeholder(tf.int32, shape=[None])
      weights = tf.placeholder(tf.float32, shape=[])
      acc = classification.accuracy(pred, labels)
      result = session.run(acc,
                           feed_dict={
                               pred: [1, 0, 1, 0],
                               labels: [1, 1, 0, 0],
                               weights: 3.0,
                           })
      self.assertEqual(result, 0.5)


if __name__ == '__main__':
  tf.test.main()
