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
"""Tests for Categorical distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def make_categorical(batch_shape, num_classes, dtype=tf.int32):
  logits = tf.random_uniform(
      list(batch_shape) + [num_classes], -10, 10, dtype=tf.float32) - 50.
  return tf.contrib.distributions.Categorical(logits, dtype=dtype)


class CategoricalTest(tf.test.TestCase):

  def testLogits(self):
    logits = np.log([0.2, 0.8]) - 50.
    dist = tf.contrib.distributions.Categorical(logits)
    with self.test_session():
      self.assertAllClose(logits, dist.logits.eval())

  def testShapes(self):
    with self.test_session():
      for batch_shape in ([], [1], [2, 3, 4]):
        dist = make_categorical(batch_shape, 10)
        self.assertAllEqual(batch_shape, dist.get_batch_shape().as_list())
        self.assertAllEqual(batch_shape, dist.batch_shape().eval())
        self.assertAllEqual([], dist.get_event_shape().as_list())
        self.assertAllEqual([], dist.event_shape().eval())

  def testDtype(self):
    dist = make_categorical([], 5, dtype=tf.int32)
    self.assertEqual(dist.dtype, tf.int32)
    self.assertEqual(dist.dtype, dist.sample_n(5).dtype)
    self.assertEqual(dist.dtype, dist.mode().dtype)
    dist = make_categorical([], 5, dtype=tf.int64)
    self.assertEqual(dist.dtype, tf.int64)
    self.assertEqual(dist.dtype, dist.sample_n(5).dtype)
    self.assertEqual(dist.dtype, dist.mode().dtype)
    self.assertEqual(dist.logits.dtype, tf.float32)
    self.assertEqual(dist.logits.dtype, dist.entropy().dtype)
    self.assertEqual(dist.logits.dtype, dist.pmf(0).dtype)
    self.assertEqual(dist.logits.dtype, dist.log_pmf(0).dtype)

  def testPMFWithBatch(self):
    histograms = [[0.2, 0.8], [0.6, 0.4]]
    dist = tf.contrib.distributions.Categorical(tf.log(histograms) - 50.)
    with self.test_session():
      self.assertAllClose(dist.pmf([0, 1]).eval(), [0.2, 0.4])

  def testPMFNoBatch(self):
    histograms = [0.2, 0.8]
    dist = tf.contrib.distributions.Categorical(tf.log(histograms) - 50.)
    with self.test_session():
      self.assertAllClose(dist.pmf(0).eval(), 0.2)

  def testLogPMF(self):
    logits = np.log([[0.2, 0.8], [0.6, 0.4]]) - 50.
    dist = tf.contrib.distributions.Categorical(logits)
    with self.test_session():
      self.assertAllClose(dist.log_pmf([0, 1]).eval(), np.log([0.2, 0.4]))

  def testEntropyNoBatch(self):
    logits = np.log([0.2, 0.8]) - 50.
    dist = tf.contrib.distributions.Categorical(logits)
    with self.test_session():
      self.assertAllClose(
          dist.entropy().eval(),
          -(0.2 * np.log(0.2) + 0.8 * np.log(0.8)))

  def testEntropyWithBatch(self):
    logits = np.log([[0.2, 0.8], [0.6, 0.4]]) - 50.
    dist = tf.contrib.distributions.Categorical(logits)
    with self.test_session():
      self.assertAllClose(dist.entropy().eval(),
                          [-(0.2 * np.log(0.2) + 0.8 * np.log(0.8)),
                           -(0.6 * np.log(0.6) + 0.4 * np.log(0.4))])

  def testSample(self):
    with self.test_session():
      histograms = [[[0.2, 0.8], [0.4, 0.6]]]
      dist = tf.contrib.distributions.Categorical(tf.log(histograms) - 50.)
      n = 10000
      samples = dist.sample_n(n, seed=123)
      samples.set_shape([n, 1, 2])
      self.assertEqual(samples.dtype, tf.int32)
      sample_values = samples.eval()
      self.assertFalse(np.any(sample_values < 0))
      self.assertFalse(np.any(sample_values > 1))
      self.assertAllClose(
          [[0.2, 0.4]], np.mean(sample_values == 0, axis=0), atol=1e-2)
      self.assertAllClose(
          [[0.8, 0.6]], np.mean(sample_values == 1, axis=0), atol=1e-2)

  def testSampleWithSampleShape(self):
    with self.test_session():
      histograms = [[[0.2, 0.8], [0.4, 0.6]]]
      dist = tf.contrib.distributions.Categorical(tf.log(histograms) - 50.)
      samples = dist.sample((100, 100), seed=123)
      prob = dist.prob(samples)
      prob_val = prob.eval()
      self.assertAllClose([0.2**2 + 0.8**2], [prob_val[:, :, :, 0].mean()],
                          atol=1e-2)
      self.assertAllClose([0.4**2 + 0.6**2], [prob_val[:, :, :, 1].mean()],
                          atol=1e-2)

  def testLogPMFBroadcasting(self):
    with self.test_session():
      histograms = [[[0.2, 0.8], [0.4, 0.6]]]
      dist = tf.contrib.distributions.Categorical(tf.log(histograms) - 50.)

      prob = dist.prob(1)
      self.assertAllClose([[0.8, 0.6]], prob.eval())

      prob = dist.prob([1])
      self.assertAllClose([[0.8, 0.6]], prob.eval())

      prob = dist.prob([0, 1])
      self.assertAllClose([[0.2, 0.6]], prob.eval())

      prob = dist.prob([[0, 1]])
      self.assertAllClose([[0.2, 0.6]], prob.eval())

      prob = dist.prob([[[0, 1]]])
      self.assertAllClose([[[0.2, 0.6]]], prob.eval())

      prob = dist.prob([[1, 0], [0, 1]])
      self.assertAllClose([[0.8, 0.4], [0.2, 0.6]], prob.eval())

      prob = dist.prob([[[1, 1], [1, 0]], [[1, 0], [0, 1]]])
      self.assertAllClose([[[0.8, 0.6], [0.8, 0.4]], [[0.8, 0.4], [0.2, 0.6]]],
                          prob.eval())

  def testLogPMFShape(self):
    with self.test_session():
      # shape [1, 2, 2]
      histograms = [[[0.2, 0.8], [0.4, 0.6]]]
      dist = tf.contrib.distributions.Categorical(tf.log(histograms))

      log_prob = dist.log_prob([0, 1])
      self.assertEqual(2, log_prob.get_shape().ndims)
      self.assertAllEqual([1, 2], log_prob.get_shape())

      log_prob = dist.log_prob([[[1, 1], [1, 0]], [[1, 0], [0, 1]]])
      self.assertEqual(3, log_prob.get_shape().ndims)
      self.assertAllEqual([2, 2, 2], log_prob.get_shape())

  def testLogPMFShapeNoBatch(self):
      histograms = [0.2, 0.8]
      dist = tf.contrib.distributions.Categorical(tf.log(histograms))

      log_prob = dist.log_prob(0)
      self.assertEqual(0, log_prob.get_shape().ndims)
      self.assertAllEqual([], log_prob.get_shape())

      log_prob = dist.log_prob([[[1, 1], [1, 0]], [[1, 0], [0, 1]]])
      self.assertEqual(3, log_prob.get_shape().ndims)
      self.assertAllEqual([2, 2, 2], log_prob.get_shape())

  def testMode(self):
    with self.test_session():
      histograms = [[[0.2, 0.8], [0.6, 0.4]]]
      dist = tf.contrib.distributions.Categorical(tf.log(histograms) - 50.)
      self.assertAllEqual(dist.mode().eval(), [[1, 0]])

if __name__ == "__main__":
  tf.test.main()
