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

from tensorflow.contrib.distributions.python.ops import categorical
from tensorflow.contrib.distributions.python.ops import kullback_leibler
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


def make_categorical(batch_shape, num_classes, dtype=dtypes.int32):
  logits = random_ops.random_uniform(
      list(batch_shape) + [num_classes], -10, 10, dtype=dtypes.float32) - 50.
  return categorical.Categorical(logits, dtype=dtype)


class CategoricalTest(test.TestCase):

  def testP(self):
    p = [0.2, 0.8]
    dist = categorical.Categorical(probs=p)
    with self.test_session():
      self.assertAllClose(p, dist.probs.eval())
      self.assertAllEqual([2], dist.logits.get_shape())

  def testLogits(self):
    p = np.array([0.2, 0.8], dtype=np.float32)
    logits = np.log(p) - 50.
    dist = categorical.Categorical(logits=logits)
    with self.test_session():
      self.assertAllEqual([2], dist.probs.get_shape())
      self.assertAllEqual([2], dist.logits.get_shape())
      self.assertAllClose(dist.probs.eval(), p)
      self.assertAllClose(dist.logits.eval(), logits)

  def testShapes(self):
    with self.test_session():
      for batch_shape in ([], [1], [2, 3, 4]):
        dist = make_categorical(batch_shape, 10)
        self.assertAllEqual(batch_shape, dist.batch_shape)
        self.assertAllEqual(batch_shape, dist.batch_shape_tensor().eval())
        self.assertAllEqual([], dist.event_shape)
        self.assertAllEqual([], dist.event_shape_tensor().eval())
        self.assertEqual(10, dist.event_size.eval())
        # event_size is available as a constant because the shape is
        # known at graph build time.
        self.assertEqual(10, tensor_util.constant_value(dist.event_size))

      for batch_shape in ([], [1], [2, 3, 4]):
        dist = make_categorical(
            batch_shape, constant_op.constant(
                10, dtype=dtypes.int32))
        self.assertAllEqual(len(batch_shape), dist.batch_shape.ndims)
        self.assertAllEqual(batch_shape, dist.batch_shape_tensor().eval())
        self.assertAllEqual([], dist.event_shape)
        self.assertAllEqual([], dist.event_shape_tensor().eval())
        self.assertEqual(10, dist.event_size.eval())

  def testDtype(self):
    dist = make_categorical([], 5, dtype=dtypes.int32)
    self.assertEqual(dist.dtype, dtypes.int32)
    self.assertEqual(dist.dtype, dist.sample(5).dtype)
    self.assertEqual(dist.dtype, dist.mode().dtype)
    dist = make_categorical([], 5, dtype=dtypes.int64)
    self.assertEqual(dist.dtype, dtypes.int64)
    self.assertEqual(dist.dtype, dist.sample(5).dtype)
    self.assertEqual(dist.dtype, dist.mode().dtype)
    self.assertEqual(dist.probs.dtype, dtypes.float32)
    self.assertEqual(dist.logits.dtype, dtypes.float32)
    self.assertEqual(dist.logits.dtype, dist.entropy().dtype)
    self.assertEqual(
        dist.logits.dtype, dist.prob(np.array(
            0, dtype=np.int64)).dtype)
    self.assertEqual(
        dist.logits.dtype, dist.log_prob(np.array(
            0, dtype=np.int64)).dtype)

  def testUnknownShape(self):
    with self.test_session():
      logits = array_ops.placeholder(dtype=dtypes.float32)
      dist = categorical.Categorical(logits)
      sample = dist.sample()
      # Will sample class 1.
      sample_value = sample.eval(feed_dict={logits: [-1000.0, 1000.0]})
      self.assertEqual(1, sample_value)

      # Batch entry 0 will sample class 1, batch entry 1 will sample class 0.
      sample_value_batch = sample.eval(
          feed_dict={logits: [[-1000.0, 1000.0], [1000.0, -1000.0]]})
      self.assertAllEqual([1, 0], sample_value_batch)

  def testPMFWithBatch(self):
    histograms = [[0.2, 0.8], [0.6, 0.4]]
    dist = categorical.Categorical(math_ops.log(histograms) - 50.)
    with self.test_session():
      self.assertAllClose(dist.prob([0, 1]).eval(), [0.2, 0.4])

  def testPMFNoBatch(self):
    histograms = [0.2, 0.8]
    dist = categorical.Categorical(math_ops.log(histograms) - 50.)
    with self.test_session():
      self.assertAllClose(dist.prob(0).eval(), 0.2)

  def testLogPMF(self):
    logits = np.log([[0.2, 0.8], [0.6, 0.4]]) - 50.
    dist = categorical.Categorical(logits)
    with self.test_session():
      self.assertAllClose(dist.log_prob([0, 1]).eval(), np.log([0.2, 0.4]))

  def testEntropyNoBatch(self):
    logits = np.log([0.2, 0.8]) - 50.
    dist = categorical.Categorical(logits)
    with self.test_session():
      self.assertAllClose(dist.entropy().eval(),
                          -(0.2 * np.log(0.2) + 0.8 * np.log(0.8)))

  def testEntropyWithBatch(self):
    logits = np.log([[0.2, 0.8], [0.6, 0.4]]) - 50.
    dist = categorical.Categorical(logits)
    with self.test_session():
      self.assertAllClose(dist.entropy().eval(), [
          -(0.2 * np.log(0.2) + 0.8 * np.log(0.8)),
          -(0.6 * np.log(0.6) + 0.4 * np.log(0.4))
      ])

  def testEntropyGradient(self):
    with self.test_session() as sess:
      logits = constant_op.constant([[1., 2., 3.], [2., 5., 1.]])

      probabilities = nn_ops.softmax(logits)
      log_probabilities = nn_ops.log_softmax(logits)
      true_entropy = - math_ops.reduce_sum(
          probabilities * log_probabilities, axis=-1)

      categorical_distribution = categorical.Categorical(probs=probabilities)
      categorical_entropy = categorical_distribution.entropy()

      # works
      true_entropy_g = gradients_impl.gradients(true_entropy, [logits])
      categorical_entropy_g = gradients_impl.gradients(
          categorical_entropy, [logits])

      res = sess.run({"true_entropy": true_entropy,
                      "categorical_entropy": categorical_entropy,
                      "true_entropy_g": true_entropy_g,
                      "categorical_entropy_g": categorical_entropy_g})
      self.assertAllClose(res["true_entropy"],
                          res["categorical_entropy"])
      self.assertAllClose(res["true_entropy_g"],
                          res["categorical_entropy_g"])

  def testSample(self):
    with self.test_session():
      histograms = [[[0.2, 0.8], [0.4, 0.6]]]
      dist = categorical.Categorical(math_ops.log(histograms) - 50.)
      n = 10000
      samples = dist.sample(n, seed=123)
      samples.set_shape([n, 1, 2])
      self.assertEqual(samples.dtype, dtypes.int32)
      sample_values = samples.eval()
      self.assertFalse(np.any(sample_values < 0))
      self.assertFalse(np.any(sample_values > 1))
      self.assertAllClose(
          [[0.2, 0.4]], np.mean(
              sample_values == 0, axis=0), atol=1e-2)
      self.assertAllClose(
          [[0.8, 0.6]], np.mean(
              sample_values == 1, axis=0), atol=1e-2)

  def testSampleWithSampleShape(self):
    with self.test_session():
      histograms = [[[0.2, 0.8], [0.4, 0.6]]]
      dist = categorical.Categorical(math_ops.log(histograms) - 50.)
      samples = dist.sample((100, 100), seed=123)
      prob = dist.prob(samples)
      prob_val = prob.eval()
      self.assertAllClose(
          [0.2**2 + 0.8**2], [prob_val[:, :, :, 0].mean()], atol=1e-2)
      self.assertAllClose(
          [0.4**2 + 0.6**2], [prob_val[:, :, :, 1].mean()], atol=1e-2)

  def testLogPMFBroadcasting(self):
    with self.test_session():
      histograms = [[[0.2, 0.8], [0.4, 0.6]]]
      dist = categorical.Categorical(math_ops.log(histograms) - 50.)

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
      dist = categorical.Categorical(math_ops.log(histograms))

      log_prob = dist.log_prob([0, 1])
      self.assertEqual(2, log_prob.get_shape().ndims)
      self.assertAllEqual([1, 2], log_prob.get_shape())

      log_prob = dist.log_prob([[[1, 1], [1, 0]], [[1, 0], [0, 1]]])
      self.assertEqual(3, log_prob.get_shape().ndims)
      self.assertAllEqual([2, 2, 2], log_prob.get_shape())

  def testLogPMFShapeNoBatch(self):
    histograms = [0.2, 0.8]
    dist = categorical.Categorical(math_ops.log(histograms))

    log_prob = dist.log_prob(0)
    self.assertEqual(0, log_prob.get_shape().ndims)
    self.assertAllEqual([], log_prob.get_shape())

    log_prob = dist.log_prob([[[1, 1], [1, 0]], [[1, 0], [0, 1]]])
    self.assertEqual(3, log_prob.get_shape().ndims)
    self.assertAllEqual([2, 2, 2], log_prob.get_shape())

  def testMode(self):
    with self.test_session():
      histograms = [[[0.2, 0.8], [0.6, 0.4]]]
      dist = categorical.Categorical(math_ops.log(histograms) - 50.)
      self.assertAllEqual(dist.mode().eval(), [[1, 0]])

  def testCategoricalCategoricalKL(self):

    def np_softmax(logits):
      exp_logits = np.exp(logits)
      return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    with self.test_session() as sess:
      for categories in [2, 4]:
        for batch_size in [1, 10]:
          a_logits = np.random.randn(batch_size, categories)
          b_logits = np.random.randn(batch_size, categories)

          a = categorical.Categorical(logits=a_logits)
          b = categorical.Categorical(logits=b_logits)

          kl = kullback_leibler.kl(a, b)
          kl_val = sess.run(kl)
          # Make sure KL(a||a) is 0
          kl_same = sess.run(kullback_leibler.kl(a, a))

          prob_a = np_softmax(a_logits)
          prob_b = np_softmax(b_logits)
          kl_expected = np.sum(prob_a * (np.log(prob_a) - np.log(prob_b)),
                               axis=-1)

          self.assertEqual(kl.get_shape(), (batch_size,))
          self.assertAllClose(kl_val, kl_expected)
          self.assertAllClose(kl_same, np.zeros_like(kl_expected))


if __name__ == "__main__":
  test.main()
