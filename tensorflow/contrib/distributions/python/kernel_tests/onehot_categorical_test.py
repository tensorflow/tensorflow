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
"""Tests for OneHotCategorical distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.contrib.distributions.python.ops import onehot_categorical
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.platform import test


def make_onehot_categorical(batch_shape, num_classes, dtype=dtypes.int32):
  logits = random_ops.random_uniform(
      list(batch_shape) + [num_classes], -10, 10, dtype=dtypes.float32) - 50.
  return onehot_categorical.OneHotCategorical(logits, dtype=dtype)


class OneHotCategoricalTest(test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def testP(self):
    p = [0.2, 0.8]
    dist = onehot_categorical.OneHotCategorical(probs=p)
    with self.test_session():
      self.assertAllClose(p, dist.probs.eval())
      self.assertAllEqual([2], dist.logits.get_shape())

  def testLogits(self):
    p = np.array([0.2, 0.8], dtype=np.float32)
    logits = np.log(p) - 50.
    dist = onehot_categorical.OneHotCategorical(logits=logits)
    with self.test_session():
      self.assertAllEqual([2], dist.probs.get_shape())
      self.assertAllEqual([2], dist.logits.get_shape())
      self.assertAllClose(dist.probs.eval(), p)
      self.assertAllClose(dist.logits.eval(), logits)

  def testShapes(self):
    with self.test_session():
      for batch_shape in ([], [1], [2, 3, 4]):
        dist = make_onehot_categorical(batch_shape, 10)
        self.assertAllEqual(batch_shape, dist.batch_shape.as_list())
        self.assertAllEqual(batch_shape, dist.batch_shape_tensor().eval())
        self.assertAllEqual([10], dist.event_shape.as_list())
        self.assertAllEqual([10], dist.event_shape_tensor().eval())
        # event_shape is available as a constant because the shape is
        # known at graph build time.
        self.assertEqual(10,
                         tensor_util.constant_value(dist.event_shape_tensor()))

      for batch_shape in ([], [1], [2, 3, 4]):
        dist = make_onehot_categorical(
            batch_shape, constant_op.constant(10, dtype=dtypes.int32))
        self.assertAllEqual(len(batch_shape), dist.batch_shape.ndims)
        self.assertAllEqual(batch_shape, dist.batch_shape_tensor().eval())
        self.assertAllEqual([10], dist.event_shape.as_list())
        self.assertEqual(10, dist.event_shape_tensor().eval())

  def testDtype(self):
    dist = make_onehot_categorical([], 5, dtype=dtypes.int32)
    self.assertEqual(dist.dtype, dtypes.int32)
    self.assertEqual(dist.dtype, dist.sample(5).dtype)
    self.assertEqual(dist.dtype, dist.mode().dtype)
    dist = make_onehot_categorical([], 5, dtype=dtypes.int64)
    self.assertEqual(dist.dtype, dtypes.int64)
    self.assertEqual(dist.dtype, dist.sample(5).dtype)
    self.assertEqual(dist.dtype, dist.mode().dtype)
    self.assertEqual(dist.probs.dtype, dtypes.float32)
    self.assertEqual(dist.logits.dtype, dtypes.float32)
    self.assertEqual(dist.logits.dtype, dist.entropy().dtype)
    self.assertEqual(dist.logits.dtype, dist.prob(
        np.array([1]+[0]*4, dtype=np.int64)).dtype)
    self.assertEqual(dist.logits.dtype, dist.log_prob(
        np.array([1]+[0]*4, dtype=np.int64)).dtype)

  def testUnknownShape(self):
    with self.test_session():
      logits = array_ops.placeholder(dtype=dtypes.float32)
      dist = onehot_categorical.OneHotCategorical(logits)
      sample = dist.sample()
      # Will sample class 1.
      sample_value = sample.eval(feed_dict={logits: [-1000.0, 1000.0]})
      self.assertAllEqual([0, 1], sample_value)
      # Batch entry 0 will sample class 1, batch entry 1 will sample class 0.
      sample_value_batch = sample.eval(
          feed_dict={logits: [[-1000.0, 1000.0], [1000.0, -1000.0]]})
      self.assertAllEqual([[0, 1], [1, 0]], sample_value_batch)

  def testEntropyNoBatch(self):
    logits = np.log([0.2, 0.8]) - 50.
    dist = onehot_categorical.OneHotCategorical(logits)
    with self.test_session():
      self.assertAllClose(
          dist.entropy().eval(),
          -(0.2 * np.log(0.2) + 0.8 * np.log(0.8)))

  def testEntropyWithBatch(self):
    logits = np.log([[0.2, 0.8], [0.6, 0.4]]) - 50.
    dist = onehot_categorical.OneHotCategorical(logits)
    with self.test_session():
      self.assertAllClose(dist.entropy().eval(), [
          -(0.2 * np.log(0.2) + 0.8 * np.log(0.8)),
          -(0.6 * np.log(0.6) + 0.4 * np.log(0.4))
      ])

  def testPmf(self):
    # check that probability of samples correspond to their class probabilities
    with self.test_session():
      logits = self._rng.random_sample(size=(8, 2, 10))
      prob = np.exp(logits)/np.sum(np.exp(logits), axis=-1, keepdims=True)
      dist = onehot_categorical.OneHotCategorical(logits=logits)
      np_sample = dist.sample().eval()
      np_prob = dist.prob(np_sample).eval()
      expected_prob = prob[np_sample.astype(np.bool)]
      self.assertAllClose(expected_prob, np_prob.flatten())

  def testSample(self):
    with self.test_session():
      probs = [[[0.2, 0.8], [0.4, 0.6]]]
      dist = onehot_categorical.OneHotCategorical(math_ops.log(probs) - 50.)
      n = 100
      samples = dist.sample(n, seed=123)
      self.assertEqual(samples.dtype, dtypes.int32)
      sample_values = samples.eval()
      self.assertAllEqual([n, 1, 2, 2], sample_values.shape)
      self.assertFalse(np.any(sample_values < 0))
      self.assertFalse(np.any(sample_values > 1))

  def testSampleWithSampleShape(self):
    with self.test_session():
      probs = [[[0.2, 0.8], [0.4, 0.6]]]
      dist = onehot_categorical.OneHotCategorical(math_ops.log(probs) - 50.)
      samples = dist.sample((100, 100), seed=123)
      prob = dist.prob(samples)
      prob_val = prob.eval()
      self.assertAllClose([0.2**2 + 0.8**2], [prob_val[:, :, :, 0].mean()],
                          atol=1e-2)
      self.assertAllClose([0.4**2 + 0.6**2], [prob_val[:, :, :, 1].mean()],
                          atol=1e-2)

  def testCategoricalCategoricalKL(self):
    def np_softmax(logits):
      exp_logits = np.exp(logits)
      return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    with self.test_session() as sess:
      for categories in [2, 10]:
        for batch_size in [1, 2]:
          p_logits = self._rng.random_sample((batch_size, categories))
          q_logits = self._rng.random_sample((batch_size, categories))
          p = onehot_categorical.OneHotCategorical(logits=p_logits)
          q = onehot_categorical.OneHotCategorical(logits=q_logits)
          prob_p = np_softmax(p_logits)
          prob_q = np_softmax(q_logits)
          kl_expected = np.sum(
              prob_p * (np.log(prob_p) - np.log(prob_q)), axis=-1)

          kl_actual = kullback_leibler.kl_divergence(p, q)
          kl_same = kullback_leibler.kl_divergence(p, p)
          x = p.sample(int(2e4), seed=0)
          x = math_ops.cast(x, dtype=dtypes.float32)
          # Compute empirical KL(p||q).
          kl_sample = math_ops.reduce_mean(p.log_prob(x) - q.log_prob(x), 0)

          [kl_sample_, kl_actual_, kl_same_] = sess.run([kl_sample, kl_actual,
                                                         kl_same])
          self.assertEqual(kl_actual.get_shape(), (batch_size,))
          self.assertAllClose(kl_same_, np.zeros_like(kl_expected))
          self.assertAllClose(kl_actual_, kl_expected, atol=0., rtol=1e-6)
          self.assertAllClose(kl_sample_, kl_expected, atol=1e-2, rtol=0.)

  def testSampleUnbiasedNonScalarBatch(self):
    with self.test_session() as sess:
      logits = self._rng.rand(4, 3, 2).astype(np.float32)
      dist = onehot_categorical.OneHotCategorical(logits=logits)
      n = int(3e3)
      x = dist.sample(n, seed=0)
      x = math_ops.cast(x, dtype=dtypes.float32)
      sample_mean = math_ops.reduce_mean(x, 0)
      x_centered = array_ops.transpose(x - sample_mean, [1, 2, 3, 0])
      sample_covariance = math_ops.matmul(
          x_centered, x_centered, adjoint_b=True) / n
      [
          sample_mean_,
          sample_covariance_,
          actual_mean_,
          actual_covariance_,
      ] = sess.run([
          sample_mean,
          sample_covariance,
          dist.probs,
          dist.covariance(),
      ])
      self.assertAllEqual([4, 3, 2], sample_mean.get_shape())
      self.assertAllClose(actual_mean_, sample_mean_, atol=0., rtol=0.07)
      self.assertAllEqual([4, 3, 2, 2], sample_covariance.get_shape())
      self.assertAllClose(
          actual_covariance_, sample_covariance_, atol=0., rtol=0.10)

  def testSampleUnbiasedScalarBatch(self):
    with self.test_session() as sess:
      logits = self._rng.rand(3).astype(np.float32)
      dist = onehot_categorical.OneHotCategorical(logits=logits)
      n = int(1e4)
      x = dist.sample(n, seed=0)
      x = math_ops.cast(x, dtype=dtypes.float32)
      sample_mean = math_ops.reduce_mean(x, 0)  # elementwise mean
      x_centered = x - sample_mean
      sample_covariance = math_ops.matmul(
          x_centered, x_centered, adjoint_a=True) / n
      [
          sample_mean_,
          sample_covariance_,
          actual_mean_,
          actual_covariance_,
      ] = sess.run([
          sample_mean,
          sample_covariance,
          dist.probs,
          dist.covariance(),
      ])
      self.assertAllEqual([3], sample_mean.get_shape())
      self.assertAllClose(actual_mean_, sample_mean_, atol=0., rtol=0.1)
      self.assertAllEqual([3, 3], sample_covariance.get_shape())
      self.assertAllClose(
          actual_covariance_, sample_covariance_, atol=0., rtol=0.1)

if __name__ == "__main__":
  test.main()
