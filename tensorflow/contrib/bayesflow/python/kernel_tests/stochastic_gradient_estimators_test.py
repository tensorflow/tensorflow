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
"""Tests for stochastic graphs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

st = tf.contrib.bayesflow.stochastic_tensor
sge = tf.contrib.bayesflow.stochastic_gradient_estimators
dists = tf.contrib.distributions


def _vimco(loss):
  """Python implementation of VIMCO."""
  n = loss.shape[0]
  log_loss = np.log(loss)
  geometric_mean = []
  for j in range(n):
    geometric_mean.append(
        np.exp(np.mean([log_loss[i, :] for i in range(n) if i != j], 0)))
  geometric_mean = np.array(geometric_mean)

  learning_signal = []
  for j in range(n):
    learning_signal.append(
        np.sum([loss[i, :] for i in range(n) if i != j], 0))
  learning_signal = np.array(learning_signal)

  local_learning_signal = np.log(1/n * (learning_signal + geometric_mean))

  # log_mean - local_learning_signal
  log_mean = np.log(np.mean(loss, 0))
  advantage = log_mean - local_learning_signal

  return advantage


class StochasticGradientEstimatorsTest(tf.test.TestCase):

  def setUp(self):
    self._p = tf.constant(0.999999)
    self._final_loss = tf.constant(3.2)

  def _testScoreFunction(self, loss_fn, expected):
    x = st.StochasticTensor(dists.Bernoulli(p=self._p), loss_fn=loss_fn)
    sf = x.loss(self._final_loss)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(*sess.run([expected, sf]))

  def testScoreFunction(self):
    expected = tf.log(self._p) * self._final_loss
    self._testScoreFunction(sge.score_function, expected)

  def testScoreFunctionWithConstantBaseline(self):
    b = tf.constant(9.8)
    expected = tf.log(self._p) * (self._final_loss - b)
    self._testScoreFunction(
        sge.get_score_function_with_constant_baseline(b), expected)

  def testScoreFunctionWithBaselineFn(self):
    b = tf.constant(9.8)

    def baseline_fn(stoch_tensor, loss):
      self.assertTrue(isinstance(stoch_tensor, st.StochasticTensor))
      self.assertTrue(isinstance(loss, tf.Tensor))
      return b

    expected = tf.log(self._p) * (self._final_loss - b)
    self._testScoreFunction(
        sge.get_score_function_with_baseline(baseline_fn), expected)

  def testScoreFunctionWithMeanBaseline(self):
    ema_decay = 0.8
    num_steps = 6
    x = st.StochasticTensor(
        dists.Bernoulli(p=self._p),
        loss_fn=sge.get_score_function_with_baseline(
            sge.get_mean_baseline(ema_decay)))
    sf = x.loss(self._final_loss)

    # Expected EMA value
    ema = 0.
    for _ in range(num_steps):
      ema -= (1. - ema_decay) * (ema - self._final_loss)

    # Baseline is EMA with bias correction
    bias_correction = 1. - ema_decay**num_steps
    baseline = ema / bias_correction
    expected = tf.log(self._p) * (self._final_loss - baseline)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      for _ in range(num_steps - 1):
        sess.run(sf)  # run to update EMA
      self.assertAllClose(*sess.run([expected, sf]))

  def testScoreFunctionWithAdvantageFn(self):
    b = tf.constant(9.8)

    def advantage_fn(stoch_tensor, loss):
      self.assertTrue(isinstance(stoch_tensor, st.StochasticTensor))
      self.assertTrue(isinstance(loss, tf.Tensor))
      return loss - b

    expected = tf.log(self._p) * (self._final_loss - b)
    self._testScoreFunction(
        sge.get_score_function_with_advantage(advantage_fn), expected)

  def testVIMCOAdvantageFn(self):
    # simple_loss: (3, 2) with 3 samples, batch size 2
    simple_loss = np.array(
        [[1.0, 1.5],
         [1e-6, 1e4],
         [2.0, 3.0]])
    # random_loss: (100, 50, 64) with 100 samples, batch shape (50, 64)
    random_loss = 100*np.random.rand(100, 50, 64)

    advantage_fn = sge.get_vimco_advantage_fn(have_log_loss=False)

    with self.test_session() as sess:
      for loss in [simple_loss, random_loss]:
        expected = _vimco(loss)
        loss_t = tf.constant(loss, dtype=tf.float32)
        advantage_t = advantage_fn(None, loss_t)  # ST is not used
        advantage = sess.run(advantage_t)
        self.assertEqual(expected.shape, advantage_t.get_shape())
        self.assertAllClose(expected, advantage, atol=5e-5)

  def testVIMCOAdvantageGradients(self):
    loss = np.log(
        [[1.0, 1.5],
         [1e-6, 1e4],
         [2.0, 3.0]])
    advantage_fn = sge.get_vimco_advantage_fn(have_log_loss=True)

    with self.test_session():
      loss_t = tf.constant(loss, dtype=tf.float64)
      advantage_t = advantage_fn(None, loss_t)  # ST is not used
      gradient_error = tf.test.compute_gradient_error(
          loss_t, loss_t.get_shape().as_list(),
          advantage_t, advantage_t.get_shape().as_list(),
          x_init_value=loss)
      self.assertLess(gradient_error, 1e-3)

  def testVIMCOAdvantageWithSmallProbabilities(self):
    theta_value = np.random.rand(10, 100000)
    # Test with float16 dtype to ensure stability even in this extreme case.
    theta = tf.constant(theta_value, dtype=tf.float16)
    advantage_fn = sge.get_vimco_advantage_fn(have_log_loss=True)

    with self.test_session() as sess:
      log_loss = -tf.reduce_sum(theta, [1])
      advantage_t = advantage_fn(None, log_loss)
      grad_t = tf.gradients(advantage_t, theta)[0]
      advantage, grad = sess.run((advantage_t, grad_t))
      self.assertTrue(np.all(np.isfinite(advantage)))
      self.assertTrue(np.all(np.isfinite(grad)))

  def testScoreFunctionWithMeanBaselineHasUniqueVarScope(self):
    ema_decay = 0.8
    x = st.StochasticTensor(
        dists.Bernoulli(p=self._p),
        loss_fn=sge.get_score_function_with_baseline(
            sge.get_mean_baseline(ema_decay)))
    y = st.StochasticTensor(
        dists.Bernoulli(p=self._p),
        loss_fn=sge.get_score_function_with_baseline(
            sge.get_mean_baseline(ema_decay)))
    sf_x = x.loss(self._final_loss)
    sf_y = y.loss(self._final_loss)
    with self.test_session() as sess:
      # Smoke test
      sess.run(tf.global_variables_initializer())
      sess.run([sf_x, sf_y])


if __name__ == "__main__":
  tf.test.main()
