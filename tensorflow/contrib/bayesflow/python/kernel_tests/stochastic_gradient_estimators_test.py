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

import tensorflow as tf

st = tf.contrib.bayesflow.stochastic_tensor
sge = tf.contrib.bayesflow.stochastic_gradient_estimators


class StochasticGradientEstimatorsTest(tf.test.TestCase):

  def setUp(self):
    self._p = tf.constant(0.999999)
    self._final_loss = tf.constant(3.2)

  def _testScoreFunction(self, loss_fn, expected):
    x = st.BernoulliTensor(p=self._p, loss_fn=loss_fn)
    sf = x.loss(self._final_loss)
    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
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
    x = st.BernoulliTensor(
        p=self._p,
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
      sess.run(tf.initialize_all_variables())
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

  def testScoreFunctionWithMeanBaselineHasUniqueVarScope(self):
    ema_decay = 0.8
    x = st.BernoulliTensor(
        p=self._p,
        loss_fn=sge.get_score_function_with_baseline(
            sge.get_mean_baseline(ema_decay)))
    y = st.BernoulliTensor(
        p=self._p,
        loss_fn=sge.get_score_function_with_baseline(
            sge.get_mean_baseline(ema_decay)))
    sf_x = x.loss(self._final_loss)
    sf_y = y.loss(self._final_loss)
    with self.test_session() as sess:
      # Smoke test
      sess.run(tf.initialize_all_variables())
      sess.run([sf_x, sf_y])


if __name__ == "__main__":
  tf.test.main()
