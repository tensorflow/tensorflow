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
"""Tests for Monte Carlo Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

distributions = tf.contrib.distributions
layers = tf.contrib.layers
entropy = tf.contrib.bayesflow.entropy


class NormalNoEntropy(distributions.Normal):  # pylint: disable=no-init
  """Normal distribution without a `.entropy` method."""

  def entropy(self):
    return NotImplementedError('Entropy removed by gremlins')


def get_train_op(scalar_loss, optimizer='SGD', learning_rate=1.0, decay=0.0):
  global_step = tf.Variable(0)

  def decay_fn(rate, t):
    return rate * (1 + tf.to_float(t))**(-decay)

  train_op = layers.optimize_loss(
      scalar_loss,
      global_step,
      learning_rate,
      optimizer,
      learning_rate_decay_fn=decay_fn)
  return train_op


def _assert_monotonic_decreasing(array, atol=1e-5):
  array = np.asarray(array)
  _assert_monotonic_increasing(-array, atol=atol)


def _assert_monotonic_increasing(array, atol=1e-5):
  array = np.asarray(array)
  diff = np.diff(array.ravel())
  np.testing.assert_array_less(-1 * atol, diff)


class ElboRatioTest(tf.test.TestCase):
  """Show sampling converges to true KL values."""

  def setUp(self):
    self._rng = np.random.RandomState(0)

  def test_convergence_to_kl_using_sample_form_on_3dim_normal(self):
    # Test that the sample mean KL is the same as analytic when we use samples
    # to estimate every part of the KL divergence ratio.
    vector_shape = (2, 3)
    n_samples = 5000

    with self.test_session():
      q = distributions.MultivariateNormalDiag(
          mu=self._rng.rand(*vector_shape),
          diag_stdev=self._rng.rand(*vector_shape))
      p = distributions.MultivariateNormalDiag(
          mu=self._rng.rand(*vector_shape),
          diag_stdev=self._rng.rand(*vector_shape))

      # In this case, the log_ratio is the KL.
      sample_kl = -1 * entropy.elbo_ratio(
          log_p=p.log_prob,
          q=q,
          n=n_samples,
          form=entropy.ELBOForms.sample,
          seed=42)
      actual_kl = distributions.kl(q, p)

      # Relative tolerance (rtol) chosen 2 times as large as minimim needed to
      # pass.
      self.assertEqual((2,), sample_kl.get_shape())
      self.assertAllClose(actual_kl.eval(), sample_kl.eval(), rtol=0.03)

  def test_convergence_to_kl_using_analytic_entropy_form_on_3dim_normal(self):
    # Test that the sample mean KL is the same as analytic when we use an
    # analytic entropy combined with sampled cross-entropy.
    n_samples = 5000

    vector_shape = (2, 3)
    with self.test_session():
      q = distributions.MultivariateNormalDiag(
          mu=self._rng.rand(*vector_shape),
          diag_stdev=self._rng.rand(*vector_shape))
      p = distributions.MultivariateNormalDiag(
          mu=self._rng.rand(*vector_shape),
          diag_stdev=self._rng.rand(*vector_shape))

      # In this case, the log_ratio is the KL.
      sample_kl = -1 * entropy.elbo_ratio(
          log_p=p.log_prob,
          q=q,
          n=n_samples,
          form=entropy.ELBOForms.analytic_entropy,
          seed=42)
      actual_kl = distributions.kl(q, p)

      # Relative tolerance (rtol) chosen 2 times as large as minimim needed to
      # pass.
      self.assertEqual((2,), sample_kl.get_shape())
      self.assertAllClose(actual_kl.eval(), sample_kl.eval(), rtol=0.05)

  def test_sample_kl_zero_when_p_and_q_are_the_same_distribution(self):
    n_samples = 50

    vector_shape = (2, 3)
    with self.test_session():
      q = distributions.MultivariateNormalDiag(
          mu=self._rng.rand(*vector_shape),
          diag_stdev=self._rng.rand(*vector_shape))

      # In this case, the log_ratio is the KL.
      sample_kl = -1 * entropy.elbo_ratio(
          log_p=q.log_prob,
          q=q,
          n=n_samples,
          form=entropy.ELBOForms.sample,
          seed=42)

      self.assertEqual((2,), sample_kl.get_shape())
      self.assertAllClose(np.zeros(2), sample_kl.eval())


class EntropyShannonTest(tf.test.TestCase):

  def test_normal_entropy_default_form_uses_exact_entropy(self):
    with self.test_session():
      dist = distributions.Normal(mu=1.11, sigma=2.22)
      mc_entropy = entropy.entropy_shannon(dist, n=11)
      exact_entropy = dist.entropy()
      self.assertEqual(exact_entropy.get_shape(), mc_entropy.get_shape())
      self.assertAllClose(exact_entropy.eval(), mc_entropy.eval())

  def test_normal_entropy_analytic_form_uses_exact_entropy(self):
    with self.test_session():
      dist = distributions.Normal(mu=1.11, sigma=2.22)
      mc_entropy = entropy.entropy_shannon(
          dist, form=entropy.ELBOForms.analytic_entropy)
      exact_entropy = dist.entropy()
      self.assertEqual(exact_entropy.get_shape(), mc_entropy.get_shape())
      self.assertAllClose(exact_entropy.eval(), mc_entropy.eval())

  def test_normal_entropy_sample_form_gets_approximate_answer(self):
    # Tested by showing we get a good answer that is not exact.
    with self.test_session():
      dist = distributions.Normal(mu=1.11, sigma=2.22)
      mc_entropy = entropy.entropy_shannon(
          dist, n=1000, form=entropy.ELBOForms.sample, seed=0)
      exact_entropy = dist.entropy()

      self.assertEqual(exact_entropy.get_shape(), mc_entropy.get_shape())

      # Relative tolerance (rtol) chosen 2 times as large as minimim needed to
      # pass.
      self.assertAllClose(exact_entropy.eval(), mc_entropy.eval(), rtol=0.01)

      # Make sure there is some error, proving we used samples
      self.assertLess(0.0001, tf.abs(exact_entropy - mc_entropy).eval())

  def test_default_entropy_falls_back_on_sample_if_analytic_not_available(self):
    # Tested by showing we get a good answer that is not exact.
    with self.test_session():
      # NormalNoEntropy is like a Normal, but does not have .entropy method, so
      # we are forced to fall back on sample entropy.
      dist_no_entropy = NormalNoEntropy(mu=1.11, sigma=2.22)
      dist_yes_entropy = distributions.Normal(mu=1.11, sigma=2.22)

      mc_entropy = entropy.entropy_shannon(
          dist_no_entropy, n=1000, form=entropy.ELBOForms.sample, seed=0)
      exact_entropy = dist_yes_entropy.entropy()

      self.assertEqual(exact_entropy.get_shape(), mc_entropy.get_shape())

      # Relative tolerance (rtol) chosen 2 times as large as minimim needed to
      # pass.
      self.assertAllClose(exact_entropy.eval(), mc_entropy.eval(), rtol=0.01)

      # Make sure there is some error, proving we used samples
      self.assertLess(0.0001, tf.abs(exact_entropy - mc_entropy).eval())


class RenyiRatioTest(tf.test.TestCase):
  """Show renyi_ratio is minimized when the distributions match."""

  def setUp(self):
    self._rng = np.random.RandomState(0)

  def test_fitting_two_dimensional_normal_n_equals_1000(self):
    # Minmizing Renyi divergence should allow us to make one normal match
    # another one exactly.
    n = 1000
    mu_true = np.array([1.0, -1.0], dtype=np.float64)
    chol_true = np.array([[2.0, 0.0], [0.5, 1.0]], dtype=np.float64)
    with self.test_session() as sess:
      target = distributions.MultivariateNormalCholesky(mu_true, chol_true)

      # Set up q distribution by defining mean/covariance as Variables
      mu = tf.Variable(np.zeros(mu_true.shape), dtype=mu_true.dtype, name='mu')
      mat = tf.Variable(
          np.zeros(chol_true.shape), dtype=chol_true.dtype, name='mat')
      chol = distributions.matrix_diag_transform(mat, transform=tf.nn.softplus)
      q = distributions.MultivariateNormalCholesky(mu, chol)
      for alpha in [0.25, 0.75]:

        negative_renyi_divergence = entropy.renyi_ratio(
            log_p=target.log_prob, q=q, n=n, alpha=alpha, seed=0)
        train_op = get_train_op(
            tf.reduce_mean(-negative_renyi_divergence),
            optimizer='SGD',
            learning_rate=0.5,
            decay=0.1)

        tf.initialize_all_variables().run()
        renyis = []
        for step in range(1000):
          sess.run(train_op)
          if step in [1, 5, 100]:
            renyis.append(negative_renyi_divergence.eval())

        # This optimization should maximize the renyi divergence.
        _assert_monotonic_increasing(renyis, atol=0)

        # Relative tolerance (rtol) chosen 2 times as large as minimim needed to
        # pass.
        self.assertAllClose(target.mu.eval(), q.mu.eval(), rtol=0.06)
        self.assertAllClose(target.sigma.eval(), q.sigma.eval(), rtol=0.02)

  def test_divergence_between_identical_distributions_is_zero(self):
    n = 1000
    vector_shape = (2, 3)
    with self.test_session():
      q = distributions.MultivariateNormalDiag(
          mu=self._rng.rand(*vector_shape),
          diag_stdev=self._rng.rand(*vector_shape))
      for alpha in [0.25, 0.75]:

        negative_renyi_divergence = entropy.renyi_ratio(
            log_p=q.log_prob, q=q, n=n, alpha=alpha, seed=0)

        self.assertEqual((2,), negative_renyi_divergence.get_shape())
        self.assertAllClose(np.zeros(2), negative_renyi_divergence.eval())


class RenyiAlphaTest(tf.test.TestCase):

  def test_with_three_alphas(self):
    with self.test_session():
      for dtype in (tf.float32, tf.float64):
        alpha_min = tf.constant(0.0, dtype=dtype)
        alpha_max = 0.5
        decay_time = 3

        alpha_0 = entropy.renyi_alpha(
            0, decay_time, alpha_min=alpha_min, alpha_max=alpha_max)
        alpha_1 = entropy.renyi_alpha(
            1, decay_time, alpha_min=alpha_min, alpha_max=alpha_max)
        alpha_2 = entropy.renyi_alpha(
            2, decay_time, alpha_min=alpha_min, alpha_max=alpha_max)
        alpha_3 = entropy.renyi_alpha(
            3, decay_time, alpha_min=alpha_min, alpha_max=alpha_max)

        # Alpha should start at alpha_max.
        self.assertAllClose(alpha_max, alpha_0.eval(), atol=1e-5)
        # Alpha should finish at alpha_min.
        self.assertAllClose(alpha_min.eval(), alpha_3.eval(), atol=1e-5)
        # In between, alpha should be monotonically decreasing.
        _assert_monotonic_decreasing(
            [alpha_0.eval(), alpha_1.eval(), alpha_2.eval(), alpha_3.eval()])

  def test_non_scalar_input_raises(self):
    with self.test_session():
      # Good values here
      step = 0
      alpha_min = 0.0
      alpha_max = 0.5
      decay_time = 3

      # Use one bad value inside each check.
      # The "bad" value is always the non-scalar one.
      with self.assertRaisesRegexp(ValueError, 'must be scalar'):
        entropy.renyi_alpha(
            [step], decay_time, alpha_min=alpha_min, alpha_max=alpha_max).eval()

      with self.assertRaisesRegexp(ValueError, 'must be scalar'):
        entropy.renyi_alpha(
            step, [decay_time], alpha_min=alpha_min, alpha_max=alpha_max).eval()

      with self.assertRaisesRegexp(ValueError, 'must be scalar'):
        entropy.renyi_alpha(
            step, decay_time, alpha_min=[alpha_min], alpha_max=alpha_max).eval()

      with self.assertRaisesRegexp(ValueError, 'must be scalar'):
        entropy.renyi_alpha(
            step, decay_time, alpha_min=alpha_min, alpha_max=[alpha_max]).eval()

  def test_input_with_wrong_sign_raises(self):
    with self.test_session():
      # Good values here
      step = 0
      alpha_min = 0.0
      alpha_max = 0.5
      decay_time = 3

      # Use one bad value inside each check.
      # The "bad" value is always the non-scalar one.
      with self.assertRaisesOpError('decay_time must be positive'):
        entropy.renyi_alpha(
            step, 0.0, alpha_min=alpha_min, alpha_max=alpha_max).eval()

      with self.assertRaisesOpError('step must be non-negative'):
        entropy.renyi_alpha(
            -1, decay_time, alpha_min=alpha_min, alpha_max=alpha_max).eval()


if __name__ == '__main__':
  tf.test.main()
