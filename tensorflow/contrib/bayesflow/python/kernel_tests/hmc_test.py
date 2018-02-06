# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Hamiltonian Monte Carlo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import special
from scipy import stats

from tensorflow.contrib.bayesflow.python.ops import hmc

from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging


# TODO(b/66964210): Test float16.
class HMCTest(test.TestCase):

  def setUp(self):
    self._shape_param = 5.
    self._rate_param = 10.
    self._expected_x = (special.digamma(self._shape_param)
                        - np.log(self._rate_param))
    self._expected_exp_x = self._shape_param / self._rate_param

    random_seed.set_random_seed(10003)
    np.random.seed(10003)

  def assertAllFinite(self, x):
    self.assertAllEqual(np.ones_like(x).astype(bool), np.isfinite(x))

  def _log_gamma_log_prob(self, x, event_dims=()):
    """Computes log-pdf of a log-gamma random variable.

    Args:
      x: Value of the random variable.
      event_dims: Dimensions not to treat as independent.

    Returns:
      log_prob: The log-pdf up to a normalizing constant.
    """
    return math_ops.reduce_sum(self._shape_param * x -
                               self._rate_param * math_ops.exp(x),
                               event_dims)

  def _log_gamma_log_prob_grad(self, x, event_dims=()):
    """Computes log-pdf and gradient of a log-gamma random variable.

    Args:
      x: Value of the random variable.
      event_dims: Dimensions not to treat as independent. Default is (),
        i.e., all dimensions are independent.

    Returns:
      log_prob: The log-pdf up to a normalizing constant.
      grad: The gradient of the log-pdf with respect to x.
    """
    return (math_ops.reduce_sum(self._shape_param * x -
                                self._rate_param * math_ops.exp(x),
                                event_dims),
            self._shape_param - self._rate_param * math_ops.exp(x))

  def _n_event_dims(self, x_shape, event_dims):
    return np.prod([int(x_shape[i]) for i in event_dims])

  def _integrator_conserves_energy(self, x, event_dims, sess,
                                   feed_dict=None):
    def potential_and_grad(x):
      log_prob, grad = self._log_gamma_log_prob_grad(x, event_dims)
      return -log_prob, -grad

    step_size = array_ops.placeholder(np.float32, [], name='step_size')
    hmc_lf_steps = array_ops.placeholder(np.int32, [], name='hmc_lf_steps')

    if feed_dict is None:
      feed_dict = {}
    feed_dict[hmc_lf_steps] = 1000

    m = random_ops.random_normal(array_ops.shape(x))
    potential_0, grad_0 = potential_and_grad(x)
    old_energy = potential_0 + 0.5 * math_ops.reduce_sum(m * m,
                                                         event_dims)

    _, new_m, potential_1, _ = (
        hmc.leapfrog_integrator(step_size, hmc_lf_steps, x,
                                m, potential_and_grad, grad_0))

    new_energy = potential_1 + 0.5 * math_ops.reduce_sum(new_m * new_m,
                                                         event_dims)

    x_shape = sess.run(x, feed_dict).shape
    n_event_dims = self._n_event_dims(x_shape, event_dims)
    feed_dict[step_size] = 0.1 / n_event_dims
    old_energy_val, new_energy_val = sess.run([old_energy, new_energy],
                                              feed_dict)
    logging.vlog(1, 'average energy change: {}'.format(
        abs(old_energy_val - new_energy_val).mean()))

    self.assertAllEqual(np.ones_like(new_energy_val, dtype=np.bool),
                        abs(old_energy_val - new_energy_val) < 1.)

  def _integrator_conserves_energy_wrapper(self, event_dims):
    """Tests the long-term energy conservation of the leapfrog integrator.

    The leapfrog integrator is symplectic, so for sufficiently small step
    sizes it should be possible to run it more or less indefinitely without
    the energy of the system blowing up or collapsing.

    Args:
      event_dims: A tuple of dimensions that should not be treated as
        independent. This allows for multiple chains to be run independently
        in parallel. Default is (), i.e., all dimensions are independent.
    """
    with self.test_session() as sess:
      x_ph = array_ops.placeholder(np.float32, name='x_ph')

      feed_dict = {x_ph: np.zeros([50, 10, 2])}
      self._integrator_conserves_energy(x_ph, event_dims, sess, feed_dict)

  def testIntegratorEnergyConservationNullShape(self):
    self._integrator_conserves_energy_wrapper([])

  def testIntegratorEnergyConservation1(self):
    self._integrator_conserves_energy_wrapper([1])

  def testIntegratorEnergyConservation2(self):
    self._integrator_conserves_energy_wrapper([2])

  def testIntegratorEnergyConservation12(self):
    self._integrator_conserves_energy_wrapper([1, 2])

  def testIntegratorEnergyConservation012(self):
    self._integrator_conserves_energy_wrapper([0, 1, 2])

  def _chain_gets_correct_expectations(self, x, event_dims, sess,
                                       feed_dict=None):
    def log_gamma_log_prob(x):
      return self._log_gamma_log_prob(x, event_dims)

    step_size = array_ops.placeholder(np.float32, [], name='step_size')
    hmc_lf_steps = array_ops.placeholder(np.int32, [], name='hmc_lf_steps')
    hmc_n_steps = array_ops.placeholder(np.int32, [], name='hmc_n_steps')

    if feed_dict is None:
      feed_dict = {}
    feed_dict.update({step_size: 0.1,
                      hmc_lf_steps: 2,
                      hmc_n_steps: 300})

    sample_chain, acceptance_prob_chain = hmc.chain([hmc_n_steps],
                                                    step_size,
                                                    hmc_lf_steps,
                                                    x, log_gamma_log_prob,
                                                    event_dims)

    acceptance_probs, samples = sess.run([acceptance_prob_chain, sample_chain],
                                         feed_dict)
    samples = samples[feed_dict[hmc_n_steps] // 2:]
    expected_x_est = samples.mean()
    expected_exp_x_est = np.exp(samples).mean()

    logging.vlog(1, 'True      E[x, exp(x)]: {}\t{}'.format(
        self._expected_x, self._expected_exp_x))
    logging.vlog(1, 'Estimated E[x, exp(x)]: {}\t{}'.format(
        expected_x_est, expected_exp_x_est))
    self.assertNear(expected_x_est, self._expected_x, 2e-2)
    self.assertNear(expected_exp_x_est, self._expected_exp_x, 2e-2)
    self.assertTrue((acceptance_probs > 0.5).all())
    self.assertTrue((acceptance_probs <= 1.0).all())

  def _chain_gets_correct_expectations_wrapper(self, event_dims):
    with self.test_session() as sess:
      x_ph = array_ops.placeholder(np.float32, name='x_ph')

      feed_dict = {x_ph: np.zeros([50, 10, 2])}
      self._chain_gets_correct_expectations(x_ph, event_dims, sess,
                                            feed_dict)

  def testHMCChainExpectationsNullShape(self):
    self._chain_gets_correct_expectations_wrapper([])

  def testHMCChainExpectations1(self):
    self._chain_gets_correct_expectations_wrapper([1])

  def testHMCChainExpectations2(self):
    self._chain_gets_correct_expectations_wrapper([2])

  def testHMCChainExpectations12(self):
    self._chain_gets_correct_expectations_wrapper([1, 2])

  def _kernel_leaves_target_invariant(self, initial_draws, event_dims,
                                      sess, feed_dict=None):
    def log_gamma_log_prob(x):
      return self._log_gamma_log_prob(x, event_dims)

    def fake_log_prob(x):
      """Cooled version of the target distribution."""
      return 1.1 * log_gamma_log_prob(x)

    step_size = array_ops.placeholder(np.float32, [], name='step_size')

    if feed_dict is None:
      feed_dict = {}

    feed_dict[step_size] = 0.4

    sample, acceptance_probs, _, _ = hmc.kernel(step_size, 5, initial_draws,
                                                log_gamma_log_prob, event_dims)
    bad_sample, bad_acceptance_probs, _, _ = hmc.kernel(
        step_size, 5, initial_draws, fake_log_prob, event_dims)
    (acceptance_probs_val, bad_acceptance_probs_val, initial_draws_val,
     updated_draws_val, fake_draws_val) = sess.run([acceptance_probs,
                                                    bad_acceptance_probs,
                                                    initial_draws, sample,
                                                    bad_sample], feed_dict)
    # Confirm step size is small enough that we usually accept.
    self.assertGreater(acceptance_probs_val.mean(), 0.5)
    self.assertGreater(bad_acceptance_probs_val.mean(), 0.5)
    # Confirm step size is large enough that we sometimes reject.
    self.assertLess(acceptance_probs_val.mean(), 0.99)
    self.assertLess(bad_acceptance_probs_val.mean(), 0.99)
    _, ks_p_value_true = stats.ks_2samp(initial_draws_val.flatten(),
                                        updated_draws_val.flatten())
    _, ks_p_value_fake = stats.ks_2samp(initial_draws_val.flatten(),
                                        fake_draws_val.flatten())
    logging.vlog(1, 'acceptance rate for true target: {}'.format(
        acceptance_probs_val.mean()))
    logging.vlog(1, 'acceptance rate for fake target: {}'.format(
        bad_acceptance_probs_val.mean()))
    logging.vlog(1, 'K-S p-value for true target: {}'.format(ks_p_value_true))
    logging.vlog(1, 'K-S p-value for fake target: {}'.format(ks_p_value_fake))
    # Make sure that the MCMC update hasn't changed the empirical CDF much.
    self.assertGreater(ks_p_value_true, 1e-3)
    # Confirm that targeting the wrong distribution does
    # significantly change the empirical CDF.
    self.assertLess(ks_p_value_fake, 1e-6)

  def _kernel_leaves_target_invariant_wrapper(self, event_dims):
    """Tests that the kernel leaves the target distribution invariant.

    Draws some independent samples from the target distribution,
    applies an iteration of the MCMC kernel, then runs a
    Kolmogorov-Smirnov test to determine if the distribution of the
    MCMC-updated samples has changed.

    We also confirm that running the kernel with a different log-pdf
    does change the target distribution. (And that we can detect that.)

    Args:
      event_dims: A tuple of dimensions that should not be treated as
        independent. This allows for multiple chains to be run independently
        in parallel. Default is (), i.e., all dimensions are independent.
    """
    with self.test_session() as sess:
      initial_draws = np.log(np.random.gamma(self._shape_param,
                                             size=[50000, 2, 2]))
      initial_draws -= np.log(self._rate_param)
      x_ph = array_ops.placeholder(np.float32, name='x_ph')

      feed_dict = {x_ph: initial_draws}

      self._kernel_leaves_target_invariant(x_ph, event_dims, sess,
                                           feed_dict)

  def testKernelLeavesTargetInvariantNullShape(self):
    self._kernel_leaves_target_invariant_wrapper([])

  def testKernelLeavesTargetInvariant1(self):
    self._kernel_leaves_target_invariant_wrapper([1])

  def testKernelLeavesTargetInvariant2(self):
    self._kernel_leaves_target_invariant_wrapper([2])

  def testKernelLeavesTargetInvariant12(self):
    self._kernel_leaves_target_invariant_wrapper([1, 2])

  def _ais_gets_correct_log_normalizer(self, init, event_dims, sess,
                                       feed_dict=None):
    def proposal_log_prob(x):
      return math_ops.reduce_sum(-0.5 * x * x - 0.5 * np.log(2*np.pi),
                                 event_dims)

    def target_log_prob(x):
      return self._log_gamma_log_prob(x, event_dims)

    if feed_dict is None:
      feed_dict = {}

    w, _, _ = hmc.ais_chain(200, 0.5, 2, init, target_log_prob,
                            proposal_log_prob, event_dims)

    w_val = sess.run(w, feed_dict)
    init_shape = sess.run(init, feed_dict).shape
    normalizer_multiplier = np.prod([init_shape[i] for i in event_dims])

    true_normalizer = -self._shape_param * np.log(self._rate_param)
    true_normalizer += special.gammaln(self._shape_param)
    true_normalizer *= normalizer_multiplier

    n_weights = np.prod(w_val.shape)
    normalized_w = np.exp(w_val - true_normalizer)
    standard_error = np.std(normalized_w) / np.sqrt(n_weights)
    logging.vlog(1, 'True normalizer {}, estimated {}, n_weights {}'.format(
        true_normalizer, np.log(normalized_w.mean()) + true_normalizer,
        n_weights))
    self.assertNear(normalized_w.mean(), 1.0, 4.0 * standard_error)

  def _ais_gets_correct_log_normalizer_wrapper(self, event_dims):
    """Tests that AIS yields reasonable estimates of normalizers."""
    with self.test_session() as sess:
      x_ph = array_ops.placeholder(np.float32, name='x_ph')

      initial_draws = np.random.normal(size=[30, 2, 1])
      feed_dict = {x_ph: initial_draws}

      self._ais_gets_correct_log_normalizer(x_ph, event_dims, sess,
                                            feed_dict)

  def testAISNullShape(self):
    self._ais_gets_correct_log_normalizer_wrapper([])

  def testAIS1(self):
    self._ais_gets_correct_log_normalizer_wrapper([1])

  def testAIS2(self):
    self._ais_gets_correct_log_normalizer_wrapper([2])

  def testAIS12(self):
    self._ais_gets_correct_log_normalizer_wrapper([1, 2])

  def testNanRejection(self):
    """Tests that an update that yields NaN potentials gets rejected.

    We run HMC with a target distribution that returns NaN
    log-likelihoods if any element of x < 0, and unit-scale
    exponential log-likelihoods otherwise. The exponential potential
    pushes x towards 0, ensuring that any reasonably large update will
    push us over the edge into NaN territory.
    """
    def _unbounded_exponential_log_prob(x):
      """An exponential distribution with log-likelihood NaN for x < 0."""
      per_element_potentials = array_ops.where(x < 0,
                                               np.nan * array_ops.ones_like(x),
                                               -x)
      return math_ops.reduce_sum(per_element_potentials)

    with self.test_session() as sess:
      initial_x = math_ops.linspace(0.01, 5, 10)
      updated_x, acceptance_probs, _, _ = hmc.kernel(
          2., 5, initial_x, _unbounded_exponential_log_prob, [0])
      initial_x_val, updated_x_val, acceptance_probs_val = sess.run(
          [initial_x, updated_x, acceptance_probs])

      logging.vlog(1, 'initial_x = {}'.format(initial_x_val))
      logging.vlog(1, 'updated_x = {}'.format(updated_x_val))
      logging.vlog(1, 'acceptance_probs = {}'.format(acceptance_probs_val))

      self.assertAllEqual(initial_x_val, updated_x_val)
      self.assertEqual(acceptance_probs_val, 0.)

  def testNanFromGradsDontPropagate(self):
    """Test that update with NaN gradients does not cause NaN in results."""
    def _nan_log_prob_with_nan_gradient(x):
      return np.nan * math_ops.reduce_sum(x)

    with self.test_session() as sess:
      initial_x = math_ops.linspace(0.01, 5, 10)
      updated_x, acceptance_probs, new_log_prob, new_grad = hmc.kernel(
          2., 5, initial_x, _nan_log_prob_with_nan_gradient, [0])
      initial_x_val, updated_x_val, acceptance_probs_val = sess.run(
          [initial_x, updated_x, acceptance_probs])

      logging.vlog(1, 'initial_x = {}'.format(initial_x_val))
      logging.vlog(1, 'updated_x = {}'.format(updated_x_val))
      logging.vlog(1, 'acceptance_probs = {}'.format(acceptance_probs_val))

      self.assertAllEqual(initial_x_val, updated_x_val)
      self.assertEqual(acceptance_probs_val, 0.)

      self.assertAllFinite(
          gradients_impl.gradients(updated_x, initial_x)[0].eval())
      self.assertTrue(
          gradients_impl.gradients(new_grad, initial_x)[0] is None)

      # Gradients of the acceptance probs and new log prob are not finite.
      _ = new_log_prob  # Prevent unused arg error.
      # self.assertAllFinite(
      #     gradients_impl.gradients(acceptance_probs, initial_x)[0].eval())
      # self.assertAllFinite(
      #     gradients_impl.gradients(new_log_prob, initial_x)[0].eval())

  def testChainWorksIn64Bit(self):
    def log_prob(x):
      return - math_ops.reduce_sum(x * x, axis=-1)
    states, acceptance_probs = hmc.chain(
        n_iterations=10,
        step_size=np.float64(0.01),
        n_leapfrog_steps=10,
        initial_x=np.zeros(5).astype(np.float64),
        target_log_prob_fn=log_prob,
        event_dims=[-1])
    with self.test_session() as sess:
      states_, acceptance_probs_ = sess.run([states, acceptance_probs])
    self.assertEqual(np.float64, states_.dtype)
    self.assertEqual(np.float64, acceptance_probs_.dtype)

  def testChainWorksIn16Bit(self):
    def log_prob(x):
      return - math_ops.reduce_sum(x * x, axis=-1)
    states, acceptance_probs = hmc.chain(
        n_iterations=10,
        step_size=np.float16(0.01),
        n_leapfrog_steps=10,
        initial_x=np.zeros(5).astype(np.float16),
        target_log_prob_fn=log_prob,
        event_dims=[-1])
    with self.test_session() as sess:
      states_, acceptance_probs_ = sess.run([states, acceptance_probs])
    self.assertEqual(np.float16, states_.dtype)
    self.assertEqual(np.float16, acceptance_probs_.dtype)


if __name__ == '__main__':
  test.main()
