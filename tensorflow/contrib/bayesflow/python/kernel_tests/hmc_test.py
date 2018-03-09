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

import collections

import numpy as np
from scipy import stats

from tensorflow.contrib.bayesflow.python.ops import hmc
from tensorflow.contrib.bayesflow.python.ops.hmc_impl import _compute_energy_change
from tensorflow.contrib.bayesflow.python.ops.hmc_impl import _leapfrog_integrator

from tensorflow.contrib.distributions.python.ops import independent as independent_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gradients_impl as gradients_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import gamma as gamma_lib
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging_ops


def _reduce_variance(x, axis=None, keepdims=False):
  sample_mean = math_ops.reduce_mean(x, axis, keepdims=True)
  return math_ops.reduce_mean(
      math_ops.squared_difference(x, sample_mean), axis, keepdims)


class HMCTest(test.TestCase):

  def setUp(self):
    self._shape_param = 5.
    self._rate_param = 10.

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

  def _integrator_conserves_energy(self, x, independent_chain_ndims, sess,
                                   feed_dict=None):
    step_size = array_ops.placeholder(np.float32, [], name="step_size")
    hmc_lf_steps = array_ops.placeholder(np.int32, [], name="hmc_lf_steps")

    if feed_dict is None:
      feed_dict = {}
    feed_dict[hmc_lf_steps] = 1000

    event_dims = math_ops.range(independent_chain_ndims,
                                array_ops.rank(x))

    m = random_ops.random_normal(array_ops.shape(x))
    log_prob_0 = self._log_gamma_log_prob(x, event_dims)
    grad_0 = gradients_ops.gradients(log_prob_0, x)
    old_energy = -log_prob_0 + 0.5 * math_ops.reduce_sum(m**2., event_dims)

    new_m, _, log_prob_1, _ = _leapfrog_integrator(
        current_momentums=[m],
        target_log_prob_fn=lambda x: self._log_gamma_log_prob(x, event_dims),
        current_state_parts=[x],
        step_sizes=[step_size],
        num_leapfrog_steps=hmc_lf_steps,
        current_target_log_prob=log_prob_0,
        current_grads_target_log_prob=grad_0)
    new_m = new_m[0]

    new_energy = -log_prob_1 + 0.5 * math_ops.reduce_sum(new_m * new_m,
                                                         event_dims)

    x_shape = sess.run(x, feed_dict).shape
    event_size = np.prod(x_shape[independent_chain_ndims:])
    feed_dict[step_size] = 0.1 / event_size
    old_energy_, new_energy_ = sess.run([old_energy, new_energy],
                                        feed_dict)
    logging_ops.vlog(1, "average energy relative change: {}".format(
        (1. - new_energy_ / old_energy_).mean()))
    self.assertAllClose(old_energy_, new_energy_, atol=0., rtol=0.02)

  def _integrator_conserves_energy_wrapper(self, independent_chain_ndims):
    """Tests the long-term energy conservation of the leapfrog integrator.

    The leapfrog integrator is symplectic, so for sufficiently small step
    sizes it should be possible to run it more or less indefinitely without
    the energy of the system blowing up or collapsing.

    Args:
      independent_chain_ndims: Python `int` scalar representing the number of
        dims associated with independent chains.
    """
    with self.test_session(graph=ops.Graph()) as sess:
      x_ph = array_ops.placeholder(np.float32, name="x_ph")
      feed_dict = {x_ph: np.random.rand(50, 10, 2)}
      self._integrator_conserves_energy(x_ph, independent_chain_ndims,
                                        sess, feed_dict)

  def testIntegratorEnergyConservationNullShape(self):
    self._integrator_conserves_energy_wrapper(0)

  def testIntegratorEnergyConservation1(self):
    self._integrator_conserves_energy_wrapper(1)

  def testIntegratorEnergyConservation2(self):
    self._integrator_conserves_energy_wrapper(2)

  def testIntegratorEnergyConservation3(self):
    self._integrator_conserves_energy_wrapper(3)

  def testSampleChainSeedReproducibleWorksCorrectly(self):
    with self.test_session(graph=ops.Graph()) as sess:
      num_results = 10
      independent_chain_ndims = 1

      def log_gamma_log_prob(x):
        event_dims = math_ops.range(independent_chain_ndims,
                                    array_ops.rank(x))
        return self._log_gamma_log_prob(x, event_dims)

      kwargs = dict(
          target_log_prob_fn=log_gamma_log_prob,
          current_state=np.random.rand(4, 3, 2),
          step_size=0.1,
          num_leapfrog_steps=2,
          num_burnin_steps=150,
          seed=52,
      )

      samples0, kernel_results0 = hmc.sample_chain(
          **dict(list(kwargs.items()) + list(dict(
              num_results=2 * num_results,
              num_steps_between_results=0).items())))

      samples1, kernel_results1 = hmc.sample_chain(
          **dict(list(kwargs.items()) + list(dict(
              num_results=num_results,
              num_steps_between_results=1).items())))

      [
          samples0_,
          samples1_,
          target_log_prob0_,
          target_log_prob1_,
      ] = sess.run([
          samples0,
          samples1,
          kernel_results0.current_target_log_prob,
          kernel_results1.current_target_log_prob,
      ])
      self.assertAllClose(samples0_[::2], samples1_,
                          atol=1e-5, rtol=1e-5)
      self.assertAllClose(target_log_prob0_[::2], target_log_prob1_,
                          atol=1e-5, rtol=1e-5)

  def _chain_gets_correct_expectations(self, x, independent_chain_ndims,
                                       sess, feed_dict=None):
    counter = collections.Counter()
    def log_gamma_log_prob(x):
      counter["target_calls"] += 1
      event_dims = math_ops.range(independent_chain_ndims,
                                  array_ops.rank(x))
      return self._log_gamma_log_prob(x, event_dims)

    num_results = array_ops.placeholder(
        np.int32, [], name="num_results")
    step_size = array_ops.placeholder(
        np.float32, [], name="step_size")
    num_leapfrog_steps = array_ops.placeholder(
        np.int32, [], name="num_leapfrog_steps")

    if feed_dict is None:
      feed_dict = {}
    feed_dict.update({num_results: 150,
                      step_size: 0.05,
                      num_leapfrog_steps: 2})

    samples, kernel_results = hmc.sample_chain(
        num_results=num_results,
        target_log_prob_fn=log_gamma_log_prob,
        current_state=x,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
        num_burnin_steps=150,
        seed=42)

    self.assertAllEqual(dict(target_calls=2), counter)

    expected_x = (math_ops.digamma(self._shape_param)
                  - np.log(self._rate_param))

    expected_exp_x = self._shape_param / self._rate_param

    log_accept_ratio_, samples_, expected_x_ = sess.run(
        [kernel_results.log_accept_ratio, samples, expected_x],
        feed_dict)

    actual_x = samples_.mean()
    actual_exp_x = np.exp(samples_).mean()
    acceptance_probs = np.exp(np.minimum(log_accept_ratio_, 0.))

    logging_ops.vlog(1, "True      E[x, exp(x)]: {}\t{}".format(
        expected_x_, expected_exp_x))
    logging_ops.vlog(1, "Estimated E[x, exp(x)]: {}\t{}".format(
        actual_x, actual_exp_x))
    self.assertNear(actual_x, expected_x_, 2e-2)
    self.assertNear(actual_exp_x, expected_exp_x, 2e-2)
    self.assertAllEqual(np.ones_like(acceptance_probs, np.bool),
                        acceptance_probs > 0.5)
    self.assertAllEqual(np.ones_like(acceptance_probs, np.bool),
                        acceptance_probs <= 1.)

  def _chain_gets_correct_expectations_wrapper(self, independent_chain_ndims):
    with self.test_session(graph=ops.Graph()) as sess:
      x_ph = array_ops.placeholder(np.float32, name="x_ph")
      feed_dict = {x_ph: np.random.rand(50, 10, 2)}
      self._chain_gets_correct_expectations(x_ph, independent_chain_ndims,
                                            sess, feed_dict)

  def testHMCChainExpectationsNullShape(self):
    self._chain_gets_correct_expectations_wrapper(0)

  def testHMCChainExpectations1(self):
    self._chain_gets_correct_expectations_wrapper(1)

  def testHMCChainExpectations2(self):
    self._chain_gets_correct_expectations_wrapper(2)

  def testKernelResultsUsingTruncatedDistribution(self):
    def log_prob(x):
      return array_ops.where(
          x >= 0.,
          -x - x**2,  # Non-constant gradient.
          array_ops.fill(x.shape, math_ops.cast(-np.inf, x.dtype)))
    # This log_prob has the property that it is likely to attract
    # the flow toward, and below, zero...but for x <=0,
    # log_prob(x) = -inf, which should result in rejection, as well
    # as a non-finite log_prob.  Thus, this distribution gives us an opportunity
    # to test out the kernel results ability to correctly capture rejections due
    # to finite AND non-finite reasons.
    # Why use a non-constant gradient?  This ensures the leapfrog integrator
    # will not be exact.

    num_results = 1000
    # Large step size, will give rejections due to integration error in addition
    # to rejection due to going into a region of log_prob = -inf.
    step_size = 0.1
    num_leapfrog_steps = 5
    num_chains = 2

    with self.test_session(graph=ops.Graph()) as sess:

      # Start multiple independent chains.
      initial_state = ops.convert_to_tensor([0.1] * num_chains)

      states, kernel_results = hmc.sample_chain(
          num_results=num_results,
          target_log_prob_fn=log_prob,
          current_state=initial_state,
          step_size=step_size,
          num_leapfrog_steps=num_leapfrog_steps,
          seed=42)

      states_, kernel_results_ = sess.run([states, kernel_results])
      pstates_ = kernel_results_.proposed_state

      neg_inf_mask = np.isneginf(kernel_results_.proposed_target_log_prob)

      # First:  Test that the mathematical properties of the above log prob
      # function in conjunction with HMC show up as expected in kernel_results_.

      # We better have log_prob = -inf some of the time.
      self.assertLess(0, neg_inf_mask.sum())
      # We better have some rejections due to something other than -inf.
      self.assertLess(neg_inf_mask.sum(), (~kernel_results_.is_accepted).sum())
      # We better have accepted a decent amount, even near end of the chain.
      self.assertLess(
          0.1, kernel_results_.is_accepted[int(0.9 * num_results):].mean())
      # We better not have any NaNs in states or log_prob.
      # We may have some NaN in grads, which involve multiplication/addition due
      # to gradient rules.  This is the known "NaN grad issue with tf.where."
      self.assertAllEqual(np.zeros_like(states_),
                          np.isnan(kernel_results_.proposed_target_log_prob))
      self.assertAllEqual(np.zeros_like(states_),
                          np.isnan(states_))
      # We better not have any +inf in states, grads, or log_prob.
      self.assertAllEqual(np.zeros_like(states_),
                          np.isposinf(kernel_results_.proposed_target_log_prob))
      self.assertAllEqual(
          np.zeros_like(states_),
          np.isposinf(kernel_results_.proposed_grads_target_log_prob[0]))
      self.assertAllEqual(np.zeros_like(states_),
                          np.isposinf(states_))

      # Second:  Test that kernel_results is congruent with itself and
      # acceptance/rejection of states.

      # Proposed state is negative iff proposed target log prob is -inf.
      np.testing.assert_array_less(pstates_[neg_inf_mask], 0.)
      np.testing.assert_array_less(0., pstates_[~neg_inf_mask])

      # Acceptance probs are zero whenever proposed state is negative.
      acceptance_probs = np.exp(np.minimum(
          kernel_results_.log_accept_ratio, 0.))
      self.assertAllEqual(
          np.zeros_like(pstates_[neg_inf_mask]),
          acceptance_probs[neg_inf_mask])

      # The move is accepted ==> state = proposed state.
      self.assertAllEqual(
          states_[kernel_results_.is_accepted],
          pstates_[kernel_results_.is_accepted],
      )
      # The move was rejected <==> state[t] == state[t - 1].
      for t in range(1, num_results):
        for i in range(num_chains):
          if kernel_results_.is_accepted[t, i]:
            self.assertNotEqual(states_[t, i], states_[t - 1, i])
          else:
            self.assertEqual(states_[t, i], states_[t - 1, i])

  def _kernel_leaves_target_invariant(self, initial_draws,
                                      independent_chain_ndims,
                                      sess, feed_dict=None):
    def log_gamma_log_prob(x):
      event_dims = math_ops.range(independent_chain_ndims, array_ops.rank(x))
      return self._log_gamma_log_prob(x, event_dims)

    def fake_log_prob(x):
      """Cooled version of the target distribution."""
      return 1.1 * log_gamma_log_prob(x)

    step_size = array_ops.placeholder(np.float32, [], name="step_size")

    if feed_dict is None:
      feed_dict = {}

    feed_dict[step_size] = 0.4

    sample, kernel_results = hmc.kernel(
        target_log_prob_fn=log_gamma_log_prob,
        current_state=initial_draws,
        step_size=step_size,
        num_leapfrog_steps=5,
        seed=43)

    bad_sample, bad_kernel_results = hmc.kernel(
        target_log_prob_fn=fake_log_prob,
        current_state=initial_draws,
        step_size=step_size,
        num_leapfrog_steps=5,
        seed=44)

    [
        log_accept_ratio_,
        bad_log_accept_ratio_,
        initial_draws_,
        updated_draws_,
        fake_draws_,
    ] = sess.run([
        kernel_results.log_accept_ratio,
        bad_kernel_results.log_accept_ratio,
        initial_draws,
        sample,
        bad_sample,
    ], feed_dict)

    # Confirm step size is small enough that we usually accept.
    acceptance_probs = np.exp(np.minimum(log_accept_ratio_, 0.))
    bad_acceptance_probs = np.exp(np.minimum(bad_log_accept_ratio_, 0.))
    self.assertGreater(acceptance_probs.mean(), 0.5)
    self.assertGreater(bad_acceptance_probs.mean(), 0.5)

    # Confirm step size is large enough that we sometimes reject.
    self.assertLess(acceptance_probs.mean(), 0.99)
    self.assertLess(bad_acceptance_probs.mean(), 0.99)

    _, ks_p_value_true = stats.ks_2samp(initial_draws_.flatten(),
                                        updated_draws_.flatten())
    _, ks_p_value_fake = stats.ks_2samp(initial_draws_.flatten(),
                                        fake_draws_.flatten())

    logging_ops.vlog(1, "acceptance rate for true target: {}".format(
        acceptance_probs.mean()))
    logging_ops.vlog(1, "acceptance rate for fake target: {}".format(
        bad_acceptance_probs.mean()))
    logging_ops.vlog(1, "K-S p-value for true target: {}".format(
        ks_p_value_true))
    logging_ops.vlog(1, "K-S p-value for fake target: {}".format(
        ks_p_value_fake))
    # Make sure that the MCMC update hasn't changed the empirical CDF much.
    self.assertGreater(ks_p_value_true, 1e-3)
    # Confirm that targeting the wrong distribution does
    # significantly change the empirical CDF.
    self.assertLess(ks_p_value_fake, 1e-6)

  def _kernel_leaves_target_invariant_wrapper(self, independent_chain_ndims):
    """Tests that the kernel leaves the target distribution invariant.

    Draws some independent samples from the target distribution,
    applies an iteration of the MCMC kernel, then runs a
    Kolmogorov-Smirnov test to determine if the distribution of the
    MCMC-updated samples has changed.

    We also confirm that running the kernel with a different log-pdf
    does change the target distribution. (And that we can detect that.)

    Args:
      independent_chain_ndims: Python `int` scalar representing the number of
        dims associated with independent chains.
    """
    with self.test_session(graph=ops.Graph()) as sess:
      initial_draws = np.log(np.random.gamma(self._shape_param,
                                             size=[50000, 2, 2]))
      initial_draws -= np.log(self._rate_param)
      x_ph = array_ops.placeholder(np.float32, name="x_ph")

      feed_dict = {x_ph: initial_draws}

      self._kernel_leaves_target_invariant(x_ph, independent_chain_ndims,
                                           sess, feed_dict)

  def testKernelLeavesTargetInvariant1(self):
    self._kernel_leaves_target_invariant_wrapper(1)

  def testKernelLeavesTargetInvariant2(self):
    self._kernel_leaves_target_invariant_wrapper(2)

  def testKernelLeavesTargetInvariant3(self):
    self._kernel_leaves_target_invariant_wrapper(3)

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
      per_element_potentials = array_ops.where(
          x < 0.,
          array_ops.fill(array_ops.shape(x), x.dtype.as_numpy_dtype(np.nan)),
          -x)
      return math_ops.reduce_sum(per_element_potentials)

    with self.test_session(graph=ops.Graph()) as sess:
      initial_x = math_ops.linspace(0.01, 5, 10)
      updated_x, kernel_results = hmc.kernel(
          target_log_prob_fn=_unbounded_exponential_log_prob,
          current_state=initial_x,
          step_size=2.,
          num_leapfrog_steps=5,
          seed=46)
      initial_x_, updated_x_, log_accept_ratio_ = sess.run(
          [initial_x, updated_x, kernel_results.log_accept_ratio])
      acceptance_probs = np.exp(np.minimum(log_accept_ratio_, 0.))

      logging_ops.vlog(1, "initial_x = {}".format(initial_x_))
      logging_ops.vlog(1, "updated_x = {}".format(updated_x_))
      logging_ops.vlog(1, "log_accept_ratio = {}".format(log_accept_ratio_))

      self.assertAllEqual(initial_x_, updated_x_)
      self.assertEqual(acceptance_probs, 0.)

  def testNanFromGradsDontPropagate(self):
    """Test that update with NaN gradients does not cause NaN in results."""
    def _nan_log_prob_with_nan_gradient(x):
      return np.nan * math_ops.reduce_sum(x)

    with self.test_session(graph=ops.Graph()) as sess:
      initial_x = math_ops.linspace(0.01, 5, 10)
      updated_x, kernel_results = hmc.kernel(
          target_log_prob_fn=_nan_log_prob_with_nan_gradient,
          current_state=initial_x,
          step_size=2.,
          num_leapfrog_steps=5,
          seed=47)
      initial_x_, updated_x_, log_accept_ratio_ = sess.run(
          [initial_x, updated_x, kernel_results.log_accept_ratio])
      acceptance_probs = np.exp(np.minimum(log_accept_ratio_, 0.))

      logging_ops.vlog(1, "initial_x = {}".format(initial_x_))
      logging_ops.vlog(1, "updated_x = {}".format(updated_x_))
      logging_ops.vlog(1, "log_accept_ratio = {}".format(log_accept_ratio_))

      self.assertAllEqual(initial_x_, updated_x_)
      self.assertEqual(acceptance_probs, 0.)

      self.assertAllFinite(
          gradients_ops.gradients(updated_x, initial_x)[0].eval())
      self.assertAllEqual([True], [g is None for g in gradients_ops.gradients(
          kernel_results.proposed_grads_target_log_prob, initial_x)])
      self.assertAllEqual([False], [g is None for g in gradients_ops.gradients(
          kernel_results.proposed_grads_target_log_prob,
          kernel_results.proposed_state)])

      # Gradients of the acceptance probs and new log prob are not finite.
      # self.assertAllFinite(
      #     gradients_ops.gradients(acceptance_probs, initial_x)[0].eval())
      # self.assertAllFinite(
      #     gradients_ops.gradients(new_log_prob, initial_x)[0].eval())

  def _testChainWorksDtype(self, dtype):
    with self.test_session(graph=ops.Graph()) as sess:
      states, kernel_results = hmc.sample_chain(
          num_results=10,
          target_log_prob_fn=lambda x: -math_ops.reduce_sum(x**2., axis=-1),
          current_state=np.zeros(5).astype(dtype),
          step_size=0.01,
          num_leapfrog_steps=10,
          seed=48)
      states_, log_accept_ratio_ = sess.run(
          [states, kernel_results.log_accept_ratio])
      self.assertEqual(dtype, states_.dtype)
      self.assertEqual(dtype, log_accept_ratio_.dtype)

  def testChainWorksIn64Bit(self):
    self._testChainWorksDtype(np.float64)

  def testChainWorksIn16Bit(self):
    self._testChainWorksDtype(np.float16)

  def testChainWorksCorrelatedMultivariate(self):
    dtype = np.float32
    true_mean = dtype([0, 0])
    true_cov = dtype([[1, 0.5],
                      [0.5, 1]])
    num_results = 2000
    counter = collections.Counter()
    with self.test_session(graph=ops.Graph()) as sess:
      def target_log_prob(x, y):
        counter["target_calls"] += 1
        # Corresponds to unnormalized MVN.
        # z = matmul(inv(chol(true_cov)), [x, y] - true_mean)
        z = array_ops.stack([x, y], axis=-1) - true_mean
        z = array_ops.squeeze(
            gen_linalg_ops.matrix_triangular_solve(
                np.linalg.cholesky(true_cov),
                z[..., array_ops.newaxis]),
            axis=-1)
        return -0.5 * math_ops.reduce_sum(z**2., axis=-1)
      states, _ = hmc.sample_chain(
          num_results=num_results,
          target_log_prob_fn=target_log_prob,
          current_state=[dtype(-2), dtype(2)],
          step_size=[0.5, 0.5],
          num_leapfrog_steps=2,
          num_burnin_steps=200,
          num_steps_between_results=1,
          seed=54)
      self.assertAllEqual(dict(target_calls=2), counter)
      states = array_ops.stack(states, axis=-1)
      self.assertEqual(num_results, states.shape[0].value)
      sample_mean = math_ops.reduce_mean(states, axis=0)
      x = states - sample_mean
      sample_cov = math_ops.matmul(x, x, transpose_a=True) / dtype(num_results)
      [sample_mean_, sample_cov_] = sess.run([
          sample_mean, sample_cov])
      self.assertAllClose(true_mean, sample_mean_,
                          atol=0.05, rtol=0.)
      self.assertAllClose(true_cov, sample_cov_,
                          atol=0., rtol=0.1)


class _EnergyComputationTest(object):

  def testHandlesNanFromPotential(self):
    with self.test_session(graph=ops.Graph()) as sess:
      x = [1, np.inf, -np.inf, np.nan]
      target_log_prob, proposed_target_log_prob = [
          self.dtype(x.flatten()) for x in np.meshgrid(x, x)]
      num_chains = len(target_log_prob)
      dummy_momentums = [-1, 1]
      momentums = [self.dtype([dummy_momentums] * num_chains)]
      proposed_momentums = [self.dtype([dummy_momentums] * num_chains)]

      target_log_prob = ops.convert_to_tensor(target_log_prob)
      momentums = [ops.convert_to_tensor(momentums[0])]
      proposed_target_log_prob = ops.convert_to_tensor(proposed_target_log_prob)
      proposed_momentums = [ops.convert_to_tensor(proposed_momentums[0])]

      energy = _compute_energy_change(
          target_log_prob,
          momentums,
          proposed_target_log_prob,
          proposed_momentums,
          independent_chain_ndims=1)
      grads = gradients_ops.gradients(energy, momentums)

      [actual_energy, grads_] = sess.run([energy, grads])

      # Ensure energy is `inf` (note: that's positive inf) in weird cases and
      # finite otherwise.
      expected_energy = self.dtype([0] + [np.inf]*(num_chains - 1))
      self.assertAllEqual(expected_energy, actual_energy)

      # Ensure gradient is finite.
      self.assertAllEqual(np.ones_like(grads_).astype(np.bool),
                          np.isfinite(grads_))

  def testHandlesNanFromKinetic(self):
    with self.test_session(graph=ops.Graph()) as sess:
      x = [1, np.inf, -np.inf, np.nan]
      momentums, proposed_momentums = [
          [np.reshape(self.dtype(x), [-1, 1])]
          for x in np.meshgrid(x, x)]
      num_chains = len(momentums[0])
      target_log_prob = np.ones(num_chains, self.dtype)
      proposed_target_log_prob = np.ones(num_chains, self.dtype)

      target_log_prob = ops.convert_to_tensor(target_log_prob)
      momentums = [ops.convert_to_tensor(momentums[0])]
      proposed_target_log_prob = ops.convert_to_tensor(proposed_target_log_prob)
      proposed_momentums = [ops.convert_to_tensor(proposed_momentums[0])]

      energy = _compute_energy_change(
          target_log_prob,
          momentums,
          proposed_target_log_prob,
          proposed_momentums,
          independent_chain_ndims=1)
      grads = gradients_ops.gradients(energy, momentums)

      [actual_energy, grads_] = sess.run([energy, grads])

      # Ensure energy is `inf` (note: that's positive inf) in weird cases and
      # finite otherwise.
      expected_energy = self.dtype([0] + [np.inf]*(num_chains - 1))
      self.assertAllEqual(expected_energy, actual_energy)

      # Ensure gradient is finite.
      g = grads_[0].reshape([len(x), len(x)])[:, 0]
      self.assertAllEqual(np.ones_like(g).astype(np.bool), np.isfinite(g))

      # The remaining gradients are nan because the momentum was itself nan or
      # inf.
      g = grads_[0].reshape([len(x), len(x)])[:, 1:]
      self.assertAllEqual(np.ones_like(g).astype(np.bool), np.isnan(g))


class EnergyComputationTest16(test.TestCase, _EnergyComputationTest):
  dtype = np.float16


class EnergyComputationTest32(test.TestCase, _EnergyComputationTest):
  dtype = np.float32


class EnergyComputationTest64(test.TestCase, _EnergyComputationTest):
  dtype = np.float64


class _HMCHandlesLists(object):

  def testStateParts(self):
    with self.test_session(graph=ops.Graph()) as sess:
      dist_x = normal_lib.Normal(loc=self.dtype(0), scale=self.dtype(1))
      dist_y = independent_lib.Independent(
          gamma_lib.Gamma(concentration=self.dtype([1, 2]),
                          rate=self.dtype([0.5, 0.75])),
          reinterpreted_batch_ndims=1)
      def target_log_prob(x, y):
        return dist_x.log_prob(x) + dist_y.log_prob(y)
      x0 = [dist_x.sample(seed=1), dist_y.sample(seed=2)]
      samples, _ = hmc.sample_chain(
          num_results=int(2e3),
          target_log_prob_fn=target_log_prob,
          current_state=x0,
          step_size=0.85,
          num_leapfrog_steps=3,
          num_burnin_steps=int(250),
          seed=49)
      actual_means = [math_ops.reduce_mean(s, axis=0) for s in samples]
      actual_vars = [_reduce_variance(s, axis=0) for s in samples]
      expected_means = [dist_x.mean(), dist_y.mean()]
      expected_vars = [dist_x.variance(), dist_y.variance()]
      [
          actual_means_,
          actual_vars_,
          expected_means_,
          expected_vars_,
      ] = sess.run([
          actual_means,
          actual_vars,
          expected_means,
          expected_vars,
      ])
      self.assertAllClose(expected_means_, actual_means_, atol=0.05, rtol=0.16)
      self.assertAllClose(expected_vars_, actual_vars_, atol=0., rtol=0.25)


class HMCHandlesLists32(_HMCHandlesLists, test.TestCase):
  dtype = np.float32


class HMCHandlesLists64(_HMCHandlesLists, test.TestCase):
  dtype = np.float64


if __name__ == "__main__":
  test.main()
