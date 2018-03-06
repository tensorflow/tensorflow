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
"""Tests for Metropolis-Hastings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.bayesflow.python.ops import metropolis_hastings_impl as mh
from tensorflow.contrib.distributions.python.ops import mvn_tril as mvn_tril_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.platform import test


class MetropolisHastingsTest(test.TestCase):

  def testKernelStateTensor(self):
    """Test that transition kernel works with tensor input to `state`."""
    loc = variable_scope.get_variable("loc", initializer=0.)

    def target_log_prob_fn(loc):
      return normal_lib.Normal(loc=0.0, scale=0.1).log_prob(loc)

    new_state, _ = mh.kernel(
        target_log_prob_fn=target_log_prob_fn,
        proposal_fn=mh.proposal_normal(scale=0.05),
        current_state=loc,
        seed=231251)
    loc_update = loc.assign(new_state)

    init = variables.initialize_all_variables()
    with self.test_session() as sess:
      sess.run(init)
      loc_samples = []
      for _ in range(2500):
        loc_sample = sess.run(loc_update)
        loc_samples.append(loc_sample)
    loc_samples = loc_samples[500:]  # drop samples for burn-in

    self.assertAllClose(np.mean(loc_samples), 0.0, rtol=1e-5, atol=1e-1)
    self.assertAllClose(np.std(loc_samples), 0.1, rtol=1e-5, atol=1e-1)

  def testKernelStateList(self):
    """Test that transition kernel works with list input to `state`."""
    num_chains = 2
    loc_one = variable_scope.get_variable(
        "loc_one", [num_chains],
        initializer=init_ops.zeros_initializer())
    loc_two = variable_scope.get_variable(
        "loc_two", [num_chains], initializer=init_ops.zeros_initializer())

    def target_log_prob_fn(loc_one, loc_two):
      loc = array_ops.stack([loc_one, loc_two])
      log_prob = mvn_tril_lib.MultivariateNormalTriL(
          loc=constant_op.constant([0., 0.]),
          scale_tril=constant_op.constant([[0.1, 0.1], [0.0, 0.1]])).log_prob(
              loc)
      return math_ops.reduce_sum(log_prob, 0)

    def proposal_fn(loc_one, loc_two):
      loc_one_proposal = mh.proposal_normal(scale=0.05)
      loc_two_proposal = mh.proposal_normal(scale=0.05)
      loc_one_sample, _ = loc_one_proposal(loc_one)
      loc_two_sample, _ = loc_two_proposal(loc_two)
      return [loc_one_sample, loc_two_sample], None

    new_state, _ = mh.kernel(
        target_log_prob_fn=target_log_prob_fn,
        proposal_fn=proposal_fn,
        current_state=[loc_one, loc_two],
        seed=12415)
    loc_one_update = loc_one.assign(new_state[0])
    loc_two_update = loc_two.assign(new_state[1])

    init = variables.initialize_all_variables()
    with self.test_session() as sess:
      sess.run(init)
      loc_one_samples = []
      loc_two_samples = []
      for _ in range(10000):
        loc_one_sample, loc_two_sample = sess.run(
            [loc_one_update, loc_two_update])
        loc_one_samples.append(loc_one_sample)
        loc_two_samples.append(loc_two_sample)

    loc_one_samples = np.array(loc_one_samples)
    loc_two_samples = np.array(loc_two_samples)
    loc_one_samples = loc_one_samples[1000:]  # drop samples for burn-in
    loc_two_samples = loc_two_samples[1000:]  # drop samples for burn-in

    self.assertAllClose(np.mean(loc_one_samples, 0),
                        np.array([0.] * num_chains),
                        rtol=1e-5, atol=1e-1)
    self.assertAllClose(np.mean(loc_two_samples, 0),
                        np.array([0.] * num_chains),
                        rtol=1e-5, atol=1e-1)
    self.assertAllClose(np.std(loc_one_samples, 0),
                        np.array([0.1] * num_chains),
                        rtol=1e-5, atol=1e-1)
    self.assertAllClose(np.std(loc_two_samples, 0),
                        np.array([0.1] * num_chains),
                        rtol=1e-5, atol=1e-1)

  def testKernelResultsUsingTruncatedDistribution(self):
    def log_prob(x):
      return array_ops.where(
          x >= 0.,
          -x - x**2,
          array_ops.fill(x.shape, math_ops.cast(-np.inf, x.dtype)))
    # The truncated distribution has the property that it is likely to attract
    # the flow toward, and below, zero...but for x <=0,
    # log_prob(x) = -inf, which should result in rejection, as well
    # as a non-finite log_prob.  Thus, this distribution gives us an opportunity
    # to test out the kernel results ability to correctly capture rejections due
    # to finite AND non-finite reasons.

    num_results = 1000
    # Large step size, will give rejections due to going into a region of
    # log_prob = -inf.
    step_size = 0.3
    num_chains = 2

    with self.test_session(graph=ops.Graph()) as sess:

      # Start multiple independent chains.
      initial_state = ops.convert_to_tensor([0.1] * num_chains)

      states = []
      is_accepted = []
      proposed_states = []
      current_state = initial_state
      for _ in range(num_results):
        current_state, kernel_results = mh.kernel(
            target_log_prob_fn=log_prob,
            proposal_fn=mh.proposal_uniform(step_size=step_size),
            current_state=current_state,
            seed=42)
        states.append(current_state)
        proposed_states.append(kernel_results.proposed_state)
        is_accepted.append(kernel_results.is_accepted)

      states = array_ops.stack(states)
      proposed_states = array_ops.stack(proposed_states)
      is_accepted = array_ops.stack(is_accepted)
      states_, pstates_, is_accepted_ = sess.run(
          [states, proposed_states, is_accepted])

      # We better have accepted a decent amount, even near end of the chain.
      self.assertLess(
          0.1, is_accepted_[int(0.9 * num_results):].mean())
      # We better not have any NaNs in states.
      self.assertAllEqual(np.zeros_like(states_),
                          np.isnan(states_))
      # We better not have any +inf in states.
      self.assertAllEqual(np.zeros_like(states_),
                          np.isposinf(states_))

      # The move is accepted ==> state = proposed state.
      self.assertAllEqual(
          states_[is_accepted_],
          pstates_[is_accepted_],
      )

      # The move was rejected <==> state[t] == state[t - 1].
      for t in range(1, num_results):
        for i in range(num_chains):
          if is_accepted_[t, i]:
            self.assertNotEqual(states_[t, i], states_[t - 1, i])
          else:
            self.assertEqual(states_[t, i], states_[t - 1, i])

  def testDensityIncreasingStepAccepted(self):
    """Tests that if a transition increases density, it is always accepted."""
    target_log_density = lambda x: - x * x
    state = variable_scope.get_variable("state", initializer=10.)
    state_log_density = variable_scope.get_variable(
        "state_log_density",
        initializer=target_log_density(state.initialized_value()))
    log_accept_ratio = variable_scope.get_variable(
        "log_accept_ratio", initializer=0.)

    get_next_proposal = lambda x: (x - 1., None)
    step = mh.evolve(state, state_log_density, log_accept_ratio,
                     target_log_density, get_next_proposal, seed=1234)
    init = variables.initialize_all_variables()
    with self.test_session() as sess:
      sess.run(init)
      for j in range(9):
        sess.run(step)
        sample = sess.run(state)
        sample_log_density = sess.run(state_log_density)
        self.assertAlmostEqual(sample, 9 - j)
        self.assertAlmostEqual(sample_log_density, - (9 - j) * (9 - j))

  def testSampleProperties(self):
    """Tests that the samples converge to the target distribution."""

    def target_log_density(x):
      """Log-density corresponding to a normal distribution with mean = 4."""
      return - (x - 2.0) * (x - 2.0) * 0.5

    # Use the uniform random walker to generate proposals.
    proposal_fn = mh.proposal_uniform(
        step_size=1.0, seed=1234)

    state = variable_scope.get_variable("state", initializer=0.0)
    state_log_density = variable_scope.get_variable(
        "state_log_density",
        initializer=target_log_density(state.initialized_value()))
    log_accept_ratio = variable_scope.get_variable(
        "log_accept_ratio", initializer=0.)

    # Random walk MCMC converges slowly so need to put in enough iterations.
    num_iterations = 5000
    step = mh.evolve(state, state_log_density, log_accept_ratio,
                     target_log_density, proposal_fn, seed=4321)

    init = variables.global_variables_initializer()

    sample_sum, sample_sq_sum = 0.0, 0.0
    with self.test_session() as sess:
      sess.run(init)
      for _ in np.arange(num_iterations):
        # Allow for the mixing of the chain and discard these samples.
        sess.run(step)
      for _ in np.arange(num_iterations):
        sess.run(step)
        sample = sess.run(state)
        sample_sum += sample
        sample_sq_sum += sample * sample

    sample_mean = sample_sum / num_iterations
    sample_variance = sample_sq_sum / num_iterations - sample_mean * sample_mean
    # The samples have large autocorrelation which reduces the effective sample
    # size.
    self.assertAlmostEqual(sample_mean, 2.0, delta=0.1)
    self.assertAlmostEqual(sample_variance, 1.0, delta=0.1)

  def testProposalNormal(self):
    """Tests that the normal proposals are correctly distributed."""

    initial_points = array_ops.ones([10000], dtype=dtypes.float32)
    proposal_fn = mh.proposal_normal(
        scale=2.0, seed=1234)
    proposal_points, _ = proposal_fn(initial_points)

    with self.test_session() as sess:
      sample = sess.run(proposal_points)

    # It is expected that the elements in proposal_points have the same mean as
    # initial_points and have the standard deviation that was supplied to the
    # proposal scheme.
    self.assertAlmostEqual(np.mean(sample), 1.0, delta=0.1)
    self.assertAlmostEqual(np.std(sample), 2.0, delta=0.1)

  def testDocstringExample(self):
    """Tests the simplified docstring example with multiple chains."""

    n = 2  # dimension of the problem

    # Generate 300 initial values randomly. Each of these would be an
    # independent starting point for a Markov chain.
    state = variable_scope.get_variable(
        "state", initializer=random_ops.random_normal(
            [300, n], mean=3.0, dtype=dtypes.float32, seed=42))

    # Computes the log(p(x)) for the unit normal density and ignores the
    # normalization constant.
    def log_density(x):
      return  - math_ops.reduce_sum(x * x, reduction_indices=-1) / 2.0

    # Initial log-density value
    state_log_density = variable_scope.get_variable(
        "state_log_density",
        initializer=log_density(state.initialized_value()))

    # A variable to store the log_acceptance_ratio:
    log_acceptance_ratio = variable_scope.get_variable(
        "log_acceptance_ratio",
        initializer=array_ops.zeros([300], dtype=dtypes.float32))

    # Generates random proposals by moving each coordinate uniformly and
    # independently in a box of size 2 centered around the current value.
    # Returns the new point and also the log of the Hastings ratio (the
    # ratio of the probability of going from the proposal to origin and the
    # probability of the reverse transition). When this ratio is 1, the value
    # may be omitted and replaced by None.
    def random_proposal(x):
      return (x + random_ops.random_uniform(
          array_ops.shape(x), minval=-1, maxval=1,
          dtype=x.dtype, seed=12)), None

    #  Create the op to propagate the chain for 100 steps.
    stepper = mh.evolve(
        state, state_log_density, log_acceptance_ratio,
        log_density, random_proposal, n_steps=100, seed=123)
    init = variables.initialize_all_variables()
    with self.test_session() as sess:
      sess.run(init)
      # Run the chains for a total of 1000 steps.
      for _ in range(10):
        sess.run(stepper)
      samples = sess.run(state)
      covariance = np.eye(n)
      # Verify that the estimated mean and covariance are close to the true
      # values.
      self.assertAlmostEqual(
          np.max(np.abs(np.mean(samples, 0)
                        - np.zeros(n))), 0,
          delta=0.1)
      self.assertAlmostEqual(
          np.max(np.abs(np.reshape(np.cov(samples, rowvar=False), [n**2])
                        - np.reshape(covariance, [n**2]))), 0,
          delta=0.2)

if __name__ == "__main__":
  test.main()
