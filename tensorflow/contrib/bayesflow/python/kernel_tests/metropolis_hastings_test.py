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
"""Tests for metropolis_hastings.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.contrib.bayesflow.python.ops import metropolis_hastings_impl as mh
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class McmcStepTest(test.TestCase):

  def test_density_increasing_step_accepted(self):
    """Tests that if a transition increases density, it is always accepted."""
    target_log_density = lambda x: - x * x
    state = variable_scope.get_variable('state', initializer=10.)
    state_log_density = variable_scope.get_variable(
        'state_log_density',
        initializer=target_log_density(state.initialized_value()))
    log_accept_ratio = variable_scope.get_variable(
        'log_accept_ratio', initializer=0.)

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

  def test_sample_properties(self):
    """Tests that the samples converge to the target distribution."""

    def target_log_density(x):
      """Log-density corresponding to a normal distribution with mean = 4."""
      return - (x - 2.0) * (x - 2.0) * 0.5

    # Use the uniform random walker to generate proposals.
    proposal_fn = mh.uniform_random_proposal(
        step_size=1.0, seed=1234)

    state = variable_scope.get_variable('state', initializer=0.0)
    state_log_density = variable_scope.get_variable(
        'state_log_density',
        initializer=target_log_density(state.initialized_value()))

    log_accept_ratio = variable_scope.get_variable(
        'log_accept_ratio', initializer=0.)
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

  def test_normal_proposals(self):
    """Tests that the normal proposals are correctly distributed."""

    initial_points = array_ops.ones([10000], dtype=dtypes.float32)
    proposal_fn = mh.normal_random_proposal(
        scale=2.0, seed=1234)
    proposal_points, _ = proposal_fn(initial_points)

    with self.test_session() as sess:
      sample = sess.run(proposal_points)

    # It is expected that the elements in proposal_points have the same mean as
    # initial_points and have the standard deviation that was supplied to the
    # proposal scheme.
    self.assertAlmostEqual(np.mean(sample), 1.0, delta=0.1)
    self.assertAlmostEqual(np.std(sample), 2.0, delta=0.1)

  def test_docstring_example(self):
    """Tests the simplified docstring example with multiple chains."""

    n = 2  # dimension of the problem

    # Generate 300 initial values randomly. Each of these would be an
    # independent starting point for a Markov chain.
    state = variable_scope.get_variable(
        'state', initializer=random_ops.random_normal(
            [300, n], mean=3.0, dtype=dtypes.float32, seed=42))

    # Computes the log(p(x)) for the unit normal density and ignores the
    # normalization constant.
    def log_density(x):
      return  - math_ops.reduce_sum(x * x, reduction_indices=-1) / 2.0

    # Initial log-density value
    state_log_density = variable_scope.get_variable(
        'state_log_density',
        initializer=log_density(state.initialized_value()))

    # A variable to store the log_acceptance_ratio:
    log_acceptance_ratio = variable_scope.get_variable(
        'log_acceptance_ratio',
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

if __name__ == '__main__':
  test.main()
