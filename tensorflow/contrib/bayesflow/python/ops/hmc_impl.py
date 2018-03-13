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
"""Hamiltonian Monte Carlo, a gradient-based MCMC algorithm.

@@sample_chain
@@kernel
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients_impl as gradients_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import util as distributions_util

__all__ = [
    "sample_chain",
    "kernel",
]


KernelResults = collections.namedtuple(
    "KernelResults",
    [
        "log_accept_ratio",
        "current_grads_target_log_prob",  # "Current result" means "accepted".
        "current_target_log_prob",  # "Current result" means "accepted".
        "is_accepted",
        "proposed_grads_target_log_prob",
        "proposed_state",
        "proposed_target_log_prob",
    ])


def _make_dummy_kernel_results(
    dummy_state,
    dummy_target_log_prob,
    dummy_grads_target_log_prob):
  return KernelResults(
      log_accept_ratio=dummy_target_log_prob,
      current_grads_target_log_prob=dummy_grads_target_log_prob,
      current_target_log_prob=dummy_target_log_prob,
      is_accepted=array_ops.ones_like(dummy_target_log_prob, dtypes.bool),
      proposed_grads_target_log_prob=dummy_grads_target_log_prob,
      proposed_state=dummy_state,
      proposed_target_log_prob=dummy_target_log_prob,
  )


def sample_chain(
    num_results,
    target_log_prob_fn,
    current_state,
    step_size,
    num_leapfrog_steps,
    num_burnin_steps=0,
    num_steps_between_results=0,
    seed=None,
    current_target_log_prob=None,
    current_grads_target_log_prob=None,
    name=None):
  """Runs multiple iterations of one or more Hamiltonian Monte Carlo chains.

  Hamiltonian Monte Carlo (HMC) is a Markov chain Monte Carlo (MCMC) algorithm
  that takes a series of gradient-informed steps to produce a Metropolis
  proposal. This function samples from an HMC Markov chain at `current_state`
  and whose stationary distribution has log-unnormalized-density
  `target_log_prob_fn()`.

  This function samples from multiple chains in parallel. It assumes that the
  the leftmost dimensions of (each) `current_state` (part) index an independent
  chain.  The function `target_log_prob_fn()` sums log-probabilities across
  event dimensions (i.e., current state (part) rightmost dimensions). Each
  element of the output of `target_log_prob_fn()` represents the (possibly
  unnormalized) log-probability of the joint distribution over (all) the current
  state (parts).

  The `current_state` can be represented as a single `Tensor` or a `list` of
  `Tensors` which collectively represent the current state. When specifying a
  `list`, one must also specify a list of `step_size`s.

  Note: `target_log_prob_fn` is called exactly twice.

  Since HMC states are correlated, it is sometimes desirable to produce
  additional intermediate states, and then discard them, ending up with a set of
  states with decreased autocorrelation.  See [1].  Such "thinning" is made
  possible by setting `num_steps_between_results > 0`.  The chain then takes
  `num_steps_between_results` extra steps between the steps that make it into
  the results.  The extra steps are never materialized (in calls to `sess.run`),
  and thus do not increase memory requirements.

  [1]: "Statistically efficient thinning of a Markov chain sampler."
       Art B. Owen. April 2017.
       http://statweb.stanford.edu/~owen/reports/bestthinning.pdf

  #### Examples:

  ##### Sample from a diagonal-variance Gaussian.

  ```python
  tfd = tf.contrib.distributions

  def make_likelihood(true_variances):
    return tfd.MultivariateNormalDiag(
        scale_diag=tf.sqrt(true_variances))

  dims = 10
  dtype = np.float32
  true_variances = tf.linspace(dtype(1), dtype(3), dims)
  likelihood = make_likelihood(true_variances)

  states, kernel_results = hmc.sample_chain(
      num_results=1000,
      target_log_prob_fn=likelihood.log_prob,
      current_state=tf.zeros(dims),
      step_size=0.5,
      num_leapfrog_steps=2,
      num_burnin_steps=500)

  # Compute sample stats.
  sample_mean = tf.reduce_mean(states, axis=0)
  sample_var = tf.reduce_mean(
      tf.squared_difference(states, sample_mean),
      axis=0)
  ```

  ##### Sampling from factor-analysis posteriors with known factors.

  I.e.,

  ```none
  for i=1..n:
    w[i] ~ Normal(0, eye(d))            # prior
    x[i] ~ Normal(loc=matmul(w[i], F))  # likelihood
  ```

  where `F` denotes factors.

  ```python
  tfd = tf.contrib.distributions

  def make_prior(dims, dtype):
    return tfd.MultivariateNormalDiag(
        loc=tf.zeros(dims, dtype))

  def make_likelihood(weights, factors):
    return tfd.MultivariateNormalDiag(
        loc=tf.tensordot(weights, factors, axes=[[0], [-1]]))

  # Setup data.
  num_weights = 10
  num_factors = 4
  num_chains = 100
  dtype = np.float32

  prior = make_prior(num_weights, dtype)
  weights = prior.sample(num_chains)
  factors = np.random.randn(num_factors, num_weights).astype(dtype)
  x = make_likelihood(weights, factors).sample(num_chains)

  def target_log_prob(w):
    # Target joint is: `f(w) = p(w, x | factors)`.
    return prior.log_prob(w) + make_likelihood(w, factors).log_prob(x)

  # Get `num_results` samples from `num_chains` independent chains.
  chains_states, kernels_results = hmc.sample_chain(
      num_results=1000,
      target_log_prob_fn=target_log_prob,
      current_state=tf.zeros([num_chains, dims], dtype),
      step_size=0.1,
      num_leapfrog_steps=2,
      num_burnin_steps=500)

  # Compute sample stats.
  sample_mean = tf.reduce_mean(chains_states, axis=[0, 1])
  sample_var = tf.reduce_mean(
      tf.squared_difference(chains_states, sample_mean),
      axis=[0, 1])
  ```

  Args:
    num_results: Integer number of Markov chain draws.
    target_log_prob_fn: Python callable which takes an argument like
      `current_state` (or `*current_state` if it's a list) and returns its
      (possibly unnormalized) log-density under the target distribution.
    current_state: `Tensor` or Python `list` of `Tensor`s representing the
      current state(s) of the Markov chain(s). The first `r` dimensions index
      independent chains, `r = tf.rank(target_log_prob_fn(*current_state))`.
    step_size: `Tensor` or Python `list` of `Tensor`s representing the step size
      for the leapfrog integrator. Must broadcast with the shape of
      `current_state`. Larger step sizes lead to faster progress, but too-large
      step sizes make rejection exponentially more likely. When possible, it's
      often helpful to match per-variable step sizes to the standard deviations
      of the target distribution in each variable.
    num_leapfrog_steps: Integer number of steps to run the leapfrog integrator
      for. Total progress per HMC step is roughly proportional to `step_size *
      num_leapfrog_steps`.
    num_burnin_steps: Integer number of chain steps to take before starting to
      collect results.
      Default value: 0 (i.e., no burn-in).
    num_steps_between_results: Integer number of chain steps between collecting
      a result. Only one out of every `num_steps_between_samples + 1` steps is
      included in the returned results.  The number of returned chain states is
      still equal to `num_results`.  Default value: 0 (i.e., no thinning).
    seed: Python integer to seed the random number generator.
    current_target_log_prob: (Optional) `Tensor` representing the value of
      `target_log_prob_fn` at the `current_state`. The only reason to specify
      this argument is to reduce TF graph size.
      Default value: `None` (i.e., compute as needed).
    current_grads_target_log_prob: (Optional) Python list of `Tensor`s
      representing gradient of `target_log_prob` at the `current_state` and wrt
      the `current_state`. Must have same shape as `current_state`. The only
      reason to specify this argument is to reduce TF graph size.
      Default value: `None` (i.e., compute as needed).
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., "hmc_sample_chain").

  Returns:
    next_states: Tensor or Python list of `Tensor`s representing the
      state(s) of the Markov chain(s) at each result step. Has same shape as
      input `current_state` but with a prepended `num_results`-size dimension.
    kernel_results: `collections.namedtuple` of internal calculations used to
      advance the chain.
  """
  with ops.name_scope(
      name, "hmc_sample_chain",
      [num_results, current_state, step_size, num_leapfrog_steps,
       num_burnin_steps, num_steps_between_results, seed,
       current_target_log_prob, current_grads_target_log_prob]):
    with ops.name_scope("initialize"):
      [
          current_state,
          step_size,
          current_target_log_prob,
          current_grads_target_log_prob,
      ] = _prepare_args(
          target_log_prob_fn,
          current_state,
          step_size,
          current_target_log_prob,
          current_grads_target_log_prob)
      num_results = ops.convert_to_tensor(
          num_results,
          dtype=dtypes.int32,
          name="num_results")
      num_leapfrog_steps = ops.convert_to_tensor(
          num_leapfrog_steps,
          dtype=dtypes.int32,
          name="num_leapfrog_steps")
      num_burnin_steps = ops.convert_to_tensor(
          num_burnin_steps,
          dtype=dtypes.int32,
          name="num_burnin_steps")
      num_steps_between_results = ops.convert_to_tensor(
          num_steps_between_results,
          dtype=dtypes.int32,
          name="num_steps_between_results")

    def _run_chain(num_steps, current_state, kernel_results):
      """Runs the chain(s) for `num_steps`."""
      def _loop_body(iter_, current_state, kernel_results):
        return [iter_ + 1] + list(kernel(
            target_log_prob_fn,
            current_state,
            step_size,
            num_leapfrog_steps,
            seed,
            kernel_results.current_target_log_prob,
            kernel_results.current_grads_target_log_prob))
      while_loop_kwargs = dict(
          cond=lambda iter_, *args: iter_ < num_steps,
          body=_loop_body,
          loop_vars=[
              np.int32(0),
              current_state,
              kernel_results,
          ],
      )
      if seed is not None:
        while_loop_kwargs["parallel_iterations"] = 1
      return control_flow_ops.while_loop(
          **while_loop_kwargs)[1:]  # Lop-off "iter_".

    def _scan_body(args_list, iter_):
      """Closure which implements `tf.scan` body."""
      current_state, kernel_results = args_list
      return _run_chain(
          1 + array_ops.where(math_ops.equal(iter_, 0),
                              num_burnin_steps,
                              num_steps_between_results),
          current_state,
          kernel_results)

    scan_kwargs = dict(
        fn=_scan_body,
        elems=math_ops.range(num_results),  # iter_: used to choose burnin.
        initializer=[
            current_state,
            _make_dummy_kernel_results(
                current_state,
                current_target_log_prob,
                current_grads_target_log_prob),
        ])
    if seed is not None:
      scan_kwargs["parallel_iterations"] = 1
    return functional_ops.scan(**scan_kwargs)


def kernel(target_log_prob_fn,
           current_state,
           step_size,
           num_leapfrog_steps,
           seed=None,
           current_target_log_prob=None,
           current_grads_target_log_prob=None,
           name=None):
  """Runs one iteration of Hamiltonian Monte Carlo.

  Hamiltonian Monte Carlo (HMC) is a Markov chain Monte Carlo (MCMC)
  algorithm that takes a series of gradient-informed steps to produce
  a Metropolis proposal. This function applies one step of HMC to
  randomly update the variable `x`.

  This function can update multiple chains in parallel. It assumes that all
  leftmost dimensions of `current_state` index independent chain states (and are
  therefore updated independently). The output of `target_log_prob_fn()` should
  sum log-probabilities across all event dimensions. Slices along the rightmost
  dimensions may have different target distributions; for example,
  `current_state[0, :]` could have a different target distribution from
  `current_state[1, :]`. This is up to `target_log_prob_fn()`. (The number of
  independent chains is `tf.size(target_log_prob_fn(*current_state))`.)

  #### Examples:

  ##### Simple chain with warm-up.

  ```python
  tfd = tf.contrib.distributions

  # Tuning acceptance rates:
  dtype = np.float32
  target_accept_rate = 0.631
  num_warmup_iter = 500
  num_chain_iter = 500

  x = tf.get_variable(name="x", initializer=dtype(1))
  step_size = tf.get_variable(name="step_size", initializer=dtype(1))

  target = tfd.Normal(loc=dtype(0), scale=dtype(1))

  next_x, other_results = hmc.kernel(
      target_log_prob_fn=target.log_prob,
      current_state=x,
      step_size=step_size,
      num_leapfrog_steps=3)[:4]

  x_update = x.assign(next_x)

  step_size_update = step_size.assign_add(
      step_size * tf.where(
          tf.exp(tf.minimum(other_results.log_accept_ratio), 0.) >
              target_accept_rate,
          0.01, -0.01))

  warmup = tf.group([x_update, step_size_update])

  tf.global_variables_initializer().run()

  sess.graph.finalize()  # No more graph building.

  # Warm up the sampler and adapt the step size
  for _ in xrange(num_warmup_iter):
    sess.run(warmup)

  # Collect samples without adapting step size
  samples = np.zeros([num_chain_iter])
  for i in xrange(num_chain_iter):
    _, x_, target_log_prob_, grad_ = sess.run([
        x_update,
        x,
        other_results.target_log_prob,
        other_results.grads_target_log_prob])
    samples[i] = x_

  print(samples.mean(), samples.std())
  ```

  ##### Sample from more complicated posterior.

  I.e.,

  ```none
    W ~ MVN(loc=0, scale=sigma * eye(dims))
    for i=1...num_samples:
        X[i] ~ MVN(loc=0, scale=eye(dims))
      eps[i] ~ Normal(loc=0, scale=1)
        Y[i] = X[i].T * W + eps[i]
  ```

  ```python
  tfd = tf.contrib.distributions

  def make_training_data(num_samples, dims, sigma):
    dt = np.asarray(sigma).dtype
    zeros = tf.zeros(dims, dtype=dt)
    x = tfd.MultivariateNormalDiag(
        loc=zeros).sample(num_samples, seed=1)
    w = tfd.MultivariateNormalDiag(
        loc=zeros,
        scale_identity_multiplier=sigma).sample(seed=2)
    noise = tfd.Normal(
        loc=dt(0),
        scale=dt(1)).sample(num_samples, seed=3)
    y = tf.tensordot(x, w, axes=[[1], [0]]) + noise
    return y, x, w

  def make_prior(sigma, dims):
    # p(w | sigma)
    return tfd.MultivariateNormalDiag(
        loc=tf.zeros([dims], dtype=sigma.dtype),
        scale_identity_multiplier=sigma)

  def make_likelihood(x, w):
    # p(y | x, w)
    return tfd.MultivariateNormalDiag(
        loc=tf.tensordot(x, w, axes=[[1], [0]]))

  # Setup assumptions.
  dtype = np.float32
  num_samples = 150
  dims = 10
  num_iters = int(5e3)

  true_sigma = dtype(0.5)
  y, x, true_weights = make_training_data(num_samples, dims, true_sigma)

  # Estimate of `log(true_sigma)`.
  log_sigma = tf.get_variable(name="log_sigma", initializer=dtype(0))
  sigma = tf.exp(log_sigma)

  # State of the Markov chain.
  weights = tf.get_variable(
      name="weights",
      initializer=np.random.randn(dims).astype(dtype))

  prior = make_prior(sigma, dims)

  def joint_log_prob_fn(w):
    # f(w) = log p(w, y | x)
    return prior.log_prob(w) + make_likelihood(x, w).log_prob(y)

  weights_update = weights.assign(
      hmc.kernel(target_log_prob_fn=joint_log_prob,
                 current_state=weights,
                 step_size=0.1,
                 num_leapfrog_steps=5)[0])

  with tf.control_dependencies([weights_update]):
    loss = -prior.log_prob(weights)

  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  log_sigma_update = optimizer.minimize(loss, var_list=[log_sigma])

  sess.graph.finalize()  # No more graph building.

  tf.global_variables_initializer().run()

  sigma_history = np.zeros(num_iters, dtype)
  weights_history = np.zeros([num_iters, dims], dtype)

  for i in xrange(num_iters):
    _, sigma_, weights_, _ = sess.run([log_sigma_update, sigma, weights])
    weights_history[i, :] = weights_
    sigma_history[i] = sigma_

  true_weights_ = sess.run(true_weights)

  # Should converge to something close to true_sigma.
  plt.plot(sigma_history);
  plt.ylabel("sigma");
  plt.xlabel("iteration");
  ```

  Args:
    target_log_prob_fn: Python callable which takes an argument like
      `current_state` (or `*current_state` if it's a list) and returns its
      (possibly unnormalized) log-density under the target distribution.
    current_state: `Tensor` or Python `list` of `Tensor`s representing the
      current state(s) of the Markov chain(s). The first `r` dimensions index
      independent chains, `r = tf.rank(target_log_prob_fn(*current_state))`.
    step_size: `Tensor` or Python `list` of `Tensor`s representing the step size
      for the leapfrog integrator. Must broadcast with the shape of
      `current_state`. Larger step sizes lead to faster progress, but too-large
      step sizes make rejection exponentially more likely. When possible, it's
      often helpful to match per-variable step sizes to the standard deviations
      of the target distribution in each variable.
    num_leapfrog_steps: Integer number of steps to run the leapfrog integrator
      for. Total progress per HMC step is roughly proportional to `step_size *
      num_leapfrog_steps`.
    seed: Python integer to seed the random number generator.
    current_target_log_prob: (Optional) `Tensor` representing the value of
      `target_log_prob_fn` at the `current_state`. The only reason to
      specify this argument is to reduce TF graph size.
      Default value: `None` (i.e., compute as needed).
    current_grads_target_log_prob: (Optional) Python list of `Tensor`s
      representing gradient of `current_target_log_prob` at the `current_state`
      and wrt the `current_state`. Must have same shape as `current_state`. The
      only reason to specify this argument is to reduce TF graph size.
      Default value: `None` (i.e., compute as needed).
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., "hmc_kernel").

  Returns:
    next_state: Tensor or Python list of `Tensor`s representing the state(s)
      of the Markov chain(s) at each result step. Has same shape as
      `current_state`.
    kernel_results: `collections.namedtuple` of internal calculations used to
      advance the chain.

  Raises:
    ValueError: if there isn't one `step_size` or a list with same length as
      `current_state`.
  """
  with ops.name_scope(
      name, "hmc_kernel",
      [current_state, step_size, num_leapfrog_steps, seed,
       current_target_log_prob, current_grads_target_log_prob]):
    with ops.name_scope("initialize"):
      [current_state_parts, step_sizes, current_target_log_prob,
       current_grads_target_log_prob] = _prepare_args(
           target_log_prob_fn, current_state, step_size,
           current_target_log_prob, current_grads_target_log_prob,
           maybe_expand=True)
      independent_chain_ndims = distributions_util.prefer_static_rank(
          current_target_log_prob)
      current_momentums = []
      for s in current_state_parts:
        current_momentums.append(random_ops.random_normal(
            shape=array_ops.shape(s),
            dtype=s.dtype.base_dtype,
            seed=seed))
        seed = distributions_util.gen_new_seed(
            seed, salt="hmc_kernel_momentums")

      num_leapfrog_steps = ops.convert_to_tensor(
          num_leapfrog_steps,
          dtype=dtypes.int32,
          name="num_leapfrog_steps")
    [
        proposed_momentums,
        proposed_state_parts,
        proposed_target_log_prob,
        proposed_grads_target_log_prob,
    ] = _leapfrog_integrator(current_momentums,
                             target_log_prob_fn,
                             current_state_parts,
                             step_sizes,
                             num_leapfrog_steps,
                             current_target_log_prob,
                             current_grads_target_log_prob)

    energy_change = _compute_energy_change(current_target_log_prob,
                                           current_momentums,
                                           proposed_target_log_prob,
                                           proposed_momentums,
                                           independent_chain_ndims)
    log_accept_ratio = -energy_change

    # u < exp(log_accept_ratio),  where u~Uniform[0,1)
    # ==> log(u) < log_accept_ratio
    random_value = random_ops.random_uniform(
        shape=array_ops.shape(energy_change),
        dtype=energy_change.dtype,
        seed=seed)
    random_negative = math_ops.log(random_value)
    is_accepted = random_negative < log_accept_ratio

    accepted_target_log_prob = array_ops.where(is_accepted,
                                               proposed_target_log_prob,
                                               current_target_log_prob)

    next_state_parts = [_choose(is_accepted,
                                proposed_state_part,
                                current_state_part,
                                independent_chain_ndims)
                        for current_state_part, proposed_state_part
                        in zip(current_state_parts, proposed_state_parts)]

    accepted_grads_target_log_prob = [
        _choose(is_accepted,
                proposed_grad,
                grad,
                independent_chain_ndims)
        for proposed_grad, grad
        in zip(proposed_grads_target_log_prob, current_grads_target_log_prob)]

    maybe_flatten = lambda x: x if _is_list_like(current_state) else x[0]
    return [
        maybe_flatten(next_state_parts),
        KernelResults(
            log_accept_ratio=log_accept_ratio,
            current_grads_target_log_prob=accepted_grads_target_log_prob,
            current_target_log_prob=accepted_target_log_prob,
            is_accepted=is_accepted,
            proposed_grads_target_log_prob=proposed_grads_target_log_prob,
            proposed_state=maybe_flatten(proposed_state_parts),
            proposed_target_log_prob=proposed_target_log_prob,
        ),
    ]


def _leapfrog_integrator(current_momentums,
                         target_log_prob_fn,
                         current_state_parts,
                         step_sizes,
                         num_leapfrog_steps,
                         current_target_log_prob=None,
                         current_grads_target_log_prob=None,
                         name=None):
  """Applies `num_leapfrog_steps` of the leapfrog integrator.

  Assumes a simple quadratic kinetic energy function: `0.5 ||momentum||**2`.

  #### Examples:

  ##### Simple quadratic potential.

  ```python
  tfd = tf.contrib.distributions

  dims = 10
  num_iter = int(1e3)
  dtype = np.float32

  position = tf.placeholder(np.float32)
  momentum = tf.placeholder(np.float32)

  [
      next_momentums,
      next_positions,
  ] = hmc._leapfrog_integrator(
      current_momentums=[momentum],
      target_log_prob_fn=tfd.MultivariateNormalDiag(
          loc=tf.zeros(dims, dtype)).log_prob,
      current_state_parts=[position],
      step_sizes=0.1,
      num_leapfrog_steps=3)[:2]

  sess.graph.finalize()  # No more graph building.

  momentum_ = np.random.randn(dims).astype(dtype)
  position_ = np.random.randn(dims).astype(dtype)

  positions = np.zeros([num_iter, dims], dtype)
  for i in xrange(num_iter):
    position_, momentum_ = sess.run(
        [next_momentums[0], next_position[0]],
        feed_dict={position: position_, momentum: momentum_})
    positions[i] = position_

  plt.plot(positions[:, 0]);  # Sinusoidal.
  ```

  Args:
    current_momentums: Tensor containing the value(s) of the momentum
      variable(s) to update.
    target_log_prob_fn: Python callable which takes an argument like
      `*current_state_parts` and returns its (possibly unnormalized) log-density
      under the target distribution.
    current_state_parts: Python `list` of `Tensor`s representing the current
      state(s) of the Markov chain(s). The first `independent_chain_ndims` of
      the `Tensor`(s) index different chains.
    step_sizes: Python `list` of `Tensor`s representing the step size for the
      leapfrog integrator. Must broadcast with the shape of
      `current_state_parts`.  Larger step sizes lead to faster progress, but
      too-large step sizes make rejection exponentially more likely. When
      possible, it's often helpful to match per-variable step sizes to the
      standard deviations of the target distribution in each variable.
    num_leapfrog_steps: Integer number of steps to run the leapfrog integrator
      for. Total progress per HMC step is roughly proportional to `step_size *
      num_leapfrog_steps`.
    current_target_log_prob: (Optional) `Tensor` representing the value of
      `target_log_prob_fn(*current_state_parts)`. The only reason to specify
      this argument is to reduce TF graph size.
      Default value: `None` (i.e., compute as needed).
    current_grads_target_log_prob: (Optional) Python list of `Tensor`s
      representing gradient of `target_log_prob_fn(*current_state_parts`) wrt
      `current_state_parts`. Must have same shape as `current_state_parts`. The
      only reason to specify this argument is to reduce TF graph size.
      Default value: `None` (i.e., compute as needed).
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., "hmc_leapfrog_integrator").

  Returns:
    proposed_momentums: Updated value of the momentum.
    proposed_state_parts: Tensor or Python list of `Tensor`s representing the
      state(s) of the Markov chain(s) at each result step. Has same shape as
      input `current_state_parts`.
    proposed_target_log_prob: `Tensor` representing the value of
      `target_log_prob_fn` at `next_state`.
    proposed_grads_target_log_prob: Gradient of `proposed_target_log_prob` wrt
      `next_state`.

  Raises:
    ValueError: if `len(momentums) != len(state_parts)`.
    ValueError: if `len(state_parts) != len(step_sizes)`.
    ValueError: if `len(state_parts) != len(grads_target_log_prob)`.
    TypeError: if `not target_log_prob.dtype.is_floating`.
  """
  def _loop_body(step,
                 current_momentums,
                 current_state_parts,
                 ignore_current_target_log_prob,  # pylint: disable=unused-argument
                 current_grads_target_log_prob):
    return [step + 1] + list(_leapfrog_step(current_momentums,
                                            target_log_prob_fn,
                                            current_state_parts,
                                            step_sizes,
                                            current_grads_target_log_prob))

  with ops.name_scope(
      name, "hmc_leapfrog_integrator",
      [current_momentums, current_state_parts, step_sizes, num_leapfrog_steps,
       current_target_log_prob, current_grads_target_log_prob]):
    if len(current_momentums) != len(current_state_parts):
      raise ValueError("`momentums` must be in one-to-one correspondence "
                       "with `state_parts`")
    num_leapfrog_steps = ops.convert_to_tensor(num_leapfrog_steps,
                                               name="num_leapfrog_steps")
    current_target_log_prob, current_grads_target_log_prob = (
        _maybe_call_fn_and_grads(
            target_log_prob_fn,
            current_state_parts,
            current_target_log_prob,
            current_grads_target_log_prob))
    return control_flow_ops.while_loop(
        cond=lambda iter_, *args: iter_ < num_leapfrog_steps,
        body=_loop_body,
        loop_vars=[
            np.int32(0),  # iter_
            current_momentums,
            current_state_parts,
            current_target_log_prob,
            current_grads_target_log_prob,
        ],
        back_prop=False)[1:]  # Lop-off "iter_".


def _leapfrog_step(current_momentums,
                   target_log_prob_fn,
                   current_state_parts,
                   step_sizes,
                   current_grads_target_log_prob,
                   name=None):
  """Applies one step of the leapfrog integrator."""
  with ops.name_scope(
      name, "_leapfrog_step",
      [current_momentums, current_state_parts, step_sizes,
       current_grads_target_log_prob]):
    proposed_momentums = [m + 0.5 * ss * g for m, ss, g
                          in zip(current_momentums,
                                 step_sizes,
                                 current_grads_target_log_prob)]
    proposed_state_parts = [x + ss * m for x, ss, m
                            in zip(current_state_parts,
                                   step_sizes,
                                   proposed_momentums)]
    proposed_target_log_prob = target_log_prob_fn(*proposed_state_parts)
    if not proposed_target_log_prob.dtype.is_floating:
      raise TypeError("`target_log_prob_fn` must produce a `Tensor` "
                      "with `float` `dtype`.")
    proposed_grads_target_log_prob = gradients_ops.gradients(
        proposed_target_log_prob, proposed_state_parts)
    if any(g is None for g in proposed_grads_target_log_prob):
      raise ValueError(
          "Encountered `None` gradient. Does your target `target_log_prob_fn` "
          "access all `tf.Variable`s via `tf.get_variable`?\n"
          "  current_state_parts: {}\n"
          "  proposed_state_parts: {}\n"
          "  proposed_grads_target_log_prob: {}".format(
              current_state_parts,
              proposed_state_parts,
              proposed_grads_target_log_prob))
    proposed_momentums = [m + 0.5 * ss * g for m, ss, g
                          in zip(proposed_momentums,
                                 step_sizes,
                                 proposed_grads_target_log_prob)]
    return [
        proposed_momentums,
        proposed_state_parts,
        proposed_target_log_prob,
        proposed_grads_target_log_prob,
    ]


def _compute_energy_change(current_target_log_prob,
                           current_momentums,
                           proposed_target_log_prob,
                           proposed_momentums,
                           independent_chain_ndims,
                           name=None):
  """Helper to `kernel` which computes the energy change."""
  with ops.name_scope(
      name, "compute_energy_change",
      ([current_target_log_prob, proposed_target_log_prob,
        independent_chain_ndims] +
       current_momentums + proposed_momentums)):
    # Abbreviate lk0=log_kinetic_energy and lk1=proposed_log_kinetic_energy
    # since they're a mouthful and lets us inline more.
    lk0, lk1 = [], []
    for current_momentum, proposed_momentum in zip(current_momentums,
                                                   proposed_momentums):
      axis = math_ops.range(independent_chain_ndims,
                            array_ops.rank(current_momentum))
      lk0.append(_log_sum_sq(current_momentum, axis))
      lk1.append(_log_sum_sq(proposed_momentum, axis))

    lk0 = -np.log(2.) + math_ops.reduce_logsumexp(array_ops.stack(lk0, axis=-1),
                                                  axis=-1)
    lk1 = -np.log(2.) + math_ops.reduce_logsumexp(array_ops.stack(lk1, axis=-1),
                                                  axis=-1)
    lp0 = -current_target_log_prob   # potential
    lp1 = -proposed_target_log_prob  # proposed_potential
    x = array_ops.stack([lp1, math_ops.exp(lk1), -lp0, -math_ops.exp(lk0)],
                        axis=-1)

    # The sum is NaN if any element is NaN or we see both +Inf and -Inf.
    # Thus we will replace such rows with infinite energy change which implies
    # rejection. Recall that float-comparisons with NaN are always False.
    is_sum_determinate = (
        math_ops.reduce_all(math_ops.is_finite(x) | (x >= 0.), axis=-1) &
        math_ops.reduce_all(math_ops.is_finite(x) | (x <= 0.), axis=-1))
    is_sum_determinate = array_ops.tile(
        is_sum_determinate[..., array_ops.newaxis],
        multiples=array_ops.concat([
            array_ops.ones(array_ops.rank(is_sum_determinate),
                           dtype=dtypes.int32),
            [4],
        ], axis=0))
    x = array_ops.where(is_sum_determinate,
                        x,
                        array_ops.fill(array_ops.shape(x),
                                       value=x.dtype.as_numpy_dtype(np.inf)))

    return math_ops.reduce_sum(x, axis=-1)


def _choose(is_accepted,
            accepted,
            rejected,
            independent_chain_ndims,
            name=None):
  """Helper to `kernel` which expand_dims `is_accepted` to apply tf.where."""
  def _expand_is_accepted_like(x):
    with ops.name_scope("_choose"):
      expand_shape = array_ops.concat([
          array_ops.shape(is_accepted),
          array_ops.ones([array_ops.rank(x) - array_ops.rank(is_accepted)],
                         dtype=dtypes.int32),
      ], axis=0)
      multiples = array_ops.concat([
          array_ops.ones([array_ops.rank(is_accepted)], dtype=dtypes.int32),
          array_ops.shape(x)[independent_chain_ndims:],
      ], axis=0)
      m = array_ops.tile(array_ops.reshape(is_accepted, expand_shape),
                         multiples)
      m.set_shape(x.shape)
      return m
  with ops.name_scope(name, "_choose", values=[
      is_accepted, accepted, rejected, independent_chain_ndims]):
    return array_ops.where(_expand_is_accepted_like(accepted),
                           accepted,
                           rejected)


def _maybe_call_fn_and_grads(fn,
                             fn_arg_list,
                             fn_result=None,
                             grads_fn_result=None,
                             description="target_log_prob"):
  """Helper which computes `fn_result` and `grads` if needed."""
  fn_arg_list = (list(fn_arg_list) if _is_list_like(fn_arg_list)
                 else [fn_arg_list])
  if fn_result is None:
    fn_result = fn(*fn_arg_list)
  if not fn_result.dtype.is_floating:
    raise TypeError("`{}` must be a `Tensor` with `float` `dtype`.".format(
        description))
  if grads_fn_result is None:
    grads_fn_result = gradients_ops.gradients(
        fn_result, fn_arg_list)
  if len(fn_arg_list) != len(grads_fn_result):
    raise ValueError("`{}` must be in one-to-one correspondence with "
                     "`grads_{}`".format(*[description]*2))
  if any(g is None for g in grads_fn_result):
    raise ValueError("Encountered `None` gradient.")
  return fn_result, grads_fn_result


def _prepare_args(target_log_prob_fn, state, step_size,
                  target_log_prob=None, grads_target_log_prob=None,
                  maybe_expand=False, description="target_log_prob"):
  """Helper which processes input args to meet list-like assumptions."""
  state_parts = list(state) if _is_list_like(state) else [state]
  state_parts = [ops.convert_to_tensor(s, name="state")
                 for s in state_parts]
  target_log_prob, grads_target_log_prob = _maybe_call_fn_and_grads(
      target_log_prob_fn,
      state_parts,
      target_log_prob,
      grads_target_log_prob,
      description)
  step_sizes = list(step_size) if _is_list_like(step_size) else [step_size]
  step_sizes = [
      ops.convert_to_tensor(
          s, name="step_size", dtype=target_log_prob.dtype)
      for s in step_sizes]
  if len(step_sizes) == 1:
    step_sizes *= len(state_parts)
  if len(state_parts) != len(step_sizes):
    raise ValueError("There should be exactly one `step_size` or it should "
                     "have same length as `current_state`.")
  maybe_flatten = lambda x: x if maybe_expand or _is_list_like(state) else x[0]
  return [
      maybe_flatten(state_parts),
      maybe_flatten(step_sizes),
      target_log_prob,
      grads_target_log_prob,
  ]


def _is_list_like(x):
  """Helper which returns `True` if input is `list`-like."""
  return isinstance(x, (tuple, list))


def _log_sum_sq(x, axis=None):
  """Computes log(sum(x**2))."""
  return math_ops.reduce_logsumexp(2. * math_ops.log(math_ops.abs(x)), axis)
