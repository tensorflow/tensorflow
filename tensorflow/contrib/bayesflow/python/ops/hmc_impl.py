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

@@chain
@@update
@@leapfrog_integrator
@@leapfrog_step
@@ais_chain
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import tf_logging as logging

__all__ = [
    'chain',
    'kernel',
    'leapfrog_integrator',
    'leapfrog_step',
    'ais_chain'
]


def _make_potential_and_grad(target_log_prob_fn):
  def potential_and_grad(x):
    log_prob_result = -target_log_prob_fn(x)
    grad_result = gradients_impl.gradients(math_ops.reduce_sum(log_prob_result),
                                           x)[0]
    return log_prob_result, grad_result
  return potential_and_grad


def chain(n_iterations, step_size, n_leapfrog_steps, initial_x,
          target_log_prob_fn, event_dims=(), name=None):
  """Runs multiple iterations of one or more Hamiltonian Monte Carlo chains.

  Hamiltonian Monte Carlo (HMC) is a Markov chain Monte Carlo (MCMC)
  algorithm that takes a series of gradient-informed steps to produce
  a Metropolis proposal. This function samples from an HMC Markov
  chain whose initial state is `initial_x` and whose stationary
  distribution has log-density `target_log_prob_fn()`.

  This function can update multiple chains in parallel. It assumes
  that all dimensions of `initial_x` not specified in `event_dims` are
  independent, and should therefore be updated independently. The
  output of `target_log_prob_fn()` should sum log-probabilities across
  all event dimensions. Slices along dimensions not in `event_dims`
  may have different target distributions; this is up to
  `target_log_prob_fn()`.

  This function basically just wraps `hmc.kernel()` in a tf.scan() loop.

  Args:
    n_iterations: Integer number of Markov chain updates to run.
    step_size: Scalar step size or array of step sizes for the
      leapfrog integrator. Broadcasts to the shape of
      `initial_x`. Larger step sizes lead to faster progress, but
      too-large step sizes make rejection exponentially more likely.
      When possible, it's often helpful to match per-variable step
      sizes to the standard deviations of the target distribution in
      each variable.
    n_leapfrog_steps: Integer number of steps to run the leapfrog
      integrator for. Total progress per HMC step is roughly
      proportional to step_size * n_leapfrog_steps.
    initial_x: Tensor of initial state(s) of the Markov chain(s).
    target_log_prob_fn: Python callable which takes an argument like `initial_x`
      and returns its (possibly unnormalized) log-density under the target
      distribution.
    event_dims: List of dimensions that should not be treated as
      independent. This allows for multiple chains to be run independently
      in parallel. Default is (), i.e., all dimensions are independent.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    acceptance_probs: Tensor with the acceptance probabilities for each
      iteration. Has shape matching `target_log_prob_fn(initial_x)`.
    chain_states: Tensor with the state of the Markov chain at each iteration.
      Has shape `[n_iterations, initial_x.shape[0],...,initial_x.shape[-1]`.

  #### Examples:

  ```python
  # Sampling from a standard normal (note `log_joint()` is unnormalized):
  def log_joint(x):
    return tf.reduce_sum(-0.5 * tf.square(x))
  chain, acceptance_probs = hmc.chain(1000, 0.5, 2, tf.zeros(10), log_joint,
                                      event_dims=[0])
  # Discard first half of chain as warmup/burn-in
  warmed_up = chain[500:]
  mean_est = tf.reduce_mean(warmed_up, 0)
  var_est = tf.reduce_mean(tf.square(warmed_up), 0) - tf.square(mean_est)
  ```

  ```python
  # Sampling from a diagonal-variance Gaussian:
  variances = tf.linspace(1., 3., 10)
  def log_joint(x):
    return tf.reduce_sum(-0.5 / variances * tf.square(x))
  chain, acceptance_probs = hmc.chain(1000, 0.5, 2, tf.zeros(10), log_joint,
                                      event_dims=[0])
  # Discard first half of chain as warmup/burn-in
  warmed_up = chain[500:]
  mean_est = tf.reduce_mean(warmed_up, 0)
  var_est = tf.reduce_mean(tf.square(warmed_up), 0) - tf.square(mean_est)
  ```

  ```python
  # Sampling from factor-analysis posteriors with known factors W:
  # mu[i, j] ~ Normal(0, 1)
  # x[i] ~ Normal(matmul(mu[i], W), I)
  def log_joint(mu, x, W):
    prior = -0.5 * tf.reduce_sum(tf.square(mu), 1)
    x_mean = tf.matmul(mu, W)
    likelihood = -0.5 * tf.reduce_sum(tf.square(x - x_mean), 1)
    return prior + likelihood
  chain, acceptance_probs = hmc.chain(1000, 0.1, 2,
                                      tf.zeros([x.shape[0], W.shape[0]]),
                                      lambda mu: log_joint(mu, x, W),
                                      event_dims=[1])
  # Discard first half of chain as warmup/burn-in
  warmed_up = chain[500:]
  mean_est = tf.reduce_mean(warmed_up, 0)
  var_est = tf.reduce_mean(tf.square(warmed_up), 0) - tf.square(mean_est)
  ```

  ```python
  # Sampling from the posterior of a Bayesian regression model.:

  # Run 100 chains in parallel, each with a different initialization.
  initial_beta = tf.random_normal([100, x.shape[1]])
  chain, acceptance_probs = hmc.chain(1000, 0.1, 10, initial_beta,
                                      log_joint_partial, event_dims=[1])
  # Discard first halves of chains as warmup/burn-in
  warmed_up = chain[500:]
  # Averaging across samples within a chain and across chains
  mean_est = tf.reduce_mean(warmed_up, [0, 1])
  var_est = tf.reduce_mean(tf.square(warmed_up), [0, 1]) - tf.square(mean_est)
  ```
  """
  with ops.name_scope(name, 'hmc_chain', [n_iterations, step_size,
                                          n_leapfrog_steps, initial_x]):
    initial_x = ops.convert_to_tensor(initial_x, name='initial_x')
    non_event_shape = array_ops.shape(target_log_prob_fn(initial_x))

    def body(a, _):
      updated_x, acceptance_probs, log_prob, grad = kernel(
          step_size, n_leapfrog_steps, a[0], target_log_prob_fn, event_dims,
          a[2], a[3])
      return updated_x, acceptance_probs, log_prob, grad

    potential_and_grad = _make_potential_and_grad(target_log_prob_fn)
    potential, grad = potential_and_grad(initial_x)
    return functional_ops.scan(body, array_ops.zeros(n_iterations),
                               (initial_x, array_ops.zeros(non_event_shape),
                                -potential, -grad))[:2]


def ais_chain(n_iterations, step_size, n_leapfrog_steps, initial_x,
              target_log_prob_fn, proposal_log_prob_fn, event_dims=(),
              name=None):
  """Runs annealed importance sampling (AIS) to estimate normalizing constants.

  This routine uses Hamiltonian Monte Carlo to sample from a series of
  distributions that slowly interpolates between an initial "proposal"
  distribution

  `exp(proposal_log_prob_fn(x) - proposal_log_normalizer)`

  and the target distribution

  `exp(target_log_prob_fn(x) - target_log_normalizer)`,

  accumulating importance weights along the way. The product of these
  importance weights gives an unbiased estimate of the ratio of the
  normalizing constants of the initial distribution and the target
  distribution:

  E[exp(w)] = exp(target_log_normalizer - proposal_log_normalizer).

  Args:
    n_iterations: Integer number of Markov chain updates to run. More
      iterations means more expense, but smoother annealing between q
      and p, which in turn means exponentially lower variance for the
      normalizing constant estimator.
    step_size: Scalar step size or array of step sizes for the
      leapfrog integrator. Broadcasts to the shape of
      `initial_x`. Larger step sizes lead to faster progress, but
      too-large step sizes make rejection exponentially more likely.
      When possible, it's often helpful to match per-variable step
      sizes to the standard deviations of the target distribution in
      each variable.
    n_leapfrog_steps: Integer number of steps to run the leapfrog
      integrator for. Total progress per HMC step is roughly
      proportional to step_size * n_leapfrog_steps.
    initial_x: Tensor of initial state(s) of the Markov chain(s). Must
      be a sample from q, or results will be incorrect.
    target_log_prob_fn: Python callable which takes an argument like `initial_x`
      and returns its (possibly unnormalized) log-density under the target
      distribution.
    proposal_log_prob_fn: Python callable that returns the log density of the
      initial distribution.
    event_dims: List of dimensions that should not be treated as
      independent. This allows for multiple chains to be run independently
      in parallel. Default is (), i.e., all dimensions are independent.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    ais_weights: Tensor with the estimated weight(s). Has shape matching
      `target_log_prob_fn(initial_x)`.
    chain_states: Tensor with the state(s) of the Markov chain(s) the final
      iteration. Has shape matching `initial_x`.
    acceptance_probs: Tensor with the acceptance probabilities for the final
      iteration. Has shape matching `target_log_prob_fn(initial_x)`.

  #### Examples:

  ```python
  # Estimating the normalizing constant of a log-gamma distribution:
  def proposal_log_prob(x):
    # Standard normal log-probability. This is properly normalized.
    return tf.reduce_sum(-0.5 * tf.square(x) - 0.5 * np.log(2 * np.pi), 1)
  def target_log_prob(x):
    # Unnormalized log-gamma(2, 3) distribution.
    # True normalizer is (lgamma(2) - 2 * log(3)) * x.shape[1]
    return tf.reduce_sum(2. * x - 3. * tf.exp(x), 1)
  # Run 100 AIS chains in parallel
  initial_x = tf.random_normal([100, 20])
  w, _, _ = hmc.ais_chain(1000, 0.2, 2, initial_x, target_log_prob,
                          proposal_log_prob, event_dims=[1])
  log_normalizer_estimate = tf.reduce_logsumexp(w) - np.log(100)
  ```

  ```python
  # Estimating the marginal likelihood of a Bayesian regression model:
  base_measure = -0.5 * np.log(2 * np.pi)
  def proposal_log_prob(x):
    # Standard normal log-probability. This is properly normalized.
    return tf.reduce_sum(-0.5 * tf.square(x) + base_measure, 1)
  def regression_log_joint(beta, x, y):
    # This function returns a vector whose ith element is log p(beta[i], y | x).
    # Each row of beta corresponds to the state of an independent Markov chain.
    log_prior = tf.reduce_sum(-0.5 * tf.square(beta) + base_measure, 1)
    means = tf.matmul(beta, x, transpose_b=True)
    log_likelihood = tf.reduce_sum(-0.5 * tf.square(y - means) +
                                   base_measure, 1)
    return log_prior + log_likelihood
  def log_joint_partial(beta):
    return regression_log_joint(beta, x, y)
  # Run 100 AIS chains in parallel
  initial_beta = tf.random_normal([100, x.shape[1]])
  w, beta_samples, _ = hmc.ais_chain(1000, 0.1, 2, initial_beta,
                                     log_joint_partial, proposal_log_prob,
                                     event_dims=[1])
  log_normalizer_estimate = tf.reduce_logsumexp(w) - np.log(100)
  ```
  """
  with ops.name_scope(name, 'hmc_ais_chain',
                      [n_iterations, step_size, n_leapfrog_steps, initial_x]):
    non_event_shape = array_ops.shape(target_log_prob_fn(initial_x))

    beta_series = math_ops.linspace(0., 1., n_iterations+1)[1:]
    def _body(a, beta):  # pylint: disable=missing-docstring
      def log_prob_beta(x):
        return ((1 - beta) * proposal_log_prob_fn(x) +
                beta * target_log_prob_fn(x))
      last_x = a[0]
      w = a[2]
      w += (1. / n_iterations) * (target_log_prob_fn(last_x) -
                                  proposal_log_prob_fn(last_x))
      # TODO(b/66917083): There's an opportunity for gradient reuse here.
      updated_x, acceptance_probs, _, _ = kernel(step_size, n_leapfrog_steps,
                                                 last_x, log_prob_beta,
                                                 event_dims)
      return updated_x, acceptance_probs, w

    x, acceptance_probs, w = functional_ops.scan(
        _body, beta_series, (initial_x, array_ops.zeros(non_event_shape),
                             array_ops.zeros(non_event_shape)))
  return w[-1], x[-1], acceptance_probs[-1]


def kernel(step_size, n_leapfrog_steps, x, target_log_prob_fn, event_dims=(),
           x_log_prob=None, x_grad=None, name=None):
  """Runs one iteration of Hamiltonian Monte Carlo.

  Hamiltonian Monte Carlo (HMC) is a Markov chain Monte Carlo (MCMC)
  algorithm that takes a series of gradient-informed steps to produce
  a Metropolis proposal. This function applies one step of HMC to
  randomly update the variable `x`.

  This function can update multiple chains in parallel. It assumes
  that all dimensions of `x` not specified in `event_dims` are
  independent, and should therefore be updated independently. The
  output of `target_log_prob_fn()` should sum log-probabilities across
  all event dimensions. Slices along dimensions not in `event_dims`
  may have different target distributions; for example, if
  `event_dims == (1,)`, then `x[0, :]` could have a different target
  distribution from x[1, :]. This is up to `target_log_prob_fn()`.

  Args:
    step_size: Scalar step size or array of step sizes for the
      leapfrog integrator. Broadcasts to the shape of
      `x`. Larger step sizes lead to faster progress, but
      too-large step sizes make rejection exponentially more likely.
      When possible, it's often helpful to match per-variable step
      sizes to the standard deviations of the target distribution in
      each variable.
    n_leapfrog_steps: Integer number of steps to run the leapfrog
      integrator for. Total progress per HMC step is roughly
      proportional to step_size * n_leapfrog_steps.
    x: Tensor containing the value(s) of the random variable(s) to update.
    target_log_prob_fn: Python callable which takes an argument like `initial_x`
      and returns its (possibly unnormalized) log-density under the target
      distribution.
    event_dims: List of dimensions that should not be treated as
      independent. This allows for multiple chains to be run independently
      in parallel. Default is (), i.e., all dimensions are independent.
    x_log_prob (optional): Tensor containing the cached output of a previous
      call to `target_log_prob_fn()` evaluated at `x` (such as that provided by
      a previous call to `kernel()`). Providing `x_log_prob` and
      `x_grad` saves one gradient computation per call to `kernel()`.
    x_grad (optional): Tensor containing the cached gradient of
      `target_log_prob_fn()` evaluated at `x` (such as that provided by
      a previous call to `kernel()`). Providing `x_log_prob` and
      `x_grad` saves one gradient computation per call to `kernel()`.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    updated_x: The updated variable(s) x. Has shape matching `initial_x`.
    acceptance_probs: Tensor with the acceptance probabilities for the final
      iteration. This is useful for diagnosing step size problems etc. Has
      shape matching `target_log_prob_fn(initial_x)`.
    new_log_prob: The value of `target_log_prob_fn()` evaluated at `updated_x`.
    new_grad: The value of the gradient of `target_log_prob_fn()` evaluated at
      `updated_x`.

  #### Examples:

  ```python
  # Tuning acceptance rates:
  target_accept_rate = 0.631
  def target_log_prob(x):
    # Standard normal
    return tf.reduce_sum(-0.5 * tf.square(x))
  initial_x = tf.zeros([10])
  initial_log_prob = target_log_prob(initial_x)
  initial_grad = tf.gradients(initial_log_prob, initial_x)[0]
  # Algorithm state
  x = tf.Variable(initial_x, name='x')
  step_size = tf.Variable(1., name='step_size')
  last_log_prob = tf.Variable(initial_log_prob, name='last_log_prob')
  last_grad = tf.Variable(initial_grad, name='last_grad')
  # Compute updates
  new_x, acceptance_prob, log_prob, grad = hmc.kernel(step_size, 3, x,
                                                      target_log_prob,
                                                      event_dims=[0],
                                                      x_log_prob=last_log_prob)
  x_update = tf.assign(x, new_x)
  log_prob_update = tf.assign(last_log_prob, log_prob)
  grad_update = tf.assign(last_grad, grad)
  step_size_update = tf.assign(step_size,
                               tf.where(acceptance_prob > target_accept_rate,
                                        step_size * 1.01, step_size / 1.01))
  adaptive_updates = [x_update, log_prob_update, grad_update, step_size_update]
  sampling_updates = [x_update, log_prob_update, grad_update]

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  # Warm up the sampler and adapt the step size
  for i in xrange(500):
    sess.run(adaptive_updates)
  # Collect samples without adapting step size
  samples = np.zeros([500, 10])
  for i in xrange(500):
    x_val, _ = sess.run([new_x, sampling_updates])
    samples[i] = x_val
  ```

  ```python
  # Empirical-Bayes estimation of a hyperparameter by MCMC-EM:

  # Problem setup
  N = 150
  D = 10
  x = np.random.randn(N, D).astype(np.float32)
  true_sigma = 0.5
  true_beta = true_sigma * np.random.randn(D).astype(np.float32)
  y = x.dot(true_beta) + np.random.randn(N).astype(np.float32)

  def log_prior(beta, log_sigma):
    return tf.reduce_sum(-0.5 / tf.exp(2 * log_sigma) * tf.square(beta) -
                         log_sigma)
  def regression_log_joint(beta, log_sigma, x, y):
    # This function returns log p(beta | log_sigma) + log p(y | x, beta).
    means = tf.matmul(tf.expand_dims(beta, 0), x, transpose_b=True)
    means = tf.squeeze(means)
    log_likelihood = tf.reduce_sum(-0.5 * tf.square(y - means))
    return log_prior(beta, log_sigma) + log_likelihood
  def log_joint_partial(beta):
    return regression_log_joint(beta, log_sigma, x, y)
  # Our estimate of log(sigma)
  log_sigma = tf.Variable(0., name='log_sigma')
  # The state of the Markov chain
  beta = tf.Variable(tf.random_normal([x.shape[1]]), name='beta')
  new_beta, _, _, _ = hmc.kernel(0.1, 5, beta, log_joint_partial,
                                 event_dims=[0])
  beta_update = tf.assign(beta, new_beta)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  with tf.control_dependencies([beta_update]):
    log_sigma_update = optimizer.minimize(-log_prior(beta, log_sigma),
                                          var_list=[log_sigma])

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  log_sigma_history = np.zeros(1000)
  for i in xrange(1000):
    log_sigma_val, _ = sess.run([log_sigma, log_sigma_update])
    log_sigma_history[i] = log_sigma_val
  # Should converge to something close to true_sigma
  plt.plot(np.exp(log_sigma_history))
  ```
  """
  with ops.name_scope(name, 'hmc_kernel', [step_size, n_leapfrog_steps, x]):
    potential_and_grad = _make_potential_and_grad(target_log_prob_fn)

    x_shape = array_ops.shape(x)
    m = random_ops.random_normal(x_shape)

    kinetic_0 = 0.5 * math_ops.reduce_sum(math_ops.square(m), event_dims)

    if (x_log_prob is not None) and (x_grad is not None):
      log_potential_0, grad_0 = -x_log_prob, -x_grad  # pylint: disable=invalid-unary-operand-type
    else:
      if x_log_prob is not None:
        logging.warn('x_log_prob was provided, but x_grad was not,'
                     ' so x_log_prob was not used.')
      if x_grad is not None:
        logging.warn('x_grad was provided, but x_log_prob was not,'
                     ' so x_grad was not used.')
      log_potential_0, grad_0 = potential_and_grad(x)

    new_x, new_m, log_potential_1, grad_1 = leapfrog_integrator(
        step_size, n_leapfrog_steps, x, m, potential_and_grad, grad_0)

    kinetic_1 = 0.5 * math_ops.reduce_sum(math_ops.square(new_m), event_dims)

    # TODO(mhoffman): It seems like there may be an opportunity for nans here.
    # I'm delaying addressing this because we're going to refactor this part
    # to use the more general Metropolis abstraction anyway.
    acceptance_probs = math_ops.exp(math_ops.minimum(0., log_potential_0 -
                                                     log_potential_1 +
                                                     kinetic_0 - kinetic_1))
    accepted = math_ops.cast(
        random_ops.random_uniform(array_ops.shape(acceptance_probs)) <
        acceptance_probs, np.float32)
    new_log_prob = (-log_potential_0 * (1. - accepted) -
                    log_potential_1 * accepted)

    # TODO(b/65738010): This should work, but it doesn't for now.
    # reduced_shape = math_ops.reduced_shape(x_shape, event_dims)
    reduced_shape = array_ops.shape(math_ops.reduce_sum(x, event_dims,
                                                        keep_dims=True))
    accepted = array_ops.reshape(accepted, reduced_shape)
    new_x = x * (1. - accepted) + new_x * accepted
    new_grad = -grad_0 * (1. - accepted) - grad_1 * accepted

  return new_x, acceptance_probs, new_log_prob, new_grad


def leapfrog_integrator(step_size, n_steps, initial_position, initial_momentum,
                        potential_and_grad, initial_grad, name=None):
  """Applies `n_steps` steps of the leapfrog integrator.

  This just wraps `leapfrog_step()` in a `tf.while_loop()`, reusing
  gradient computations where possible.

  Args:
    step_size: Scalar step size or array of step sizes for the
      leapfrog integrator. Broadcasts to the shape of
      `initial_position`. Larger step sizes lead to faster progress, but
      too-large step sizes lead to larger discretization error and
      worse energy conservation.
    n_steps: Number of steps to run the leapfrog integrator.
    initial_position: Tensor containing the value(s) of the position variable(s)
      to update.
    initial_momentum: Tensor containing the value(s) of the momentum variable(s)
      to update.
    potential_and_grad: Python callable that takes a position tensor like
      `initial_position` and returns the potential energy and its gradient at
      that position.
    initial_grad: Tensor with the value of the gradient of the potential energy
      at `initial_position`.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    updated_position: Updated value of the position.
    updated_momentum: Updated value of the momentum.
    new_potential: Potential energy of the new position. Has shape matching
      `potential_and_grad(initial_position)`.
    new_grad: Gradient from potential_and_grad() evaluated at the new position.
      Has shape matching `initial_position`.

  Example: Simple quadratic potential.
  ```python
  def potential_and_grad(position):
    return tf.reduce_sum(0.5 * tf.square(position)), position
  position = tf.placeholder(np.float32)
  momentum = tf.placeholder(np.float32)
  potential, grad = potential_and_grad(position)
  new_position, new_momentum, new_potential, new_grad = hmc.leapfrog_integrator(
    0.1, 3, position, momentum, potential_and_grad, grad)

  sess = tf.Session()
  position_val = np.random.randn(10)
  momentum_val = np.random.randn(10)
  potential_val, grad_val = sess.run([potential, grad],
                                     {position: position_val})
  positions = np.zeros([100, 10])
  for i in xrange(100):
    position_val, momentum_val, potential_val, grad_val = sess.run(
      [new_position, new_momentum, new_potential, new_grad],
      {position: position_val, momentum: momentum_val})
    positions[i] = position_val
  # Should trace out sinusoidal dynamics.
  plt.plot(positions[:, 0])
  ```
  """
  def leapfrog_wrapper(step_size, x, m, grad, l):
    x, m, _, grad = leapfrog_step(step_size, x, m, potential_and_grad, grad)
    return step_size, x, m, grad, l + 1

  def counter_fn(a, b, c, d, counter):  # pylint: disable=unused-argument
    return counter < n_steps

  with ops.name_scope(name, 'leapfrog_integrator',
                      [step_size, n_steps, initial_position, initial_momentum,
                       initial_grad]):
    _, new_x, new_m, new_grad, _ = control_flow_ops.while_loop(
        counter_fn, leapfrog_wrapper, [step_size, initial_position,
                                       initial_momentum, initial_grad,
                                       array_ops.constant(0)], back_prop=False)
    # We're counting on the runtime to eliminate this redundant computation.
    new_potential, new_grad = potential_and_grad(new_x)
  return new_x, new_m, new_potential, new_grad


def leapfrog_step(step_size, position, momentum, potential_and_grad, grad,
                  name=None):
  """Applies one step of the leapfrog integrator.

  Assumes a simple quadratic kinetic energy function: 0.5 * ||momentum||^2.

  Args:
    step_size: Scalar step size or array of step sizes for the
      leapfrog integrator. Broadcasts to the shape of
      `position`. Larger step sizes lead to faster progress, but
      too-large step sizes lead to larger discretization error and
      worse energy conservation.
    position: Tensor containing the value(s) of the position variable(s)
      to update.
    momentum: Tensor containing the value(s) of the momentum variable(s)
      to update.
    potential_and_grad: Python callable that takes a position tensor like
      `position` and returns the potential energy and its gradient at that
      position.
    grad: Tensor with the value of the gradient of the potential energy
      at `position`.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    updated_position: Updated value of the position.
    updated_momentum: Updated value of the momentum.
    new_potential: Potential energy of the new position. Has shape matching
      `potential_and_grad(position)`.
    new_grad: Gradient from potential_and_grad() evaluated at the new position.
      Has shape matching `position`.

  Example: Simple quadratic potential.
  ```python
  def potential_and_grad(position):
    # Simple quadratic potential
    return tf.reduce_sum(0.5 * tf.square(position)), position
  position = tf.placeholder(np.float32)
  momentum = tf.placeholder(np.float32)
  potential, grad = potential_and_grad(position)
  new_position, new_momentum, new_potential, new_grad = hmc.leapfrog_step(
    0.1, position, momentum, potential_and_grad, grad)

  sess = tf.Session()
  position_val = np.random.randn(10)
  momentum_val = np.random.randn(10)
  potential_val, grad_val = sess.run([potential, grad],
                                     {position: position_val})
  positions = np.zeros([100, 10])
  for i in xrange(100):
    position_val, momentum_val, potential_val, grad_val = sess.run(
      [new_position, new_momentum, new_potential, new_grad],
      {position: position_val, momentum: momentum_val})
    positions[i] = position_val
  # Should trace out sinusoidal dynamics.
  plt.plot(positions[:, 0])
  ```
  """
  with ops.name_scope(name, 'leapfrog_step', [step_size, position, momentum,
                                              grad]):
    momentum -= 0.5 * step_size * grad
    position += step_size * momentum
    potential, grad = potential_and_grad(position)
    momentum -= 0.5 * step_size * grad

  return position, momentum, potential, grad
