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
"""Functions to create a Markov Chain Monte Carlo Metropolis step.

@@evolve
@@uniform_random_proposal
@@normal_random_proposal
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops

__all__ = [
    'evolve',
    'uniform_random_proposal',
    'normal_random_proposal',
]


def _single_iteration(current_state, current_log_density,
                      log_unnormalized_prob_fn, proposal_fn, seed=None,
                      name='None'):
  """Performs a single Metropolis-Hastings step.

  Args:
    current_state: Float-like `Tensor` (i.e., `dtype` is either
      `tf.float16`, `tf.float32` or `tf.float64`) of any shape that can
      be consumed by the `log_unnormalized_prob_fn` and `proposal_fn`
      callables.
    current_log_density: Float-like `Tensor` with `dtype` and shape equivalent
      to `log_unnormalized_prob_fn(current_state)`, i.e., matching the result of
      `log_unnormalized_prob_fn` invoked at `current_state`.
    log_unnormalized_prob_fn: A Python callable evaluated at
      `current_state` and returning a float-like `Tensor` of log target-density
      up to a normalizing constant. In other words,
      `log_unnormalized_prob_fn(x) = log(g(x))`, where
      `target_density = g(x)/Z` for some constant `A`. The shape of the input
      tensor is the same as the shape of the `current_state`. The shape of the
      output tensor is either
        (a). Same as the input shape if the density being sampled is one
          dimensional, or
        (b). If the density is defined for `events` of shape
          `event_shape = [E1, E2, ... Ee]`, then the input tensor should be of
          shape `batch_shape + event_shape`, where `batch_shape = [B1, ..., Bb]`
          and the result must be of shape [B1, ..., Bb]. For example, if the
          distribution that is being sampled is a 10 dimensional normal,
          then the input tensor may be of shape [100, 10] or [30, 20, 10]. The
          last dimension will then be 'consumed' by `log_unnormalized_prob_fn`
          and it should return tensors of shape [100] and [30, 20] respectively.
    proposal_fn: A callable accepting a real valued `Tensor` of current sample
      points and returning a tuple of two `Tensors`. The first element of the
      pair is a `Tensor` containing the proposal state and should have
      the same shape as the input `Tensor`. The second element of the pair gives
      the log of the ratio of the probability of transitioning from the
      proposal points to the input points and the probability of transitioning
      from the input points to the proposal points. If the proposal is
      symmetric (e.g., random walk, where the proposal is either
      normal or uniform centered at `current_state`), i.e.,
      Probability(Proposal -> Current) = Probability(Current -> Proposal)
      the second value should be set to `None` instead of explicitly supplying a
      tensor of zeros. In addition to being convenient, this also leads to a
      more efficient graph.
    seed: `int` or None. The random seed for this `Op`. If `None`, no seed is
      applied.
    name: Python `str` name prefix for ops managed by this function.

  Returns:
    next_state: `Tensor` with `dtype` and shape matching `current_state`.
      Created by propagating the chain by one step, starting from
      `current_state`.
    next_log_density: `Tensor` with `dtype` and shape matching
      `current_log_density`, which is equal to the value of the unnormalized
      `log_unnormalized_prob_fn` computed at `next_state`.
    log_accept_ratio: `Tensor` with `dtype` and shape matching
      `current_log_density`. Stands for the log of Metropolis-Hastings
      acceptance ratio used in generating the `next_state`.
  """

  with ops.name_scope(name, 'single_iteration', [current_state]):
    # The proposed state and the log of the corresponding Hastings ratio.
    proposal_state, log_transit_ratio = proposal_fn(current_state)

    # If the log ratio is None, assume that the transitions are symmetric,
    # i.e., Prob(Current -> Proposed) = Prob(Proposed -> Current).
    if log_transit_ratio is None:
      log_transit_ratio = 0.

    # Log-density of the proposal state.
    proposal_log_density = log_unnormalized_prob_fn(proposal_state)

    # Ops to compute the log of the acceptance ratio. Recall that the
    # acceptance ratio is: [Prob(Proposed) / Prob(Current)] *
    # [Prob(Proposed -> Current) / Prob(Current -> Proposed)]. The log of the
    # second term is the log_transit_ratio.
    with ops.name_scope('accept_reject'):
      # The log of the acceptance ratio.
      log_accept_ratio = (proposal_log_density - current_log_density
                          + log_transit_ratio)

      # A proposal is accepted or rejected depending on the acceptance ratio.
      # If the acceptance ratio is greater than 1 then it is always accepted.
      # If the acceptance ratio is less than 1 then the proposal is accepted
      # with probability = acceptance ratio. As we are working in log space to
      # prevent over/underflows, this logic is expressed in log terms below.
      # If a proposal is accepted we place a True in the acceptance state
      # tensor and if it is to be rejected we place a False.
      # The log_draws below have to be compared to the log_accept_ratio so we
      # make sure that they have the same data type.
      log_draws = math_ops.log(random_ops.random_uniform(
          array_ops.shape(current_log_density), seed=seed,
          dtype=log_accept_ratio.dtype))
      is_proposal_accepted = log_draws < log_accept_ratio

    # The acceptance state decides which elements of the current state are to
    # be replaced with the corresponding elements in the proposal state.
    with ops.name_scope(name, 'metropolis_single_step',
                        [current_state, current_log_density]):
      next_log_density = array_ops.where(is_proposal_accepted,
                                         proposal_log_density,
                                         current_log_density)
      next_state = array_ops.where(is_proposal_accepted, proposal_state,
                                   current_state)

    return next_state, next_log_density, log_accept_ratio


def evolve(initial_sample,
           initial_log_density,
           initial_log_accept_ratio,
           log_unnormalized_prob_fn,
           proposal_fn,
           n_steps=1,
           seed=None,
           name=None):
  """Performs `n_steps` of the Metropolis-Hastings update.

  Given a probability density function, `f(x)` and a proposal scheme which
  generates new points from old, this `Op` returns a tensor
  which may be used to generate approximate samples from the target distribution
  using the Metropolis-Hastings algorithm. These samples are from a Markov chain
  whose equilibrium distribution matches the target distribution.

  The probability distribution may have an unknown normalization constan.
  We parameterize the probability density as follows:
    ```
      f(x) = exp(L(x) + constant)
    ```
  Here `L(x)` is any continuous function with an (possibly unknown but finite)
  upper bound, i.e. there exists a number beta such that
  `L(x)< beta < infinity` for all x. The constant is the normalization needed
  to make `f(x)` a probability density (as opposed to just a finite measure).

  Although `initial_sample` can be arbitrary, a poor choice may result in a
  slow-to-mix chain. In many cases the best choice is the one that maximizes
  the target density, i.e., choose `initial_sample` such that
  `f(initial_sample) >= f(x)` for all `x`.


  If the support of the distribution is a strict subset of R^n (but of non zero
  measure), then the unnormalized log-density `L(x)` should return `-infinity`
  outside the support domain. This effectively forces the sampler to only
  explore points in the regions of finite support.

  Usage:
  This function is meant to be wrapped up with some of the common proposal
  schemes (e.g. random walk, Langevin diffusion etc) to produce a more user
  friendly interface. However, it may also be used to create bespoke samplers.

  The following example, demonstrates the use to generate a 1000 uniform random
  walk Metropolis samplers run in parallel for the normal target distribution.
  ```python
    n = 3  # dimension of the problem

    # Generate 1000 initial values randomly. Each of these would be an
    # independent starting point for a Markov chain.
    state = tf.get_variable(
        'state',initializer=tf.random_normal([1000, n], mean=3.0,
                                             dtype=tf.float64, seed=42))

    # Computes the log(p(x)) for the unit normal density and ignores the
    # normalization constant.
    def log_density(x):
      return  - tf.reduce_sum(x * x, reduction_indices=-1) / 2.0

    # Initial log-density value
    state_log_density = tf.get_variable(
        'state_log_density', initializer=log_density(state.initialized_value()))

    # A variable to store the log_acceptance_ratio:
    log_acceptance_ratio = tf.get_variable(
        'log_acceptance_ratio', initializer=tf.zeros([1000], dtype=tf.float64))

    # Generates random proposals by moving each coordinate uniformly and
    # independently in a box of size 2 centered around the current value.
    # Returns the new point and also the log of the Hastings ratio (the
    # ratio of the probability of going from the proposal to origin and the
    # probability of the reverse transition). When this ratio is 1, the value
    # may be omitted and replaced by None.
    def random_proposal(x):
      return (x + tf.random_uniform(tf.shape(x), minval=-1, maxval=1,
                                    dtype=x.dtype, seed=12)), None

    #  Create the op to propagate the chain for 100 steps.
    stepper = mh.evolve(
        state, state_log_density, log_acceptance_ratio,
        log_density, random_proposal, n_steps=100, seed=123)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
      sess.run(init)
      # Run the chains for a total of 1000 steps and print out the mean across
      # the chains every 100 iterations.
      for n_iter in range(10):
        # Executing the stepper advances the chain to the next state.
        sess.run(stepper)
        # Print out the current value of the mean(sample) for every dimension.
        print(np.mean(sess.run(state), 0))
      # Estimated covariance matrix
      samples = sess.run(state)
      print('')
      print(np.cov(samples, rowvar=False))
  ```

  Args:
    initial_sample: A float-like `tf.Variable` of any shape that can
      be consumed by the `log_unnormalized_prob_fn` and `proposal_fn`
      callables.
    initial_log_density: Float-like `tf.Variable` with `dtype` and shape
      equivalent  to `log_unnormalized_prob_fn(initial_sample)`, i.e., matching
        the result of `log_unnormalized_prob_fn` invoked at `current_state`.
    initial_log_accept_ratio: A `tf.Variable` with `dtype` and shape matching
      `initial_log_density`. Stands for the log of Metropolis-Hastings
      acceptance ratio after propagating the chain for `n_steps`.
    log_unnormalized_prob_fn: A Python callable evaluated at
      `current_state` and returning a float-like `Tensor` of log target-density
      up to a normalizing constant. In other words,
      `log_unnormalized_prob_fn(x) = log(g(x))`, where
      `target_density = g(x)/Z` for some constant `A`. The shape of the input
      tensor is the same as the shape of the `current_state`. The shape of the
      output tensor is either
        (a). Same as the input shape if the density being sampled is one
          dimensional, or
        (b). If the density is defined for `events` of shape
          `event_shape = [E1, E2, ... Ee]`, then the input tensor should be of
          shape `batch_shape + event_shape`, here `batch_shape = [B1, ..., Bb]`
          and the result must be of shape [B1, ..., Bb]. For example, if the
          distribution that is being sampled is a 10 dimensional normal,
          then the input tensor may be of shape [100, 10] or [30, 20, 10]. The
          last dimension will then be 'consumed' by `log_unnormalized_prob_fn`
          and it should return tensors of shape [100] and [30, 20] respectively.
    proposal_fn: A callable accepting a real valued `Tensor` of current sample
      points and returning a tuple of two `Tensors`. The first element of the
      pair should be a `Tensor` containing the proposal state and should have
      the same shape as the input `Tensor`. The second element of the pair gives
      the log of the ratio of the probability of transitioning from the
      proposal points to the input points and the probability of transitioning
      from the input points to the proposal points. If the proposal is
      symmetric, i.e.
      Probability(Proposal -> Current) = Probability(Current -> Proposal)
      the second value should be set to None instead of explicitly supplying a
      tensor of zeros. In addition to being convenient, this also leads to a
      more efficient graph.
    n_steps: A positive `int` or a scalar `int32` tensor. Sets the number of
      iterations of the chain.
    seed: `int` or None. The random seed for this `Op`. If `None`, no seed is
      applied.
    name: A string that sets the name for this `Op`.

  Returns:
    forward_step: an `Op` to step the Markov chain forward for `n_steps`.
  """

  with ops.name_scope(name, 'metropolis_hastings', [initial_sample]):
    current_state = initial_sample
    current_log_density = initial_log_density
    log_accept_ratio = initial_log_accept_ratio

    # Stop condition for the while_loop
    def stop_condition(i, _):
      return i < n_steps

    def step(i, loop_vars):
      """Wrap `_single_iteration` for `while_loop`."""
      state = loop_vars[0]
      state_log_density = loop_vars[1]
      return i + 1, list(_single_iteration(state, state_log_density,
                                           log_unnormalized_prob_fn,
                                           proposal_fn, seed=seed))

    loop_vars = [current_state, current_log_density, log_accept_ratio]
    # Build an `Op` to evolve the Markov chain for `n_steps`
    (_, [end_state, end_log_density, end_log_acceptance]) = (
        control_flow_ops.while_loop(
            stop_condition, step,
            (0, loop_vars),
            parallel_iterations=1, swap_memory=1))

    forward_step = control_flow_ops.group(
        state_ops.assign(current_log_density, end_log_density),
        state_ops.assign(current_state, end_state),
        state_ops.assign(log_accept_ratio, end_log_acceptance))

    return forward_step


def uniform_random_proposal(step_size=1.,
                            seed=None,
                            name=None):
  """Returns a callable that adds a random uniform tensor to the input.

  This function returns a callable that accepts one `Tensor` argument of any
  shape and a real data type (i.e. `tf.float32` or `tf.float64`). It adds a
  sample from a random uniform distribution drawn from [-stepsize, stepsize]
  to its input. It also returns the log of the ratio of the probability of
  moving from the input point to the proposed point, but since this log ratio is
  identically equal to 0 (because the probability of drawing a value `x` from
  the symmetric uniform distribution is the same as the probability of drawing
  `-x`), it simply returns None for the second element of the returned tuple.

  Args:
    step_size: A positive `float` or a scalar tensor of real dtype
      controlling the scale of the uniform distribution.
      If step_size = a, then draws are made uniformly from [-a, a].
    seed: `int` or None. The random seed for this `Op`. If `None`, no seed is
      applied.
    name: A string that sets the name for this `Op`.

  Returns:
    proposal_fn:  A callable accepting one float-like `Tensor` and returning a
    2-tuple. The first value in the tuple is a `Tensor` of the same shape and
    dtype as the input argument and the second element of the tuple is None.
  """

  with ops.name_scope(name, 'uniform_random_proposal', [step_size]):
    def proposal_fn(input_state, name=None):
      """Adds a uniform perturbation to the input state.

      Args:
        input_state: A `Tensor` of any shape and real dtype.
        name: A string that sets the name for this `Op`.

      Returns:
        proposal_state:  A float-like `Tensot` with `dtype` and shape matching
          `input_state`.
        log_transit_ratio: `None`. Proposal is symmetric.
      """
      with ops.name_scope(name, 'proposer', [input_state]):
        input_state = ops.convert_to_tensor(input_state, name='input_state')
        return input_state + random_ops.random_uniform(
            array_ops.shape(input_state),
            minval=-step_size,
            maxval=step_size,
            seed=seed), None
    return proposal_fn


def normal_random_proposal(scale=1.,
                           seed=None,
                           name=None):
  """Returns a callable that adds a random normal tensor to the input.

  This function returns a callable that accepts one `Tensor` argument of any
  shape and a real data type (i.e. `tf.float32` or `tf.float64`). The callable
  adds a sample from a normal distribution with the supplied standard deviation
  and zero mean to its input argument (called the proposal point).
  The callable returns a tuple with the proposal point as the first element.
  The second element is identically `None`. It is included so the callable is
  compatible with the expected signature of the proposal scheme argument in the
  `metropolis_hastings` function. A value of `None` indicates that the
  probability of going from the input point to the proposal point is equal to
  the probability of going from the proposal point to the input point.

  Args:
    scale: A positive `float` or a scalar tensor of any real dtype controlling
      the scale of the normal distribution.
    seed: `int` or None. The random seed for this `Op`. If `None`, no seed is
      applied.
    name: A string that sets the name for this `Op`.

  Returns:
    proposal_fn: A callable accepting one float-like `Tensor` and returning a
    2-tuple. The first value in the tuple is a `Tensor` of the same shape and
    dtype as the input argument and the second element of the tuple is None.
  """

  with ops.name_scope(name, 'normal_random_proposal', [scale]):
    def proposal_fn(input_state, name=None):
      """Adds a normal perturbation to the input state.

      Args:
        input_state: A `Tensor` of any shape and real dtype.
        name: A string that sets the name for this `Op`.

      Returns:
        proposal_state:  A float-like `Tensot` with `dtype` and shape matching
          `input_state`.
        log_transit_ratio: `None`. Proposal is symmetric.
      """

      with ops.name_scope(name, 'proposer', [input_state]):
        input_state = ops.convert_to_tensor(input_state, name='input_state')
        return input_state + random_ops.random_normal(
            array_ops.shape(input_state),
            mean=0.,
            stddev=scale,
            seed=seed), None
    return proposal_fn
