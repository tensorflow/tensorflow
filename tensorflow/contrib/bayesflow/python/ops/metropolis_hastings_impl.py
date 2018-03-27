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
"""Metropolis-Hastings and proposal distributions.

@@kernel
@@evolve
@@proposal_uniform
@@proposal_normal
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops

__all__ = [
    "kernel",
    "evolve",
    "proposal_uniform",
    "proposal_normal",
]


KernelResults = collections.namedtuple(
    "KernelResults",
    [
        "log_accept_ratio",
        "current_target_log_prob",  # "Current result" means "accepted".
        "is_accepted",
        "proposed_state",
    ])


def kernel(target_log_prob_fn,
           proposal_fn,
           current_state,
           seed=None,
           current_target_log_prob=None,
           name=None):
  """Runs the Metropolis-Hastings transition kernel.

  This function can update multiple chains in parallel. It assumes that all
  leftmost dimensions of `current_state` index independent chain states (and are
  therefore updated independently). The output of `target_log_prob_fn()` should
  sum log-probabilities across all event dimensions. Slices along the rightmost
  dimensions may have different target distributions; for example,
  `current_state[0, :]` could have a different target distribution from
  `current_state[1, :]`. This is up to `target_log_prob_fn()`. (The number of
  independent chains is `tf.size(target_log_prob_fn(*current_state))`.)

  Args:
    target_log_prob_fn: Python callable which takes an argument like
      `current_state` (or `*current_state` if it's a list) and returns its
      (possibly unnormalized) log-density under the target distribution.
    proposal_fn: Python callable which takes an argument like `current_state`
      (or `*current_state` if it's a list) and returns a tuple of proposed
      states of same shape as `state`, and a log ratio `Tensor` of same shape
      as `current_target_log_prob`. The log ratio is the log-probability of
      `state` given proposed states minus the log-probability of proposed
      states given `state`. If the proposal is symmetric, set the second value
      to `None`: this enables more efficient computation than explicitly
      supplying a tensor of zeros.
    current_state: `Tensor` or Python `list` of `Tensor`s representing the
      current state(s) of the Markov chain(s). The first `r` dimensions index
      independent chains, `r = tf.rank(target_log_prob_fn(*current_state))`.
    seed: Python integer to seed the random number generator.
    current_target_log_prob: (Optional) `Tensor` representing the value of
      `target_log_prob_fn` at the `current_state`. The only reason to
      specify this argument is to reduce TF graph size.
      Default value: `None` (i.e., compute as needed).
    name: A name of the operation (optional).

  Returns:
    next_state: Tensor or Python list of `Tensor`s representing the state(s)
      of the Markov chain(s) at each result step. Has same shape as
      `current_state`.
    kernel_results: `collections.namedtuple` of internal calculations used to
      advance the chain.

  #### Examples

  We illustrate Metropolis-Hastings on a Normal likelihood with
  unknown mean.

  ```python
  tfd = tf.contrib.distributions
  tfp = tf.contrib.bayesflow

  loc = tf.get_variable("loc", initializer=1.)
  x = tf.constant([0.0] * 50)

  def make_target_log_prob_fn(x):
    def target_log_prob_fn(loc):
      prior = tfd.Normal(loc=0., scale=1.)
      likelihood = tfd.Independent(
        tfd.Normal(loc=loc, scale=0.1),
        reinterpreted_batch_ndims=1)
      return prior.log_prob(loc) + likelihood.log_prob(x)
    return target_log_prob_fn

  next_state, kernel_results = tfp.metropolis_hastings.kernel(
      target_log_prob_fn=make_target_log_prob_fn(x),
      proposal_fn=tfp.metropolis_hastings.proposal_normal(),
      current_state=loc)
  loc_update = loc.assign(next_state)
  ```

  We illustrate Metropolis-Hastings on a Normal likelihood with
  unknown mean and variance. We apply 4 chains.

  ```python
  tfd = tf.contrib.distributions
  tfp = tf.contrib.bayesflow

  num_chains = 4
  loc = tf.get_variable("loc", shape=[num_chains],
                        initializer=tf.random_normal_initializer())
  scale = tf.get_variable("scale", shape=[num_chains],
                          initializer=tf.ones_initializer())
  x = tf.constant([0.0] * 50)

  def make_target_log_prob_fn(x):
    data = tf.reshape(x, shape=[-1, 1])
    def target_log_prob_fn(loc, scale):
      prior_loc = tfd.Normal(loc=0., scale=1.)
      prior_scale = tfd.InverseGamma(concentration=1., rate=1.)
      likelihood = tfd.Independent(
        tfd.Normal(loc=loc, scale=scale),
        reinterpreted_batch_ndims=1)
      return (prior_loc.log_prob(loc) +
              prior_scale.log_prob(scale) +
              likelihood.log_prob(data))
    return target_log_prob_fn

  def proposal_fn(loc, scale):
    loc_proposal = tfp.metropolis_hastings.proposal_normal()
    scale_proposal = tfp.metropolis_hastings.proposal_uniform(minval=-1.)
    proposed_loc, _ = loc_proposal(loc)
    proposed_scale, _ = scale_proposal(scale)
    proposed_scale = tf.maximum(proposed_scale, 0.01)
    return [proposed_loc, proposed_scale], None

  next_state, kernel_results = tfp.metropolis_hastings.kernel(
      target_log_prob_fn=make_target_log_prob_fn(x),
      proposal_fn=proposal_fn,
      current_state=[loc, scale])
  train_op = tf.group(loc.assign(next_state[0]),
                      scale.assign(next_state[1]))
  ```

  """
  with ops.name_scope(
      name, "metropolis_hastings_kernel",
      [current_state, seed, current_target_log_prob]):
    with ops.name_scope("initialize"):
      maybe_expand = lambda x: list(x) if _is_list_like(x) else [x]
      current_state_parts = maybe_expand(current_state)
      if current_target_log_prob is None:
        current_target_log_prob = target_log_prob_fn(*current_state_parts)

    proposed_state, log_transit_ratio = proposal_fn(*current_state_parts)
    proposed_state_parts = maybe_expand(proposed_state)

    proposed_target_log_prob = target_log_prob_fn(*proposed_state_parts)

    with ops.name_scope(
        "accept_reject",
        [current_state_parts, proposed_state_parts,
         current_target_log_prob, proposed_target_log_prob]):
      log_accept_ratio = proposed_target_log_prob - current_target_log_prob
      if log_transit_ratio is not None:
        # If the log_transit_ratio is None, then assume the proposal is
        # symmetric, i.e.,
        #   log p(old | new) - log p(new | old) = 0.
        log_accept_ratio += log_transit_ratio

      # u < exp(log_accept_ratio),  where u~Uniform[0,1)
      # ==> log(u) < log_accept_ratio
      random_value = random_ops.random_uniform(
          array_ops.shape(log_accept_ratio),
          dtype=log_accept_ratio.dtype,
          seed=seed)
      random_negative = math_ops.log(random_value)
      is_accepted = random_negative < log_accept_ratio
      next_state_parts = [array_ops.where(is_accepted,
                                          proposed_state_part,
                                          current_state_part)
                          for proposed_state_part, current_state_part in
                          zip(proposed_state_parts, current_state_parts)]
      accepted_log_prob = array_ops.where(is_accepted,
                                          proposed_target_log_prob,
                                          current_target_log_prob)
    maybe_flatten = lambda x: x if _is_list_like(current_state) else x[0]
    return [
        maybe_flatten(next_state_parts),
        KernelResults(
            log_accept_ratio=log_accept_ratio,
            current_target_log_prob=accepted_log_prob,
            is_accepted=is_accepted,
            proposed_state=maybe_flatten(proposed_state_parts),
        ),
    ]


def evolve(initial_sample,
           initial_log_density,
           initial_log_accept_ratio,
           target_log_prob_fn,
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

  ```none
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
      "state",
      initializer=tf.random_normal([1000, n],
                                   mean=3.0,
                                   dtype=tf.float64,
                                   seed=42))

  # Computes the log(p(x)) for the unit normal density and ignores the
  # normalization constant.
  def log_density(x):
    return -tf.reduce_sum(x * x, reduction_indices=-1) / 2.0

  # Initial log-density value
  state_log_density = tf.get_variable(
      "state_log_density",
      initializer=log_density(state.initialized_value()))

  # A variable to store the log_acceptance_ratio:
  log_acceptance_ratio = tf.get_variable(
      "log_acceptance_ratio",
      initializer=tf.zeros([1000], dtype=tf.float64))

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
    print(np.cov(samples, rowvar=False))
  ```

  Args:
    initial_sample: A float-like `tf.Variable` of any shape that can
      be consumed by the `target_log_prob_fn` and `proposal_fn`
      callables.
    initial_log_density: Float-like `tf.Variable` with `dtype` and shape
      equivalent  to `target_log_prob_fn(initial_sample)`, i.e., matching
        the result of `target_log_prob_fn` invoked at `current_state`.
    initial_log_accept_ratio: A `tf.Variable` with `dtype` and shape matching
      `initial_log_density`. Stands for the log of Metropolis-Hastings
      acceptance ratio after propagating the chain for `n_steps`.
    target_log_prob_fn: A Python callable evaluated at
      `current_state` and returning a float-like `Tensor` of log target-density
      up to a normalizing constant. In other words,
      `target_log_prob_fn(x) = log(g(x))`, where
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
          last dimension will then be 'consumed' by `target_log_prob_fn`
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

  with ops.name_scope(name, "metropolis_hastings", [initial_sample]):
    current_state = initial_sample
    current_target_log_prob = initial_log_density
    log_accept_ratio = initial_log_accept_ratio

    def step(i, current_state, current_target_log_prob, log_accept_ratio):
      """Wrap single Markov chain iteration in `while_loop`."""
      next_state, kernel_results = kernel(
          target_log_prob_fn=target_log_prob_fn,
          proposal_fn=proposal_fn,
          current_state=current_state,
          current_target_log_prob=current_target_log_prob,
          seed=seed)
      accepted_log_prob = kernel_results.current_target_log_prob
      log_accept_ratio = kernel_results.log_accept_ratio
      return i + 1, next_state, accepted_log_prob, log_accept_ratio

    (_, accepted_state, accepted_target_log_prob, accepted_log_accept_ratio) = (
        control_flow_ops.while_loop(
            cond=lambda i, *ignored_args: i < n_steps,
            body=step,
            loop_vars=[
                0,  # i
                current_state,
                current_target_log_prob,
                log_accept_ratio,
            ],
            parallel_iterations=1 if seed is not None else 10,
            # TODO(b/73775595): Confirm optimal setting of swap_memory.
            swap_memory=1))

    forward_step = control_flow_ops.group(
        state_ops.assign(current_target_log_prob, accepted_target_log_prob),
        state_ops.assign(current_state, accepted_state),
        state_ops.assign(log_accept_ratio, accepted_log_accept_ratio))

    return forward_step


def proposal_uniform(step_size=1.,
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

  with ops.name_scope(name, "proposal_uniform", [step_size]):
    step_size = ops.convert_to_tensor(step_size, name="step_size")

    def proposal_fn(input_state, name=None):
      """Adds a uniform perturbation to the input state.

      Args:
        input_state: A `Tensor` of any shape and real dtype.
        name: A string that sets the name for this `Op`.

      Returns:
        proposal_state:  A float-like `Tensor` with `dtype` and shape matching
          `input_state`.
        log_transit_ratio: `None`. Proposal is symmetric.
      """
      with ops.name_scope(name, "proposer", [input_state]):
        input_state = ops.convert_to_tensor(input_state, name="input_state")
        return input_state + random_ops.random_uniform(
            array_ops.shape(input_state),
            minval=-step_size,
            maxval=step_size,
            seed=seed), None
    return proposal_fn


def proposal_normal(scale=1.,
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

  with ops.name_scope(name, "proposal_normal", [scale]):
    scale = ops.convert_to_tensor(scale, name="scale")

    def proposal_fn(input_state, name=None):
      """Adds a normal perturbation to the input state.

      Args:
        input_state: A `Tensor` of any shape and real dtype.
        name: A string that sets the name for this `Op`.

      Returns:
        proposal_state:  A float-like `Tensor` with `dtype` and shape matching
          `input_state`.
        log_transit_ratio: `None`. Proposal is symmetric.
      """

      with ops.name_scope(name, "proposer", [input_state]):
        input_state = ops.convert_to_tensor(input_state, name="input_state")
        return input_state + random_ops.random_normal(
            array_ops.shape(input_state),
            mean=0.,
            stddev=scale,
            dtype=scale.dtype,
            seed=seed), None
    return proposal_fn


def _is_list_like(x):
  """Helper which returns `True` if input is `list`-like."""
  return isinstance(x, (tuple, list))
