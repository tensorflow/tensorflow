# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for Markov Chain Monte Carlo (MCMC) sampling.

@@potential_scale_reduction
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

__all__ = [
    "potential_scale_reduction",
]


def potential_scale_reduction(state, independent_chain_ndims=1, name=None):
  """Gelman and Rubin's potential scale reduction factor for chain convergence.

  Given `N > 1` samples from each of `C > 1` independent chains, the potential
  scale reduction factor, commonly referred to as R-hat, measures convergence of
  the chains (to the same target) by testing for equality of means.
  Specifically, R-hat measures the degree to which variance (of the means)
  between chains exceeds what one would expect if the chains were identically
  distributed.  See [1], [2].

  Some guidelines:

  * The initial state of the chains should be drawn from a distribution
    overdispersed with respect to the target.
  * If all chains converge to the target, then as `N --> infinity`, R-hat --> 1.
    Before that, R-hat > 1 (except in pathological cases, e.g. if the chain
    paths were identical).
  * The above holds for any number of chains `C > 1`.  Increasing `C` does
    improves effectiveness of the diagnostic.
  * Sometimes, R-hat < 1.2 is used to indicate approximate convergence, but of
    course this is problem depedendent.  See [2].
  * R-hat only measures non-convergence of the mean. If higher moments, or other
    statistics are desired, a different diagnostic should be used.  See [2].

  #### Examples

  Diagnosing convergence by monitoring 10 chains that each attempt to
  sample from a 2-variate normal.

  ```python
  tfd = tf.contrib.distributions
  tfb = tf.contrib.bayesflow

  target = tfd.MultivariateNormalDiag(scale_diag=[1., 2.])

  # Get 10 (2x) overdispersed initial states.
  initial_state = target.sample(10) * 2.
  ==> (10, 2)

  # Get 1000 samples from the 10 independent chains.
  state = tfb.hmc.sample_chain(
      num_results=1000,
      target_log_prob_fn=target.log_prob,
      current_state=initial_state,
      step_size=0.05,
      num_leapfrog_steps=20,
      num_burnin_steps=200)
  state.shape
  ==> (1000, 10, 2)

  rhat = tfb.mcmc_diagnostics.potential_scale_reduction(
      state, independent_chain_ndims=1)

  # The second dimension needed a longer burn-in.
  rhat.eval()
  ==> [1.05, 1.3]
  ```

  To see why R-hat is reasonable, let `X` be a random variable drawn uniformly
  from the combined states (combined over all chains).  Then, in the limit
  `N, C --> infinity`, with `E`, `Var` denoting expectation and variance,

  ```R-hat = ( E[Var[X | chain]] + Var[E[X | chain]] ) / E[Var[X | chain]].```

  Using the law of total variance, the numerator is the variance of the combined
  states, and the denominator is the total variance minus the variance of the
  the individual chain means.  If the chains are all drawing from the same
  distribution, they will have the same mean, and thus the ratio should be one.

  [1] "Inference from Iterative Simulation Using Multiple Sequences"
      Andrew Gelman and Donald B. Rubin
      Statist. Sci. Volume 7, Number 4 (1992), 457-472.
  [2] "General Methods for Monitoring Convergence of Iterative Simulations"
      Stephen P. Brooks and Andrew Gelman
      Journal of Computational and Graphical Statistics, 1998. Vol 7, No. 4.

  Args:
    state:  `Tensor` or Python `list` of `Tensor`s representing the state(s) of
      a Markov Chain at each result step.  The `ith` state is assumed to have
      shape `[Ni, Ci1, Ci2,...,CiD] + A`.
      Dimension `0` indexes the `Ni > 1` result steps of the Markov Chain.
      Dimensions `1` through `D` index the `Ci1 x ... x CiD` independent
      chains to be tested for convergence to the same target.
      The remaining dimensions, `A`, can have any shape (even empty).
    independent_chain_ndims: Integer type `Tensor` with value `>= 1` giving the
      number of giving the number of dimensions, from `dim = 1` to `dim = D`,
      holding independent chain results to be tested for convergence.
    name: `String` name to prepend to created ops.  Default:
      `potential_scale_reduction`.

  Returns:
    `Tensor` or Python `list` of `Tensor`s representing the R-hat statistic for
    the state(s).  Same `dtype` as `state`, and shape equal to
    `state.shape[1 + independent_chain_ndims:]`.

  Raises:
    ValueError:  If `independent_chain_ndims < 1`.
  """
  # tensor_util.constant_value returns None iff a constant value (as a numpy
  # array) is not efficiently computable.  Therefore, we try constant_value then
  # check for None.
  icn_const_ = tensor_util.constant_value(
      ops.convert_to_tensor(independent_chain_ndims))
  if icn_const_ is not None:
    independent_chain_ndims = icn_const_
    if icn_const_ < 1:
      raise ValueError(
          "Argument `independent_chain_ndims` must be `>= 1`, found: {}".format(
              independent_chain_ndims))
  with ops.name_scope(
      name,
      "potential_scale_reduction",
      values=[state, independent_chain_ndims]):
    if _is_list_like(state):
      return [
          _potential_scale_reduction_single_state(s, independent_chain_ndims)
          for s in state
      ]
    return _potential_scale_reduction_single_state(state,
                                                   independent_chain_ndims)


def _potential_scale_reduction_single_state(state, independent_chain_ndims):
  """potential_scale_reduction for one single state `Tensor`."""
  # We assume exactly one leading dimension indexes e.g. correlated samples from
  # each Markov chain.
  state = ops.convert_to_tensor(state, name="state")
  sample_ndims = 1

  sample_axis = math_ops.range(0, sample_ndims)
  chain_axis = math_ops.range(sample_ndims,
                              sample_ndims + independent_chain_ndims)
  sample_and_chain_axis = math_ops.range(0,
                                         sample_ndims + independent_chain_ndims)

  n = _axis_size(state, sample_axis)
  m = _axis_size(state, chain_axis)

  # In the language of [2],
  # B / n is the between chain variance, the variance of the chain means.
  # W is the within sequence variance, the mean of the chain variances.
  b_div_n = _reduce_variance(
      math_ops.reduce_mean(state, sample_axis, keepdims=True),
      sample_and_chain_axis,
      biased=False)
  w = math_ops.reduce_mean(
      _reduce_variance(state, sample_axis, keepdims=True, biased=True),
      sample_and_chain_axis)

  # sigma^2_+ is an estimate of the true variance, which would be unbiased if
  # each chain was drawn from the target.  c.f. "law of total variance."
  sigma_2_plus = w + b_div_n

  return ((m + 1.) / m) * sigma_2_plus / w - (n - 1.) / (m * n)


def effective_sample_size(state,
                          independent_chain_ndims=1,
                          max_lags=None,
                          max_lags_threshold=None,
                          name="effective_sample_size"):
  if max_lags is not None and max_lags_threshold is not None:
    raise ValueError(
        "Expected at most one of max_lags, max_lags_threshold to be provided.  "
        "Found: {}, {}".format(max_lags, max_lags_threshold))
  with ops.name_scope(
      name,
      values=[state, independent_chain_ndims, max_lags, max_lags_threshold]):
    pass


# TODO(b/72873233) Move some variant of this to sample_stats.
def _reduce_variance(x, axis=None, biased=True, keepdims=False):
  with ops.name_scope("reduce_variance"):
    x = ops.convert_to_tensor(x, name="x")
    mean = math_ops.reduce_mean(x, axis=axis, keepdims=True)
    biased_var = math_ops.reduce_mean(
        math_ops.squared_difference(x, mean), axis=axis, keepdims=keepdims)
    if biased:
      return biased_var
    n = _axis_size(x, axis)
    return (n / (n - 1.)) * biased_var


def _axis_size(x, axis=None):
  """Get number of elements of `x` in `axis`, as type `x.dtype`."""
  if axis is None:
    return math_ops.cast(array_ops.size(x), x.dtype)
  return math_ops.cast(
      math_ops.reduce_prod(array_ops.gather(array_ops.shape(x), axis)), x.dtype)


def _is_list_like(x):
  """Helper which returns `True` if input is `list`-like."""
  return isinstance(x, (tuple, list))
