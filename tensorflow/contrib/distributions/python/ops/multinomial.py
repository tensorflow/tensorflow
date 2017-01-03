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
"""The Multinomial distribution class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


_multinomial_prob_note = """
For each batch of counts `[n_1,...,n_k]`, `P[counts]` is the probability
that after sampling `n` draws from this Multinomial distribution, the
number of draws falling in class `j` is `n_j`.  Note that different
sequences of draws can result in the same counts, thus the probability
includes a combinatorial coefficient.

Note that input "counts" must be a non-negative tensor with dtype `dtype`
and whose shape can be broadcast with `self.p` and `self.n`.  For fixed
leading dimensions, the last dimension represents counts for the
corresponding Multinomial distribution in `self.p`. `counts` is only legal
if it sums up to `n` and its components are equal to integer values.
"""


class Multinomial(distribution.Distribution):
  """Multinomial distribution.

  This distribution is parameterized by a vector `p` of probability
  parameters for `k` classes and `n`, the counts per each class..

  #### Mathematical details

  The Multinomial is a distribution over k-class count data, meaning
  for each k-tuple of non-negative integer `counts = [n_1,...,n_k]`, we have a
  probability of these draws being made from the distribution.  The distribution
  has hyperparameters `p = (p_1,...,p_k)`, and probability mass
  function (pmf):

  ```pmf(counts) = n! / (n_1!...n_k!) * (p_1)^n_1*(p_2)^n_2*...(p_k)^n_k```

  where above `n = sum_j n_j`, `n!` is `n` factorial.

  #### Examples

  Create a 3-class distribution, with the 3rd class is most likely to be drawn,
  using logits..

  ```python
  logits = [-50., -43, 0]
  dist = Multinomial(n=4., logits=logits)
  ```

  Create a 3-class distribution, with the 3rd class is most likely to be drawn.

  ```python
  p = [.2, .3, .5]
  dist = Multinomial(n=4., p=p)
  ```

  The distribution functions can be evaluated on counts.

  ```python
  # counts same shape as p.
  counts = [1., 0, 3]
  dist.prob(counts)  # Shape []

  # p will be broadcast to [[.2, .3, .5], [.2, .3, .5]] to match counts.
  counts = [[1., 2, 1], [2, 2, 0]]
  dist.prob(counts)  # Shape [2]

  # p will be broadcast to shape [5, 7, 3] to match counts.
  counts = [[...]]  # Shape [5, 7, 3]
  dist.prob(counts)  # Shape [5, 7]
  ```

  Create a 2-batch of 3-class distributions.

  ```python
  p = [[.1, .2, .7], [.3, .3, .4]]  # Shape [2, 3]
  dist = Multinomial(n=[4., 5], p=p)

  counts = [[2., 1, 1], [3, 1, 1]]
  dist.prob(counts)  # Shape [2]
  ```
  """

  def __init__(self,
               n,
               logits=None,
               p=None,
               validate_args=False,
               allow_nan_stats=True,
               name="Multinomial"):
    """Initialize a batch of Multinomial distributions.

    Args:
      n:  Non-negative floating point tensor with shape broadcastable to
        `[N1,..., Nm]` with `m >= 0`. Defines this as a batch of
        `N1 x ... x Nm` different Multinomial distributions.  Its components
        should be equal to integer values.
      logits: Floating point tensor representing the log-odds of a
        positive event with shape broadcastable to `[N1,..., Nm, k], m >= 0`,
        and the same dtype as `n`. Defines this as a batch of `N1 x ... x Nm`
        different `k` class Multinomial distributions. Only one of `logits` or
        `p` should be passed in.
      p:  Positive floating point tensor with shape broadcastable to
        `[N1,..., Nm, k]` `m >= 0` and same dtype as `n`.  Defines this as
        a batch of `N1 x ... x Nm` different `k` class Multinomial
        distributions. `p`'s components in the last portion of its shape should
        sum up to 1. Only one of `logits` or `p` should be passed in.
      validate_args: `Boolean`, default `False`.  Whether to assert valid
        values for parameters `n` and `p`, and `x` in `prob` and `log_prob`.
        If `False`, correct behavior is not guaranteed.
      allow_nan_stats: `Boolean`, default `True`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member.  If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to prefix Ops created by this distribution class.

    Examples:

    ```python
    # Define 1-batch of 2-class multinomial distribution,
    # also known as a Binomial distribution.
    dist = Multinomial(n=2., p=[.1, .9])

    # Define a 2-batch of 3-class distributions.
    dist = Multinomial(n=[4., 5], p=[[.1, .3, .6], [.4, .05, .55]])
    ```

    """
    parameters = locals()
    parameters.pop("self")
    with ops.name_scope(name, values=[n, p]) as ns:
      with ops.control_dependencies([
          check_ops.assert_non_negative(
              n, message="n has negative components."),
          distribution_util.assert_integer_form(
              n, message="n has non-integer components.")
      ] if validate_args else []):
        self._logits, self._p = distribution_util.get_logits_and_prob(
            name=name, logits=logits, p=p, validate_args=validate_args,
            multidimensional=True)
        self._n = array_ops.identity(n, name="convert_n")
        self._mean_val = array_ops.expand_dims(n, -1) * self._p
        self._broadcast_shape = math_ops.reduce_sum(
            self._mean_val, reduction_indices=[-1], keep_dims=False)
    super(Multinomial, self).__init__(
        dtype=self._p.dtype,
        is_continuous=False,
        is_reparameterized=False,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._p, self._n, self._mean_val,
                       self._logits, self._broadcast_shape],
        name=ns)

  @property
  def n(self):
    """Number of trials."""
    return self._n

  @property
  def logits(self):
    """Vector of coordinatewise logits."""
    return self._logits

  @property
  def p(self):
    """Vector of probabilities summing to one.

    Each element is the probability of drawing that coordinate."""
    return self._p

  def _batch_shape(self):
    return array_ops.shape(self._broadcast_shape)

  def _get_batch_shape(self):
    return self._broadcast_shape.get_shape()

  def _event_shape(self):
    return array_ops.gather(array_ops.shape(self._mean_val),
                            [array_ops.rank(self._mean_val) - 1])

  def _get_event_shape(self):
    return self._mean_val.get_shape().with_rank_at_least(1)[-1:]

  def _sample_n(self, n, seed=None):
    n_draws = math_ops.cast(self.n, dtype=dtypes.int32)
    if self.n.get_shape().ndims is not None:
      if self.n.get_shape().ndims != 0:
        raise NotImplementedError(
            "Sample only supported for scalar number of draws.")
    elif self.validate_args:
      is_scalar = check_ops.assert_rank(
          n_draws, 0,
          message="Sample only supported for scalar number of draws.")
      n_draws = control_flow_ops.with_dependencies([is_scalar], n_draws)
    k = self.event_shape()[0]
    # Flatten batch dims so logits has shape [B, k],
    # where B = reduce_prod(self.batch_shape()).
    logits = array_ops.reshape(self.logits, [-1, k])
    draws = random_ops.multinomial(logits=logits,
                                   num_samples=n * n_draws,
                                   seed=seed)
    draws = array_ops.reshape(draws, shape=[-1, n, n_draws])
    x = math_ops.reduce_sum(array_ops.one_hot(draws, depth=k),
                            reduction_indices=-2)  # shape: [B, n, k]
    x = array_ops.transpose(x, perm=[1, 0, 2])
    final_shape = array_ops.concat_v2([[n], self.batch_shape(), [k]], 0)
    return array_ops.reshape(x, final_shape)

  @distribution_util.AppendDocstring(_multinomial_prob_note)
  def _log_prob(self, counts):
    counts = self._assert_valid_sample(counts)
    log_unnormalized_prob = math_ops.reduce_sum(
        counts * math_ops.log(self.p),
        reduction_indices=[-1])
    log_normalizer = -distribution_util.log_combinations(self.n, counts)
    return log_unnormalized_prob - log_normalizer

  @distribution_util.AppendDocstring(_multinomial_prob_note)
  def _prob(self, counts):
    return math_ops.exp(self._log_prob(counts))

  def _mean(self):
    return array_ops.identity(self._mean_val)

  def _variance(self):
    p = self.p * array_ops.expand_dims(array_ops.ones_like(self.n), -1)
    outer_prod = math_ops.matmul(
        array_ops.expand_dims(self._mean_val, -1), array_ops.expand_dims(p, -2))
    return array_ops.matrix_set_diag(-outer_prod,
                                     self._mean_val - self._mean_val * p)

  def _assert_valid_sample(self, counts):
    """Check counts for proper shape, values, then return tensor version."""
    if not self.validate_args: return counts
    return control_flow_ops.with_dependencies([
        check_ops.assert_non_negative(
            counts, message="counts has negative components."),
        check_ops.assert_equal(
            self.n, math_ops.reduce_sum(counts, reduction_indices=[-1]),
            message="counts do not sum to n."),
        distribution_util.assert_integer_form(
            counts, message="counts have non-integer components.")
    ], counts)
