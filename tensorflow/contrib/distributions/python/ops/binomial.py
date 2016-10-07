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
"""The Binomial distribution class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

_binomial_prob_note = """
For each batch member of counts `value`, `P[counts]` is the probability that
after sampling `n` draws from this Binomial distribution, the number of
successes is `k`.  Note that different sequences of draws can result in the
same counts, thus the probability includes a combinatorial coefficient.

`value` must be a non-negative tensor with dtype `dtype` and whose shape
can be broadcast with `self.p` and `self.n`. `counts` is only legal if it is
less than or equal to `n` and its components are equal to integer
values.
"""


class Binomial(distribution.Distribution):
  """Binomial distribution.

  This distribution is parameterized by a vector `p` of probabilities and `n`,
  the total counts.

  #### Mathematical details

  The Binomial is a distribution over the number of successes in `n` independent
  trials, with each trial having the same probability of success `p`.
  The probability mass function (pmf):

  ```pmf(k) = n! / (k! * (n - k)!) * (p)^k * (1 - p)^(n - k)```

  #### Examples

  Create a single distribution, corresponding to 5 coin flips.

  ```python
  dist = Binomial(n=5., p=.5)
  ```

  Create a single distribution (using logits), corresponding to 5 coin flips.

  ```python
  dist = Binomial(n=5., logits=0.)
  ```

  Creates 3 distributions with the third distribution most likely to have
  successes.

  ```python
  p = [.2, .3, .8]
  # n will be broadcast to [4., 4., 4.], to match p.
  dist = Binomial(n=4., p=p)
  ```

  The distribution functions can be evaluated on counts.

  ```python
  # counts same shape as p.
  counts = [1., 2, 3]
  dist.prob(counts)  # Shape [3]

  # p will be broadcast to [[.2, .3, .8], [.2, .3, .8]] to match counts.
  counts = [[1., 2, 1], [2, 2, 4]]
  dist.prob(counts)  # Shape [2, 3]

  # p will be broadcast to shape [5, 7, 3] to match counts.
  counts = [[...]]  # Shape [5, 7, 3]
  dist.prob(counts)  # Shape [5, 7, 3]
  ```
  """

  def __init__(self,
               n,
               logits=None,
               p=None,
               validate_args=False,
               allow_nan_stats=True,
               name="Binomial"):
    """Initialize a batch of Binomial distributions.

    Args:
      n:  Non-negative floating point tensor with shape broadcastable to
        `[N1,..., Nm]` with `m >= 0` and the same dtype as `p` or `logits`.
        Defines this as a batch of `N1 x ... x Nm` different Binomial
        distributions. Its components should be equal to integer values.
      logits: Floating point tensor representing the log-odds of a
        positive event with shape broadcastable to `[N1,..., Nm]` `m >= 0`, and
        the same dtype as `n`. Each entry represents logits for the probability
        of success for independent Binomial distributions.
      p:  Positive floating point tensor with shape broadcastable to
        `[N1,..., Nm]` `m >= 0`, `p in [0, 1]`. Each entry represents the
        probability of success for independent Binomial distributions.
      validate_args: `Boolean`, default `False`.  Whether to assert valid values
        for parameters `n`, `p`, and `x` in `prob` and `log_prob`.
        If `False` and inputs are invalid, correct behavior is not guaranteed.
      allow_nan_stats: `Boolean`, default `True`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member.  If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to prefix Ops created by this distribution class.

    Examples:

    ```python
    # Define 1-batch of a binomial distribution.
    dist = Binomial(n=2., p=.9)

    # Define a 2-batch.
    dist = Binomial(n=[4., 5], p=[.1, .3])
    ```

    """
    self._logits, self._p = distribution_util.get_logits_and_prob(
        name=name, logits=logits, p=p, validate_args=validate_args)
    with ops.name_scope(name, values=[n]) as ns:
      with ops.control_dependencies([
          check_ops.assert_non_negative(
              n, message="n has negative components."),
          distribution_util.assert_integer_form(
              n, message="n has non-integer components."),
      ] if validate_args else []):
        self._n = array_ops.identity(n, name="n")
        super(Binomial, self).__init__(
            dtype=self._p.dtype,
            parameters={"n": self._n, "p": self._p, "logits": self._logits},
            is_continuous=False,
            is_reparameterized=False,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=ns)

  @property
  def n(self):
    """Number of trials."""
    return self._n

  @property
  def logits(self):
    """Log-odds."""
    return self._logits

  @property
  def p(self):
    """Probability of success."""
    return self._p

  def _batch_shape(self):
    return array_ops.shape(self._n + self._p)

  def _get_batch_shape(self):
    return common_shapes.broadcast_shape(self.n.get_shape(),
                                         self.p.get_shape())

  def _event_shape(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _get_event_shape(self):
    return tensor_shape.scalar()

  @distribution_util.AppendDocstring(_binomial_prob_note)
  def _log_prob(self, counts):
    counts = self._check_counts(counts)
    prob_prob = (counts * math_ops.log(self.p) +
                 (self.n - counts) * math_ops.log(1. - self.p))
    combinations = (math_ops.lgamma(self.n + 1) -
                    math_ops.lgamma(counts + 1) -
                    math_ops.lgamma(self.n - counts + 1))
    log_prob = prob_prob + combinations
    return log_prob

  @distribution_util.AppendDocstring(_binomial_prob_note)
  def _prob(self, counts):
    return math_ops.exp(self._log_prob(counts))

  def _mean(self):
    return self._n * self._p

  def _variance(self):
    return self._n * self._p * (1 - self._p)

  def _std(self):
    return math_ops.sqrt(self._variance())

  @distribution_util.AppendDocstring(
      """Note that when `(n + 1) * p` is an integer, there are actually two
      modes.  Namely, `(n + 1) * p` and `(n + 1) * p - 1` are both modes. Here
      we return only the larger of the two modes.""")
  def _mode(self):
    return math_ops.floor((self._n + 1) * self._p)

  @distribution_util.AppendDocstring(
      """Check counts for proper shape, values, then return tensor version.""")
  def _check_counts(self, counts):
    counts = ops.convert_to_tensor(counts, name="counts_before_deps")
    if not self.validate_args:
      return counts
    return control_flow_ops.with_dependencies([
        check_ops.assert_non_negative(
            counts, message="counts has negative components."),
        check_ops.assert_less_equal(
            counts, self._n, message="counts are not less than or equal to n."),
        distribution_util.assert_integer_form(
            counts, message="counts have non-integer components.")], counts)
