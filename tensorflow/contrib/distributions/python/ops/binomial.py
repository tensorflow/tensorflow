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

# pylint: disable=line-too-long

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

# pylint: enable=line-too-long


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
               validate_args=True,
               allow_nan_stats=False,
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
      validate_args: Whether to assert valid values for parameters `n` and `p`,
        and `x` in `prob` and `log_prob`.  If `False`, correct behavior is not
        guaranteed.
      allow_nan_stats:  Boolean, default `False`.  If `False`, raise an
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

    with ops.op_scope([n], name):
      with ops.control_dependencies([
          check_ops.assert_non_negative(
              n, message="n has negative components."),
          distribution_util.assert_integer_form(
              n, message="n has non-integer components."
          )] if validate_args else []):
        self._n = array_ops.identity(n, name="convert_n")

        self._name = name
        self._validate_args = validate_args
        self._allow_nan_stats = allow_nan_stats

        self._mean = self._n * self._p
        self._get_batch_shape = self._mean.get_shape()
        self._get_event_shape = tensor_shape.TensorShape([])

  @property
  def name(self):
    """Name to prepend to all ops."""
    return self._name

  @property
  def dtype(self):
    """dtype of samples from this distribution."""
    return self._p.dtype

  @property
  def validate_args(self):
    """Boolean describing behavior on invalid input."""
    return self._validate_args

  @property
  def allow_nan_stats(self):
    """Boolean describing behavior when a stat is undefined for batch member."""
    return self._allow_nan_stats

  def batch_shape(self, name="batch_shape"):
    """Batch dimensions of this instance as a 1-D int32 `Tensor`.

    The product of the dimensions of the `batch_shape` is the number of
    independent distributions of this kind the instance represents.

    Args:
      name: name to give to the op

    Returns:
      `Tensor` `batch_shape`
    """
    return array_ops.shape(self._mean)

  def get_batch_shape(self):
    """`TensorShape` available at graph construction time.

    Same meaning as `batch_shape`. May be only partially defined.

    Returns:
      batch shape
    """
    return self._get_batch_shape

  def event_shape(self, name="event_shape"):
    """Shape of a sample from a single distribution as a 1-D int32 `Tensor`.

    Args:
      name: name to give to the op

    Returns:
      `Tensor` `event_shape`
    """
    with ops.name_scope(self.name):
      with ops.op_scope([], name):
        return constant_op.constant([], name=name, dtype=dtypes.int32)

  def get_event_shape(self):
    """`TensorShape` available at graph construction time.

    Same meaning as `event_shape`. May be only partially defined.

    Returns:
      event shape
    """
    return self._get_event_shape

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

  def mean(self, name="mean"):
    """Mean of the distribution."""
    with ops.name_scope(self.name):
      return array_ops.identity(self._mean, name=name)

  def variance(self, name="variance"):
    """Variance of the distribution."""
    with ops.name_scope(self.name):
      with ops.op_scope([self._n, self._p], name):
        return self._n * self._p * (1 - self._p)

  def std(self, name="std"):
    """Standard deviation of the distribution."""
    with ops.name_scope(self.name):
      with ops.op_scope([self._n, self._p], name):
        return math_ops.sqrt(self.variance())

  def mode(self, name="mode"):
    """Mode of the distribution.

    Note that when `(n + 1) * p` is an integer, there are actually two modes.
    Namely, `(n + 1) * p` and `(n + 1) * p - 1` are both modes. Here we return
    only the larger of the two modes.

    Args:
      name: The name for this op.

    Returns:
      The mode of the Binomial distribution.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._n, self._p], name):
        return math_ops.floor((self._n + 1) * self._p)

  def log_prob(self, counts, name="log_prob"):
    """`Log(P[counts])`, computed for every batch member.

    For each batch member of counts `k`, `P[counts]` is the probability that
    after sampling `n` draws from this Binomial distribution, the number of
    successes is `k`.  Note that different sequences of draws can result in the
    same counts, thus the probability includes a combinatorial coefficient.

    Args:
      counts:  Non-negative tensor with dtype `dtype` and whose shape can be
        broadcast with `self.p` and `self.n`. `counts` is only legal if it is
        less than or equal to `n` and its components are equal to integer
        values.
      name:  Name to give this Op, defaults to "log_prob".

    Returns:
      Log probabilities for each record, shape `[N1,...,Nm]`.
    """
    n = self._n
    p = self._p
    with ops.name_scope(self.name):
      with ops.op_scope([self._n, self._p, counts], name):
        counts = self._check_counts(counts)

        prob_prob = counts * math_ops.log(p) + (
            n - counts) * math_ops.log(1 - p)

        combinations = math_ops.lgamma(n + 1) - math_ops.lgamma(
            counts + 1) - math_ops.lgamma(n - counts + 1)
        log_prob = prob_prob + combinations
        return log_prob

  def prob(self, counts, name="prob"):
    """`P[counts]`, computed for every batch member.


    For each batch member of counts `k`, `P[counts]` is the probability that
    after sampling `n` draws from this Binomial distribution, the number of
    successes is `k`.  Note that different sequences of draws can result in the
    same counts, thus the probability includes a combinatorial coefficient.

    Args:
      counts:  Non-negative tensor with dtype `dtype` and whose shape can be
        broadcast with `self.p` and `self.n`. `counts` is only legal if it is
        less than or equal to `n` and its components are equal to integer
        values.
      name:  Name to give this Op, defaults to "prob".

    Returns:
      Probabilities for each record, shape `[N1,...,Nm]`.
    """
    return super(Binomial, self).prob(counts, name=name)

  @property
  def is_continuous(self):
    return False

  @property
  def is_reparameterized(self):
    return False

  def _check_counts(self, counts):
    """Check counts for proper shape, values, then return tensor version."""
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
