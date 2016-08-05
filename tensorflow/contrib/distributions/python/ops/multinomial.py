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

# pylint: disable=line-too-long

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

# pylint: enable=line-too-long


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
               validate_args=True,
               allow_nan_stats=False,
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
        different `k` class Multinomial distributions.
      p:  Positive floating point tensor with shape broadcastable to
        `[N1,..., Nm, k]` `m >= 0` and same dtype as `n`.  Defines this as
        a batch of `N1 x ... x Nm` different `k` class Multinomial
        distributions. `p`'s components in the last portion of its shape should
        sum up to 1.
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
    # Define 1-batch of 2-class multinomial distribution,
    # also known as a Binomial distribution.
    dist = Multinomial(n=2., p=[.1, .9])

    # Define a 2-batch of 3-class distributions.
    dist = Multinomial(n=[4., 5], p=[[.1, .3, .6], [.4, .05, .55]])
    ```

    """

    self._logits, self._p = distribution_util.get_logits_and_prob(
        name=name, logits=logits, p=p, validate_args=validate_args,
        multidimensional=True)
    with ops.op_scope([n, self._p], name):
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

        self._mean = array_ops.expand_dims(n, -1) * self._p
        # Only used for inferring shape.
        self._broadcast_shape = math_ops.reduce_sum(self._mean,
                                                    reduction_indices=[-1],
                                                    keep_dims=False)

        self._get_batch_shape = self._broadcast_shape.get_shape()
        self._get_event_shape = (
            self._mean.get_shape().with_rank_at_least(1)[-1:])

  @property
  def n(self):
    """Number of trials."""
    return self._n

  @property
  def p(self):
    """Event probabilities."""
    return self._p

  @property
  def logits(self):
    """Log-odds."""
    return self._logits

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
    with ops.name_scope(self.name):
      with ops.op_scope([self._broadcast_shape], name):
        return array_ops.shape(self._broadcast_shape)

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
      with ops.op_scope([self._mean], name):
        return array_ops.gather(array_ops.shape(self._mean),
                                [array_ops.rank(self._mean) - 1])

  def get_event_shape(self):
    """`TensorShape` available at graph construction time.

    Same meaning as `event_shape`. May be only partially defined.

    Returns:
      event shape
    """
    return self._get_event_shape

  def mean(self, name="mean"):
    """Mean of the distribution."""
    with ops.name_scope(self.name):
      return array_ops.identity(self._mean, name=name)

  def variance(self, name="variance"):
    """Variance of the distribution."""
    with ops.name_scope(self.name):
      with ops.op_scope([self._n, self._p, self._mean], name):
        p = array_ops.expand_dims(
            self._p * array_ops.expand_dims(
                array_ops.ones_like(self._n), -1), -1)
        variance = -math_ops.batch_matmul(
            array_ops.expand_dims(self._mean, -1), p, adj_y=True)
        variance += array_ops.batch_matrix_diag(self._mean)
        return variance

  def log_prob(self, counts, name="log_prob"):
    """`Log(P[counts])`, computed for every batch member.

    For each batch of counts `[n_1,...,n_k]`, `P[counts]` is the probability
    that after sampling `n` draws from this Multinomial distribution, the
    number of draws falling in class `j` is `n_j`.  Note that different
    sequences of draws can result in the same counts, thus the probability
    includes a combinatorial coefficient.

    Args:
      counts:  Non-negative tensor with dtype `dtype` and whose shape can
        be broadcast with `self.p` and `self.n`.  For fixed leading dimensions,
        the last dimension represents counts for the corresponding Multinomial
        distribution in `self.p`. `counts` is only legal if it sums up to `n`
        and its components are equal to integer values.
      name:  Name to give this Op, defaults to "log_prob".

    Returns:
      Log probabilities for each record, shape `[N1,...,Nm]`.
    """
    n = self._n
    p = self._p
    with ops.name_scope(self.name):
      with ops.op_scope([n, p, counts], name):
        counts = self._check_counts(counts)

        prob_prob = math_ops.reduce_sum(counts * math_ops.log(self._p),
                                        reduction_indices=[-1])
        log_prob = prob_prob + distribution_util.log_combinations(
            n, counts)
        return log_prob

  def prob(self, counts, name="prob"):
    """`P[counts]`, computed for every batch member.

    For each batch of counts `[n_1,...,n_k]`, `P[counts]` is the probability
    that after sampling `n` draws from this Multinomial distribution, the
    number of draws falling in class `j` is `n_j`.  Note that different
    sequences of draws can result in the same counts, thus the probability
    includes a combinatorial coefficient.

    Args:
      counts:  Non-negative tensor with dtype `dtype` and whose shape can
        be broadcast with `self.p` and `self.n`.  For fixed leading dimensions,
        the last dimension represents counts for the corresponding Multinomial
        distribution in `self.p`. `counts` is only legal if it sums up to `n`
        and its components are equal to integer values.
      name:  Name to give this Op, defaults to "prob".

    Returns:
      Probabilities for each record, shape `[N1,...,Nm]`.
    """
    return super(Multinomial, self).prob(counts, name=name)

  @property
  def is_continuous(self):
    return False

  @property
  def is_reparameterized(self):
    return False

  def _check_counts(self, counts):
    """Check counts for proper shape, values, then return tensor version."""
    counts = ops.convert_to_tensor(counts, name="counts_before_deps")
    candidate_n = math_ops.reduce_sum(counts, reduction_indices=[-1])
    if not self.validate_args:
      return counts

    return control_flow_ops.with_dependencies([
        check_ops.assert_non_negative(
            counts, message="counts has negative components."),
        check_ops.assert_equal(
            self._n, candidate_n, message="counts do not sum to n."),
        distribution_util.assert_integer_form(
            counts, message="counts have non-integer components.")], counts)
