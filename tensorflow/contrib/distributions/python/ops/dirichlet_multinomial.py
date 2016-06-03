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
"""The Dirichlet Multinomial distribution class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=line-too-long

from tensorflow.contrib.distributions.python.ops import distribution  # pylint: disable=line-too-long
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops

# pylint: enable=line-too-long


def _assert_integer_form(x):
  """Check x for integer components (or floats that are equal to integers)."""
  x = ops.convert_to_tensor(x, name='x')
  casted_x = math_ops.to_int64(x)
  return check_ops.assert_equal(x, math_ops.cast(
      math_ops.round(casted_x), x.dtype))


def _check_alpha(alpha):
  """Check alpha for proper shape, values, then return tensor version."""
  alpha = ops.convert_to_tensor(alpha, name='alpha_before_deps')
  return control_flow_ops.with_dependencies(
      [check_ops.assert_rank_at_least(alpha, 1),
       check_ops.assert_positive(alpha)], alpha)


def _check_n(n):
  """Check n for proper shape, values, then return tensor version."""
  n = ops.convert_to_tensor(n, name='n_before_deps')
  return control_flow_ops.with_dependencies(
      [check_ops.assert_non_negative(n), _assert_integer_form(n)], n)


def _log_combinations(n, counts, name='log_combinations'):
  """Log number of ways counts could have come in."""
  # First a bit about the number of ways counts could have come in:
  # E.g. if counts = [1, 2], then this is 3 choose 2.
  # In general, this is (sum counts)! / sum(counts!)
  # The sum should be along the last dimension of counts.  This is the
  # "distribution" dimension. Here n a priori represents the sum of counts.
  with ops.op_scope([counts], name):
    # To compute factorials, use the fact that Gamma(n + 1) = n!
    # Compute two terms, each a sum over counts.  Compute each for each
    # batch member.
    # Log Gamma((sum counts) + 1) = Log((sum counts)!)
    total_permutations = math_ops.lgamma(n + 1)
    # sum(Log Gamma(counts + 1)) = Log sum(counts!)
    counts_factorial = math_ops.lgamma(counts + 1)
    redundant_permutations = math_ops.reduce_sum(counts_factorial,
                                                 reduction_indices=[-1])
    return total_permutations - redundant_permutations


class DirichletMultinomial(distribution.DiscreteDistribution):
  """DirichletMultinomial mixture distribution.

  This distribution is parameterized by a vector `alpha` of concentration
  parameters for `k` classes and `n`, the counts per each class..

  #### Mathematical details

  The Dirichlet Multinomial is a distribution over k-class count data, meaning
  for each k-tuple of non-negative integer `counts = [c_1,...,c_k]`, we have a
  probability of these draws being made from the distribution.  The distribution
  has hyperparameters `alpha = (alpha_1,...,alpha_k)`, and probability mass
  function (pmf):

  ```pmf(counts) = N! / (n_1!...n_k!) * Beta(alpha + c) / Beta(alpha)```

  where above `N = sum_j n_j`, `N!` is `N` factorial, and
  `Beta(x) = prod_j Gamma(x_j) / Gamma(sum_j x_j)` is the multivariate beta
  function.

  This is a mixture distribution in that `M` samples can be produced by:
    1. Choose class probabilities `p = (p_1,...,p_k) ~ Dir(alpha)`
    2. Draw integers `m = (n_1,...,n_k) ~ Multinomial(N, p)`

  This class provides methods to create indexed batches of Dirichlet
  Multinomial distributions.  If the provided `alpha` is rank 2 or higher, for
  every fixed set of leading dimensions, the last dimension represents one
  single Dirichlet Multinomial distribution.  When calling distribution
  functions (e.g. `dist.pmf(counts)`), `alpha` and `counts` are broadcast to the
  same shape (if possible).  In all cases, the last dimension of alpha/counts
  represents single Dirichlet Multinomial distributions.

  #### Examples

  ```python
  alpha = [1, 2, 3]
  n = 2
  dist = DirichletMultinomial(n, alpha)
  ```

  Creates a 3-class distribution, with the 3rd class is most likely to be drawn.
  The distribution functions can be evaluated on counts.

  ```python
  # counts same shape as alpha.
  counts = [0, 0, 2]
  dist.pmf(counts)  # Shape []

  # alpha will be broadcast to [[1, 2, 3], [1, 2, 3]] to match counts.
  counts = [[1, 1, 0], [1, 0, 1]]
  dist.pmf(counts)  # Shape [2]

  # alpha will be broadcast to shape [5, 7, 3] to match counts.
  counts = [[...]]  # Shape [5, 7, 3]
  dist.pmf(counts)  # Shape [5, 7]
  ```

  Creates a 2-batch of 3-class distributions.

  ```python
  alpha = [[1, 2, 3], [4, 5, 6]]  # Shape [2, 3]
  n = [3, 3]
  dist = DirichletMultinomial(n, alpha)

  # counts will be broadcast to [[2, 1, 0], [2, 1, 0]] to match alpha.
  counts = [2, 1, 0]
  dist.pmf(counts)  # Shape [2]
  ```
  """

  # TODO(b/27419586) Change docstring for dtype of alpha once int allowed.
  def __init__(self,
               n,
               alpha,
               name='DirichletMultinomial',
               allow_arbitrary_counts=False):
    """Initialize a batch of DirichletMultinomial distributions.

    Args:
      n:  Non-negative `float` or `double` tensor with shape
        broadcastable to `[N1,..., Nm]` with `m >= 0`.  Defines this as a batch
        of `N1 x ... x Nm` different Dirichlet multinomial distributions. Its
        components should be equal to integral values.
      alpha:  Positive `float` or `double` tensor with shape broadcastable to
        `[N1,..., Nm, k]` `m >= 0`.  Defines this as a batch of `N1 x ... x Nm`
         different `k` class Dirichlet multinomial distributions.
      name: The name to prefix Ops created by this distribution class.
      allow_arbitrary_counts: Boolean. This represents whether the pmf/cdf
        allows for the `counts` tensor to be non-integral values.
        The pmf/cdf are functions that can be evaluated at non-integral values,
        but are only a distribution over non-negative integers.

    Examples:

    ```python
    # Define 1-batch of 2-class Dirichlet multinomial distribution,
    # also known as a beta-binomial.
    dist = DirichletMultinomial(2.0, [1.1, 2.0])

    # Define a 2-batch of 3-class distributions.
    dist = DirichletMultinomial([3., 4], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    ```
    """
    with ops.op_scope([n, alpha], name):
      # Broadcasting works because:
      # * The broadcasting convention is to prepend dimensions of size [1], and
      #   we use the last dimension for the distribution, wherease
      #   the batch dimensions are the leading dimensions, which forces the
      #   distribution dimension to be defined explicitly (i.e. it cannot be
      #   created automatically by prepending).  This forces enough
      #   explicitivity.
      #   * All calls involving `counts` eventually require a broadcast between
      #   `counts` and alpha.
      self._alpha = _check_alpha(alpha)
      self._name = name

      n = _check_n(n)
      n = math_ops.cast(n, self._alpha.dtype)
      self._n = n

      self._allow_arbitrary_counts = allow_arbitrary_counts

      self._alpha_sum = math_ops.reduce_sum(
          self._alpha, reduction_indices=[-1], keep_dims=False)

      self._get_batch_shape = self._alpha_sum.get_shape()

      # event shape depends only on alpha, not "n".
      self._get_event_shape = self._alpha.get_shape().with_rank_at_least(1)[-1:]

  @property
  def n(self):
    """Parameter defining this distribution."""
    return self._n

  @property
  def alpha(self):
    """Parameter defining this distribution."""
    return self._alpha

  @property
  def name(self):
    """Name to prepend to all ops."""
    return self._name

  @property
  def dtype(self):
    """dtype of samples from this distribution."""
    return self._alpha.dtype

  def mean(self, name='mean'):
    """Class means for every batch member."""
    alpha = self._alpha
    alpha_sum = self._alpha_sum
    n = self._n
    with ops.name_scope(self.name):
      with ops.op_scope([alpha, alpha_sum, n], name):
        mean_no_n = alpha / array_ops.expand_dims(alpha_sum, -1)
        return array_ops.expand_dims(n, -1) * mean_no_n

  def batch_shape(self, name='batch_shape'):
    """Batch dimensions of this instance as a 1-D int32 `Tensor`.

    The product of the dimensions of the `batch_shape` is the number of
    independent distributions of this kind the instance represents.

    Args:
      name: name to give to the op

    Returns:
      `Tensor` `batch_shape`
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._alpha_sum], name):
        return array_ops.shape(self._alpha_sum)

  def get_batch_shape(self):
    """`TensorShape` available at graph construction time.

    Same meaning as `batch_shape`. May be only partially defined.

    Returns:
      batch shape
    """
    return self._get_batch_shape

  def event_shape(self, name='event_shape'):
    """Shape of a sample from a single distribution as a 1-D int32 `Tensor`.

    Args:
      name: name to give to the op

    Returns:
      `Tensor` `event_shape`
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._alpha], name):
        return array_ops.reverse(array_ops.shape(self._alpha), [True])[0]

  def get_event_shape(self):
    """`TensorShape` available at graph construction time.

    Same meaning as `event_shape`. May be only partially defined.

    Returns:
      event shape
    """
    return self._get_event_shape

  def cdf(self, x, name='cdf'):
    raise NotImplementedError(
        'DirichletMultinomial does not have a well-defined cdf.')

  def log_cdf(self, x, name='log_cdf'):
    raise NotImplementedError(
        'DirichletMultinomial does not have a well-defined cdf.')

  def log_pmf(self, counts, name='log_pmf'):
    """`Log(P[counts])`, computed for every batch member.

    For each batch of counts `[n_1,...,n_k]`, `P[counts]` is the probability
    that after sampling `n` draws from this Dirichlet Multinomial
    distribution, the number of draws falling in class `j` is `n_j`.  Note that
    different sequences of draws can result in the same counts, thus the
    probability includes a combinatorial coefficient.

    Args:
      counts:  Non-negative `float` or `double` tensor whose shape can
        be broadcast with `self.alpha`.  For fixed leading dimensions, the last
        dimension represents counts for the corresponding Dirichlet Multinomial
        distribution in `self.alpha`. `counts` is only legal if it sums up to
        `n` and its components are equal to integral values. The second
        condition is relaxed if `allow_arbitrary_counts` is set.
      name:  Name to give this Op, defaults to "log_pmf".

    Returns:
      Log probabilities for each record, shape `[N1,...,Nn]`.
    """
    n = self._n
    alpha = self._alpha
    with ops.name_scope(self.name):
      with ops.op_scope([n, alpha, counts], name):
        counts = self._check_counts(counts)
        # Use the same dtype as alpha for computations.
        counts = math_ops.cast(counts, self.dtype)

        ordered_pmf = (special_math_ops.lbeta(alpha + counts) -
                       special_math_ops.lbeta(alpha))
        log_pmf = ordered_pmf + _log_combinations(n, counts)
        # If alpha = counts = [[]], ordered_pmf carries the right shape, which
        # is [].  However, since reduce_sum([[]]) = [0], log_combinations = [0],
        # which is not correct.  Luckily, [] + [0] = [], so the sum is fine, but
        # shape must be inferred from ordered_pmf. We must also make this
        # broadcastable with n, so this is multiplied by n to ensure the shape
        # is correctly inferred.
        # Note also that tf.constant([]).get_shape() =
        # TensorShape([Dimension(0)])
        broadcasted_tensor = ordered_pmf * n
        log_pmf.set_shape(broadcasted_tensor.get_shape())
        return log_pmf

  def pmf(self, counts, name='pmf'):
    """`P[counts]`, computed for every batch member.

    For each batch of counts `[c_1,...,c_k]`, `P[counts]` is the probability
    that after sampling `sum_j c_j` draws from this Dirichlet Multinomial
    distribution, the number of draws falling in class `j` is `c_j`.  Note that
    different sequences of draws can result in the same counts, thus the
    probability includes a combinatorial coefficient.

    Args:
      counts:  Non-negative `float`, `double` tensor whose shape can
        be broadcast with `self.alpha`.  For fixed leading dimensions, the last
        dimension represents counts for the corresponding Dirichlet Multinomial
        distribution in `self.alpha`. `counts` is only legal if it sums up to
        `n` and its components are equal to integral values. The second
        condition is relaxed if `allow_arbitrary_counts` is set.
      name:  Name to give this Op, defaults to "pmf".

    Returns:
      Probabilities for each record, shape `[N1,...,Nn]`.
    """
    return super(DirichletMultinomial, self).pmf(counts, name=name)

  def _check_counts(self, counts):
    """Check counts for proper shape, values, then return tensor version."""
    counts = ops.convert_to_tensor(counts, name='counts_before_deps')
    candidate_n = math_ops.reduce_sum(counts, reduction_indices=[-1])
    dependencies = [check_ops.assert_non_negative(counts),
                    check_ops.assert_equal(self._n,
                                           math_ops.cast(candidate_n,
                                                         self._n.dtype))]
    if not self._allow_arbitrary_counts:
      dependencies += [_assert_integer_form(counts)]

    return control_flow_ops.with_dependencies(dependencies, counts)
