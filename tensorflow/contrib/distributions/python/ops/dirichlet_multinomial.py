# Copyright 2016 Google Inc. All Rights Reserved.
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
"""The Dirichlet Multinomial distribution class.

@@DirichletMultinomial
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops


def _check_alpha(alpha):
  """Check alpha for proper shape, values, then return tensor version."""
  alpha = ops.convert_to_tensor(alpha, name='alpha_before_deps')
  return control_flow_ops.with_dependencies(
      [check_ops.assert_rank_at_least(alpha, 1),
       check_ops.assert_positive(alpha)], alpha)


def _log_combinations(counts, name='log_combinations'):
  """Log number of ways counts could have come in."""
  # Firt a bit about the number of ways counts could have come in:
  # E.g. if counts = [1, 2], then this is 3 choose 2.
  # In general, this is (sum counts)! / sum(counts!)
  # The sum should be along the last dimension of counts.  This is the
  # "distribution" dimension.
  with ops.op_scope([counts], name):
    last_dim = array_ops.rank(counts) - 1
    # To compute factorials, use the fact that Gamma(n + 1) = n!
    # Compute two terms, each a sum over counts.  Compute each for each
    # batch member.
    # Log Gamma((sum counts) + 1) = Log((sum counts)!)
    sum_of_counts = math_ops.reduce_sum(counts, reduction_indices=last_dim)
    total_permutations = math_ops.lgamma(sum_of_counts + 1)
    # sum(Log Gamma(counts + 1)) = Log sum(counts!)
    counts_factorial = math_ops.lgamma(counts + 1)
    redundant_permutations = math_ops.reduce_sum(counts_factorial,
                                                 reduction_indices=last_dim)
    return total_permutations - redundant_permutations


class DirichletMultinomial(object):
  """DirichletMultinomial mixture distribution.

  The Dirichlet Multinomial is a distribution over k-class count data, meaning
  for each k-tuple of non-negative integer `counts = [c_1,...,c_k]`, we have a
  probability of these draws being made from the distribution.  The distribution
  has hyperparameters `alpha = (alpha_1,...,alpha_k)`, and probability mass
  function (pmf):

  ```pmf(counts) = C! / (c_1!...c_k!) * Beta(alpha + c) / Beta(alpha)```

  where above `C = sum_j c_j`, `N!` is `N` factorial, and
  `Beta(x) = prod_j Gamma(x_j) / Gamma(sum_j x_j)` is the multivariate beta
  function.

  This is a mixture distribution in that `N` samples can be produced by:
    1. Choose class probabilities `p = (p_1,...,p_k) ~ Dir(alpha)`
    2. Draw integers `m = (m_1,...,m_k) ~ Multinomial(p, N)`

  This class provides methods to create indexed batches of Dirichlet
  Multinomial distributions.  If the provided `alpha` is rank 2 or higher, for
  every fixed set of leading dimensions, the last dimension represents one
  single Dirichlet Multinomial distribution.  When calling distribution
  functions (e.g. `dist.pdf(counts)`), `alpha` and `counts` are broadcast to the
  same shape (if possible).  In all cases, the last dimension of alpha/counts
  represents single Dirichlet Multinomial distributions.

  Examples:

  ```python
  alpha = [1, 2, 3]
  dist = DirichletMultinomial(alpha)
  ```

  Creates a 3-class distribution, with the 3rd class is most likely to be drawn.
  The distribution functions can be evaluated on counts.

  ```python
  # counts same shape as alpha.
  counts = [0, 2, 0]
  dist.pdf(counts)  # Shape []

  # alpha will be broadcast to [[1, 2, 3], [1, 2, 3]] to match counts.
  counts = [[11, 22, 33], [44, 55, 66]]
  dist.pdf(counts)  # Shape [2]

  # alpha will be broadcast to shape [5, 7, 3] to match counts.
  counts = [[...]]  # Shape [5, 7, 3]
  dist.pdf(counts)  # Shape [5, 7]
  ```

  Creates a 2-batch of 3-class distributions.

  ```python
  alpha = [[1, 2, 3], [4, 5, 6]]  # Shape [2, 3]
  dist = DirichletMultinomial(alpha)

  # counts will be broadcast to [[11, 22, 33], [11, 22, 33]] to match alpha.
  counts = [11, 22, 33]
  dist.pdf(counts)  # Shape [2]
  ```
  """

  # TODO(b/27419586) Change docstring for dtype of alpha once int allowed.
  def __init__(self, alpha):
    """Initialize a batch of DirichletMultinomial distributions.

    Args:
      alpha:  Shape `[N1,..., Nn, k]` positive `float` or `double` tensor with
        `n >= 0`.  Defines this as a batch of `N1 x ... x Nn` different `k`
        class Dirichlet multinomial distributions.

    Examples:

    ```python
    # Define 1-batch of 2-class Dirichlet multinomial distribution,
    # also known as a beta-binomial.
    dist = DirichletMultinomial([1.1, 2.0])

    # Define a 2-batch of 3-class distributions.
    dist = DirichletMultinomial([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    ```
    """
    # Broadcasting works because:
    # * The broadcasting convention is to prepend dimensions of size [1], and
    #   we use the last dimension for the distribution, wherease
    #   the batch dimensions are the leading dimensions, which forces the
    #   distribution dimension to be defined explicitly (i.e. it cannot be
    #   created automatically by prepending).  This forces enough explicitivity.
    # * All calls involving `counts` eventually require a broadcast between
    #   `counts` and alpha.
    self._alpha = _check_alpha(alpha)

    self._num_classes = self._get_num_classes()
    self._dist_indices = self._get_dist_indices()

  @property
  def alpha(self):
    """Parameters defining this distribution."""
    return self._alpha

  @property
  def dtype(self):
    return self._alpha.dtype

  @property
  def mean(self):
    """Class means for every batch member."""
    with ops.name_scope('mean'):
      alpha_sum = math_ops.reduce_sum(self._alpha,
                                      reduction_indices=self._dist_indices,
                                      keep_dims=True)
      mean = math_ops.truediv(self._alpha, alpha_sum)
      mean.set_shape(self._alpha.get_shape())
      return mean

  def _get_dist_indices(self):
    """Dimensions corresponding to individual distributions."""
    # Reshape the scalar to a rank 1 tensor.
    return array_ops.reshape(array_ops.rank(self._alpha) - 1, [-1])

  def _get_num_classes(self):
    return ops.convert_to_tensor(
        array_ops.reverse(
            array_ops.shape(self._alpha), [True])[0],
        name='num_classes')

  @property
  def num_classes(self):
    """Tensor providing number of classes in each batch member."""
    return self._num_classes

  def cdf(self, x):
    raise NotImplementedError(
        'DirichletMultinomial does not have a well-defined cdf.')

  def log_cdf(self, x):
    raise NotImplementedError(
        'DirichletMultinomial does not have a well-defined cdf.')

  def log_pmf(self, counts, name=None):
    """`Log(P[counts])`, computed for every batch member.

    For each batch of counts `[c_1,...,c_k]`, `P[counts]` is the probability
    that after sampling `sum_j c_j` draws from this Dirichlet Multinomial
    distribution, the number of draws falling in class `j` is `c_j`.  Note that
    different sequences of draws can result in the same counts, thus the
    probability includes a combinatorial coefficient.

    Args:
      counts:  Non-negative `float`, `double`, or `int` tensor whose shape can
        be broadcast with `self.alpha`.  For fixed leading dimensions, the last
        dimension represents counts for the corresponding Dirichlet Multinomial
        distribution in `self.alpha`.
      name:  Name to give this Op, defaults to "log_pmf".

    Returns:
      Log probabilities for each record, shape `[N1,...,Nn]`.
    """
    alpha = self._alpha
    with ops.op_scope([alpha, counts], name, 'log_pmf'):
      counts = self._check_counts(counts)
      ordered_pmf = (special_math_ops.lbeta(alpha + counts) -
                     special_math_ops.lbeta(alpha))
      log_pmf = ordered_pmf + _log_combinations(counts)
      # If alpha = counts = [[]], ordered_pmf carries the right shape, which is
      # [].  However, since reduce_sum([[]]) = [0], log_combinations = [0],
      # which is not correct.  Luckily, [] + [0] = [], so the sum is fine, but
      # shape must be inferred from ordered_pmf.
      # Note also that tf.constant([]).get_shape() = TensorShape([Dimension(0)])
      log_pmf.set_shape(ordered_pmf.get_shape())
      return log_pmf

  def pmf(self, counts, name=None):
    """`P[counts]`, computed for every batch member.

    For each batch of counts `[c_1,...,c_k]`, `P[counts]` is the probability
    that after sampling `sum_j c_j` draws from this Dirichlet Multinomial
    distribution, the number of draws falling in class `j` is `c_j`.  Note that
    different sequences of draws can result in the same counts, thus the
    probability includes a combinatorial coefficient.

    Args:
      counts:  Non-negative `float`, `double`, or `int` tensor whose shape can
        be broadcast with `self.alpha`.  For fixed leading dimensions, the last
        dimension represents counts for the corresponding Dirichlet Multinomial
        distribution in `self.alpha`.
      name:  Name to give this Op, defaults to "pmf".

    Returns:
      Probabilities for each record, shape `[N1,...,Nn]`.
    """
    with ops.name_scope('pmf' if name is None else name):
      return math_ops.exp(self.log_pmf(counts))

  def _check_counts(self, counts):
    """Check counts for proper shape, values, then return tensor version."""
    counts = ops.convert_to_tensor(counts, name='counts_before_deps')
    counts = math_ops.cast(counts, self.dtype)
    return control_flow_ops.with_dependencies(
        [check_ops.assert_non_negative(counts)], counts)
