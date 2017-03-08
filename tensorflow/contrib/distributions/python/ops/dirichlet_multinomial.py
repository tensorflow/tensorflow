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

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops


_dirichlet_multinomial_prob_note = """
For each batch of counts `[n_1,...,n_k]`, `P[counts]` is the probability
that after sampling `n` draws from this Dirichlet Multinomial
distribution, the number of draws falling in class `j` is `n_j`.  Note that
different sequences of draws can result in the same counts, thus the
probability includes a combinatorial coefficient.

Note that input, "counts", must be a non-negative tensor with dtype `dtype`
and whose shape can be broadcast with `self.alpha`.  For fixed leading
dimensions, the last dimension represents counts for the corresponding
Dirichlet Multinomial distribution in `self.alpha`. `counts` is only legal if
it sums up to `n` and its components are equal to integer values.
"""


class DirichletMultinomial(distribution.Distribution):
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
               validate_args=False,
               allow_nan_stats=True,
               name="DirichletMultinomial"):
    """Initialize a batch of DirichletMultinomial distributions.

    Args:
      n:  Non-negative floating point tensor, whose dtype is the same as
        `alpha`. The shape is broadcastable to `[N1,..., Nm]` with `m >= 0`.
        Defines this as a batch of `N1 x ... x Nm` different Dirichlet
        multinomial distributions. Its components should be equal to integer
        values.
      alpha: Positive floating point tensor, whose dtype is the same as
        `n` with shape broadcastable to `[N1,..., Nm, k]` `m >= 0`.  Defines
        this as a batch of `N1 x ... x Nm` different `k` class Dirichlet
        multinomial distributions.
      validate_args: `Boolean`, default `False`.  Whether to assert valid
        values for parameters `alpha` and `n`, and `x` in `prob` and
        `log_prob`.  If `False`, correct behavior is not guaranteed.
      allow_nan_stats: `Boolean`, default `True`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member.  If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to prefix Ops created by this distribution class.

    Examples:

    ```python
    # Define 1-batch of 2-class Dirichlet multinomial distribution,
    # also known as a beta-binomial.
    dist = DirichletMultinomial(2.0, [1.1, 2.0])

    # Define a 2-batch of 3-class distributions.
    dist = DirichletMultinomial([3., 4], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    ```

    """
    parameters = locals()
    parameters.pop("self")
    with ops.name_scope(name, values=[n, alpha]) as ns:
      # Broadcasting works because:
      # * The broadcasting convention is to prepend dimensions of size [1], and
      #   we use the last dimension for the distribution, wherease
      #   the batch dimensions are the leading dimensions, which forces the
      #   distribution dimension to be defined explicitly (i.e. it cannot be
      #   created automatically by prepending).  This forces enough
      #   explicitivity.
      #   * All calls involving `counts` eventually require a broadcast between
      #   `counts` and alpha.
      self._alpha = self._assert_valid_alpha(alpha, validate_args)
      self._n = self._assert_valid_n(n, validate_args)
      self._alpha_sum = math_ops.reduce_sum(
          self._alpha, reduction_indices=[-1], keep_dims=False)
    super(DirichletMultinomial, self).__init__(
        dtype=self._alpha.dtype,
        is_continuous=False,
        is_reparameterized=False,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._alpha, self._n, self._alpha_sum],
        name=ns)

  @property
  def n(self):
    """Parameter defining this distribution."""
    return self._n

  @property
  def alpha(self):
    """Parameter defining this distribution."""
    return self._alpha

  @property
  def alpha_sum(self):
    """Summation of alpha parameter."""
    return self._alpha_sum

  def _batch_shape(self):
    return array_ops.shape(self.alpha_sum)

  def _get_batch_shape(self):
    return self.alpha_sum.get_shape()

  def _event_shape(self):
    return array_ops.reverse(array_ops.shape(self.alpha), [True])[0]

  def _get_event_shape(self):
    # Event shape depends only on alpha, not "n".
    return self.alpha.get_shape().with_rank_at_least(1)[-1:]

  @distribution_util.AppendDocstring(_dirichlet_multinomial_prob_note)
  def _log_prob(self, counts):
    counts = self._assert_valid_counts(counts)
    ordered_prob = (special_math_ops.lbeta(self.alpha + counts) -
                    special_math_ops.lbeta(self.alpha))
    log_prob = ordered_prob + distribution_util.log_combinations(
        self.n, counts)
    return log_prob

  @distribution_util.AppendDocstring(_dirichlet_multinomial_prob_note)
  def _prob(self, counts):
    return math_ops.exp(self._log_prob(counts))

  def _mean(self):
    normalized_alpha = self.alpha / array_ops.expand_dims(self.alpha_sum, -1)
    return array_ops.expand_dims(self.n, -1) * normalized_alpha

  @distribution_util.AppendDocstring(
      """The variance for each batch member is defined as the following:

      ```
      Var(X_j) = n * alpha_j / alpha_0 * (1 - alpha_j / alpha_0) *
      (n + alpha_0) / (1 + alpha_0)
      ```

      where `alpha_0 = sum_j alpha_j`.

      The covariance between elements in a batch is defined as:

      ```
      Cov(X_i, X_j) = -n * alpha_i * alpha_j / alpha_0 ** 2 *
      (n + alpha_0) / (1 + alpha_0)
      ```
      """)
  def _variance(self):
    alpha_sum = array_ops.expand_dims(self.alpha_sum, -1)
    normalized_alpha = self.alpha / alpha_sum
    variance = -math_ops.matmul(
        array_ops.expand_dims(normalized_alpha, -1),
        array_ops.expand_dims(normalized_alpha, -2))
    variance = array_ops.matrix_set_diag(variance, normalized_alpha *
                                         (1. - normalized_alpha))
    shared_factor = (self.n * (alpha_sum + self.n) /
                     (alpha_sum + 1) * array_ops.ones_like(self.alpha))
    variance *= array_ops.expand_dims(shared_factor, -1)
    return variance

  def _assert_valid_counts(self, counts):
    """Check counts for proper shape, values, then return tensor version."""
    counts = ops.convert_to_tensor(counts, name="counts")
    if not self.validate_args:
      return counts
    candidate_n = math_ops.reduce_sum(counts, reduction_indices=[-1])
    return control_flow_ops.with_dependencies([
        check_ops.assert_non_negative(counts),
        check_ops.assert_equal(
            self._n, candidate_n,
            message="counts do not sum to n"),
        distribution_util.assert_integer_form(counts)], counts)

  def _assert_valid_alpha(self, alpha, validate_args):
    alpha = ops.convert_to_tensor(alpha, name="alpha")
    if not validate_args:
      return alpha
    return control_flow_ops.with_dependencies(
        [check_ops.assert_rank_at_least(alpha, 1),
         check_ops.assert_positive(alpha)], alpha)

  def _assert_valid_n(self, n, validate_args):
    n = ops.convert_to_tensor(n, name="n")
    if not validate_args:
      return n
    return control_flow_ops.with_dependencies(
        [check_ops.assert_non_negative(n),
         distribution_util.assert_integer_form(n)], n)
