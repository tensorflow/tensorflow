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
"""The Dirichlet distribution class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=line-too-long

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import special_math_ops

# pylint: enable=line-too-long


class Dirichlet(distribution.Distribution):
  """Dirichlet distribution.

  This distribution is parameterized by a vector `alpha` of concentration
  parameters for `k` classes.

  #### Mathematical details

  The Dirichlet is a distribution over the standard n-simplex, where the
  standard n-simplex is defined by:
  ```{ (x_1, ..., x_n) in R^(n+1) | sum_j x_j = 1 and x_j >= 0 for all j }```.
  The distribution has hyperparameters `alpha = (alpha_1,...,alpha_k)`,
  and probability mass function (prob):

  ```prob(x) = 1 / Beta(alpha) * prod_j x_j^(alpha_j - 1)```

  where `Beta(x) = prod_j Gamma(x_j) / Gamma(sum_j x_j)` is the multivariate
  beta function.


  This class provides methods to create indexed batches of Dirichlet
  distributions.  If the provided `alpha` is rank 2 or higher, for
  every fixed set of leading dimensions, the last dimension represents one
  single Dirichlet distribution.  When calling distribution
  functions (e.g. `dist.prob(x)`), `alpha` and `x` are broadcast to the
  same shape (if possible).  In all cases, the last dimension of alpha/x
  represents single Dirichlet distributions.

  #### Examples

  ```python
  alpha = [1, 2, 3]
  dist = Dirichlet(alpha)
  ```

  Creates a 3-class distribution, with the 3rd class is most likely to be drawn.
  The distribution functions can be evaluated on x.

  ```python
  # x same shape as alpha.
  x = [.2, .3, .5]
  dist.prob(x)  # Shape []

  # alpha will be broadcast to [[1, 2, 3], [1, 2, 3]] to match x.
  x = [[.1, .4, .5], [.2, .3, .5]]
  dist.prob(x)  # Shape [2]

  # alpha will be broadcast to shape [5, 7, 3] to match x.
  x = [[...]]  # Shape [5, 7, 3]
  dist.prob(x)  # Shape [5, 7]
  ```

  Creates a 2-batch of 3-class distributions.

  ```python
  alpha = [[1, 2, 3], [4, 5, 6]]  # Shape [2, 3]
  dist = Dirichlet(alpha)

  # x will be broadcast to [[2, 1, 0], [2, 1, 0]] to match alpha.
  x = [.2, .3, .5]
  dist.prob(x)  # Shape [2]
  ```

  """

  def __init__(self,
               alpha,
               validate_args=True,
               allow_nan_stats=False,
               name="Dirichlet"):
    """Initialize a batch of Dirichlet distributions.

    Args:
      alpha:  Positive floating point tensor with shape broadcastable to
        `[N1,..., Nm, k]` `m >= 0`.  Defines this as a batch of `N1 x ... x Nm`
         different `k` class Dirichlet distributions.
      validate_args: Whether to assert valid values for parameters `alpha` and
        `x` in `prob` and `log_prob`.  If `False`, correct behavior is not
        guaranteed.
      allow_nan_stats:  Boolean, default `False`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member.  If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to prefix Ops created by this distribution class.

    Examples:

    ```python
    # Define 1-batch of 2-class Dirichlet distributions,
    # also known as a Beta distribution.
    dist = Dirichlet([1.1, 2.0])

    # Define a 2-batch of 3-class distributions.
    dist = Dirichlet([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    ```

    """
    with ops.op_scope([alpha], name):
      alpha = ops.convert_to_tensor(alpha, name="alpha_before_deps")
      with ops.control_dependencies([
          check_ops.assert_positive(alpha), check_ops.assert_rank_at_least(
              alpha, 1)
      ] if validate_args else []):
        alpha = array_ops.identity(alpha, name="alpha")

      self._alpha = alpha
      self._name = name

      # Used for mean/mode/variance/entropy computations
      self._alpha_0 = math_ops.reduce_sum(alpha,
                                          reduction_indices=[-1],
                                          keep_dims=False)

      self._get_batch_shape = self._alpha_0.get_shape()
      self._get_event_shape = self._alpha.get_shape().with_rank_at_least(1)[-1:]
      self._validate_args = validate_args
      self._allow_nan_stats = allow_nan_stats

  @property
  def alpha(self):
    """Shape parameter."""
    return self._alpha

  @property
  def name(self):
    """Name to prepend to all ops."""
    return self._name

  @property
  def dtype(self):
    """dtype of samples from this distribution."""
    return self._alpha.dtype

  @property
  def allow_nan_stats(self):
    """Boolean describing behavior when a stat is undefined for batch member."""
    return self._allow_nan_stats

  @property
  def validate_args(self):
    """Boolean describing behavior on invalid input."""
    return self._validate_args

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
      with ops.op_scope([self._alpha], name):
        return array_ops.shape(self._alpha_0)

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
      with ops.op_scope([self._alpha], name):
        return array_ops.gather(array_ops.shape(self._alpha),
                                [array_ops.rank(self._alpha) - 1])

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
      with ops.op_scope([self._alpha, self._alpha_0], name):
        return self._alpha / array_ops.expand_dims(self._alpha_0, -1)

  def variance(self, name="variance"):
    """Variance of the distribution."""
    with ops.name_scope(self.name):
      with ops.op_scope([self._alpha, self._alpha_0], name):
        alpha = array_ops.expand_dims(self._alpha, -1)
        alpha_0 = array_ops.expand_dims(self._alpha_0, -1)

        expanded_alpha_0 = array_ops.expand_dims(alpha_0, -1)

        variance = -math_ops.batch_matmul(alpha, alpha, adj_y=True) / (
            expanded_alpha_0 ** 2 * (expanded_alpha_0 + 1))
        diagonal = self._alpha / (alpha_0 * (alpha_0 + 1))
        variance += array_ops.batch_matrix_diag(diagonal)
        return variance

  def std(self, name="std"):
    """Standard deviation of the distribution."""
    with ops.name_scope(self.name):
      with ops.op_scope([], name):
        return math_ops.sqrt(self.variance())

  def mode(self, name="mode"):
    """Mode of the distribution.

    Note that the mode for the Beta distribution is only defined
    when `alpha > 1`. This returns the mode when `alpha > 1`,
    and NaN otherwise. If `self.allow_nan_stats` is `False`, an exception
    will be raised rather than returning `NaN`.

    Args:
      name: The name for this op.

    Returns:
      Mode of the Dirichlet distribution.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._alpha, self._alpha_0], name):
        one = constant_op.constant(1, self.dtype)
        mode = (self._alpha - 1)/ (
            array_ops.expand_dims(self._alpha_0, -1) - math_ops.cast(
                self.event_shape()[0], self.dtype))

        if self.allow_nan_stats:
          return math_ops.select(
              math_ops.greater(self._alpha, 1),
              mode,
              (constant_op.constant(float("NaN"), dtype=self.dtype) *
               array_ops.ones_like(self._alpha, dtype=self.dtype)))
        else:
          return control_flow_ops.with_dependencies([
              check_ops.assert_less(
                  one, self._alpha,
                  message="mode not defined for components of alpha <= 1")
          ], mode)

  def entropy(self, name="entropy"):
    """Entropy of the distribution in nats."""
    with ops.name_scope(self.name):
      with ops.op_scope([self._alpha, self._alpha_0], name):
        alpha = self._alpha
        alpha_0 = self._alpha_0

        entropy = special_math_ops.lbeta(alpha)
        entropy += (alpha_0 - math_ops.cast(
            self.event_shape()[0], self.dtype)) * math_ops.digamma(
                alpha_0)
        entropy += -math_ops.reduce_sum(
            (alpha - 1) * math_ops.digamma(alpha),
            reduction_indices=[-1],
            keep_dims=False)
        return entropy

  def cdf(self, x, name="cdf"):
    """Cumulative distribution function."""
    raise NotImplementedError("Dirichlet does not have a well-defined cdf.")

  def log_cdf(self, x, name="log_cdf"):
    """Log CDF."""
    raise NotImplementedError("Dirichlet does not have a well-defined cdf.")

  def log_prob(self, x, name="log_prob"):
    """`Log(P[counts])`, computed for every batch member.

    Args:
      x:  Non-negative tensor with dtype `dtype` and whose shape can
        be broadcast with `self.alpha`.  For fixed leading dimensions, the last
        dimension represents counts for the corresponding Dirichlet distribution
        in `self.alpha`. `x` is only legal if it sums up to one.
      name:  Name to give this Op, defaults to "log_prob".

    Returns:
      Log probabilities for each record, shape `[N1,...,Nm]`.
    """
    alpha = self._alpha
    with ops.name_scope(self.name):
      with ops.op_scope([alpha, x], name):
        x = self._check_x(x)

        unnorm_prob = (alpha - 1) * math_ops.log(x)
        log_prob = math_ops.reduce_sum(
            unnorm_prob, reduction_indices=[-1],
            keep_dims=False) - special_math_ops.lbeta(alpha)

        return log_prob

  def prob(self, x, name="prob"):
    """`P[x]`, computed for every batch member.

    Args:
      x:  Non-negative tensor with dtype `dtype` and whose shape can
        be broadcast with `self.alpha`.  For fixed leading dimensions, the last
        dimension represents x for the corresponding Dirichlet distribution in
        `self.alpha` and `self.beta`. `x` is only legal if it sums up to one.
      name:  Name to give this Op, defaults to "prob".

    Returns:
      Probabilities for each record, shape `[N1,...,Nm]`.
    """
    return super(Dirichlet, self).prob(x, name=name)

  def sample_n(self, n, seed=None, name="sample_n"):
    """Sample `n` observations from the distributions.

    Args:
      n: `Scalar`, type int32, the number of observations to sample.
      seed: Python integer, the random seed.
      name: The name to give this op.

    Returns:
      samples: `[n, ...]`, a `Tensor` of `n` samples for each
        of the distributions determined by broadcasting the hyperparameters.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self.alpha, n], name):
        gamma_sample = random_ops.random_gamma(
            [n,], self.alpha, dtype=self.dtype, seed=seed)
        n_val = tensor_util.constant_value(n)
        final_shape = tensor_shape.vector(n_val).concatenate(
            self.alpha.get_shape())

        gamma_sample.set_shape(final_shape)
        return gamma_sample / math_ops.reduce_sum(
            gamma_sample, reduction_indices=[-1], keep_dims=True)

  @property
  def is_continuous(self):
    return True

  @property
  def is_reparameterized(self):
    return False

  def _check_x(self, x):
    """Check x for proper shape, values, then return tensor version."""
    x = ops.convert_to_tensor(x, name="x_before_deps")
    candidate_one = math_ops.reduce_sum(x, reduction_indices=[-1])
    one = constant_op.constant(1., self.dtype)
    dependencies = [check_ops.assert_positive(x), check_ops.assert_less(
        x, one, message="x has components greater than or equal to 1"),
                    distribution_util.assert_close(one, candidate_one)
                   ] if self.validate_args else []
    return control_flow_ops.with_dependencies(dependencies, x)
