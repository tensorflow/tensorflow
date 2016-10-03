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

import numpy as np

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import special_math_ops


_dirichlet_prob_note = """
Note that the input must be a non-negative tensor with dtype `dtype` and whose
shape can be broadcast with `self.alpha`.  For fixed leading dimensions, the
last dimension represents counts for the corresponding Dirichlet distribution
in `self.alpha`. `x` is only legal if it sums up to one.
"""


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
               validate_args=False,
               allow_nan_stats=True,
               name="Dirichlet"):
    """Initialize a batch of Dirichlet distributions.

    Args:
      alpha:  Positive floating point tensor with shape broadcastable to
        `[N1,..., Nm, k]` `m >= 0`.  Defines this as a batch of `N1 x ... x Nm`
         different `k` class Dirichlet distributions.
      validate_args: `Boolean`, default `False`.  Whether to assert valid values
        for parameters `alpha` and `x` in `prob` and `log_prob`.  If `False`,
        correct behavior is not guaranteed.
      allow_nan_stats: `Boolean`, default `True`.  If `False`, raise an
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
    with ops.name_scope(name, values=[alpha]) as ns:
      alpha = ops.convert_to_tensor(alpha, name="alpha")
      with ops.control_dependencies([
          check_ops.assert_positive(alpha),
          check_ops.assert_rank_at_least(alpha, 1)
      ] if validate_args else []):
        self._alpha = array_ops.identity(alpha, name="alpha")
        self._alpha_sum = math_ops.reduce_sum(alpha,
                                              reduction_indices=[-1],
                                              keep_dims=False)
        super(Dirichlet, self).__init__(
            dtype=self._alpha.dtype,
            parameters={"alpha": self._alpha, "alpha_sum": self._alpha_sum},
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            is_continuous=True,
            is_reparameterized=False,
            name=ns)

  @property
  def alpha(self):
    """Shape parameter."""
    return self._alpha

  @property
  def alpha_sum(self):
    """Sum of shape parameter."""
    return self._alpha_sum

  def _batch_shape(self):
    return array_ops.shape(self.alpha_sum)

  def _get_batch_shape(self):
    return self.alpha_sum.get_shape()

  def _event_shape(self):
    return array_ops.gather(array_ops.shape(self.alpha),
                            [array_ops.rank(self.alpha) - 1])

  def _get_event_shape(self):
    return self.alpha.get_shape().with_rank_at_least(1)[-1:]

  def _sample_n(self, n, seed=None):
    gamma_sample = random_ops.random_gamma(
        [n,], self.alpha, dtype=self.dtype, seed=seed)
    return gamma_sample / math_ops.reduce_sum(
        gamma_sample, reduction_indices=[-1], keep_dims=True)

  @distribution_util.AppendDocstring(_dirichlet_prob_note)
  def _log_prob(self, x):
    x = ops.convert_to_tensor(x, name="x")
    x = self._assert_valid_sample(x)
    unnorm_prob = (self.alpha - 1.) * math_ops.log(x)
    log_prob = math_ops.reduce_sum(
        unnorm_prob, reduction_indices=[-1],
        keep_dims=False) - special_math_ops.lbeta(self.alpha)
    return log_prob

  @distribution_util.AppendDocstring(_dirichlet_prob_note)
  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))

  def _entropy(self):
    entropy = special_math_ops.lbeta(self.alpha)
    entropy += math_ops.digamma(self.alpha_sum) * (
        self.alpha_sum - math_ops.cast(self.event_shape()[0], self.dtype))
    entropy += -math_ops.reduce_sum(
        (self.alpha - 1.) * math_ops.digamma(self.alpha),
        reduction_indices=[-1],
        keep_dims=False)
    return entropy

  def _mean(self):
    return self.alpha / array_ops.expand_dims(self.alpha_sum, -1)

  def _variance(self):
    scale = self.alpha_sum * math_ops.sqrt(1. + self.alpha_sum)
    alpha = self.alpha / scale
    outer_prod = -math_ops.batch_matmul(
        array_ops.expand_dims(alpha, dim=-1),  # column
        array_ops.expand_dims(alpha, dim=-2))  # row
    return array_ops.matrix_set_diag(outer_prod,
                                     alpha * (self.alpha_sum / scale - alpha))

  def _std(self):
    return math_ops.sqrt(self._variance())

  @distribution_util.AppendDocstring(
      """Note that the mode for the Dirichlet distribution is only defined
      when `alpha > 1`. This returns the mode when `alpha > 1`,
      and NaN otherwise. If `self.allow_nan_stats` is `False`, an exception
      will be raised rather than returning `NaN`.""")
  def _mode(self):
    mode = ((self.alpha - 1.) /
            (array_ops.expand_dims(self.alpha_sum, dim=-1) -
             math_ops.cast(self.event_shape()[0], self.dtype)))
    if self.allow_nan_stats:
      nan = np.array(np.nan, dtype=self.dtype.as_numpy_dtype())
      shape = array_ops.concat(0, (self.batch_shape(), self.event_shape()))
      return math_ops.select(
          math_ops.greater(self.alpha, 1.),
          mode,
          array_ops.fill(shape, nan, name="nan"))
    else:
      return control_flow_ops.with_dependencies([
          check_ops.assert_less(
              array_ops.ones((), dtype=self.dtype), self.alpha,
              message="mode not defined for components of alpha <= 1")
      ], mode)

  def _assert_valid_sample(self, x):
    if not self.validate_args: return x
    return control_flow_ops.with_dependencies([
        check_ops.assert_positive(x),
        distribution_util.assert_close(
            array_ops.ones((), dtype=self.dtype),
            math_ops.reduce_sum(x, reduction_indices=[-1])),
    ], x)
