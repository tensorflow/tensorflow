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
"""The InverseGamma distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import distribution  # pylint: disable=line-too-long
from tensorflow.contrib.framework.python.framework import tensor_util as contrib_tensor_util  # pylint: disable=line-too-long
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


class InverseGamma(distribution.Distribution):
  """The `InverseGamma` distribution with parameter alpha and beta.

  The parameters are the shape and inverse scale parameters alpha, beta.

  The PDF of this distribution is:

  ```pdf(x) = (beta^alpha)/Gamma(alpha)(x^(-alpha-1))e^(-beta/x), x > 0```

  and the CDF of this distribution is:

  ```cdf(x) =  GammaInc(alpha, beta / x) / Gamma(alpha), x > 0```

  where GammaInc is the upper incomplete Gamma function.

  Examples:

  ```python
  dist = InverseGamma(alpha=3.0, beta=2.0)
  dist2 = InverseGamma(alpha=[3.0, 4.0], beta=[2.0, 3.0])
  ```

  """

  def __init__(self,
               alpha,
               beta,
               validate_args=True,
               allow_nan_stats=False,
               name="InverseGamma"):
    """Construct InverseGamma distributions with parameters `alpha` and `beta`.

    The parameters `alpha` and `beta` must be shaped in a way that supports
    broadcasting (e.g. `alpha + beta` is a valid operation).

    Args:
      alpha: Floating point tensor, the shape params of the
        distribution(s).
        alpha must contain only positive values.
      beta: Floating point tensor, the scale params of the distribution(s).
        beta must contain only positive values.
      validate_args: Whether to assert that `a > 0, b > 0`, and that `x > 0` in
        the methods `prob(x)` and `log_prob(x)`.  If `validate_args` is `False`
        and the inputs are invalid, correct behavior is not guaranteed.
      allow_nan_stats:  Boolean, default `False`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member.  If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to prepend to all ops created by this distribution.

    Raises:
      TypeError: if `alpha` and `beta` are different dtypes.
    """
    self._allow_nan_stats = allow_nan_stats
    self._validate_args = validate_args
    with ops.op_scope([alpha, beta], name) as scope:
      self._name = scope
      with ops.control_dependencies([check_ops.assert_positive(
          alpha), check_ops.assert_positive(beta)] if validate_args else []):
        alpha = array_ops.identity(alpha, name="alpha")
        beta = array_ops.identity(beta, name="beta")

        contrib_tensor_util.assert_same_float_dtype((alpha, beta))
        self._broadcast_tensor = alpha + beta

    self._get_batch_shape = self._broadcast_tensor.get_shape()
    self._get_event_shape = tensor_shape.TensorShape([])

    self._alpha = alpha
    self._beta = beta

  @property
  def allow_nan_stats(self):
    """Boolean describing behavior when a stat is undefined for batch member."""
    return self._allow_nan_stats

  @property
  def validate_args(self):
    """Boolean describing behavior on invalid input."""
    return self._validate_args

  @property
  def name(self):
    """Name to prepend to all ops."""
    return self._name

  @property
  def dtype(self):
    """dtype of samples from this distribution."""
    return self._alpha.dtype

  @property
  def alpha(self):
    """Shape parameter."""
    return self._alpha

  @property
  def beta(self):
    """Scale parameter."""
    return self._beta

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
      with ops.op_scope([self._broadcast_tensor], name):
        return array_ops.shape(self._broadcast_tensor)

  def get_batch_shape(self):
    """`TensorShape` available at graph construction time.

    Same meaning as `batch_shape`. May be only partially defined.

    Returns:
      `TensorShape` object.
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
        return constant_op.constant([], dtype=dtypes.int32)

  def get_event_shape(self):
    """`TensorShape` available at graph construction time.

    Same meaning as `event_shape`. May be only partially defined.

    Returns:
      `TensorShape` object.
    """
    return self._get_event_shape

  def mean(self, name="mean"):
    """Mean of each batch member.

    The mean of an inverse gamma distribution is `beta / (alpha - 1)`,
    when `alpha > 1`, and `NaN` otherwise.  If `self.allow_nan_stats` is
    `False`, an exception will be raised rather than returning `NaN`

    Args:
      name: A name to give this op.

    Returns:
      The mean for every batch member, a `Tensor` with same `dtype` as self.
    """
    alpha = self._alpha
    beta = self._beta
    with ops.name_scope(self.name):
      with ops.op_scope([alpha, beta], name):
        mean_if_defined = beta / (alpha - 1.0)
        if self.allow_nan_stats:
          alpha_gt_1 = alpha > 1.0
          nan = np.nan * self._ones()
          return math_ops.select(alpha_gt_1, mean_if_defined, nan)
        else:
          one = constant_op.constant(1.0, dtype=self.dtype)
          return control_flow_ops.with_dependencies(
              [check_ops.assert_less(
                  one, alpha,
                  message="mean not defined for components of alpha <= 1")],
              mean_if_defined)

  def mode(self, name="mode"):
    """Mode of each batch member.

    The mode of an inverse gamma distribution is `beta / (alpha + 1)`.

    Args:
      name: A name to give this op.

    Returns:
      The mode for every batch member, a `Tensor` with same `dtype` as self.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._alpha, self._beta], name):
        return self._beta / (self._alpha + 1.0)

  def variance(self, name="variance"):
    """Variance of each batch member.

    Variance for inverse gamma is defined only for `alpha > 2`. If
    `self.allow_nan_stats` is `False`, an exception will be raised rather
    than returning `NaN`.

    Args:
      name: A name to give this op.

    Returns:
      The variance for every batch member, a `Tensor` with same `dtype` as self.
    """
    alpha = self._alpha
    beta = self._beta
    with ops.name_scope(self.name):
      with ops.op_scope([alpha, beta], name):
        var_if_defined = (math_ops.square(self._beta) /
                          (math_ops.square(self._alpha - 1.0) *
                           (self._alpha - 2.0)))
        if self.allow_nan_stats:
          alpha_gt_2 = alpha > 2.0
          nan = np.nan * self._ones()
          return math_ops.select(alpha_gt_2, var_if_defined, nan)
        else:
          two = constant_op.constant(2.0, dtype=self.dtype)
          return control_flow_ops.with_dependencies(
              [check_ops.assert_less(
                  two, alpha,
                  message="variance not defined for components of alpha <= 2")],
              var_if_defined)

  def log_prob(self, x, name="log_prob"):
    """Log prob of observations in `x` under these InverseGamma distribution(s).

    Args:
      x: tensor of dtype `dtype`, must be broadcastable with `alpha` and `beta`.
      name: The name to give this op.

    Returns:
      log_prob: tensor of dtype `dtype`, the log-PDFs of `x`.

    Raises:
      TypeError: if `x` and `alpha` are different dtypes.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._alpha, self._beta, x], name):
        alpha = self._alpha
        beta = self._beta
        x = ops.convert_to_tensor(x)
        x = control_flow_ops.with_dependencies([check_ops.assert_positive(x)] if
                                               self.validate_args else [], x)
        contrib_tensor_util.assert_same_float_dtype(tensors=[x,],
                                                    dtype=self.dtype)

        return (alpha * math_ops.log(beta) - math_ops.lgamma(self._alpha) -
                (alpha + 1) * math_ops.log(x) - beta / x)

  def prob(self, x, name="prob"):
    """Pdf of observations in `x` under these Gamma distribution(s).

    Args:
      x: tensor of dtype `dtype`, must be broadcastable with `alpha` and `beta`.
      name: The name to give this op.

    Returns:
      prob: tensor of dtype `dtype`, the PDFs of `x`

    Raises:
      TypeError: if `x` and `alpha` are different dtypes.
    """
    return super(InverseGamma, self).prob(x, name)

  def log_cdf(self, x, name="log_cdf"):
    """Log CDF of observations `x` under these InverseGamma distribution(s).

    Args:
      x: tensor of dtype `dtype`, must be broadcastable with `alpha` and `beta`.
      name: The name to give this op.

    Returns:
      log_cdf: tensor of dtype `dtype`, the log-CDFs of `x`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._alpha, self._beta, x], name):
        x = ops.convert_to_tensor(x)
        x = control_flow_ops.with_dependencies([check_ops.assert_positive(x)] if
                                               self.validate_args else [], x)
        contrib_tensor_util.assert_same_float_dtype(tensors=[x,],
                                                    dtype=self.dtype)
        # Note that igammac returns the upper regularized incomplete gamma
        # function Q(a, x), which is what we want for the CDF.
        return math_ops.log(math_ops.igammac(self._alpha, self._beta / x))

  def cdf(self, x, name="cdf"):
    """CDF of observations `x` under these InverseGamma distribution(s).

    Args:
      x: tensor of dtype `dtype`, must be broadcastable with `alpha` and `beta`.
      name: The name to give this op.

    Returns:
      cdf: tensor of dtype `dtype`, the CDFs of `x`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._alpha, self._beta, x], name):
        return math_ops.igammac(self._alpha, self._beta / x)

  def entropy(self, name="entropy"):
    """The entropy of these InverseGamma distribution(s).

    This is defined to be

    ```
    entropy = alpha - log(beta) + log(Gamma(alpha))
                 + (1-alpha)digamma(alpha)
    ```

    where digamma(alpha) is the digamma function.

    Args:
      name: The name to give this op.

    Returns:
      entropy: tensor of dtype `dtype`, the entropy.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._alpha, self._beta], name):
        alpha = self._alpha
        beta = self._beta
        return (alpha + math_ops.log(beta) + math_ops.lgamma(alpha) -
                (1 + alpha) * math_ops.digamma(alpha))

  def sample_n(self, n, seed=None, name="sample_n"):
    """Draws `n` samples from these InverseGamma distribution(s).

    See the doc for tf.random_gamma for further details on sampling strategy.

    Args:
      n: Python integer, the number of observations to sample from each
        distribution.
      seed: Python integer, the random seed for this operation.
      name: Optional name for the operation.

    Returns:
      samples: a `Tensor` of shape `(n,) + self.batch_shape + self.event_shape`
          with values of type `self.dtype`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([n, self._alpha, self._beta], name):
        one = constant_op.constant(1.0, dtype=self.dtype)
        return one / random_ops.random_gamma([n],
                                             self._alpha,
                                             beta=self._beta,
                                             dtype=self.dtype,
                                             seed=seed)

  @property
  def is_reparameterized(self):
    return False

  def _ones(self):
    return array_ops.ones_like(self._alpha + self._beta, dtype=self.dtype)

  @property
  def is_continuous(self):
    return True
