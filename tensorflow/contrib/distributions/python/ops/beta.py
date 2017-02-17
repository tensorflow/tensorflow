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
"""The Beta distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.contrib.distributions.python.ops import kullback_leibler
from tensorflow.contrib.framework.python.framework import tensor_util as contrib_tensor_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops


__all__ = [
    "Beta",
    "BetaWithSoftplusConcentration",
]


_beta_sample_note = """Note: `x` must have dtype `self.dtype` and be in
`[0, 1].` It must have a shape compatible with `self.batch_shape()`."""


class Beta(distribution.Distribution):
  """Beta distribution.

  The Beta distribution is defined over the `(0, 1)` interval using parameters
  `concentration1` (aka "alpha") and `concentration0` (aka "beta").

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; alpha, beta) = x**(alpha - 1) (1 - x)**(beta - 1) / Z
  Z = Gamma(alpha) Gamma(beta) / Gamma(alpha + beta)
  ```

  where:

  * `concentration1 = alpha`,
  * `concentration0 = beta`,
  * `Z` is the normalization constant, and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  The concentration parameters represent mean total counts of a `1` or a `0`,
  i.e.,

  ```none
  concentration1 = alpha = mean * total_concentration
  concentration0 = beta  = (1. - mean) * total_concentration
  ```

  where `mean` in `(0, 1)` and `total_concentration` is a positive real number
  representing a mean `total_count = concentration1 + concentration0`.

  Distribution parameters are automatically broadcast in all functions; see
  examples for details.

  #### Examples

  ```python
  # Create a batch of three Beta distributions.
  alpha = [1, 2, 3]
  beta = [1, 2, 3]
  dist = Beta(alpha, beta)

  dist.sample([4, 5])  # Shape [4, 5, 3]

  # `x` has three batch entries, each with two samples.
  x = [[.1, .4, .5],
       [.2, .3, .5]]
  # Calculate the probability of each pair of samples under the corresponding
  # distribution in `dist`.
  dist.prob(x)         # Shape [2, 3]
  ```

  ```python
  # Create batch_shape=[2, 3] via parameter broadcast:
  alpha = [[1.], [2]]      # Shape [2, 1]
  beta = [3., 4, 5]        # Shape [3]
  dist = Beta(alpha, beta)

  # alpha broadcast as: [[1., 1, 1,],
  #                      [2, 2, 2]]
  # beta broadcast as:  [[3., 4, 5],
  #                      [3, 4, 5]]
  # batch_Shape [2, 3]
  dist.sample([4, 5])  # Shape [4, 5, 2, 3]

  x = [.2, .3, .5]
  # x will be broadcast as [[.2, .3, .5],
  #                         [.2, .3, .5]],
  # thus matching batch_shape [2, 3].
  dist.prob(x)         # Shape [2, 3]
  ```

  """

  def __init__(self,
               concentration1=None,
               concentration0=None,
               validate_args=False,
               allow_nan_stats=True,
               name="Beta"):
    """Initialize a batch of Beta distributions.

    Args:
      concentration1: Positive floating-point `Tensor` indicating mean
        number of successes; aka "alpha". Implies `self.dtype` and
        `self.batch_shape`, i.e.,
        `concentration1.shape = [N1, N2, ..., Nm] = self.batch_shape`.
      concentration0: Positive floating-point `Tensor` indicating mean
        number of failures; aka "beta". Otherwise has same semantics as
        `concentration1`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = locals()
    with ops.name_scope(name, values=[concentration1,
                                      concentration0]) as ns:
      self._concentration1 = self._maybe_assert_valid_concentration(
          ops.convert_to_tensor(concentration1, name="concentration1"),
          validate_args)
      self._concentration0 = self._maybe_assert_valid_concentration(
          ops.convert_to_tensor(concentration0, name="concentration0"),
          validate_args)
      contrib_tensor_util.assert_same_float_dtype([
          self._concentration1, self._concentration0])
      self._total_concentration = self._concentration1 + self._concentration0
    super(Beta, self).__init__(
        dtype=self._total_concentration.dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        is_continuous=True,
        reparameterization_type=distribution.NOT_REPARAMETERIZED,
        parameters=parameters,
        graph_parents=[self._concentration1,
                       self._concentration0,
                       self._total_concentration],
        name=ns)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(zip(
        ["concentration1", "concentration0"],
        [ops.convert_to_tensor(sample_shape, dtype=dtypes.int32)] * 2))

  @property
  def concentration1(self):
    """Concentration parameter associated with a `1` outcome."""
    return self._concentration1

  @property
  def concentration0(self):
    """Concentration parameter associated with a `0` outcome."""
    return self._concentration0

  @property
  def total_concentration(self):
    """Sum of concentration parameters."""
    return self._total_concentration

  def _batch_shape_tensor(self):
    return array_ops.shape(self.total_concentration)

  def _batch_shape(self):
    return self.total_concentration.get_shape()

  def _event_shape_tensor(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    expanded_concentration1 = array_ops.ones_like(
        self.total_concentration, dtype=self.dtype) * self.concentration1
    expanded_concentration0 = array_ops.ones_like(
        self.total_concentration, dtype=self.dtype) * self.concentration0
    gamma1_sample = random_ops.random_gamma(
        shape=[n],
        alpha=expanded_concentration1,
        dtype=self.dtype,
        seed=seed)
    gamma2_sample = random_ops.random_gamma(
        shape=[n],
        alpha=expanded_concentration0,
        dtype=self.dtype,
        seed=distribution_util.gen_new_seed(seed, "beta"))
    beta_sample = gamma1_sample / (gamma1_sample + gamma2_sample)
    return beta_sample

  @distribution_util.AppendDocstring(_beta_sample_note)
  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()

  @distribution_util.AppendDocstring(_beta_sample_note)
  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))

  @distribution_util.AppendDocstring(_beta_sample_note)
  def _log_cdf(self, x):
    return math_ops.log(self._cdf(x))

  @distribution_util.AppendDocstring(_beta_sample_note)
  def _cdf(self, x):
    return math_ops.betainc(self.concentration1, self.concentration0, x)

  def _log_unnormalized_prob(self, x):
    x = self._maybe_assert_valid_sample(x)
    return ((self.concentration1 - 1.) * math_ops.log(x)
            + (self.concentration0 - 1.) * math_ops.log1p(-x))

  def _log_normalization(self):
    return (math_ops.lgamma(self.concentration1)
            + math_ops.lgamma(self.concentration0)
            - math_ops.lgamma(self.total_concentration))

  def _entropy(self):
    return (
        self._log_normalization()
        - (self.concentration1 - 1.) * math_ops.digamma(self.concentration1)
        - (self.concentration0 - 1.) * math_ops.digamma(self.concentration0)
        + ((self.total_concentration - 2.) *
           math_ops.digamma(self.total_concentration)))

  def _mean(self):
    return self._concentration1 / self._total_concentration

  def _variance(self):
    return self._mean() * (1. - self._mean()) / (1. + self.total_concentration)

  @distribution_util.AppendDocstring(
      """Note: The mode is undefined when `concentration1 <= 1` or
      `concentration0 <= 1`. If `self.allow_nan_stats` is `True`, `NaN`
      is used for undefined modes. If `self.allow_nan_stats` is `False` an
      exception is raised when one or more modes are undefined.""")
  def _mode(self):
    mode = (self.concentration1 - 1.) / (self.total_concentration - 2.)
    if self.allow_nan_stats:
      nan = array_ops.fill(
          self.batch_shape_tensor(),
          np.array(np.nan, dtype=self.dtype.as_numpy_dtype()),
          name="nan")
      is_defined = math_ops.logical_and(self.concentration1 > 1.,
                                        self.concentration0 > 1.)
      return array_ops.where(is_defined, mode, nan)
    return control_flow_ops.with_dependencies([
        check_ops.assert_less(
            array_ops.ones([], dtype=self.dtype),
            self.concentration1,
            message="Mode undefined for concentration1 <= 1."),
        check_ops.assert_less(
            array_ops.ones([], dtype=self.dtype),
            self.concentration0,
            message="Mode undefined for concentration0 <= 1.")
    ], mode)

  def _maybe_assert_valid_concentration(self, concentration, validate_args):
    """Checks the validity of a concentration parameter."""
    if not validate_args:
      return concentration
    return control_flow_ops.with_dependencies([
        check_ops.assert_positive(
            concentration,
            message="Concentration parameter must be positive."),
    ], concentration)

  def _maybe_assert_valid_sample(self, x):
    """Checks the validity of a sample."""
    if not self.validate_args:
      return x
    return control_flow_ops.with_dependencies([
        check_ops.assert_positive(
            x,
            message="sample must be positive"),
        check_ops.assert_less(
            x, array_ops.ones([], self.dtype),
            message="sample must be no larger than `1`."),
    ], x)


class BetaWithSoftplusConcentration(Beta):
  """Beta with softplus transform of `concentration1` and `concentration0`."""

  def __init__(self,
               concentration1,
               concentration0,
               validate_args=False,
               allow_nan_stats=True,
               name="BetaWithSoftplusConcentration"):
    parameters = locals()
    with ops.name_scope(name, values=[concentration1,
                                      concentration0]) as ns:
      super(BetaWithSoftplusConcentration, self).__init__(
          concentration1=nn.softplus(concentration1,
                                     name="softplus_concentration1"),
          concentration0=nn.softplus(concentration0,
                                     name="softplus_concentration0"),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=ns)
    self._parameters = parameters


@kullback_leibler.RegisterKL(Beta, Beta)
def _kl_beta_beta(d1, d2, name=None):
  """Calculate the batchwise KL divergence KL(d1 || d2) with d1 and d2 Beta.

  Args:
    d1: instance of a Beta distribution object.
    d2: instance of a Beta distribution object.
    name: (optional) Name to use for created operations.
      default is "kl_beta_beta".

  Returns:
    Batchwise KL(d1 || d2)
  """
  def delta(fn, is_property=True):
    fn1 = getattr(d1, fn)
    fn2 = getattr(d2, fn)
    return (fn2 - fn1) if is_property else (fn2() - fn1())
  with ops.name_scope(name, "kl_beta_beta", values=[
      d1.concentration1,
      d1.concentration0,
      d1.total_concentration,
      d2.concentration1,
      d2.concentration0,
      d2.total_concentration,
  ]):
    return (delta("_log_normalization", is_property=False)
            - math_ops.digamma(d1.concentration1) * delta("concentration1")
            - math_ops.digamma(d1.concentration0) * delta("concentration0")
            + (math_ops.digamma(d1.total_concentration)
               * delta("total_concentration")))
