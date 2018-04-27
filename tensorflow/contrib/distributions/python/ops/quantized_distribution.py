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
"""Quantized distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import distribution as distributions
from tensorflow.python.ops.distributions import util as distribution_util

__all__ = ["QuantizedDistribution"]


def _logsum_expbig_minus_expsmall(big, small):
  """Stable evaluation of `Log[exp{big} - exp{small}]`.

  To work correctly, we should have the pointwise relation:  `small <= big`.

  Args:
    big: Floating-point `Tensor`
    small: Floating-point `Tensor` with same `dtype` as `big` and broadcastable
      shape.

  Returns:
    `Tensor` of same `dtype` of `big` and broadcast shape.
  """
  with ops.name_scope("logsum_expbig_minus_expsmall", values=[small, big]):
    return math_ops.log(1. - math_ops.exp(small - big)) + big


_prob_base_note = """
For whole numbers `y`,

```
P[Y = y] := P[X <= low],  if y == low,
         := P[X > high - 1],  y == high,
         := 0, if j < low or y > high,
         := P[y - 1 < X <= y],  all other y.
```

"""

_prob_note = _prob_base_note + """
The base distribution's `cdf` method must be defined on `y - 1`. If the
base distribution has a `survival_function` method, results will be more
accurate for large values of `y`, and in this case the `survival_function` must
also be defined on `y - 1`.
"""

_log_prob_note = _prob_base_note + """
The base distribution's `log_cdf` method must be defined on `y - 1`. If the
base distribution has a `log_survival_function` method results will be more
accurate for large values of `y`, and in this case the `log_survival_function`
must also be defined on `y - 1`.
"""


_cdf_base_note = """

For whole numbers `y`,

```
cdf(y) := P[Y <= y]
        = 1, if y >= high,
        = 0, if y < low,
        = P[X <= y], otherwise.
```

Since `Y` only has mass at whole numbers, `P[Y <= y] = P[Y <= floor(y)]`.
This dictates that fractional `y` are first floored to a whole number, and
then above definition applies.
"""

_cdf_note = _cdf_base_note + """
The base distribution's `cdf` method must be defined on `y - 1`.
"""

_log_cdf_note = _cdf_base_note + """
The base distribution's `log_cdf` method must be defined on `y - 1`.
"""


_sf_base_note = """

For whole numbers `y`,

```
survival_function(y) := P[Y > y]
                      = 0, if y >= high,
                      = 1, if y < low,
                      = P[X <= y], otherwise.
```

Since `Y` only has mass at whole numbers, `P[Y <= y] = P[Y <= floor(y)]`.
This dictates that fractional `y` are first floored to a whole number, and
then above definition applies.
"""

_sf_note = _sf_base_note + """
The base distribution's `cdf` method must be defined on `y - 1`.
"""

_log_sf_note = _sf_base_note + """
The base distribution's `log_cdf` method must be defined on `y - 1`.
"""


class QuantizedDistribution(distributions.Distribution):
  """Distribution representing the quantization `Y = ceiling(X)`.

  #### Definition in terms of sampling.

  ```
  1. Draw X
  2. Set Y <-- ceiling(X)
  3. If Y < low, reset Y <-- low
  4. If Y > high, reset Y <-- high
  5. Return Y
  ```

  #### Definition in terms of the probability mass function.

  Given scalar random variable `X`, we define a discrete random variable `Y`
  supported on the integers as follows:

  ```
  P[Y = j] := P[X <= low],  if j == low,
           := P[X > high - 1],  j == high,
           := 0, if j < low or j > high,
           := P[j - 1 < X <= j],  all other j.
  ```

  Conceptually, without cutoffs, the quantization process partitions the real
  line `R` into half open intervals, and identifies an integer `j` with the
  right endpoints:

  ```
  R = ... (-2, -1](-1, 0](0, 1](1, 2](2, 3](3, 4] ...
  j = ...      -1      0     1     2     3     4  ...
  ```

  `P[Y = j]` is the mass of `X` within the `jth` interval.
  If `low = 0`, and `high = 2`, then the intervals are redrawn
  and `j` is re-assigned:

  ```
  R = (-infty, 0](0, 1](1, infty)
  j =          0     1     2
  ```

  `P[Y = j]` is still the mass of `X` within the `jth` interval.

  #### Caveats

  Since evaluation of each `P[Y = j]` involves a cdf evaluation (rather than
  a closed form function such as for a Poisson), computations such as mean and
  entropy are better done with samples or approximations, and are not
  implemented by this class.
  """

  def __init__(self,
               distribution,
               low=None,
               high=None,
               validate_args=False,
               name="QuantizedDistribution"):
    """Construct a Quantized Distribution representing `Y = ceiling(X)`.

    Some properties are inherited from the distribution defining `X`. Example:
    `allow_nan_stats` is determined for this `QuantizedDistribution` by reading
    the `distribution`.

    Args:
      distribution:  The base distribution class to transform. Typically an
        instance of `Distribution`.
      low: `Tensor` with same `dtype` as this distribution and shape
        able to be added to samples. Should be a whole number. Default `None`.
        If provided, base distribution's `prob` should be defined at
        `low`.
      high: `Tensor` with same `dtype` as this distribution and shape
        able to be added to samples. Should be a whole number. Default `None`.
        If provided, base distribution's `prob` should be defined at
        `high - 1`.
        `high` must be strictly greater than `low`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: If `dist_cls` is not a subclass of
          `Distribution` or continuous.
      NotImplementedError:  If the base distribution does not implement `cdf`.
    """
    parameters = locals()
    values = (
        list(distribution.parameters.values()) +
        [low, high])
    with ops.name_scope(name, values=values) as name:
      self._dist = distribution

      if low is not None:
        low = ops.convert_to_tensor(low, name="low")
      if high is not None:
        high = ops.convert_to_tensor(high, name="high")
      check_ops.assert_same_float_dtype(
          tensors=[self.distribution, low, high])

      # We let QuantizedDistribution access _graph_parents since this class is
      # more like a baseclass.
      graph_parents = self._dist._graph_parents  # pylint: disable=protected-access

      checks = []
      if validate_args and low is not None and high is not None:
        message = "low must be strictly less than high."
        checks.append(
            check_ops.assert_less(
                low, high, message=message))
      self._validate_args = validate_args  # self._check_integer uses this.
      with ops.control_dependencies(checks if validate_args else []):
        if low is not None:
          self._low = self._check_integer(low)
          graph_parents += [self._low]
        else:
          self._low = None
        if high is not None:
          self._high = self._check_integer(high)
          graph_parents += [self._high]
        else:
          self._high = None

    super(QuantizedDistribution, self).__init__(
        dtype=self._dist.dtype,
        reparameterization_type=distributions.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=self._dist.allow_nan_stats,
        parameters=parameters,
        graph_parents=graph_parents,
        name=name)

  def _batch_shape_tensor(self):
    return self.distribution.batch_shape_tensor()

  def _batch_shape(self):
    return self.distribution.batch_shape

  def _event_shape_tensor(self):
    return self.distribution.event_shape_tensor()

  def _event_shape(self):
    return self.distribution.event_shape

  def _sample_n(self, n, seed=None):
    low = self._low
    high = self._high
    with ops.name_scope("transform"):
      n = ops.convert_to_tensor(n, name="n")
      x_samps = self.distribution.sample(n, seed=seed)
      ones = array_ops.ones_like(x_samps)

      # Snap values to the intervals (j - 1, j].
      result_so_far = math_ops.ceil(x_samps)

      if low is not None:
        result_so_far = array_ops.where(result_so_far < low,
                                        low * ones, result_so_far)

      if high is not None:
        result_so_far = array_ops.where(result_so_far > high,
                                        high * ones, result_so_far)

      return result_so_far

  @distribution_util.AppendDocstring(_log_prob_note)
  def _log_prob(self, y):
    if not hasattr(self.distribution, "_log_cdf"):
      raise NotImplementedError(
          "'log_prob' not implemented unless the base distribution implements "
          "'log_cdf'")
    y = self._check_integer(y)
    try:
      return self._log_prob_with_logsf_and_logcdf(y)
    except NotImplementedError:
      return self._log_prob_with_logcdf(y)

  def _log_prob_with_logcdf(self, y):
    return _logsum_expbig_minus_expsmall(self.log_cdf(y), self.log_cdf(y - 1))

  def _log_prob_with_logsf_and_logcdf(self, y):
    """Compute log_prob(y) using log survival_function and cdf together."""
    # There are two options that would be equal if we had infinite precision:
    # Log[ sf(y - 1) - sf(y) ]
    #   = Log[ exp{logsf(y - 1)} - exp{logsf(y)} ]
    # Log[ cdf(y) - cdf(y - 1) ]
    #   = Log[ exp{logcdf(y)} - exp{logcdf(y - 1)} ]
    logsf_y = self.log_survival_function(y)
    logsf_y_minus_1 = self.log_survival_function(y - 1)
    logcdf_y = self.log_cdf(y)
    logcdf_y_minus_1 = self.log_cdf(y - 1)

    # Important:  Here we use select in a way such that no input is inf, this
    # prevents the troublesome case where the output of select can be finite,
    # but the output of grad(select) will be NaN.

    # In either case, we are doing Log[ exp{big} - exp{small} ]
    # We want to use the sf items precisely when we are on the right side of the
    # median, which occurs when logsf_y < logcdf_y.
    big = array_ops.where(logsf_y < logcdf_y, logsf_y_minus_1, logcdf_y)
    small = array_ops.where(logsf_y < logcdf_y, logsf_y, logcdf_y_minus_1)

    return _logsum_expbig_minus_expsmall(big, small)

  @distribution_util.AppendDocstring(_prob_note)
  def _prob(self, y):
    if not hasattr(self.distribution, "_cdf"):
      raise NotImplementedError(
          "'prob' not implemented unless the base distribution implements "
          "'cdf'")
    y = self._check_integer(y)
    try:
      return self._prob_with_sf_and_cdf(y)
    except NotImplementedError:
      return self._prob_with_cdf(y)

  def _prob_with_cdf(self, y):
    return self.cdf(y) - self.cdf(y - 1)

  def _prob_with_sf_and_cdf(self, y):
    # There are two options that would be equal if we had infinite precision:
    # sf(y - 1) - sf(y)
    # cdf(y) - cdf(y - 1)
    sf_y = self.survival_function(y)
    sf_y_minus_1 = self.survival_function(y - 1)
    cdf_y = self.cdf(y)
    cdf_y_minus_1 = self.cdf(y - 1)

    # sf_prob has greater precision iff we're on the right side of the median.
    return array_ops.where(
        sf_y < cdf_y,  # True iff we're on the right side of the median.
        sf_y_minus_1 - sf_y,
        cdf_y - cdf_y_minus_1)

  @distribution_util.AppendDocstring(_log_cdf_note)
  def _log_cdf(self, y):
    low = self._low
    high = self._high

    # Recall the promise:
    # cdf(y) := P[Y <= y]
    #         = 1, if y >= high,
    #         = 0, if y < low,
    #         = P[X <= y], otherwise.

    # P[Y <= j] = P[floor(Y) <= j] since mass is only at integers, not in
    # between.
    j = math_ops.floor(y)

    result_so_far = self.distribution.log_cdf(j)

    # Broadcast, because it's possible that this is a single distribution being
    # evaluated on a number of samples, or something like that.
    j += array_ops.zeros_like(result_so_far)

    # Re-define values at the cutoffs.
    if low is not None:
      neg_inf = -np.inf * array_ops.ones_like(result_so_far)
      result_so_far = array_ops.where(j < low, neg_inf, result_so_far)
    if high is not None:
      result_so_far = array_ops.where(j >= high,
                                      array_ops.zeros_like(result_so_far),
                                      result_so_far)

    return result_so_far

  @distribution_util.AppendDocstring(_cdf_note)
  def _cdf(self, y):
    low = self._low
    high = self._high

    # Recall the promise:
    # cdf(y) := P[Y <= y]
    #         = 1, if y >= high,
    #         = 0, if y < low,
    #         = P[X <= y], otherwise.

    # P[Y <= j] = P[floor(Y) <= j] since mass is only at integers, not in
    # between.
    j = math_ops.floor(y)

    # P[X <= j], used when low < X < high.
    result_so_far = self.distribution.cdf(j)

    # Broadcast, because it's possible that this is a single distribution being
    # evaluated on a number of samples, or something like that.
    j += array_ops.zeros_like(result_so_far)

    # Re-define values at the cutoffs.
    if low is not None:
      result_so_far = array_ops.where(j < low,
                                      array_ops.zeros_like(result_so_far),
                                      result_so_far)
    if high is not None:
      result_so_far = array_ops.where(j >= high,
                                      array_ops.ones_like(result_so_far),
                                      result_so_far)

    return result_so_far

  @distribution_util.AppendDocstring(_log_sf_note)
  def _log_survival_function(self, y):
    low = self._low
    high = self._high

    # Recall the promise:
    # survival_function(y) := P[Y > y]
    #                       = 0, if y >= high,
    #                       = 1, if y < low,
    #                       = P[X > y], otherwise.

    # P[Y > j] = P[ceiling(Y) > j] since mass is only at integers, not in
    # between.
    j = math_ops.ceil(y)

    # P[X > j], used when low < X < high.
    result_so_far = self.distribution.log_survival_function(j)

    # Broadcast, because it's possible that this is a single distribution being
    # evaluated on a number of samples, or something like that.
    j += array_ops.zeros_like(result_so_far)

    # Re-define values at the cutoffs.
    if low is not None:
      result_so_far = array_ops.where(j < low,
                                      array_ops.zeros_like(result_so_far),
                                      result_so_far)
    if high is not None:
      neg_inf = -np.inf * array_ops.ones_like(result_so_far)
      result_so_far = array_ops.where(j >= high, neg_inf, result_so_far)

    return result_so_far

  @distribution_util.AppendDocstring(_sf_note)
  def _survival_function(self, y):
    low = self._low
    high = self._high

    # Recall the promise:
    # survival_function(y) := P[Y > y]
    #                       = 0, if y >= high,
    #                       = 1, if y < low,
    #                       = P[X > y], otherwise.

    # P[Y > j] = P[ceiling(Y) > j] since mass is only at integers, not in
    # between.
    j = math_ops.ceil(y)

    # P[X > j], used when low < X < high.
    result_so_far = self.distribution.survival_function(j)

    # Broadcast, because it's possible that this is a single distribution being
    # evaluated on a number of samples, or something like that.
    j += array_ops.zeros_like(result_so_far)

    # Re-define values at the cutoffs.
    if low is not None:
      result_so_far = array_ops.where(j < low,
                                      array_ops.ones_like(result_so_far),
                                      result_so_far)
    if high is not None:
      result_so_far = array_ops.where(j >= high,
                                      array_ops.zeros_like(result_so_far),
                                      result_so_far)

    return result_so_far

  def _check_integer(self, value):
    with ops.name_scope("check_integer", values=[value]):
      value = ops.convert_to_tensor(value, name="value")
      if not self.validate_args:
        return value
      dependencies = [distribution_util.assert_integer_form(
          value, message="value has non-integer components.")]
      return control_flow_ops.with_dependencies(dependencies, value)

  @property
  def distribution(self):
    """Base distribution, p(x)."""
    return self._dist
