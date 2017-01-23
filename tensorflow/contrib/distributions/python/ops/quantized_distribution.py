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

from tensorflow.contrib.distributions.python.ops import distribution as distributions
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.contrib.framework.python.framework import tensor_util as contrib_tensor_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

__all__ = ["QuantizedDistribution"]


def _logsum_expbig_minus_expsmall(big, small):
  """Stable evaluation of `Log[exp{big} - exp{small}]`.

  To work correctly, we should have the pointwise relation:  `small <= big`.

  Args:
    big: Numeric `Tensor`
    small: Numeric `Tensor` with same `dtype` as `big` and broadcastable shape.

  Returns:
    `Tensor` of same `dtype` of `big` and broadcast shape.
  """
  with ops.name_scope("logsum_expbig_minus_expsmall", values=[small, big]):
    return math_ops.log(1. - math_ops.exp(small - big)) + big


_prob_base_note = """
For whole numbers `y`,

```
P[Y = y] := P[X <= lower_cutoff],  if y == lower_cutoff,
         := P[X > upper_cutoff - 1],  y == upper_cutoff,
         := 0, if j < lower_cutoff or y > upper_cutoff,
         := P[y - 1 < X <= y],  all other y.
```

"""

_prob_note = _prob_base_note + """
The base distribution's `cdf` method must be defined on `y - 1`.  If the
base distribution has a `survival_function` method, results will be more
accurate for large values of `y`, and in this case the `survival_function` must
also be defined on `y - 1`.
"""

_log_prob_note = _prob_base_note + """
The base distribution's `log_cdf` method must be defined on `y - 1`.  If the
base distribution has a `log_survival_function` method results will be more
accurate for large values of `y`, and in this case the `log_survival_function`
must also be defined on `y - 1`.
"""


_cdf_base_note = """

For whole numbers `y`,

```
cdf(y) := P[Y <= y]
        = 1, if y >= upper_cutoff,
        = 0, if y < lower_cutoff,
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
                      = 0, if y >= upper_cutoff,
                      = 1, if y < lower_cutoff,
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
  3. If Y < lower_cutoff, reset Y <-- lower_cutoff
  4. If Y > upper_cutoff, reset Y <-- upper_cutoff
  5. Return Y
  ```

  #### Definition in terms of the probability mass function.

  Given scalar random variable `X`, we define a discrete random variable `Y`
  supported on the integers as follows:

  ```
  P[Y = j] := P[X <= lower_cutoff],  if j == lower_cutoff,
           := P[X > upper_cutoff - 1],  j == upper_cutoff,
           := 0, if j < lower_cutoff or j > upper_cutoff,
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
  If `lower_cutoff = 0`, and `upper_cutoff = 2`, then the intervals are redrawn
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
               lower_cutoff=None,
               upper_cutoff=None,
               validate_args=False,
               name="QuantizedDistribution"):
    """Construct a Quantized Distribution representing `Y = ceiling(X)`.

    Some properties are inherited from the distribution defining `X`. Example:
    `allow_nan_stats` is determined for this `QuantizedDistribution` by reading
    the `distribution`.

    Args:
      distribution:  The base distribution class to transform. Typically an
        instance of `Distribution`.
      lower_cutoff:  `Tensor` with same `dtype` as this distribution and shape
        able to be added to samples.  Should be a whole number.  Default `None`.
        If provided, base distribution's pdf/pmf should be defined at
        `lower_cutoff`.
      upper_cutoff:  `Tensor` with same `dtype` as this distribution and shape
        able to be added to samples.  Should be a whole number.  Default `None`.
        If provided, base distribution's pdf/pmf should be defined at
        `upper_cutoff - 1`.
        `upper_cutoff` must be strictly greater than `lower_cutoff`.
      validate_args: Python boolean.  Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
      name: The name for the distribution.

    Raises:
      TypeError: If `dist_cls` is not a subclass of
          `Distribution` or continuous.
      NotImplementedError:  If the base distribution does not implement `cdf`.
    """
    parameters = locals()
    parameters.pop("self")
    values = (
        list(distribution.parameters.values()) +
        [lower_cutoff, upper_cutoff])
    with ops.name_scope(name, values=values) as ns:
      self._dist = distribution

      if lower_cutoff is not None:
        lower_cutoff = ops.convert_to_tensor(lower_cutoff, name="lower_cutoff")
      if upper_cutoff is not None:
        upper_cutoff = ops.convert_to_tensor(upper_cutoff, name="upper_cutoff")
      contrib_tensor_util.assert_same_float_dtype(
          tensors=[self.distribution, lower_cutoff, upper_cutoff])

      # We let QuantizedDistribution access _graph_parents since this class is
      # more like a baseclass.
      graph_parents = self._dist._graph_parents  # pylint: disable=protected-access

      checks = []
      if lower_cutoff is not None and upper_cutoff is not None:
        message = "lower_cutoff must be strictly less than upper_cutoff."
        checks.append(
            check_ops.assert_less(
                lower_cutoff, upper_cutoff, message=message))
      self._validate_args = validate_args  # self._check_integer uses this.
      with ops.control_dependencies(checks if validate_args else []):
        if lower_cutoff is not None:
          self._lower_cutoff = self._check_integer(lower_cutoff)
          graph_parents += [self._lower_cutoff]
        else:
          self._lower_cutoff = None
        if upper_cutoff is not None:
          self._upper_cutoff = self._check_integer(upper_cutoff)
          graph_parents += [self._upper_cutoff]
        else:
          self._upper_cutoff = None

    super(QuantizedDistribution, self).__init__(
        dtype=self._dist.dtype,
        is_continuous=False,
        is_reparameterized=False,
        validate_args=validate_args,
        allow_nan_stats=self._dist.allow_nan_stats,
        parameters=parameters,
        graph_parents=graph_parents,
        name=ns)

  def _batch_shape(self):
    return self.distribution.batch_shape()

  def _get_batch_shape(self):
    return self.distribution.get_batch_shape()

  def _event_shape(self):
    return self.distribution.event_shape()

  def _get_event_shape(self):
    return self.distribution.get_event_shape()

  def _sample_n(self, n, seed=None):
    lower_cutoff = self._lower_cutoff
    upper_cutoff = self._upper_cutoff
    with ops.name_scope("transform"):
      n = ops.convert_to_tensor(n, name="n")
      x_samps = self.distribution.sample(n, seed=seed)
      ones = array_ops.ones_like(x_samps)

      # Snap values to the intervals (j - 1, j].
      result_so_far = math_ops.ceil(x_samps)

      if lower_cutoff is not None:
        result_so_far = array_ops.where(result_so_far < lower_cutoff,
                                        lower_cutoff * ones, result_so_far)

      if upper_cutoff is not None:
        result_so_far = array_ops.where(result_so_far > upper_cutoff,
                                        upper_cutoff * ones, result_so_far)

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
    lower_cutoff = self._lower_cutoff
    upper_cutoff = self._upper_cutoff

    # Recall the promise:
    # cdf(y) := P[Y <= y]
    #         = 1, if y >= upper_cutoff,
    #         = 0, if y < lower_cutoff,
    #         = P[X <= y], otherwise.

    # P[Y <= j] = P[floor(Y) <= j] since mass is only at integers, not in
    # between.
    j = math_ops.floor(y)

    result_so_far = self.distribution.log_cdf(j)

    # Broadcast, because it's possible that this is a single distribution being
    # evaluated on a number of samples, or something like that.
    j += array_ops.zeros_like(result_so_far)

    # Re-define values at the cutoffs.
    if lower_cutoff is not None:
      neg_inf = -np.inf * array_ops.ones_like(result_so_far)
      result_so_far = array_ops.where(j < lower_cutoff, neg_inf, result_so_far)
    if upper_cutoff is not None:
      result_so_far = array_ops.where(j >= upper_cutoff,
                                      array_ops.zeros_like(result_so_far),
                                      result_so_far)

    return result_so_far

  @distribution_util.AppendDocstring(_cdf_note)
  def _cdf(self, y):
    lower_cutoff = self._lower_cutoff
    upper_cutoff = self._upper_cutoff

    # Recall the promise:
    # cdf(y) := P[Y <= y]
    #         = 1, if y >= upper_cutoff,
    #         = 0, if y < lower_cutoff,
    #         = P[X <= y], otherwise.

    # P[Y <= j] = P[floor(Y) <= j] since mass is only at integers, not in
    # between.
    j = math_ops.floor(y)

    # P[X <= j], used when lower_cutoff < X < upper_cutoff.
    result_so_far = self.distribution.cdf(j)

    # Broadcast, because it's possible that this is a single distribution being
    # evaluated on a number of samples, or something like that.
    j += array_ops.zeros_like(result_so_far)

    # Re-define values at the cutoffs.
    if lower_cutoff is not None:
      result_so_far = array_ops.where(j < lower_cutoff,
                                      array_ops.zeros_like(result_so_far),
                                      result_so_far)
    if upper_cutoff is not None:
      result_so_far = array_ops.where(j >= upper_cutoff,
                                      array_ops.ones_like(result_so_far),
                                      result_so_far)

    return result_so_far

  @distribution_util.AppendDocstring(_log_sf_note)
  def _log_survival_function(self, y):
    lower_cutoff = self._lower_cutoff
    upper_cutoff = self._upper_cutoff

    # Recall the promise:
    # survival_function(y) := P[Y > y]
    #                       = 0, if y >= upper_cutoff,
    #                       = 1, if y < lower_cutoff,
    #                       = P[X > y], otherwise.

    # P[Y > j] = P[ceiling(Y) > j] since mass is only at integers, not in
    # between.
    j = math_ops.ceil(y)

    # P[X > j], used when lower_cutoff < X < upper_cutoff.
    result_so_far = self.distribution.log_survival_function(j)

    # Broadcast, because it's possible that this is a single distribution being
    # evaluated on a number of samples, or something like that.
    j += array_ops.zeros_like(result_so_far)

    # Re-define values at the cutoffs.
    if lower_cutoff is not None:
      result_so_far = array_ops.where(j < lower_cutoff,
                                      array_ops.zeros_like(result_so_far),
                                      result_so_far)
    if upper_cutoff is not None:
      neg_inf = -np.inf * array_ops.ones_like(result_so_far)
      result_so_far = array_ops.where(j >= upper_cutoff, neg_inf, result_so_far)

    return result_so_far

  @distribution_util.AppendDocstring(_sf_note)
  def _survival_function(self, y):
    lower_cutoff = self._lower_cutoff
    upper_cutoff = self._upper_cutoff

    # Recall the promise:
    # survival_function(y) := P[Y > y]
    #                       = 0, if y >= upper_cutoff,
    #                       = 1, if y < lower_cutoff,
    #                       = P[X > y], otherwise.

    # P[Y > j] = P[ceiling(Y) > j] since mass is only at integers, not in
    # between.
    j = math_ops.ceil(y)

    # P[X > j], used when lower_cutoff < X < upper_cutoff.
    result_so_far = self.distribution.survival_function(j)

    # Broadcast, because it's possible that this is a single distribution being
    # evaluated on a number of samples, or something like that.
    j += array_ops.zeros_like(result_so_far)

    # Re-define values at the cutoffs.
    if lower_cutoff is not None:
      result_so_far = array_ops.where(j < lower_cutoff,
                                      array_ops.ones_like(result_so_far),
                                      result_so_far)
    if upper_cutoff is not None:
      result_so_far = array_ops.where(j >= upper_cutoff,
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
