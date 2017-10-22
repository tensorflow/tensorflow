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
"""The Folded Normal distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import special_math


__all__ = [
    "FoldedNormal",
]


class FoldedNormal(distribution.Distribution):
  """The Folded Normal distribution with loc `loc` and scale `scale`.

  #### Mathematical details

  The folded normal is a transformation of the normal distribution. So
  if some random variable `X` has normal distribution,
  ```none
  X ~ Normal(loc, scale)
  Y = |X|
  ```
  Then `Y` will have folded normal distribution. The probability density function (pdf) is:

  ```none
  # for x > 0
  pdf(x; loc, scale) = (1 / (scale * sqrt(2 *pi)) *
    ( exp(-0.5 * ((x - loc) /scale) ** 2) +
      exp(-0.5 * ((x + loc) /scale) ** 2) )
  )
  ```

  Where `loc = mu` is the mean and `scale = sigma` is the std. deviation of
  the underlying normal distribution.

  When `loc = 0.0`, the resulting distribution is known also as the half
  normal distribution, and is a common scale prior in bayesian statistics.

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  # Define a single scalar FoldedNormal distribution.
  dist = tf.contrib.distributions.FoldedNormal(loc=0., scale=3.)

  # Evaluate the cdf at 1, returning a scalar.
  dist.cdf(1.)

  # Define a batch of two scalar valued FoldedNormals.
  # The first has location 1 and scale 11, the second 2 and 22.
  dist = tf.distributions.FoldedNormal(loc=[1, 2.], scale=[11, 22.])

  # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
  # returning a length two tensor.
  dist.prob([0, 1.5])

  # Get 3 samples, returning a 3 x 2 tensor.
  dist.sample([3])
  ```

  Arguments are broadcast when possible.

  ```python
  # Define a batch of two scalar valued FoldedNormals.
  # Both have location 1, but different scales.
  dist = tf.contrib.distributions.FoldedNormal(loc=1., scale=[11, 22.])

  # Evaluate the pdf of both distributions on the same point, 3.0,
  # returning a length 2 tensor.
  dist.prob(3.0)
  ```

  """

  def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name="FoldedNormal"):
    """Construct FoldedNormals with location and scale `loc` and `scale`.

    The parameters `loc` and `scale` must be shaped in a way that supports
    broadcasting (e.g. `loc + scale` is a valid operation).

    Args:
      loc: Floating point tensor; the locations of the distribution(s).
      scale: Floating point tensor; the scales of the distribution(s).
        Must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if `loc` and `scale` have different `dtype`.
    """
    parameters = locals()
    with ops.name_scope(name, values=[loc, scale]):
      with ops.control_dependencies([check_ops.assert_positive(scale)] if
                                    validate_args else []):
        self._loc = array_ops.identity(loc, name="loc")
        self._scale = array_ops.identity(scale, name="scale")
        check_ops.assert_same_float_dtype([self._loc, self._scale])
    super(FoldedNormal, self).__init__(
        dtype=self._scale.dtype,
        reparameterization_type=distribution.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._loc, self._scale],
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("loc", "scale"), ([ops.convert_to_tensor(
            sample_shape, dtype=dtypes.int32)] * 2)))

  @property
  def loc(self):
    """Distribution parameter for the location."""
    return self._loc

  @property
  def scale(self):
    """Distribution parameter for the scale."""
    return self._scale

  def _batch_shape_tensor(self):
    return array_ops.broadcast_dynamic_shape(
        array_ops.shape(self.loc),
        array_ops.shape(self.scale))

  def _batch_shape(self):
    return array_ops.broadcast_static_shape(
        self.loc.get_shape(),
        self.scale.get_shape())

  def _event_shape_tensor(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    pass

  def _log_prob(self, x):
    return math_ops.log(self._prob(x))

  def _prob(self, x):
    coeff = 1 / (self.scale * math.sqrt(2 * math.pi))
    pos_portion = math_ops.exp(-0.5 * math_ops.square(self._pos_z))
    neg_portion = math_ops.exp(-0.5 * math_ops.square(self._neg_z))
    return coeff * (pos_portion + neg_portion)

  def _log_cdf(self, x):
    pass

  def _cdf(self, x):
    pass

  def _log_survival_function(self, x):
    pass

  def _survival_function(self, x):
    pass

  def _log_unnormalized_prob(self, x):
    pass

  def _log_normalization(self):
    pass

  def _entropy(self):
    pass

  def _mean(self):
    pass

  def _quantile(self, p):
    pass

  def _stddev(self):
    pass

  def _mode(self):
    pass

  def _pos_z(self, x):
    """Standardize input `x` to a unit normal."""
    with ops.name_scope("standardize", values=[x]):
      return (x - self.loc) / self.scale

  def _neg_z(self, x):
    """Standardize input `x` to a unit normal around -self.loc."""
    with ops.name_scope("standardize_neg", values=[x]):
      return (x + self.loc) / self.scale
