# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""The Cauchy distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution

__all__ = [
    "Cauchy",
]


class Cauchy(distribution.Distribution):
  """The Cauchy distribution with location `loc` and scale `scale`.

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x; loc, scale) = 1 / (pi scale (1 + z**2))
  z = (x - loc) / scale
  ```
  where `loc` is the location, and `scale` is the scale.

  The Cauchy distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e.
  `Y ~ Cauchy(loc, scale)` is equivalent to,

  ```none
  X ~ Cauchy(loc=0, scale=1)
  Y = loc + scale * X
  ```

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  tfd = tf.contrib.distributions

  # Define a single scalar Cauchy distribution.
  dist = tfd.Cauchy(loc=0., scale=3.)

  # Evaluate the cdf at 1, returning a scalar.
  dist.cdf(1.)

  # Define a batch of two scalar valued Cauchy distributions.
  dist = tfd.Cauchy(loc=[1, 2.], scale=[11, 22.])

  # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
  # returning a length two tensor.
  dist.prob([0, 1.5])

  # Get 3 samples, returning a 3 x 2 tensor.
  dist.sample([3])

  # Arguments are broadcast when possible.
  # Define a batch of two scalar valued Cauchy distributions.
  # Both have median 1, but different scales.
  dist = tfd.Cauchy(loc=1., scale=[11, 22.])

  # Evaluate the pdf of both distributions on the same point, 3.0,
  # returning a length 2 tensor.
  dist.prob(3.)
  ```

  """

  def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name="Cauchy"):
    """Construct Cauchy distributions.

    The parameters `loc` and `scale` must be shaped in a way that supports
    broadcasting (e.g. `loc + scale` is a valid operation).

    Args:
      loc: Floating point tensor; the modes of the distribution(s).
      scale: Floating point tensor; the locations of the distribution(s).
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
      with ops.control_dependencies([check_ops.assert_positive(scale)]
                                    if validate_args else []):
        self._loc = array_ops.identity(loc, name="loc")
        self._scale = array_ops.identity(scale, name="scale")
        check_ops.assert_same_float_dtype([self._loc, self._scale])
    super(Cauchy, self).__init__(
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
        zip(("loc", "scale"),
            ([ops.convert_to_tensor(sample_shape, dtype=dtypes.int32)] * 2)))

  @property
  def loc(self):
    """Distribution parameter for the mean."""
    return self._loc

  @property
  def scale(self):
    """Distribution parameter for standard deviation."""
    return self._scale

  def _batch_shape_tensor(self):
    return array_ops.broadcast_dynamic_shape(
        array_ops.shape(self.loc), array_ops.shape(self.scale))

  def _batch_shape(self):
    return array_ops.broadcast_static_shape(self.loc.shape, self.scale.shape)

  def _event_shape_tensor(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    shape = array_ops.concat([[n], self.batch_shape_tensor()], 0)
    probs = random_ops.random_uniform(
        shape=shape, minval=0., maxval=1., dtype=self.dtype, seed=seed)
    return self._quantile(probs)

  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()

  def _cdf(self, x):
    return math_ops.atan(self._z(x)) / np.pi + 0.5

  def _log_cdf(self, x):
    return math_ops.log1p(2 / np.pi * math_ops.atan(self._z(x))) - np.log(2)

  def _log_unnormalized_prob(self, x):
    return -math_ops.log1p(math_ops.square(self._z(x)))

  def _log_normalization(self):
    return np.log(np.pi) + math_ops.log(self.scale)

  def _entropy(self):
    h = np.log(4 * np.pi) + math_ops.log(self.scale)
    return h * array_ops.ones_like(self.loc)

  def _quantile(self, p):
    return self.loc + self.scale * math_ops.tan(np.pi * (p - 0.5))

  def _mode(self):
    return self.loc * array_ops.ones_like(self.scale)

  def _z(self, x):
    """Standardize input `x`."""
    with ops.name_scope("standardize", values=[x]):
      return (x - self.loc) / self.scale

  def _inv_z(self, z):
    """Reconstruct input `x` from a its normalized version."""
    with ops.name_scope("reconstruct", values=[z]):
      return z * self.scale + self.loc

  def _mean(self):
    if self.allow_nan_stats:
      return array_ops.fill(self.batch_shape_tensor(),
                            self.dtype.as_numpy_dtype(np.nan))
    else:
      raise ValueError("`mean` is undefined for Cauchy distribution.")

  def _stddev(self):
    if self.allow_nan_stats:
      return array_ops.fill(self.batch_shape_tensor(),
                            self.dtype.as_numpy_dtype(np.nan))
    else:
      raise ValueError("`stddev` is undefined for Cauchy distribution.")
