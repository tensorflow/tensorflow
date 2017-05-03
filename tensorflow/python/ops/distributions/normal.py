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
"""The Normal (Gaussian) distribution class."""

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
    "Normal",
    "NormalWithSoftplusScale",
]


class Normal(distribution.Distribution):
  """The Normal distribution with location `loc` and `scale` parameters.

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x; mu, sigma) = exp(-0.5 (x - mu)**2 / sigma**2) / Z
  Z = (2 pi sigma**2)**0.5
  ```

  where `loc = mu` is the mean, `scale = sigma` is the std. deviation, and, `Z`
  is the normalization constant.

  The Normal distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ Normal(loc=0, scale=1)
  Y = loc + scale * X
  ```

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  # Define a single scalar Normal distribution.
  dist = tf.distributions.Normal(loc=0., scale=3.)

  # Evaluate the cdf at 1, returning a scalar.
  dist.cdf(1.)

  # Define a batch of two scalar valued Normals.
  # The first has mean 1 and standard deviation 11, the second 2 and 22.
  dist = tf.distributions.Normal(loc=[1, 2.], scale=[11, 22.])

  # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
  # returning a length two tensor.
  dist.prob([0, 1.5])

  # Get 3 samples, returning a 3 x 2 tensor.
  dist.sample([3])
  ```

  Arguments are broadcast when possible.

  ```python
  # Define a batch of two scalar valued Normals.
  # Both have mean 1, but different standard deviations.
  dist = tf.distributions.Normal(loc=1., scale=[11, 22.])

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
               name="Normal"):
    """Construct Normal distributions with mean and stddev `loc` and `scale`.

    The parameters `loc` and `scale` must be shaped in a way that supports
    broadcasting (e.g. `loc + scale` is a valid operation).

    Args:
      loc: Floating point tensor; the means of the distribution(s).
      scale: Floating point tensor; the stddevs of the distribution(s).
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
    super(Normal, self).__init__(
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
    """Distribution parameter for the mean."""
    return self._loc

  @property
  def scale(self):
    """Distribution parameter for standard deviation."""
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
    shape = array_ops.concat([[n], self.batch_shape_tensor()], 0)
    sampled = random_ops.random_normal(
        shape=shape, mean=0., stddev=1., dtype=self.loc.dtype, seed=seed)
    return sampled * self.scale + self.loc

  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()

  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))

  def _log_cdf(self, x):
    return special_math.log_ndtr(self._z(x))

  def _cdf(self, x):
    return special_math.ndtr(self._z(x))

  def _log_survival_function(self, x):
    return special_math.log_ndtr(-self._z(x))

  def _survival_function(self, x):
    return special_math.ndtr(-self._z(x))

  def _log_unnormalized_prob(self, x):
    return -0.5 * math_ops.square(self._z(x))

  def _log_normalization(self):
    return 0.5 * math.log(2. * math.pi) + math_ops.log(self.scale)

  def _entropy(self):
    # Use broadcasting rules to calculate the full broadcast scale.
    scale = self.scale * array_ops.ones_like(self.loc)
    return 0.5 * math.log(2. * math.pi * math.e) + math_ops.log(scale)

  def _mean(self):
    return self.loc * array_ops.ones_like(self.scale)

  def _quantile(self, p):
    return self._inv_z(special_math.ndtri(p))

  def _stddev(self):
    return self.scale * array_ops.ones_like(self.loc)

  def _mode(self):
    return self._mean()

  def _z(self, x):
    """Standardize input `x` to a unit normal."""
    with ops.name_scope("standardize", values=[x]):
      return (x - self.loc) / self.scale

  def _inv_z(self, z):
    """Reconstruct input `x` from a its normalized version."""
    with ops.name_scope("reconstruct", values=[z]):
      return z * self.scale + self.loc


class NormalWithSoftplusScale(Normal):
  """Normal with softplus applied to `scale`."""

  def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name="NormalWithSoftplusScale"):
    parameters = locals()
    with ops.name_scope(name, values=[scale]):
      super(NormalWithSoftplusScale, self).__init__(
          loc=loc,
          scale=nn.softplus(scale, name="softplus_scale"),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)
    self._parameters = parameters


@kullback_leibler.RegisterKL(Normal, Normal)
def _kl_normal_normal(n_a, n_b, name=None):
  """Calculate the batched KL divergence KL(n_a || n_b) with n_a and n_b Normal.

  Args:
    n_a: instance of a Normal distribution object.
    n_b: instance of a Normal distribution object.
    name: (optional) Name to use for created operations.
      default is "kl_normal_normal".

  Returns:
    Batchwise KL(n_a || n_b)
  """
  with ops.name_scope(name, "kl_normal_normal", [n_a.loc, n_b.loc]):
    one = constant_op.constant(1, dtype=n_a.dtype)
    two = constant_op.constant(2, dtype=n_a.dtype)
    half = constant_op.constant(0.5, dtype=n_a.dtype)
    s_a_squared = math_ops.square(n_a.scale)
    s_b_squared = math_ops.square(n_b.scale)
    ratio = s_a_squared / s_b_squared
    return (math_ops.square(n_a.loc - n_b.loc) / (two * s_b_squared) +
            half * (ratio - one - math_ops.log(ratio)))
