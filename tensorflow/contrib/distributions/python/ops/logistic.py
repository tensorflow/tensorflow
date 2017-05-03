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
"""The Logistic distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution


class Logistic(distribution.Distribution):
  """The Logistic distribution with location `loc` and `scale` parameters.

  #### Mathematical details

  The cumulative density function of this distribution is:

  ```none
  cdf(x; mu, sigma) = 1 / (1 + exp(-(x - mu) / sigma))
  ```

  where `loc = mu` and `scale = sigma`.

  The Logistic distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ Logistic(loc=0, scale=1)
  Y = loc + scale * X
  ```

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  # Define a single scalar Logistic distribution.
  dist = tf.contrib.distributions.Logistic(loc=0., scale=3.)

  # Evaluate the cdf at 1, returning a scalar.
  dist.cdf(1.)

  # Define a batch of two scalar valued Logistics.
  # The first has mean 1 and scale 11, the second 2 and 22.
  dist = tf.contrib.distributions.Logistic(loc=[1, 2.], scale=[11, 22.])

  # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
  # returning a length two tensor.
  dist.prob([0, 1.5])

  # Get 3 samples, returning a 3 x 2 tensor.
  dist.sample([3])
  ```

  Arguments are broadcast when possible.

  ```python
  # Define a batch of two scalar valued Logistics.
  # Both have mean 1, but different scales.
  dist = tf.contrib.distributions.Logistic(loc=1., scale=[11, 22.])

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
               name="Logistic"):
    """Construct Logistic distributions with mean and scale `loc` and `scale`.

    The parameters `loc` and `scale` must be shaped in a way that supports
    broadcasting (e.g. `loc + scale` is a valid operation).

    Args:
      loc: Floating point tensor, the means of the distribution(s).
      scale: Floating point tensor, the scales of the distribution(s). Must
        contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: The name to give Ops created by the initializer.

    Raises:
      TypeError: if loc and scale are different dtypes.
    """
    parameters = locals()
    with ops.name_scope(name, values=[loc, scale]):
      with ops.control_dependencies([check_ops.assert_positive(scale)] if
                                    validate_args else []):
        self._loc = array_ops.identity(loc, name="loc")
        self._scale = array_ops.identity(scale, name="scale")
        check_ops.assert_same_float_dtype([self._loc, self._scale])
    super(Logistic, self).__init__(
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
    """Distribution parameter for scale."""
    return self._scale

  def _batch_shape_tensor(self):
    return array_ops.broadcast_dynamic_shape(
        array_ops.shape(self.loc), array_ops.shape(self.scale))

  def _batch_shape(self):
    return array_ops.broadcast_static_shape(
        self.loc.get_shape(), self.scale.get_shape())

  def _event_shape_tensor(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    # Uniform variates must be sampled from the open-interval `(0, 1)` rather
    # than `[0, 1)`. To do so, we use `np.finfo(self.dtype.as_numpy_dtype).tiny`
    # because it is the smallest, positive, "normal" number. A "normal" number
    # is such that the mantissa has an implicit leading 1. Normal, positive
    # numbers x, y have the reasonable property that, `x + y >= max(x, y)`. In
    # this case, a subnormal number (i.e., np.nextafter) can cause us to sample
    # 0.
    uniform = random_ops.random_uniform(
        shape=array_ops.concat([[n], self.batch_shape_tensor()], 0),
        minval=np.finfo(self.dtype.as_numpy_dtype).tiny,
        maxval=1.,
        dtype=self.dtype,
        seed=seed)
    sampled = math_ops.log(uniform) - math_ops.log1p(-1. * uniform)
    return sampled * self.scale + self.loc

  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()

  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))

  def _log_cdf(self, x):
    return nn_ops.softplus(-self._z(x))

  def _cdf(self, x):
    return math_ops.sigmoid(self._z(x))

  def _log_survival_function(self, x):
    return nn_ops.softplus(self._z(x))

  def _survival_function(self, x):
    return math_ops.sigmoid(-self._z(x))

  def _log_unnormalized_prob(self, x):
    z = self._z(x)
    return - z - 2. * nn_ops.softplus(-z)

  def _log_normalization(self):
    return math_ops.log(self.scale)

  def _entropy(self):
    # Use broadcasting rules to calculate the full broadcast sigma.
    scale = self.scale * array_ops.ones_like(self.loc)
    return 2 + math_ops.log(scale)

  def _mean(self):
    return self.loc * array_ops.ones_like(self.scale)

  def _stddev(self):
    return self.scale * array_ops.ones_like(self.loc) * math.pi / math.sqrt(3)

  def _mode(self):
    return self._mean()

  def _z(self, x):
    """Standardize input `x` to a unit logistic."""
    with ops.name_scope("standardize", values=[x]):
      return (x - self.loc) / self.scale
