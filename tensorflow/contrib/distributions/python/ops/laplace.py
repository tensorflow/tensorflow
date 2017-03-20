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
"""The Laplace distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.distributions.python.ops import special_math
from tensorflow.contrib.framework.python.framework import tensor_util as contrib_tensor_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops


__all__ = [
    "Laplace",
    "LaplaceWithSoftplusScale",
]


class Laplace(distribution.Distribution):
  """The Laplace distribution with location `loc` and `scale` parameters.

  #### Mathematical details

  The probability density function (pdf) of this distribution is,

  ```none
  pdf(x; mu, sigma) = exp(-|x - mu| / sigma) / Z
  Z = 2 sigma
  ```

  where `loc = mu`, `scale = sigma`, and `Z` is the normalization constant.

  Note that the Laplace distribution can be thought of two exponential
  distributions spliced together "back-to-back."

  The Lpalce distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ Laplace(loc=0, scale=1)
  Y = loc + scale * X
  ```

  """

  def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name="Laplace"):
    """Construct Laplace distribution with parameters `loc` and `scale`.

    The parameters `loc` and `scale` must be shaped in a way that supports
    broadcasting (e.g., `loc / scale` is a valid operation).

    Args:
      loc: Floating point tensor which characterizes the location (center)
        of the distribution.
      scale: Positive floating point tensor which characterizes the spread of
        the distribution.
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
      TypeError: if `loc` and `scale` are of different dtype.
    """
    parameters = locals()
    with ops.name_scope(name, values=[loc, scale]):
      with ops.control_dependencies([check_ops.assert_positive(scale)] if
                                    validate_args else []):
        self._loc = array_ops.identity(loc, name="loc")
        self._scale = array_ops.identity(scale, name="scale")
        contrib_tensor_util.assert_same_float_dtype([self._loc, self._scale])
      super(Laplace, self).__init__(
          dtype=self._loc.dtype,
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
    shape = array_ops.concat([[n], self.batch_shape_tensor()], 0)
    # Uniform variates must be sampled from the open-interval `(-1, 1)` rather
    # than `[-1, 1)`. In the case of `(0, 1)` we'd use
    # `np.finfo(self.dtype.as_numpy_dtype).tiny` because it is the smallest,
    # positive, "normal" number. However, the concept of subnormality exists
    # only at zero; here we need the smallest usable number larger than -1,
    # i.e., `-1 + eps/2`.
    uniform_samples = random_ops.random_uniform(
        shape=shape,
        minval=np.nextafter(self.dtype.as_numpy_dtype(-1.),
                            self.dtype.as_numpy_dtype(0.)),
        maxval=1.,
        dtype=self.dtype,
        seed=seed)
    return (self.loc - self.scale * math_ops.sign(uniform_samples) *
            math_ops.log1p(-math_ops.abs(uniform_samples)))

  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()

  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))

  def _log_cdf(self, x):
    return special_math.log_cdf_laplace(self._z(x))

  def _log_survival_function(self, x):
    return special_math.log_cdf_laplace(-self._z(x))

  def _cdf(self, x):
    z = self._z(x)
    return (0.5 + 0.5 * math_ops.sign(z) *
            (1. - math_ops.exp(-math_ops.abs(z))))

  def _log_unnormalized_prob(self, x):
    return -math_ops.abs(self._z(x))

  def _log_normalization(self):
    return math.log(2.) + math_ops.log(self.scale)

  def _entropy(self):
    # Use broadcasting rules to calculate the full broadcast scale.
    scale = self.scale + array_ops.zeros_like(self.loc)
    return math.log(2.) + 1. + math_ops.log(scale)

  def _mean(self):
    return self.loc + array_ops.zeros_like(self.scale)

  def _stddev(self):
    return math.sqrt(2.) * self.scale + array_ops.zeros_like(self.loc)

  def _median(self):
    return self._mean()

  def _mode(self):
    return self._mean()

  def _z(self, x):
    return (x - self.loc) / self.scale


class LaplaceWithSoftplusScale(Laplace):
  """Laplace with softplus applied to `scale`."""

  def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name="LaplaceWithSoftplusScale"):
    parameters = locals()
    with ops.name_scope(name, values=[loc, scale]):
      super(LaplaceWithSoftplusScale, self).__init__(
          loc=loc,
          scale=nn.softplus(scale, name="softplus_scale"),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)
    self._parameters = parameters
