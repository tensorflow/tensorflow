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


class Laplace(distribution.Distribution):
  """The Laplace distribution with location and scale > 0 parameters.

  #### Mathematical details

  The PDF of this distribution is:

  ```f(x | mu, b, b > 0) = 0.5 / b exp(-|x - mu| / b)```

  Note that the Laplace distribution can be thought of two exponential
  distributions spliced together "back-to-back."
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
      validate_args: `Boolean`, default `False`.  Whether to validate input
        with asserts.  If `validate_args` is `False`, and the inputs are
        invalid, correct behavior is not guaranteed.
      allow_nan_stats: `Boolean`, default `True`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member.  If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.

    Raises:
      TypeError: if `loc` and `scale` are of different dtype.
    """
    parameters = locals()
    parameters.pop("self")
    with ops.name_scope(name, values=[loc, scale]) as ns:
      with ops.control_dependencies([check_ops.assert_positive(scale)] if
                                    validate_args else []):
        self._loc = array_ops.identity(loc, name="loc")
        self._scale = array_ops.identity(scale, name="scale")
        contrib_tensor_util.assert_same_float_dtype((self._loc, self._scale))
      super(Laplace, self).__init__(
          dtype=self._loc.dtype,
          is_continuous=True,
          is_reparameterized=True,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          graph_parents=[self._loc, self._scale],
          name=ns)

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

  def _batch_shape(self):
    return array_ops.broadcast_dynamic_shape(
        array_ops.shape(self.loc), array_ops.shape(self.scale))

  def _get_batch_shape(self):
    return array_ops.broadcast_static_shape(
        self.loc.get_shape(), self.scale.get_shape())

  def _event_shape(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _get_event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    shape = array_ops.concat(([n], self.batch_shape()), 0)
    # Sample uniformly-at-random from the open-interval (-1, 1).
    uniform_samples = random_ops.random_uniform(
        shape=shape,
        minval=np.nextafter(self.dtype.as_numpy_dtype(-1.),
                            self.dtype.as_numpy_dtype(0.)),
        maxval=1.,
        dtype=self.dtype,
        seed=seed)
    return (self.loc - self.scale * math_ops.sign(uniform_samples) *
            math_ops.log(1. - math_ops.abs(uniform_samples)))

  def _log_prob(self, x):
    return (-math.log(2.) - math_ops.log(self.scale) -
            math_ops.abs(x - self.loc) / self.scale)

  def _prob(self, x):
    return 0.5 / self.scale * math_ops.exp(
        -math_ops.abs(x - self.loc) / self.scale)

  def _log_cdf(self, x):
    return math_ops.log(self.cdf(x))

  def _cdf(self, x):
    y = x - self.loc
    return (0.5 + 0.5 * math_ops.sign(y) *
            (1. - math_ops.exp(-math_ops.abs(y) / self.scale)))

  def _entropy(self):
    # Use broadcasting rules to calculate the full broadcast scale.
    scale = self.scale + array_ops.zeros_like(self.loc)
    return math.log(2.) + 1. + math_ops.log(scale)

  def _mean(self):
    return self.loc + array_ops.zeros_like(self.scale)

  def _variance(self):
    return math_ops.square(self._std())

  def _std(self):
    return math.sqrt(2.) * self.scale + array_ops.zeros_like(self.loc)

  def _median(self):
    return self._mean()

  def _mode(self):
    return self._mean()


class LaplaceWithSoftplusScale(Laplace):
  """Laplace with softplus applied to `scale`."""

  def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name="LaplaceWithSoftplusScale"):
    parameters = locals()
    parameters.pop("self")
    with ops.name_scope(name, values=[loc, scale]) as ns:
      super(LaplaceWithSoftplusScale, self).__init__(
          loc=loc,
          scale=nn.softplus(scale, name="softplus_scale"),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=ns)
    self._parameters = parameters
