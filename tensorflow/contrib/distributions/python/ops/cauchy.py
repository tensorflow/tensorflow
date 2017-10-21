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
"""The Cauchy distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import util as distribution_util

__all__ = [
    "Cauchy",
]


class Cauchy(distribution.Distribution):
  """The Cauchy distribution with location `loc` and scale `scale`.

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x; mu, gamma) = 1 / (pi*gamma*(1+((x-mu)/gamma)**2))
  ```
  where `loc = mu` is the mean, `scale = gamma` is the std. deviation.

  The Cauchy distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e.

  ```none
  X ~ Cauchy(loc=0, scale=1)
  Y ~ Cauchy(loc=loc, scale=scale)
  Y = loc + scale * X
  ```

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  # Define a single scalar Cauchy distribution.
  dist = Cauchy(loc=0., scale=3.)

  # Evaluate the cdf at 1, returning a scalar.
  dist.cdf(1.)

  # Define a batch of two scalar valued Normals.
  dist = Cauchy(loc=[1, 2.], scale=[11, 22.])

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
               name="Cauchy"):
    """Construct Cauchy distributions with loc and and scale `loc` and `scale`.

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
      with ops.control_dependencies([check_ops.assert_positive(scale)] if
                                    validate_args else []):
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
