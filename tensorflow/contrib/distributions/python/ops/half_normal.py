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
    (exp(-0.5 * ((x - loc) /scale) ** 2) + exp(-0.5 * ((x + loc) /scale) ** 2))
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
  dist = tf.distributions.Normal(loc=1., scale=[11, 22.])

  # Evaluate the pdf of both distributions on the same point, 3.0,
  # returning a length 2 tensor.
  dist.prob(3.0)
  ```

  """
  pass
