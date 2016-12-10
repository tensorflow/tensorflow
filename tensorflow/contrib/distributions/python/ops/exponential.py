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
"""The Exponential distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import gamma
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops


class Exponential(gamma.Gamma):
  """The Exponential distribution with rate parameter lam.

  The PDF of this distribution is:

  ```prob(x) = (lam * e^(-lam * x)), x > 0```

  Note that the Exponential distribution is a special case of the Gamma
  distribution, with Exponential(lam) = Gamma(1, lam).
  """

  def __init__(self,
               lam,
               validate_args=False,
               allow_nan_stats=True,
               name="Exponential"):
    """Construct Exponential distribution with parameter `lam`.

    Args:
      lam: Floating point tensor, the rate of the distribution(s).
        `lam` must contain only positive values.
      validate_args: `Boolean`, default `False`.  Whether to assert that
        `lam > 0`, and that `x > 0` in the methods `prob(x)` and `log_prob(x)`.
        If `validate_args` is `False` and the inputs are invalid, correct
        behavior is not guaranteed.
      allow_nan_stats: `Boolean`, default `True`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member. If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to prepend to all ops created by this distribution.
    """
    parameters = locals()
    parameters.pop("self")
    # Even though all statistics of are defined for valid inputs, this is not
    # true in the parent class "Gamma."  Therefore, passing
    # allow_nan_stats=True
    # through to the parent class results in unnecessary asserts.
    with ops.name_scope(name, values=[lam]) as ns:
      self._lam = ops.convert_to_tensor(lam, name="lam")
    super(Exponential, self).__init__(
        alpha=array_ops.ones((), dtype=self._lam.dtype),
        beta=self._lam,
        allow_nan_stats=allow_nan_stats,
        validate_args=validate_args,
        name=ns)
    # While the Gamma distribution is not reparameterizeable, the
    # exponential distribution is.
    self._is_reparameterized = True
    self._parameters = parameters
    self._graph_parents += [self._lam]

  @staticmethod
  def _param_shapes(sample_shape):
    return {"lam": ops.convert_to_tensor(sample_shape, dtype=dtypes.int32)}

  @property
  def lam(self):
    return self._lam

  def _sample_n(self, n, seed=None):
    shape = array_ops.concat_v2(([n], array_ops.shape(self._lam)), 0)
    # Sample uniformly-at-random from the open-interval (0, 1).
    sampled = random_ops.random_uniform(
        shape,
        minval=np.nextafter(self.dtype.as_numpy_dtype(0.),
                            self.dtype.as_numpy_dtype(1.)),
        maxval=array_ops.ones((), dtype=self.dtype),
        seed=seed,
        dtype=self.dtype)
    return -math_ops.log(sampled) / self._lam


class ExponentialWithSoftplusLam(Exponential):
  """Exponential with softplus transform on `lam`."""

  def __init__(self,
               lam,
               validate_args=False,
               allow_nan_stats=True,
               name="ExponentialWithSoftplusLam"):
    parameters = locals()
    parameters.pop("self")
    with ops.name_scope(name, values=[lam]) as ns:
      super(ExponentialWithSoftplusLam, self).__init__(
          lam=nn.softplus(lam),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=ns)
    self._parameters = parameters
