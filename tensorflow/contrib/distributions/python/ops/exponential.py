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
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


class Exponential(gamma.Gamma):
  """The Exponential distribution with rate parameter lam.

  The PDF of this distribution is:

  ```prob(x) = (lam * e^(-lam * x)), x > 0```

  Note that the Exponential distribution is a special case of the Gamma
  distribution, with Exponential(lam) = Gamma(1, lam).
  """

  def __init__(
      self, lam, validate_args=True, allow_nan_stats=False, name="Exponential"):
    """Construct Exponential distribution with parameter `lam`.

    Args:
      lam: Floating point tensor, the rate of the distribution(s).
        `lam` must contain only positive values.
      validate_args: Whether to assert that `lam > 0`, and that `x > 0` in the
        methods `prob(x)` and `log_prob(x)`.  If `validate_args` is `False`
        and the inputs are invalid, correct behavior is not guaranteed.
      allow_nan_stats:  Boolean, default `False`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member. If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to prepend to all ops created by this distribution.
    """
    # Even though all statistics of are defined for valid inputs, this is not
    # true in the parent class "Gamma."  Therefore, passing
    # allow_nan_stats=False
    # through to the parent class results in unnecessary asserts.
    with ops.op_scope([lam], name):
      lam = ops.convert_to_tensor(lam)
      self._lam = lam
      super(Exponential, self).__init__(
          alpha=constant_op.constant(1.0, dtype=lam.dtype),
          beta=lam,
          allow_nan_stats=allow_nan_stats,
          validate_args=validate_args)

  @property
  def lam(self):
    return self._lam

  @property
  def is_reparameterized(self):
    # While the Gamma distribution is not reparameterizeable, the
    # exponential distribution is.
    return True

  def sample_n(self, n, seed=None, name="sample_n"):
    """Sample `n` observations from the Exponential Distributions.

    Args:
      n: `Scalar`, type int32, the number of observations to sample.
      seed: Python integer, the random seed.
      name: The name to give this op.

    Returns:
      samples: `[n, ...]`, a `Tensor` of `n` samples for each
        of the distributions determined by the hyperparameters.
    """
    broadcast_shape = self._lam.get_shape()
    with ops.op_scope([self.lam, n], name, "ExponentialSample"):
      n = ops.convert_to_tensor(n, name="n")
      shape = array_ops.concat(0, ([n], array_ops.shape(self._lam)))
      # Sample uniformly-at-random from the open-interval (0, 1).
      sampled = random_ops.random_uniform(
          shape, minval=np.nextafter(
              self.dtype.as_numpy_dtype(0.), self.dtype.as_numpy_dtype(1.)),
          maxval=constant_op.constant(1.0, dtype=self.dtype),
          seed=seed,
          dtype=self.dtype)

      n_val = tensor_util.constant_value(n)
      final_shape = tensor_shape.vector(n_val).concatenate(broadcast_shape)
      sampled.set_shape(final_shape)

      return -math_ops.log(sampled) / self._lam
