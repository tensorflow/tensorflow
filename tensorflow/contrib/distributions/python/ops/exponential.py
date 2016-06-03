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

from tensorflow.contrib.distributions.python.ops import gamma
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


class Exponential(gamma.Gamma):
  """The Exponential distribution with rate parameter lam.

  The PDF of this distribution is:

  ```pdf(x) = (lam * e^(-lam * x)), x > 0```

  Note that the Exponential distribution is a special case of the Gamma
  distribution, with Exponential(lam) = Gamma(1, lam).
  """

  def __init__(self, lam, name="Exponential"):
    with ops.op_scope([lam], name, "init"):
      lam = ops.convert_to_tensor(lam)
      self._lam = lam
      super(Exponential, self).__init__(
          alpha=math_ops.cast(1.0, dtype=lam.dtype),
          beta=lam)

  @property
  def lam(self):
    return self._lam

  @property
  def is_reparameterized(self):
    # While the Gamma distribution is not reparameterizeable, the
    # exponential distribution is.
    return True

  def sample(self, n, seed=None, name=None):
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
      shape = array_ops.concat(
          0, [array_ops.pack([n]), array_ops.shape(self._lam)])
      sampled = random_ops.random_uniform(
          shape, maxval=math_ops.cast(1.0, dtype=self.dtype),
          dtype=self.dtype)

      n_val = tensor_util.constant_value(n)
      final_shape = tensor_shape.vector(n_val).concatenate(broadcast_shape)
      sampled.set_shape(final_shape)

      return -math_ops.log(sampled) / self._lam
