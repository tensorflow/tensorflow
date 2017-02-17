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
"""Softplus bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.contrib.distributions.python.ops.bijectors import bijector
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


__all__ = [
    "Softplus",
]


class Softplus(bijector.Bijector):
  """Bijector which computes `Y = g(X) = Log[1 + exp(X)]`.

  The softplus `Bijector` has the following two useful properties:

  * The domain is the positive real numbers
  * `softplus(x) approx x`, for large `x`, so it does not overflow as easily as
    the `Exp` `Bijector`.

    Example Use:

    ```python
    # Create the Y=g(X)=softplus(X) transform which works only on Tensors with 1
    # batch ndim and 2 event ndims (i.e., vector of matrices).
    softplus = Softplus(event_ndims=2)
    x = [[[1., 2],
           [3, 4]],
          [[5, 6],
           [7, 8]]]
    log(1 + exp(x)) == softplus.forward(x)
    log(exp(x) - 1) == softplus.inverse(x)
    ```

    Note: log(.) and exp(.) are applied element-wise but the Jacobian is a
    reduction over the event space.
  """

  def __init__(self,
               event_ndims=0,
               validate_args=False,
               name="softplus"):
    super(Softplus, self).__init__(
        event_ndims=event_ndims,
        validate_args=validate_args,
        name=name)

  def _forward(self, x):
    return nn_ops.softplus(x)

  def _inverse(self, y):
    return distribution_util.softplus_inverse(y)

  def _inverse_log_det_jacobian(self, y):
    # Could also do:
    #   ildj = math_ops.reduce_sum(y - distribution_util.softplus_inverse(y),
    #                              axis=event_dims)
    # but the following is more numerically stable. Ie,
    # Y = Log[1 + exp{X}] ==> X = Log[exp{Y} - 1]
    # ==> dX/dY = exp{Y} / (exp{Y} - 1)
    #           = 1 / (1 - exp{-Y}),
    # which is the most stable for large Y > 0. For small Y, we use
    # 1 - exp{-Y} approx Y.
    return -math_ops.reduce_sum(math_ops.log(-math_ops.expm1(-y)),
                                axis=self._event_dims_tensor(y))

  def _forward_log_det_jacobian(self, x):
    return -math_ops.reduce_sum(nn_ops.softplus(-x),
                                axis=self._event_dims_tensor(x))
