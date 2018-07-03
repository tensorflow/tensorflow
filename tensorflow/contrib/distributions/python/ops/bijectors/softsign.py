# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Softsign bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bijector
from tensorflow.python.util import deprecation


__all__ = [
    "Softsign",
]


class Softsign(bijector.Bijector):
  """Bijector which computes `Y = g(X) = X / (1 + |X|)`.

  The softsign `Bijector` has the following two useful properties:

  * The domain is all real numbers
  * `softsign(x) approx sgn(x)`, for large `|x|`.

  #### Examples

  ```python
  # Create the Y = softsign(X) transform.
  softsign = Softsign()
  x = [[[1., 2],
        [3, 4]],
       [[5, 6],
        [7, 8]]]
  x / (1 + abs(x)) == softsign.forward(x)
  x / (1 - abs(x)) == softsign.inverse(x)
  ```
  """

  @deprecation.deprecated(
      "2018-10-01",
      "The TensorFlow Distributions library has moved to "
      "TensorFlow Probability "
      "(https://github.com/tensorflow/probability). You "
      "should update all references to use `tfp.distributions` "
      "instead of `tf.contrib.distributions`.",
      warn_once=True)
  def __init__(self, validate_args=False, name="softsign"):
    super(Softsign, self).__init__(
        forward_min_event_ndims=0,
        validate_args=validate_args,
        name=name)

  def _forward(self, x):
    return x / (1. + math_ops.abs(x))

  def _inverse(self, y):
    y = self._maybe_assert_valid_y(y)
    return y / (1. - math_ops.abs(y))

  def _forward_log_det_jacobian(self, x):
    return -2. * math_ops.log1p(math_ops.abs(x))

  def _inverse_log_det_jacobian(self, y):
    y = self._maybe_assert_valid_y(y)
    return -2. * math_ops.log1p(-math_ops.abs(y))

  def _maybe_assert_valid_y(self, y):
    if not self.validate_args:
      return y
    is_valid = [
        check_ops.assert_greater(
            y, math_ops.cast(-1., dtype=y.dtype.base_dtype),
            message="Inverse transformation input must be greater than -1."),
        check_ops.assert_less(
            y, math_ops.cast(1., dtype=y.dtype.base_dtype),
            message="Inverse transformation input must be less than 1.")
    ]

    return control_flow_ops.with_dependencies(is_valid, y)
