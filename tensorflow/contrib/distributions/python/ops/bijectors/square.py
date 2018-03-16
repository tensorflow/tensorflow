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
"""Square bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bijector


__all__ = [
    "Square",
]


class Square(bijector.Bijector):
  """Compute `g(X) = X^2`; X is a positive real number.

  g is a bijection between the non-negative real numbers (R_+) and the
  non-negative real numbers.

  Examples:

  ```python
  bijector.Square().forward(x=[[1., 0], [2, 1]])
  # Result: [[1., 0], [4, 1]], i.e., x^2

  bijector.Square().inverse(y=[[1., 4], [9, 1]])
  # Result: [[1., 2], [3, 1]], i.e., sqrt(y).
  ```

  """

  def __init__(self, validate_args=False, name="square"):
    """Instantiates the `Square` bijector.

    Args:
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    self._name = name
    super(Square, self).__init__(
        event_ndims=0,
        validate_args=validate_args,
        name=name)

  def _forward(self, x):
    x = self._maybe_assert_valid(x)
    return math_ops.square(x)

  def _inverse(self, y):
    y = self._maybe_assert_valid(y)
    return math_ops.sqrt(y)

  def _forward_log_det_jacobian(self, x):
    x = self._maybe_assert_valid(x)
    return np.log(2.) + math_ops.log(x)

  def _maybe_assert_valid(self, t):
    if not self.validate_args:
      return t
    is_valid = check_ops.assert_non_negative(
        t, message="All elements must be non-negative.")
    return control_flow_ops.with_dependencies([is_valid], t)

