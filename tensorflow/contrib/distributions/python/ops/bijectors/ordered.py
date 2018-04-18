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
"""Ordered bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.distributions import bijector


__all__ = [
    "Ordered",
]


class Ordered(bijector.Bijector):
  """Bijector which maps a tensor x_k that has increasing elements in the last
  dimension to an unconstrained tensor y_k.

  The inverse of the bijector applied to a normal random vector `X ~ N(0, 1)`
  gives back a sorted random vector with the same distribution `Y ~ N(0, 1)`

  On the last dimension of the tensor, Ordered bijector performs:
  `y[0] = x[0]`
  `y[1:] = math_ops.log(x[1:] - x[:-1])`

  Example Use:

  ```python
  bijector.Ordered().forward(tf.log([2, 3, 4]))
  # Result: [0.6931472, 3.6931472, 7.693147]

  bijector.Ordered().inverse([0.2, 0.3, 0.4])
  # Result: tf.log([2, 3, 4])
  ```
  """

  def __init__(self,
               validate_args=False,
               name="ordered"):
    self._graph_parents = []
    self._name = name
    super(Ordered, self).__init__(
        forward_min_event_ndims=1,
        validate_args=validate_args,
        name=name)

  def _forward_event_shape(self, input_shape):
    if input_shape.ndims is None or input_shape[-1] is None:
      return input_shape
    return tensor_shape.TensorShape([input_shape[-1]])

  def _forward_event_shape_tensor(self, input_shape):
    return (input_shape[-1])[..., array_ops.newaxis]

  def _inverse_event_shape(self, output_shape):
    if output_shape.ndims is None or output_shape[-1] is None:
      return output_shape
    if output_shape[-1] <= 1:
      raise ValueError("output_shape[-1] = %d <= 1" % output_shape[-1])
    return tensor_shape.TensorShape([output_shape[-1]])

  def _inverse_event_shape_tensor(self, output_shape):
    if self.validate_args:
      is_greater_one = check_ops.assert_greater(
          output_shape[-1], 1, message="Need last dimension greater than 1.")
      output_shape = control_flow_ops.with_dependencies(
          [is_greater_one], output_shape)
    return (output_shape[-1])[..., array_ops.newaxis]

  def _forward(self, x):
    x = self._maybe_assert_valid_x(x)
    y0 = array_ops.expand_dims(x[..., 0], -1)
    yk = math_ops.log(x[..., 1:] - x[..., :-1])
    y = array_ops.concat([y0, yk], axis=-1)
    return y

  def _inverse(self, y):
    x0 = array_ops.expand_dims(y[..., 0], -1)
    xk = math_ops.exp(y[..., 1:])
    x = array_ops.concat([x0, xk], axis=-1)
    return math_ops.cumsum(x, axis=-1)

  def _inverse_log_det_jacobian(self, y):
    return math_ops.reduce_sum(y[..., 1:], axis=-1)

  def _forward_log_det_jacobian(self, x):
    pass

  def _maybe_assert_valid_x(self, x):
    if not self.validate_args:
      return x
    is_valid = check_ops.assert_positive(
        x[..., 1:] - x[..., :-1],
        message="Forward transformation input must be strictly increasing.")
    return control_flow_ops.with_dependencies([is_valid], x)