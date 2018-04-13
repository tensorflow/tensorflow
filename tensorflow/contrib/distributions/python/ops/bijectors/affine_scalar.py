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
"""Affine bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bijector


__all__ = [
    "AffineScalar",
]


class AffineScalar(bijector.Bijector):
  """Compute `Y = g(X; shift, scale) = scale * X + shift`.

  Examples:

  ```python
  # Y = X
  b = AffineScalar()

  # Y = X + shift
  b = AffineScalar(shift=[1., 2, 3])

  # Y = 2 * X + shift
  b = AffineScalar(
    shift=[1., 2, 3],
    scale=2.)
  ```

  """

  def __init__(self,
               shift=None,
               scale=None,
               validate_args=False,
               name="affine_scalar"):
    """Instantiates the `AffineScalar` bijector.

    This `Bijector` is initialized with `shift` `Tensor` and `scale` arguments,
    giving the forward operation:

    ```none
    Y = g(X) = scale * X + shift
    ```

    if `scale` is not specified, then the bijector has the semantics of
    `scale = 1.`. Similarly, if `shift` is not specified, then the bijector
    has the semantics of `shift = 0.`.

    Args:
      shift: Floating-point `Tensor`. If this is set to `None`, no shift is
        applied.
      scale: Floating-point `Tensor`. If this is set to `None`, no scale is
        applied.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    self._graph_parents = []
    self._name = name
    self._validate_args = validate_args

    with self._name_scope("init", values=[scale, shift]):
      self._shift = shift
      self._scale = scale

      if self._shift is not None:
        self._shift = ops.convert_to_tensor(shift, name="shift")

      if self._scale is not None:
        self._scale = ops.convert_to_tensor(self._scale, name="scale")
        if validate_args:
          self._scale = control_flow_ops.with_dependencies(
              [check_ops.assert_none_equal(
                  self._scale,
                  array_ops.zeros([], dtype=self._scale.dtype))],
              self._scale)

      super(AffineScalar, self).__init__(
          forward_min_event_ndims=0,
          is_constant_jacobian=True,
          validate_args=validate_args,
          name=name)

  @property
  def shift(self):
    """The `shift` `Tensor` in `Y = scale @ X + shift`."""
    return self._shift

  @property
  def scale(self):
    """The `scale` `LinearOperator` in `Y = scale @ X + shift`."""
    return self._scale

  def _forward(self, x):
    y = array_ops.identity(x)
    if self.scale is not None:
      y *= self.scale
    if self.shift is not None:
      y += self.shift
    return y

  def _inverse(self, y):
    x = array_ops.identity(y)
    if self.shift is not None:
      x -= self.shift
    if self.scale is not None:
      x /= self.scale
    return x

  def _forward_log_det_jacobian(self, x):
    # is_constant_jacobian = True for this bijector, hence the
    # `log_det_jacobian` need only be specified for a single input, as this will
    # be tiled to match `event_ndims`.
    if self.scale is None:
      return constant_op.constant(0., dtype=x.dtype.base_dtype)

    return math_ops.log(math_ops.abs(self.scale))
