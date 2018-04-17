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
"""Invert bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.distributions import bijector as bijector_lib

__all__ = [
    "Invert",
]


class Invert(bijector_lib.Bijector):
  """Bijector which inverts another Bijector.

  Example Use: [ExpGammaDistribution (see Background & Context)](
  https://reference.wolfram.com/language/ref/ExpGammaDistribution.html)
  models `Y=log(X)` where `X ~ Gamma`.

  ```python
  exp_gamma_distribution = TransformedDistribution(
    distribution=Gamma(concentration=1., rate=2.),
    bijector=bijector.Invert(bijector.Exp())
  ```

  """

  def __init__(self, bijector, validate_args=False, name=None):
    """Creates a `Bijector` which swaps the meaning of `inverse` and `forward`.

    Note: An inverted bijector's `inverse_log_det_jacobian` is often more
    efficient if the base bijector implements `_forward_log_det_jacobian`. If
    `_forward_log_det_jacobian` is not implemented then the following code is
    used:

    ```python
    y = self.inverse(x, **kwargs)
    return -self.inverse_log_det_jacobian(y, **kwargs)
    ```

    Args:
      bijector: Bijector instance.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str`, name given to ops managed by this object.
    """

    if not bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError(
          "Invert is not implemented for non-injective bijectors.")

    self._bijector = bijector
    super(Invert, self).__init__(
        graph_parents=bijector.graph_parents,
        forward_min_event_ndims=bijector.inverse_min_event_ndims,
        inverse_min_event_ndims=bijector.forward_min_event_ndims,
        is_constant_jacobian=bijector.is_constant_jacobian,
        validate_args=validate_args,
        dtype=bijector.dtype,
        name=name or "_".join(["invert", bijector.name]))

  def _forward_event_shape(self, input_shape):
    return self.bijector._inverse_event_shape(input_shape)  # pylint: disable=protected-access

  def _forward_event_shape_tensor(self, input_shape):
    return self.bijector._inverse_event_shape_tensor(input_shape)  # pylint: disable=protected-access

  def _inverse_event_shape(self, output_shape):
    return self.bijector._forward_event_shape(output_shape)  # pylint: disable=protected-access

  def _inverse_event_shape_tensor(self, output_shape):
    return self.bijector._forward_event_shape_tensor(output_shape)  # pylint: disable=protected-access

  @property
  def bijector(self):
    return self._bijector

  def _forward(self, x, **kwargs):
    return self.bijector._inverse(x, **kwargs)  # pylint: disable=protected-access

  def _inverse(self, y, **kwargs):
    return self.bijector._forward(y, **kwargs)  # pylint: disable=protected-access

  def _inverse_log_det_jacobian(self, y, **kwargs):
    return self.bijector._forward_log_det_jacobian(y, **kwargs)  # pylint: disable=protected-access

  def _forward_log_det_jacobian(self, x, **kwargs):
    return self.bijector._inverse_log_det_jacobian(x, **kwargs)  # pylint: disable=protected-access
