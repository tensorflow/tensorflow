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
"""Inline bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.distributions import bijector


__all__ = [
    "Inline",
]


class Inline(bijector.Bijector):
  """Bijector constructed from custom callables.

  Example Use:

  ```python
  exp = Inline(
    forward_fn=tf.exp,
    inverse_fn=tf.log,
    inverse_log_det_jacobian_fn=(
      lambda y: -tf.reduce_sum(tf.log(y), axis=-1)),
    name="exp")
  ```

  The above example is equivalent to the `Bijector` `Exp()`.
  """

  def __init__(self,
               forward_fn=None,
               inverse_fn=None,
               inverse_log_det_jacobian_fn=None,
               forward_log_det_jacobian_fn=None,
               forward_event_shape_fn=None,
               forward_event_shape_tensor_fn=None,
               inverse_event_shape_fn=None,
               inverse_event_shape_tensor_fn=None,
               is_constant_jacobian=False,
               validate_args=False,
               forward_min_event_ndims=None,
               inverse_min_event_ndims=None,
               name="inline"):
    """Creates a `Bijector` from callables.

    Args:
      forward_fn: Python callable implementing the forward transformation.
      inverse_fn: Python callable implementing the inverse transformation.
      inverse_log_det_jacobian_fn: Python callable implementing the
        log o det o jacobian of the inverse transformation.
      forward_log_det_jacobian_fn: Python callable implementing the
        log o det o jacobian of the forward transformation.
      forward_event_shape_fn: Python callable implementing non-identical
        static event shape changes. Default: shape is assumed unchanged.
      forward_event_shape_tensor_fn: Python callable implementing non-identical
        event shape changes. Default: shape is assumed unchanged.
      inverse_event_shape_fn: Python callable implementing non-identical
        static event shape changes. Default: shape is assumed unchanged.
      inverse_event_shape_tensor_fn: Python callable implementing non-identical
        event shape changes. Default: shape is assumed unchanged.
      is_constant_jacobian: Python `bool` indicating that the Jacobian is
        constant for all input arguments.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      forward_min_event_ndims: Python `int` indicating the minimal
        dimensionality this bijector acts on.
      inverse_min_event_ndims: Python `int` indicating the minimal
        dimensionality this bijector acts on.
      name: Python `str`, name given to ops managed by this object.
    """
    super(Inline, self).__init__(
        forward_min_event_ndims=forward_min_event_ndims,
        inverse_min_event_ndims=inverse_min_event_ndims,
        is_constant_jacobian=is_constant_jacobian,
        validate_args=validate_args,
        name=name)
    self._forward_fn = forward_fn
    self._inverse_fn = inverse_fn
    self._inverse_log_det_jacobian_fn = inverse_log_det_jacobian_fn
    self._forward_log_det_jacobian_fn = forward_log_det_jacobian_fn
    self._forward_event_shape_fn = forward_event_shape_fn
    self._forward_event_shape_tensor_fn = forward_event_shape_tensor_fn
    self._inverse_event_shape_fn = inverse_event_shape_fn
    self._inverse_event_shape_tensor_fn = inverse_event_shape_tensor_fn

  def _forward_event_shape(self, input_shape):
    if self._forward_event_shape_fn is None:
      # By default assume shape doesn't change.
      return input_shape
    return self._forward_event_shape_fn(input_shape)

  def _forward_event_shape_tensor(self, input_shape):
    if self._forward_event_shape_tensor_fn is None:
      # By default assume shape doesn't change.
      return input_shape
    return self._forward_event_shape_tensor_fn(input_shape)

  def _inverse_event_shape(self, output_shape):
    if self._inverse_event_shape_fn is None:
      # By default assume shape doesn't change.
      return output_shape
    return self._inverse_event_shape_fn(output_shape)

  def _inverse_event_shape_tensor(self, output_shape):
    if self._inverse_event_shape_tensor_fn is None:
      # By default assume shape doesn't change.
      return output_shape
    return self._inverse_event_shape_tensor_fn(output_shape)

  def _forward(self, x, **kwargs):
    if not callable(self._forward_fn):
      raise NotImplementedError(
          "forward_fn is not a callable function.")
    return self._forward_fn(x, **kwargs)

  def _inverse(self, y, **kwargs):
    if not callable(self._inverse_fn):
      raise NotImplementedError(
          "inverse_fn is not a callable function.")
    return self._inverse_fn(y, **kwargs)

  def _inverse_log_det_jacobian(self, y, **kwargs):
    if not callable(self._inverse_log_det_jacobian_fn):
      raise NotImplementedError(
          "inverse_log_det_jacobian_fn is not a callable function.")
    return self._inverse_log_det_jacobian_fn(y, **kwargs)

  def _forward_log_det_jacobian(self, x, **kwargs):
    if not callable(self._forward_log_det_jacobian_fn):
      raise NotImplementedError(
          "forward_log_det_jacobian_fn is not a callable function.")
    return self._forward_log_det_jacobian_fn(x, **kwargs)
