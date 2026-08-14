# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Numerically stable helpers for Euclidean norm gradients."""

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops


def _norm_is_safe_per_output(norm, dtype):
  """Checks which norms are outside underflow and overflow regions."""
  real_dtype = dtype.real_dtype
  finfo_dtype = (
      np.float32 if real_dtype == dtypes.bfloat16 else
      real_dtype.as_numpy_dtype
  )
  safe_norm_threshold = np.sqrt(np.finfo(finfo_dtype).tiny)
  if dtype.is_complex:
    real_norm = math_ops.real(norm)
    norm_is_finite = math_ops.logical_and(
        math_ops.is_finite(real_norm),
        math_ops.is_finite(math_ops.imag(norm)),
    )
  else:
    real_norm = norm
    norm_is_finite = math_ops.is_finite(norm)
  return math_ops.logical_and(
      norm_is_finite,
      math_ops.greater(math_ops.abs(real_norm), safe_norm_threshold),
  )


def _component_abs(tensor):
  """Returns a scaling magnitude that cannot overflow for complex inputs."""
  if tensor.dtype.is_complex:
    return math_ops.maximum(
        math_ops.abs(math_ops.real(tensor)),
        math_ops.abs(math_ops.imag(tensor)),
    )
  return math_ops.abs(tensor)


def _norm_is_safe(norm, dtype):
  """Checks whether all norms avoid underflow and overflow regions."""
  return math_ops.reduce_all(_norm_is_safe_per_output(norm, dtype))


def _reduction_scale(tensor, axis):
  """Returns a non-differentiable component scale for each reduction."""
  return array_ops.stop_gradient(
      math_ops.reduce_max(_component_abs(tensor), axis, keepdims=True)
  )


def _divide_by_real_denominator(tensor, denominator):
  """Divides real or complex components without squaring the denominator."""
  if tensor.dtype.is_complex:
    return math_ops.complex(
        math_ops.div_no_nan(math_ops.real(tensor), denominator),
        math_ops.div_no_nan(math_ops.imag(tensor), denominator),
    )
  return math_ops.div_no_nan(tensor, denominator)


def _scaled_euclidean_norm_direction(tensor, axis, scale):
  """Computes a Euclidean norm direction after scaling input components."""
  if tensor.dtype.real_dtype in (dtypes.float16, dtypes.bfloat16):
    calculation_tensor = math_ops.cast(tensor, dtypes.float32)
    calculation_scale = math_ops.cast(scale, dtypes.float32)
  else:
    calculation_tensor = tensor
    calculation_scale = scale
  scaled_tensor = _divide_by_real_denominator(
      calculation_tensor, calculation_scale
  )
  if calculation_tensor.dtype.is_complex:
    squared_components = (
        math_ops.square(math_ops.real(scaled_tensor))
        + math_ops.square(math_ops.imag(scaled_tensor))
    )
  else:
    squared_components = math_ops.square(scaled_tensor)
  scaled_norm = math_ops.sqrt(
      math_ops.reduce_sum(
          squared_components,
          axis,
          keepdims=True,
      )
  )
  direction = _divide_by_real_denominator(scaled_tensor, scaled_norm)
  return math_ops.cast(direction, tensor.dtype)


def safe_euclidean_norm(tensor, norm, axis):
  """Preserves `norm`'s value while substituting a stable risky-case gradient."""

  def scaled_result():
    scale = _reduction_scale(tensor, axis)
    finite_scale = math_ops.is_finite(scale)
    direction = _scaled_euclidean_norm_direction(tensor, axis, scale)
    direction = array_ops.where_v2(
        finite_scale, direction, array_ops.zeros_like(tensor)
    )
    finite_tensor = array_ops.where_v2(
        finite_scale, tensor, array_ops.zeros_like(tensor)
    )
    zero_with_gradient = finite_tensor - array_ops.stop_gradient(finite_tensor)
    gradient_surrogate = math_ops.reduce_sum(
        zero_with_gradient * math_ops.conj(array_ops.stop_gradient(direction)),
        axis,
        keepdims=True,
    )
    use_scaled_gradient = math_ops.logical_and(
        math_ops.logical_not(_norm_is_safe_per_output(norm, tensor.dtype)),
        finite_scale,
    )
    stable_result = array_ops.stop_gradient(norm) + gradient_surrogate
    return array_ops.where_v2(use_scaled_gradient, stable_result, norm)

  return cond.cond(
      _norm_is_safe(norm, tensor.dtype),
      lambda: norm,
      scaled_result,
  )


def safe_euclidean_norm_grad(tensor, norm, axis):
  """Returns a stable Euclidean norm gradient direction.

  For ordinary finite norms this computes `tensor / norm`. If the norm is
  zero, close enough to zero that squaring may have underflowed, or non-finite
  despite finite input components, the direction is recomputed after scaling
  by the largest input component. Scaling keeps the squared reduction in a
  representable range without changing the direction for any nonzero finite
  input. Inputs containing non-finite components retain the direct behavior.

  Args:
    tensor: The input to the Euclidean norm.
    norm: The norm result, with reduced dimensions retained.
    axis: The dimensions reduced by the norm.

  Returns:
    A tensor with the same shape and dtype as `tensor` containing the gradient
    direction. The minimum-norm subgradient zero is returned at the origin.
  """
  def direct_direction():
    return math_ops.truediv(tensor, norm)

  def scaled_direction():
    scale = _reduction_scale(tensor, axis)
    direction = _scaled_euclidean_norm_direction(tensor, axis, scale)
    direct = direct_direction()
    fallback_direction = array_ops.where_v2(
        math_ops.is_finite(scale), direction, direct
    )
    return array_ops.where_v2(
        _norm_is_safe_per_output(norm, tensor.dtype),
        direct,
        fallback_direction,
    )

  return cond.cond(
      _norm_is_safe(norm, tensor.dtype), direct_direction, scaled_direction
  )
