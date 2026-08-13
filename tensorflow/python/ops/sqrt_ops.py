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
"""Public square-root operation with dtype-specific gradient handling."""

from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


@custom_gradient.custom_gradient
def _safe_subnormal_derivative(
    x, safe_mantissa_bits, is_positive_subnormal
):
  """Computes a subnormal sqrt derivative with stable higher gradients."""
  zero_f64 = constant_op.constant(0.0, dtype=dtypes.float64)
  one_f64 = constant_op.constant(1.0, dtype=dtypes.float64)
  mantissa = gen_math_ops.cast(safe_mantissa_bits, dtypes.float64)
  value = (
      constant_op.constant(2.0**536, dtype=dtypes.float64)
      / gen_math_ops.sqrt(mantissa)
  )

  def grad_fn(output_grad):
    has_gradient = gen_math_ops.logical_and(
        is_positive_subnormal,
        gen_math_ops.not_equal(output_grad, zero_f64),
    )
    safe_x = array_ops.where_v2(has_gradient, x, one_f64)
    safe_value = array_ops.where_v2(has_gradient, value, zero_f64)
    return output_grad * (-0.5 * safe_value / safe_x), None, None

  return value, grad_fn


@custom_gradient.custom_gradient
def _stable_x_derivative(x, result):
  """Computes the sqrt input derivative with stable higher gradients."""
  zero_f64 = constant_op.constant(0.0, dtype=dtypes.float64)
  one_f64 = constant_op.constant(1.0, dtype=dtypes.float64)
  value = -0.5 * result / x

  def grad_fn(output_grad):
    has_gradient = gen_math_ops.not_equal(output_grad, zero_f64)
    safe_x = array_ops.where_v2(has_gradient, x, one_f64)
    safe_value = array_ops.where_v2(has_gradient, value, zero_f64)
    return (
        output_grad * (-safe_value / safe_x),
        output_grad * (-0.5 / safe_x),
    )

  return value, grad_fn


@custom_gradient.custom_gradient
def _sqrt_grad_with_subnormal_fallback(x, grad):
  """Computes a sqrt gradient that is stable for float64 subnormals."""
  # A positive subnormal has no exponent bits and represents
  # mantissa * 2**-1074. Therefore, 1 / (2 * sqrt(x)) is
  # 2**536 / sqrt(mantissa).
  zero_i64 = constant_op.constant(0, dtype=dtypes.int64)
  one_i64 = constant_op.constant(1, dtype=dtypes.int64)
  zero_f64 = constant_op.constant(0.0, dtype=dtypes.float64)
  one_f64 = constant_op.constant(1.0, dtype=dtypes.float64)
  x_bits = array_ops.bitcast(x, dtypes.int64)
  min_normal_bits = constant_op.constant(1 << 52, dtype=dtypes.int64)
  is_positive_subnormal = gen_math_ops.logical_and(
      gen_math_ops.greater(x_bits, zero_i64),
      gen_math_ops.less(x_bits, min_normal_bits),
  )
  safe_mantissa_bits = array_ops.where_v2(
      is_positive_subnormal, x_bits, one_i64
  )
  subnormal_derivative = _safe_subnormal_derivative(
      x, safe_mantissa_bits, is_positive_subnormal
  )
  is_not_positive_subnormal = gen_math_ops.logical_not(
      is_positive_subnormal
  )
  safe_ordinary_x = array_ops.where_v2(
      is_not_positive_subnormal, x, one_f64
  )
  ordinary_y = gen_math_ops.sqrt(safe_ordinary_x)
  ordinary_result = gen_math_ops.sqrt_grad(ordinary_y, grad)
  result = array_ops.where_v2(
      is_positive_subnormal,
      grad * subnormal_derivative,
      ordinary_result,
  )

  def grad_fn(output_grad):
    # Express the subnormal second derivative in terms of the first so it
    # overflows with the correct sign. Returning it for x directly avoids
    # differentiating through the discrete bit representation.
    use_x_gradient = gen_math_ops.logical_or(
        is_not_positive_subnormal, gen_math_ops.not_equal(grad, zero_f64)
    )
    safe_x = array_ops.where_v2(use_x_gradient, x, one_f64)
    safe_result = array_ops.where_v2(use_x_gradient, result, zero_f64)
    x_derivative = _stable_x_derivative(safe_x, safe_result)
    ordinary_derivative = 0.5 / ordinary_y
    grad_derivative = array_ops.where_v2(
        is_positive_subnormal,
        subnormal_derivative,
        ordinary_derivative,
    )
    return (
        output_grad * x_derivative,
        output_grad * grad_derivative,
    )

  return result, grad_fn


@custom_gradient.custom_gradient
def _sqrt_float64(x):
  """Computes float64 sqrt while retaining its input only for its gradient."""
  y = gen_math_ops.sqrt(x)

  def grad_fn(grad):
    return _sqrt_grad_with_subnormal_fallback(x, grad)

  return y, grad_fn


@tf_export("math.sqrt", "sqrt")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def sqrt(x, name=None):  # pylint: disable=redefined-builtin
  r"""Computes element-wise square root of the input tensor.

  Note: This operation does not support integer types.

  >>> x = tf.constant([[4.0], [16.0]])
  >>> tf.sqrt(x)
  <tf.Tensor: shape=(2, 1), dtype=float32, numpy=
    array([[2.],
           [4.]], dtype=float32)>
  >>> y = tf.constant([[-4.0], [16.0]])
  >>> tf.sqrt(y)
  <tf.Tensor: shape=(2, 1), dtype=float32, numpy=
    array([[nan],
           [ 4.]], dtype=float32)>
  >>> z = tf.constant([[-1.0], [16.0]], dtype=tf.complex128)
  >>> tf.sqrt(z)
  <tf.Tensor: shape=(2, 1), dtype=complex128, numpy=
    array([[0.0+1.j],
           [4.0+0.j]])>

  Note: In order to support complex type, please provide an input tensor
  of `complex64` or `complex128`.

  Args:
    x: A `tf.Tensor` of type `bfloat16`, `half`, `float32`, `float64`,
      `complex64`, `complex128`
    name: A name for the operation (optional).

  Returns:
    A `tf.Tensor` of same size, type and sparsity as `x`.
  """
  # SparseTensor's unary dispatcher is registered on math_ops.sqrt, so route
  # composite inputs there before attempting dense tensor conversion.
  if isinstance(x, composite_tensor.CompositeTensor):
    return math_ops.sqrt(x, name=name)
  with ops.name_scope(name, "Sqrt", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    if x.dtype == dtypes.float64:
      return _sqrt_float64(x)
    return math_ops.sqrt(x, name=name)
