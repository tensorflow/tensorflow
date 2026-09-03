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
"""Numerically stable square-root gradients for float64 subnormals."""

import functools

from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops


def _custom_gradient(function):
  """Defines a custom gradient for a single Tensor result and Tensor inputs.

  This specialized wrapper keeps sqrt gradient registration independent of
  `custom_gradient`, which depends on `backprop` and therefore on `math_grad`.
  """

  @functools.wraps(function)
  def decorated(*args):
    args = tuple(ops.convert_to_tensor(arg) for arg in args)
    result, grad_fn = function(*args)
    result = ops.convert_to_tensor(result)

    if context.executing_eagerly():
      result = array_ops.identity(result)

      def eager_tape_grad_fn(output_grad):
        return list(grad_fn(output_grad))

      record.record_operation(
          function.__name__, [result], list(args), eager_tape_grad_fn
      )
      return result

    gradient_name = "SqrtCustomGradient-%s" % ops.uid()

    def graph_tape_grad_fn(output_grad, *unused_output_grads):
      del unused_output_grads
      return [None] + list(grad_fn(output_grad))

    @ops.RegisterGradient(gradient_name)
    def internal_grad_fn(unused_op, *output_grads):
      del unused_op
      return graph_tape_grad_fn(*output_grads)

    original_tensors = [result] + list(args)
    with ops.get_default_graph().gradient_override_map(
        {"IdentityN": gradient_name}
    ):
      wrapped_tensors = array_ops.identity_n(original_tensors)
    record.record_operation(
        function.__name__,
        wrapped_tensors,
        original_tensors,
        graph_tape_grad_fn,
    )
    return wrapped_tensors[0]

  return decorated


def _divide_by_positive_subnormal(numerator, mantissa):
  """Divides by `mantissa * 2**-1074` without using a subnormal."""
  scale = constant_op.constant(2.0**537, dtype=dtypes.float64)
  # Dividing between the two scaling steps limits intermediate overflow.
  return (numerator * scale / mantissa) * scale


@_custom_gradient
def _safe_subnormal_derivative(x, mantissa, is_positive_subnormal):
  """Computes a subnormal sqrt derivative with stable higher gradients."""
  zero_f64 = constant_op.constant(0.0, dtype=dtypes.float64)
  one_f64 = constant_op.constant(1.0, dtype=dtypes.float64)
  value = (
      constant_op.constant(2.0**536, dtype=dtypes.float64)
      / gen_math_ops.sqrt(mantissa)
  )

  def grad_fn(output_grad):
    has_gradient = gen_math_ops.logical_and(
        is_positive_subnormal,
        gen_math_ops.not_equal(output_grad, zero_f64),
    )
    safe_output_grad = array_ops.where_v2(
        has_gradient, output_grad, zero_f64
    )
    safe_mantissa = array_ops.where_v2(has_gradient, mantissa, one_f64)
    safe_value = array_ops.where_v2(has_gradient, value, zero_f64)
    scaled_grad = _divide_by_positive_subnormal(
        safe_output_grad, safe_mantissa
    )
    return scaled_grad * (-0.5 * safe_value), None, None

  return value, grad_fn


@_custom_gradient
def _stable_x_derivative(x, result, is_positive_subnormal, mantissa):
  """Computes the sqrt input derivative with stable higher gradients."""
  zero_f64 = constant_op.constant(0.0, dtype=dtypes.float64)
  one_f64 = constant_op.constant(1.0, dtype=dtypes.float64)
  is_not_positive_subnormal = gen_math_ops.logical_not(
      is_positive_subnormal
  )
  safe_normal_x = array_ops.where_v2(
      is_not_positive_subnormal, x, one_f64
  )
  safe_subnormal_result = array_ops.where_v2(
      is_positive_subnormal, result, zero_f64
  )
  safe_mantissa = array_ops.where_v2(
      is_positive_subnormal, mantissa, one_f64
  )
  normal_value = -0.5 * result / safe_normal_x
  subnormal_value = -0.5 * _divide_by_positive_subnormal(
      safe_subnormal_result, safe_mantissa
  )
  value = array_ops.where_v2(
      is_positive_subnormal, subnormal_value, normal_value
  )

  def grad_fn(output_grad):
    has_gradient = gen_math_ops.not_equal(output_grad, zero_f64)
    use_subnormal_gradient = gen_math_ops.logical_and(
        is_positive_subnormal, has_gradient
    )
    use_normal_gradient = gen_math_ops.logical_and(
        is_not_positive_subnormal, has_gradient
    )
    safe_normal_output_grad = array_ops.where_v2(
        use_normal_gradient, output_grad, zero_f64
    )
    safe_normal_x = array_ops.where_v2(use_normal_gradient, x, one_f64)
    safe_normal_value = array_ops.where_v2(
        use_normal_gradient, value, zero_f64
    )
    safe_subnormal_output_grad = array_ops.where_v2(
        use_subnormal_gradient, output_grad, zero_f64
    )
    safe_subnormal_mantissa = array_ops.where_v2(
        use_subnormal_gradient, mantissa, one_f64
    )
    safe_subnormal_value = array_ops.where_v2(
        use_subnormal_gradient, value, zero_f64
    )
    scaled_subnormal_grad = _divide_by_positive_subnormal(
        safe_subnormal_output_grad, safe_subnormal_mantissa
    )
    x_gradient = array_ops.where_v2(
        is_positive_subnormal,
        scaled_subnormal_grad * -safe_subnormal_value,
        safe_normal_output_grad * (-safe_normal_value / safe_normal_x),
    )
    result_gradient = array_ops.where_v2(
        is_positive_subnormal,
        scaled_subnormal_grad * -0.5,
        safe_normal_output_grad * (-0.5 / safe_normal_x),
    )
    return (
        x_gradient,
        result_gradient,
        None,
        None,
    )

  return value, grad_fn


@_custom_gradient
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
  mantissa = gen_math_ops.cast(safe_mantissa_bits, dtypes.float64)
  subnormal_derivative = _safe_subnormal_derivative(
      x, mantissa, is_positive_subnormal
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
    x_derivative = _stable_x_derivative(
        safe_x, safe_result, is_positive_subnormal, mantissa
    )
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


def sqrt_grad(x, grad):
  """Computes the float64 sqrt gradient from its original input."""
  return _sqrt_grad_with_subnormal_fallback(x, grad)
