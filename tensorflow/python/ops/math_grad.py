# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Gradients for operators defined in math_ops.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops


def _safe_shape_div(x, y):
  """Divides `x / y` assuming `x, y >= 0`, treating `0 / 0 = 0`."""
  return x // math_ops.maximum(y, 1)


@ops.RegisterGradient("Sum")
def _SumGrad(op, grad):
  """Gradient for Sum."""
  input_shape = array_ops.shape(op.inputs[0])
  output_shape_kept_dims = math_ops.reduced_shape(input_shape, op.inputs[1])
  tile_scaling = _safe_shape_div(input_shape, output_shape_kept_dims)
  grad = array_ops.reshape(grad, output_shape_kept_dims)
  return [array_ops.tile(grad, tile_scaling), None]


def _MinOrMaxGrad(op, grad):
  """Gradient for Max or Max. Amazingly it's precisely the same code."""
  input_shape = array_ops.shape(op.inputs[0])
  output_shape_kept_dims = math_ops.reduced_shape(input_shape, op.inputs[1])
  y = op.outputs[0]
  y = array_ops.reshape(y, output_shape_kept_dims)
  grad = array_ops.reshape(grad, output_shape_kept_dims)

  # Compute the number of selected (maximum or minimum) elements in each
  # reduction dimension. If there are multiple minimum or maximum elements
  # then the gradient will be divided between them.
  indicators = math_ops.cast(math_ops.equal(y, op.inputs[0]), grad.dtype)
  num_selected = array_ops.reshape(
      math_ops.reduce_sum(indicators, op.inputs[1]),
      output_shape_kept_dims)

  return [math_ops.div(indicators, num_selected) * grad, None]


@ops.RegisterGradient("Max")
def _MaxGrad(op, grad):
  """Gradient for Max."""
  return _MinOrMaxGrad(op, grad)


@ops.RegisterGradient("Min")
def _MinGrad(op, grad):
  return _MinOrMaxGrad(op, grad)


@ops.RegisterGradient("Mean")
def _MeanGrad(op, grad):
  """Gradient for Mean."""
  sum_grad = _SumGrad(op, grad)[0]
  input_shape = array_ops.shape(op.inputs[0])
  output_shape = array_ops.shape(op.outputs[0])
  factor = _safe_shape_div(math_ops.reduce_prod(input_shape),
                           math_ops.reduce_prod(output_shape))
  return sum_grad / math_ops.cast(factor, sum_grad.dtype), None


@ops.RegisterGradient("Prod")
def _ProdGrad(op, grad):
  """Gradient for Prod."""
  # TODO(kearnes): this gives NaNs for 0s in the input tensor
  input_shape = array_ops.shape(op.inputs[0])
  output_shape_kept_dims = math_ops.reduced_shape(input_shape, op.inputs[1])
  tile_scaling = _safe_shape_div(input_shape, output_shape_kept_dims)
  grad = array_ops.reshape(grad * op.outputs[0], output_shape_kept_dims)
  grad = math_ops.div(array_ops.tile(grad, tile_scaling), op.inputs[0])
  return grad, None


@ops.RegisterGradient("SegmentSum")
def _SegmentSumGrad(op, grad):
  """Gradient for SegmentSum."""
  return array_ops.gather(grad, op.inputs[1]), None


@ops.RegisterGradient("SegmentMean")
def _SegmentMeanGrad(op, grad):
  """Gradient for SegmentMean."""
  input_rank = array_ops.rank(op.inputs[0])
  ones_shape = array_ops.concat(
      0, [array_ops.shape(op.inputs[1]),
          array_ops.fill(array_ops.expand_dims(input_rank - 1, 0), 1)])
  ones = array_ops.fill(ones_shape,
                        constant_op.constant(1, dtype=grad.dtype))
  scaled_grad = grad * math_ops.inv(math_ops.segment_sum(ones, op.inputs[1]))
  return array_ops.gather(scaled_grad, op.inputs[1]), None


@ops.RegisterGradient("SparseSegmentSum")
def _SparseSegmentSumGrad(op, grad):
  """Gradient for SparseSegmentSum."""
  input_rows = array_ops.shape(op.inputs[0])[0]
  return (math_ops.unsorted_segment_sum(
      array_ops.gather(grad, op.inputs[2]),
      op.inputs[1], input_rows), None, None)


@ops.RegisterGradient("SparseSegmentMean")
def _SparseSegmentMeanGrad(op, grad):
  """Gradient for SparseSegmentMean."""
  dim0 = array_ops.shape(op.inputs[0])[0]
  return (math_ops.sparse_segment_mean_grad(grad,
                                            op.inputs[1],
                                            op.inputs[2],
                                            dim0),
          None, None)


@ops.RegisterGradient("SparseSegmentSqrtN")
def _SparseSegmentSqrtNGrad(op, grad):
  """Gradient for SparseSegmentSqrtN."""
  dim0 = array_ops.shape(op.inputs[0])[0]
  return (math_ops.sparse_segment_sqrt_n_grad(grad,
                                              op.inputs[1],
                                              op.inputs[2],
                                              dim0),
          None, None)


def _SegmentMinOrMaxGrad(op, grad):
  """Gradient for SegmentMin and SegmentMax. Both share the same code."""
  zeros = array_ops.zeros(array_ops.shape(op.inputs[0]),
                          dtype=op.inputs[0].dtype)

  # Get the number of selected (minimum or maximum) elements in each segment.
  gathered_outputs = array_ops.gather(op.outputs[0], op.inputs[1])
  is_selected = math_ops.equal(op.inputs[0], gathered_outputs)
  num_selected = math_ops.segment_sum(math_ops.cast(is_selected, grad.dtype),
                                      op.inputs[1])

  # Compute the gradient for each segment. The gradient for the ith segment is
  # divided evenly among the selected elements in that segment.
  weighted_grads = math_ops.div(grad, num_selected)
  gathered_grads = array_ops.gather(weighted_grads, op.inputs[1])

  return math_ops.select(is_selected, gathered_grads, zeros), None


@ops.RegisterGradient("SegmentMin")
def _SegmentMinGrad(op, grad):
  """Gradient for SegmentMin."""
  return _SegmentMinOrMaxGrad(op, grad)


@ops.RegisterGradient("SegmentMax")
def _SegmentMaxGrad(op, grad):
  """Gradient for SegmentMax."""
  return _SegmentMinOrMaxGrad(op, grad)


@ops.RegisterGradient("UnsortedSegmentSum")
def _UnsortedSegmentSumGrad(op, grad):
  """Gradient for SegmentSum."""
  return array_ops.gather(grad, op.inputs[1]), None, None


@ops.RegisterGradient("Abs")
def _AbsGrad(op, grad):
  x = op.inputs[0]
  return grad * math_ops.sign(x)


@ops.RegisterGradient("Neg")
def _NegGrad(_, grad):
  """Returns -grad."""
  return - grad


@ops.RegisterGradient("Inv")
def _InvGrad(op, grad):
  """Returns -grad * (1 / x^2)."""
  y = op.outputs[0]  # y = 1 / x
  # Added control dependencies to prevent -x^2 from being computed too early.
  with ops.control_dependencies([grad.op]):
    if y.dtype.is_complex:
      y = math_ops.conj(y)
    return grad * (- math_ops.square(y))


@ops.RegisterGradient("Square")
def _SquareGrad(op, grad):
  x = op.inputs[0]
  # Added control dependencies to prevent 2*x from being computed too early.
  with ops.control_dependencies([grad.op]):
    if x.dtype.is_complex:
      x = math_ops.conj(x)
    return grad * (2.0 * x)


@ops.RegisterGradient("Sqrt")
def _SqrtGrad(op, grad):
  y = op.outputs[0]  # y = x^(1/2)
  with ops.control_dependencies([grad.op]):
    return grad * (.5 * math_ops.inv(y))


@ops.RegisterGradient("Rsqrt")
def _RsqrtGrad(op, grad):
  x = op.inputs[0]
  y = op.outputs[0]  # y = x^(-1/2)
  with ops.control_dependencies([grad.op]):
    return grad * ((-0.5) * math_ops.inv(x) * y)


@ops.RegisterGradient("Exp")
def _ExpGrad(op, grad):
  """Returns grad * exp(x)."""
  y = op.outputs[0]  # y = e^x
  with ops.control_dependencies([grad.op]):
    if y.dtype.is_complex:
      y = math_ops.conj(y)
    return grad * y


@ops.RegisterGradient("Log")
def _LogGrad(op, grad):
  """Returns grad * (1/x)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad.op]):
    return grad * math_ops.inv(x)


@ops.RegisterGradient("Tanh")
def _TanhGrad(op, grad):
  """Returns grad * (1 - tanh(x) * tanh(x))."""
  y = op.outputs[0]  # y = tanh(x)
  with ops.control_dependencies([grad.op]):
    if y.dtype.is_complex:
      y = math_ops.conj(y)
    return grad * (1 - math_ops.square(y))


@ops.RegisterGradient("Erf")
def _ErfGrad(op, grad):
  """Returns grad * 2/sqrt(pi) * exp(-x**2)."""
  x = op.inputs[0]
  two_over_root_pi = constant_op.constant(2 / np.sqrt(np.pi), dtype=grad.dtype)
  with ops.control_dependencies([grad.op]):
    return  grad * two_over_root_pi * math_ops.exp(-math_ops.square(x))


@ops.RegisterGradient("Erfc")
def _ErfcGrad(op, grad):
  """Returns -grad * 2/sqrt(pi) * exp(-x**2)."""
  x = op.inputs[0]
  minus_two_over_root_pi = constant_op.constant(-2 / np.sqrt(np.pi),
                                                dtype=grad.dtype)
  with ops.control_dependencies([grad.op]):
    return  grad * minus_two_over_root_pi * math_ops.exp(-math_ops.square(x))


@ops.RegisterGradient("Lgamma")
def _LgammaGrad(op, grad):
  """Returns grad * digamma(x)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad.op]):
    return grad * math_ops.digamma(x)


@ops.RegisterGradient("Digamma")
def _DigammaGrad(op, grad):
  """Compute gradient of the digamma function with respect to its argument."""
  x = op.inputs[0]
  with ops.control_dependencies([grad.op]):
    return grad * math_ops.polygamma(array_ops.constant(1, dtype=x.dtype), x)


@ops.RegisterGradient("Igamma")
def _IgammaGrad(op, grad):
  """Returns gradient of igamma(a, x) with respect to a and x."""
  # TODO(ebrevdo): Perhaps add the derivative w.r.t. a
  a = op.inputs[0]
  x = op.inputs[1]
  sa = array_ops.shape(a)
  sx = array_ops.shape(x)
  unused_ra, rx = gen_array_ops._broadcast_gradient_args(sa, sx)

  # Perform operations in log space before summing, because Gamma(a)
  # and Gamma'(a) can grow large.
  partial_x = math_ops.exp(-x + (a-1) * math_ops.log(x) - math_ops.lgamma(a))
  return (None,
          array_ops.reshape(math_ops.reduce_sum(partial_x * grad, rx), sx))


@ops.RegisterGradient("Igammac")
def _IgammacGrad(op, grad):
  """Returns gradient of igammac(a, x) = 1 - igamma(a, x) w.r.t. a and x."""
  return [-1 * g if g is not None else None for g in _IgammaGrad(op, grad)]


@ops.RegisterGradient("Zeta")
def _ZetaGrad(op, grad):
  """Returns gradient of zeta(x, q) with respect to x and q."""
  # TODO(tillahoffmann): Add derivative with respect to x
  x = op.inputs[0]
  q = op.inputs[1]
  # Broadcast gradients
  sx = array_ops.shape(x)
  sq = array_ops.shape(q)
  unused_rx, rq = gen_array_ops._broadcast_gradient_args(sx, sq)
  # Evaluate gradient
  with ops.control_dependencies([grad.op]):
    partial_q = -x * math_ops.zeta(x + 1, q)
    return (None,
            array_ops.reshape(math_ops.reduce_sum(partial_q * grad, rq), sq))


@ops.RegisterGradient("Polygamma")
def _PolygammaGrad(op, grad):
  """Returns gradient of psi(n, x) with respect to n and x."""
  # TODO(tillahoffmann): Add derivative with respect to n
  n = op.inputs[0]
  x = op.inputs[1]
  # Broadcast gradients
  sn = array_ops.shape(n)
  sx = array_ops.shape(x)
  unused_rn, rx = gen_array_ops._broadcast_gradient_args(sn, sx)
  # Evaluate gradient
  with ops.control_dependencies([grad.op]):
    partial_x = math_ops.polygamma(n + 1, x)
    return (None,
            array_ops.reshape(math_ops.reduce_sum(partial_x * grad, rx), sx))


@ops.RegisterGradient("Sigmoid")
def _SigmoidGrad(op, grad):
  """Returns grad * sigmoid(x) * (1 - sigmoid(x))."""
  y = op.outputs[0]  # y = sigmoid(x)
  with ops.control_dependencies([grad.op]):
    if y.dtype.is_complex:
      y = math_ops.conj(y)
    return grad * (y * (1 - y))


@ops.RegisterGradient("Sign")
def _SignGrad(op, _):
  """Returns 0."""
  x = op.inputs[0]
  return array_ops.zeros(array_ops.shape(x), dtype=x.dtype)


@ops.RegisterGradient("Sin")
def _SinGrad(op, grad):
  """Returns grad * cos(x)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad.op]):
    if x.dtype.is_complex:
      x = math_ops.conj(x)
    return grad * math_ops.cos(x)


@ops.RegisterGradient("Cos")
def _CosGrad(op, grad):
  """Returns grad * -sin(x)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad.op]):
    if x.dtype.is_complex:
      x = math_ops.conj(x)
    return -grad * math_ops.sin(x)


@ops.RegisterGradient("AddN")
def _AddNGrad(op, grad):
  """Copies the gradient to all inputs."""
  # Not broadcasting.
  return [grad] * len(op.inputs)


@ops.RegisterGradient("Add")
def _AddGrad(op, grad):
  x = op.inputs[0]
  y = op.inputs[1]
  sx = array_ops.shape(x)
  sy = array_ops.shape(y)
  rx, ry = gen_array_ops._broadcast_gradient_args(sx, sy)
  return (array_ops.reshape(math_ops.reduce_sum(grad, rx), sx),
          array_ops.reshape(math_ops.reduce_sum(grad, ry), sy))


@ops.RegisterGradient("Sub")
def _SubGrad(op, grad):
  x = op.inputs[0]
  y = op.inputs[1]
  sx = array_ops.shape(x)
  sy = array_ops.shape(y)
  rx, ry = gen_array_ops._broadcast_gradient_args(sx, sy)
  return (array_ops.reshape(math_ops.reduce_sum(grad, rx), sx),
          array_ops.reshape(-math_ops.reduce_sum(grad, ry), sy))


@ops.RegisterGradient("Mul")
def _MulGrad(op, grad):
  """The gradient of scalar multiplication."""
  x = op.inputs[0]
  y = op.inputs[1]
  assert x.dtype.base_dtype == y.dtype.base_dtype, (x.dtype, " vs. ", y.dtype)
  sx = array_ops.shape(x)
  sy = array_ops.shape(y)
  rx, ry = gen_array_ops._broadcast_gradient_args(sx, sy)
  if x.dtype.is_complex:
    x = math_ops.conj(x)
    y = math_ops.conj(y)
  return (array_ops.reshape(math_ops.reduce_sum(grad * y, rx), sx),
          array_ops.reshape(math_ops.reduce_sum(x * grad, ry), sy))


@ops.RegisterGradient("Div")
def _DivGrad(op, grad):
  x = op.inputs[0]
  y = op.inputs[1]
  sx = array_ops.shape(x)
  sy = array_ops.shape(y)
  rx, ry = gen_array_ops._broadcast_gradient_args(sx, sy)  # pylint: disable=protected-access
  return (array_ops.reshape(math_ops.reduce_sum(grad / y, rx), sx),
          array_ops.reshape(math_ops.reduce_sum(grad *
                                         (-x / math_ops.square(y)), ry), sy))


@ops.RegisterGradient("Pow")
def _PowGrad(op, grad):
  """Returns grad * (y*x^(y-1), z*log(x))."""
  x = op.inputs[0]
  y = op.inputs[1]
  z = op.outputs[0]
  sx = array_ops.shape(x)
  sy = array_ops.shape(y)
  rx, ry = gen_array_ops._broadcast_gradient_args(sx, sy)
  gx = array_ops.reshape(
      math_ops.reduce_sum(grad * y * math_ops.pow(x, y - 1), rx), sx)
  gy = array_ops.reshape(
      math_ops.reduce_sum(grad * z * math_ops.log(x), ry), sy)
  return gx, gy


def _MaximumMinimumGrad(op, grad, selector_op):
  """Factor out the code for the gradient of Maximum or Minimum."""
  x = op.inputs[0]
  y = op.inputs[1]
  gdtype = grad.dtype
  sx = array_ops.shape(x)
  sy = array_ops.shape(y)
  gradshape = array_ops.shape(grad)
  zeros = array_ops.zeros(gradshape, gdtype)
  xmask = selector_op(x, y)
  rx, ry = gen_array_ops._broadcast_gradient_args(sx, sy)
  xgrad = math_ops.select(xmask, grad, zeros)
  ygrad = math_ops.select(math_ops.logical_not(xmask), grad, zeros)
  gx = array_ops.reshape(math_ops.reduce_sum(xgrad, rx), sx)
  gy = array_ops.reshape(math_ops.reduce_sum(ygrad, ry), sy)
  return (gx, gy)


@ops.RegisterGradient("Maximum")
def _MaximumGrad(op, grad):
  """Returns grad*(x > y, x <= y) with type of grad."""
  return _MaximumMinimumGrad(op, grad, math_ops.greater_equal)


@ops.RegisterGradient("Minimum")
def _MinimumGrad(op, grad):
  """Returns grad*(x < y, x >= y) with type of grad."""
  return _MaximumMinimumGrad(op, grad, math_ops.less_equal)


@ops.RegisterGradient("SquaredDifference")
def _SquaredDifferenceGrad(op, grad):
  """Returns the gradient for (x-y)^2."""
  x = op.inputs[0]
  y = op.inputs[1]
  sx = array_ops.shape(x)
  sy = array_ops.shape(y)
  # pylint: disable=protected-access
  rx, ry = gen_array_ops._broadcast_gradient_args(sx, sy)
  # pylint: enable=protected-access
  # .op works with Tensors or IndexedSlices
  with ops.control_dependencies([grad.op]):
    # The parens ensure that if grad is IndexedSlices, it'll get multiplied by
    # Tensor (not a number like 2.0) which causes it to convert to Tensor.
    x_grad = math_ops.scalar_mul(2.0, grad) * (x - y)
  return (array_ops.reshape(math_ops.reduce_sum(x_grad, rx), sx),
          -array_ops.reshape(math_ops.reduce_sum(x_grad, ry), sy))


# Logical operations have no gradients.
ops.NoGradient("Less")
ops.NoGradient("LessEqual")
ops.NoGradient("Greater")
ops.NoGradient("GreaterEqual")
ops.NoGradient("Equal")
ops.NoGradient("NotEqual")
ops.NoGradient("LogicalAnd")
ops.NoGradient("LogicalOr")
ops.NoGradient("LogicalNot")


@ops.RegisterGradient("Select")
def _SelectGrad(op, grad):
  c = op.inputs[0]
  x = op.inputs[1]
  zeros = array_ops.zeros(array_ops.shape(x), dtype=x.dtype)
  return (None, math_ops.select(c, grad, zeros),
          math_ops.select(c, zeros, grad))


@ops.RegisterGradient("MatMul")
def _MatMulGrad(op, grad):
  t_a = op.get_attr("transpose_a")
  t_b = op.get_attr("transpose_b")
  if not t_a and not t_b:
    return (math_ops.matmul(grad, op.inputs[1], transpose_b=True),
            math_ops.matmul(op.inputs[0], grad, transpose_a=True))
  elif not t_a and t_b:
    return (math_ops.matmul(grad, op.inputs[1]),
            math_ops.matmul(grad, op.inputs[0], transpose_a=True))
  elif t_a and not t_b:
    return (math_ops.matmul(op.inputs[1], grad, transpose_b=True),
            math_ops.matmul(op.inputs[0], grad))
  elif t_a and t_b:
    return (math_ops.matmul(op.inputs[1], grad, transpose_a=True,
                            transpose_b=True),
            math_ops.matmul(grad, op.inputs[0], transpose_a=True,
                            transpose_b=True))


@ops.RegisterGradient("SparseMatMul")
def _SparseMatMulGrad(op, grad):
  """Gradient for SparseMatMul."""

  t_a = op.get_attr("transpose_a")
  t_b = op.get_attr("transpose_b")
  is_sparse = {
      op.inputs[0]: op.get_attr("a_is_sparse"),
      op.inputs[1]: op.get_attr("b_is_sparse"),
      # Use heuristic to figure out if grad might be sparse
      grad: (grad.op.type == "ReluGrad")
  }
  def _SparseMatMul(t1, t2, out_dtype,
                    transpose_a=False, transpose_b=False):
    """Helper function to create SparseMatMul op."""

    assert t1 in is_sparse and t2 in is_sparse
    t1_sparse = is_sparse[t1]
    t2_sparse = is_sparse[t2]
    if transpose_b:
      t2 = array_ops.transpose(t2)
      transpose_b = False
    prod = math_ops.matmul(t1, t2,
                           transpose_a=transpose_a,
                           transpose_b=transpose_b,
                           a_is_sparse=t1_sparse,
                           b_is_sparse=t2_sparse)
    if prod.dtype != out_dtype:
      prod = math_ops.cast(prod, out_dtype)
    return prod

  dtype_a = op.inputs[0].dtype
  dtype_b = op.inputs[1].dtype
  if not t_a and not t_b:
    return (_SparseMatMul(grad, op.inputs[1], dtype_a, transpose_b=True),
            _SparseMatMul(op.inputs[0], grad, dtype_b, transpose_a=True))
  elif not t_a and t_b:
    return (_SparseMatMul(grad, op.inputs[1], dtype_a),
            _SparseMatMul(grad, op.inputs[0], dtype_b, transpose_a=True))
  elif t_a and not t_b:
    return (_SparseMatMul(op.inputs[1], grad, dtype_a, transpose_b=True),
            _SparseMatMul(op.inputs[0], grad, dtype_b))
  elif t_a and t_b:
    return (_SparseMatMul(op.inputs[1], grad, dtype_a,
                          transpose_a=True, transpose_b=True),
            _SparseMatMul(grad, op.inputs[0], dtype_b,
                          transpose_a=True, transpose_b=True))


@ops.RegisterGradient("Floor")
def _FloorGrad(_, unused_grad):
  return [None]


@ops.RegisterGradient("BatchMatMul")
def _BatchMatMul(op, grad):
  """Returns the gradient of x and y given the gradient of x * y."""
  x = op.inputs[0]
  y = op.inputs[1]
  adj_x = op.get_attr("adj_x")
  adj_y = op.get_attr("adj_y")

  if not adj_x:
    if not adj_y:
      grad_x = math_ops.batch_matmul(grad, y, False, True)
      grad_y = math_ops.batch_matmul(x, grad, True, False)
    else:
      grad_x = math_ops.batch_matmul(grad, y, False, False)
      grad_y = math_ops.batch_matmul(grad, x, True, False)
  else:
    if not adj_y:
      grad_x = math_ops.batch_matmul(y, grad, False, True)
      grad_y = math_ops.batch_matmul(x, grad, False, False)
    else:
      grad_x = math_ops.batch_matmul(y, grad, True, True)
      grad_y = math_ops.batch_matmul(grad, x, True, True)

  return grad_x, grad_y


ops.NoGradient("Range")
ops.NoGradient("LinSpace")


@ops.RegisterGradient("Complex")
def _ComplexGrad(_, grad):
  """Returns the real and imaginary components of 'grad', respectively."""
  return math_ops.real(grad), math_ops.imag(grad)


@ops.RegisterGradient("Real")
def _RealGrad(_, grad):
  """Returns 'grad' as the real part and set the imaginary part 0."""
  zero = constant_op.constant(0, dtype=grad.dtype)
  return math_ops.complex(grad, zero)


@ops.RegisterGradient("Imag")
def _ImagGrad(_, grad):
  """Returns 'grad' as the imaginary part and set the real part 0."""
  zero = constant_op.constant(0, dtype=grad.dtype)
  return math_ops.complex(zero, grad)


@ops.RegisterGradient("Conj")
def _ConjGrad(_, grad):
  """Returns the complex conjugate of grad."""
  return math_ops.conj(grad)


@ops.RegisterGradient("ComplexAbs")
def _ComplexAbsGrad(op, grad):
  """Returns the gradient of ComplexAbs."""
  # TODO(b/27786104): The cast to complex could be removed once arithmetic
  # supports mixtures of complex64 and real values.
  return (math_ops.complex(grad, array_ops.zeros_like(grad)) *
          math_ops.sign(op.inputs[0]))


@ops.RegisterGradient("Cast")
def _CastGrad(op, grad):
  t = [dtypes.float16, dtypes.float32, dtypes.float64, dtypes.bfloat16]
  src_type = op.inputs[0].dtype.base_dtype
  dst_type = grad.dtype.base_dtype
  if src_type in t and dst_type in t:
    return math_ops.cast(grad, src_type)
  else:
    return None


@ops.RegisterGradient("FFT")
def _FFTGrad(_, grad):
  size = math_ops.cast(array_ops.size(grad), dtypes.float32)
  return math_ops.ifft(grad) * math_ops.complex(size, 0.)


@ops.RegisterGradient("IFFT")
def _IFFTGrad(_, grad):
  rsize = 1. / math_ops.cast(array_ops.size(grad), dtypes.float32)
  return math_ops.fft(grad) * math_ops.complex(rsize, 0.)


@ops.RegisterGradient("FFT2D")
def _FFT2DGrad(_, grad):
  size = math_ops.cast(array_ops.size(grad), dtypes.float32)
  return math_ops.ifft2d(grad) * math_ops.complex(size, 0.)


@ops.RegisterGradient("IFFT2D")
def _IFFT2DGrad(_, grad):
  rsize = 1. / math_ops.cast(array_ops.size(grad), dtypes.float32)
  return math_ops.fft2d(grad) * math_ops.complex(rsize, 0.)


@ops.RegisterGradient("FFT3D")
def _FFT3DGrad(_, grad):
  size = math_ops.cast(array_ops.size(grad), dtypes.float32)
  return math_ops.ifft3d(grad) * math_ops.complex(size, 0.)


@ops.RegisterGradient("IFFT3D")
def _IFFT3DGrad(_, grad):
  rsize = 1. / math_ops.cast(array_ops.size(grad), dtypes.float32)
  return math_ops.fft3d(grad) * math_ops.complex(rsize, 0.)


def _FFTSizeForGrad(grad, rank):
  return math_ops.reduce_prod(array_ops.slice(
      array_ops.reverse(
          array_ops.shape(grad), (True,)), (0,), (rank,)))


@ops.RegisterGradient("BatchFFT")
def _BatchFFTGrad(_, grad):
  size = math_ops.cast(_FFTSizeForGrad(grad, 1), dtypes.float32)
  return math_ops.batch_ifft(grad) * math_ops.complex(size, 0.)


@ops.RegisterGradient("BatchIFFT")
def _BatchIFFTGrad(_, grad):
  rsize = 1. / math_ops.cast(_FFTSizeForGrad(grad, 1), dtypes.float32)
  return math_ops.batch_fft(grad) * math_ops.complex(rsize, 0.)


@ops.RegisterGradient("BatchFFT2D")
def _BatchFFT2DGrad(_, grad):
  size = math_ops.cast(_FFTSizeForGrad(grad, 2), dtypes.float32)
  return math_ops.batch_ifft2d(grad) * math_ops.complex(size, 0.)


@ops.RegisterGradient("BatchIFFT2D")
def _BatchIFFT2DGrad(_, grad):
  rsize = 1. / math_ops.cast(_FFTSizeForGrad(grad, 2), dtypes.float32)
  return math_ops.batch_fft2d(grad) * math_ops.complex(rsize, 0.)


@ops.RegisterGradient("BatchFFT3D")
def _BatchFFT3DGrad(_, grad):
  size = math_ops.cast(_FFTSizeForGrad(grad, 3), dtypes.float32)
  return math_ops.batch_ifft3d(grad) * math_ops.complex(size, 0.)


@ops.RegisterGradient("BatchIFFT3D")
def _BatchIFFT3DGrad(_, grad):
  rsize = 1. / math_ops.cast(_FFTSizeForGrad(grad, 3), dtypes.float32)
  return math_ops.batch_fft3d(grad) * math_ops.complex(rsize, 0.)


@ops.RegisterGradient("Cross")
def _CrossGrad(op, grad):
  u = op.inputs[0]
  v = op.inputs[1]
  return (math_ops.cross(v, grad), math_ops.cross(grad, u))
