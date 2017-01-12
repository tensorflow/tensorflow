# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops


def _safe_shape_div(x, y):
  """Divides `x / y` assuming `x, y >= 0`, treating `0 / 0 = 0`."""
  return x // math_ops.maximum(y, 1)


@ops.RegisterGradient("Sum")
def _SumGrad(op, grad):
  """Gradient for Sum."""
  # Fast path for when reducing to a scalar and ndims is known: adds only
  # Reshape and Tile ops (and possibly a Shape).
  if (op.inputs[0].get_shape().ndims is not None and
      op.inputs[1].op.type == "Const"):
    rank = op.inputs[0].get_shape().ndims
    axes = tensor_util.MakeNdarray(op.inputs[1].op.get_attr("value"))
    if np.array_equal(axes, np.arange(rank)):  # Reduce all dims.
      grad = array_ops.reshape(grad, [1] * rank)
      # If shape is not fully defined (but rank is), we use Shape.
      if op.inputs[0].get_shape().is_fully_defined():
        input_shape = op.inputs[0].get_shape().as_list()
      else:
        input_shape = array_ops.shape(op.inputs[0])
      return [array_ops.tile(grad, input_shape), None]

  input_shape = array_ops.shape(op.inputs[0])
  output_shape_kept_dims = math_ops.reduced_shape(input_shape, op.inputs[1])
  tile_scaling = _safe_shape_div(input_shape, output_shape_kept_dims)
  grad = array_ops.reshape(grad, output_shape_kept_dims)
  return [array_ops.tile(grad, tile_scaling), None]


def _MinOrMaxGrad(op, grad):
  """Gradient for Min or Max. Amazingly it's precisely the same code."""
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
      math_ops.reduce_sum(indicators, op.inputs[1]), output_shape_kept_dims)

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
  factor = _safe_shape_div(
      math_ops.reduce_prod(input_shape), math_ops.reduce_prod(output_shape))
  return sum_grad / math_ops.cast(factor, sum_grad.dtype), None


@ops.RegisterGradient("Prod")
def _ProdGrad(op, grad):
  """Gradient for Prod."""
  # The gradient can be expressed by dividing the product by each entry of the
  # input tensor, but this approach can't deal with zeros in the input.
  # Here, we avoid this problem by composing the output as a product of two
  # cumprod operations.

  input_shape = array_ops.shape(op.inputs[0])
  # Reshape reduction indices for the case where the parameter is a scalar
  reduction_indices = array_ops.reshape(op.inputs[1], [-1])

  # Expand grad to full input shape
  output_shape_kept_dims = math_ops.reduced_shape(input_shape, op.inputs[1])
  tile_scaling = _safe_shape_div(input_shape, output_shape_kept_dims)
  grad = array_ops.reshape(grad, output_shape_kept_dims)
  grad = array_ops.tile(grad, tile_scaling)

  # Pack all reduced dimensions into a single one, so we can perform the
  # cumprod ops. If the reduction dims list is empty, it defaults to float32,
  # so we need to cast here.  We put all the shape-related ops on CPU to avoid
  # copying back and forth, and since listdiff is CPU only.
  with ops.device("/cpu:0"):
    reduced = math_ops.cast(reduction_indices, dtypes.int32)
    idx = math_ops.range(0, array_ops.rank(op.inputs[0]))
    other, _ = array_ops.setdiff1d(idx, reduced)
    perm = array_ops.concat([reduced, other], 0)
    reduced_num = math_ops.reduce_prod(array_ops.gather(input_shape, reduced))
    other_num = math_ops.reduce_prod(array_ops.gather(input_shape, other))
  permuted = array_ops.transpose(op.inputs[0], perm)
  permuted_shape = array_ops.shape(permuted)
  reshaped = array_ops.reshape(permuted, (reduced_num, other_num))

  # Calculate product, leaving out the current entry
  left = math_ops.cumprod(reshaped, axis=0, exclusive=True)
  right = math_ops.cumprod(reshaped, axis=0, exclusive=True, reverse=True)
  y = array_ops.reshape(left * right, permuted_shape)

  # Invert the transpose and reshape operations.
  # Make sure to set the statically known shape information through a reshape.
  out = grad * array_ops.transpose(y, array_ops.invert_permutation(perm))
  return array_ops.reshape(out, input_shape), None


@ops.RegisterGradient("SegmentSum")
def _SegmentSumGrad(op, grad):
  """Gradient for SegmentSum."""
  return array_ops.gather(grad, op.inputs[1]), None


@ops.RegisterGradient("SegmentMean")
def _SegmentMeanGrad(op, grad):
  """Gradient for SegmentMean."""
  input_rank = array_ops.rank(op.inputs[0])
  ones_shape = array_ops.concat([
      array_ops.shape(op.inputs[1]),
      array_ops.fill(array_ops.expand_dims(input_rank - 1, 0), 1)
  ], 0)
  ones = array_ops.fill(ones_shape,
                        constant_op.constant(1, dtype=grad.dtype))
  scaled_grad = math_ops.div(grad, math_ops.segment_sum(ones, op.inputs[1]))
  return array_ops.gather(scaled_grad, op.inputs[1]), None


@ops.RegisterGradient("SparseSegmentSum")
def _SparseSegmentSumGrad(op, grad):
  """Gradient for SparseSegmentSum."""
  input_rows = array_ops.shape(op.inputs[0])[0]
  return (math_ops.unsorted_segment_sum(
      array_ops.gather(grad, op.inputs[2]), op.inputs[1], input_rows), None,
          None)


@ops.RegisterGradient("SparseSegmentMean")
def _SparseSegmentMeanGrad(op, grad):
  """Gradient for SparseSegmentMean."""
  dim0 = array_ops.shape(op.inputs[0])[0]
  return (math_ops.sparse_segment_mean_grad(grad, op.inputs[1], op.inputs[2],
                                            dim0), None, None)


@ops.RegisterGradient("SparseSegmentSqrtN")
def _SparseSegmentSqrtNGrad(op, grad):
  """Gradient for SparseSegmentSqrtN."""
  dim0 = array_ops.shape(op.inputs[0])[0]
  return (math_ops.sparse_segment_sqrt_n_grad(grad, op.inputs[1], op.inputs[2],
                                              dim0), None, None)


def _SegmentMinOrMaxGrad(op, grad):
  """Gradient for SegmentMin and SegmentMax. Both share the same code."""
  zeros = array_ops.zeros(
      array_ops.shape(op.inputs[0]), dtype=op.inputs[0].dtype)

  # Get the number of selected (minimum or maximum) elements in each segment.
  gathered_outputs = array_ops.gather(op.outputs[0], op.inputs[1])
  is_selected = math_ops.equal(op.inputs[0], gathered_outputs)
  num_selected = math_ops.segment_sum(
      math_ops.cast(is_selected, grad.dtype), op.inputs[1])

  # Compute the gradient for each segment. The gradient for the ith segment is
  # divided evenly among the selected elements in that segment.
  weighted_grads = math_ops.div(grad, num_selected)
  gathered_grads = array_ops.gather(weighted_grads, op.inputs[1])

  return array_ops.where(is_selected, gathered_grads, zeros), None


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
  return -grad


@ops.RegisterGradient("Inv")
def _InvGrad(op, grad):
  """Returns -grad * (1 / x^2)."""
  y = op.outputs[0]  # y = 1 / x
  # pylint: disable=protected-access
  return gen_math_ops._reciprocal_grad(y, grad)


@ops.RegisterGradient("Reciprocal")
def _ReciprocalGrad(op, grad):
  """Returns -grad * (1 / x^2)."""
  y = op.outputs[0]  # y = 1 / x
  # pylint: disable=protected-access
  return gen_math_ops._reciprocal_grad(y, grad)


@ops.RegisterGradient("InvGrad")
def _InvGradGrad(op, grad):
  b = op.inputs[1]
  # op.output[0]: y = -b * conj(a)^2
  with ops.control_dependencies([grad.op]):
    ca = math_ops.conj(op.inputs[0])
    cg = math_ops.conj(grad)
    # pylint: disable=protected-access
    return cg * -2.0 * b * ca, gen_math_ops._reciprocal_grad(ca, grad)


@ops.RegisterGradient("ReciprocalGrad")
def _ReciprocalGradGrad(op, grad):
  b = op.inputs[1]
  # op.output[0]: y = -b * conj(a)^2
  with ops.control_dependencies([grad.op]):
    ca = math_ops.conj(op.inputs[0])
    cg = math_ops.conj(grad)
    # pylint: disable=protected-access
    return cg * -2.0 * b * ca, gen_math_ops._reciprocal_grad(ca, grad)


@ops.RegisterGradient("Square")
def _SquareGrad(op, grad):
  x = op.inputs[0]
  # Added control dependencies to prevent 2*x from being computed too early.
  with ops.control_dependencies([grad.op]):
    x = math_ops.conj(x)
    return grad * (2.0 * x)


@ops.RegisterGradient("Sqrt")
def _SqrtGrad(op, grad):
  y = op.outputs[0]  # y = x^(1/2)
  return gen_math_ops._sqrt_grad(y, grad)


@ops.RegisterGradient("SqrtGrad")
def _SqrtGradGrad(op, grad):
  a = op.inputs[0]
  y = op.outputs[0]  # y = 0.5 * b / conj(a)
  with ops.control_dependencies([grad.op]):
    ga = grad / a
    return -math_ops.conj(ga) * y, 0.5 * ga


@ops.RegisterGradient("Rsqrt")
def _RsqrtGrad(op, grad):
  """Returns -0.5 * grad * conj(y)^3."""
  y = op.outputs[0]  # y = x^(-1/2)
  return gen_math_ops._rsqrt_grad(y, grad)


@ops.RegisterGradient("RsqrtGrad")
def _RsqrtGradGrad(op, grad):
  """Returns backprop gradient for f(a,b) = -0.5 * b * conj(a)^3."""
  a = op.inputs[0]  # a = x^{-1/2}
  b = op.inputs[1]  # backprop gradient for a
  with ops.control_dependencies([grad.op]):
    ca = math_ops.conj(a)
    cg = math_ops.conj(grad)
    grad_a = -1.5 * cg * b * math_ops.square(ca)
    # pylint: disable=protected-access
    grad_b = gen_math_ops._rsqrt_grad(ca, grad)
    return grad_a, grad_b


@ops.RegisterGradient("Exp")
def _ExpGrad(op, grad):
  """Returns grad * exp(x)."""
  y = op.outputs[0]  # y = e^x
  with ops.control_dependencies([grad.op]):
    y = math_ops.conj(y)
    return grad * y


@ops.RegisterGradient("Expm1")
def _Expm1Grad(op, grad):
  """Returns grad * exp(x)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad.op]):
    x = math_ops.conj(x)
    y = math_ops.exp(x)
    return grad * y


@ops.RegisterGradient("Log")
def _LogGrad(op, grad):
  """Returns grad * (1/x)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad.op]):
    x = math_ops.conj(x)
    return grad * math_ops.reciprocal(x)


@ops.RegisterGradient("Log1p")
def _Log1pGrad(op, grad):
  """Returns grad * (1/(1 + x))."""
  x = op.inputs[0]
  with ops.control_dependencies([grad.op]):
    x = math_ops.conj(x)
    return grad * math_ops.reciprocal(1 + x)


@ops.RegisterGradient("Tanh")
def _TanhGrad(op, grad):
  """Returns grad * (1 - tanh(x) * tanh(x))."""
  y = op.outputs[0]  # y = tanh(x)
  with ops.control_dependencies([grad.op]):
    y = math_ops.conj(y)
    # pylint: disable=protected-access
    return gen_math_ops._tanh_grad(y, grad)


@ops.RegisterGradient("TanhGrad")
def _TanhGradGrad(op, grad):
  with ops.control_dependencies([grad.op]):
    a = math_ops.conj(op.inputs[0])
    b = math_ops.conj(op.inputs[1])
    # pylint: disable=protected-access
    return grad * -2.0 * b * a, gen_math_ops._tanh_grad(a, grad)


@ops.RegisterGradient("Erf")
def _ErfGrad(op, grad):
  """Returns grad * 2/sqrt(pi) * exp(-x**2)."""
  x = op.inputs[0]
  two_over_root_pi = constant_op.constant(2 / np.sqrt(np.pi), dtype=grad.dtype)
  with ops.control_dependencies([grad.op]):
    x = math_ops.conj(x)
    return grad * two_over_root_pi * math_ops.exp(-math_ops.square(x))


@ops.RegisterGradient("Erfc")
def _ErfcGrad(op, grad):
  """Returns -grad * 2/sqrt(pi) * exp(-x**2)."""
  x = op.inputs[0]
  minus_two_over_root_pi = constant_op.constant(
      -2 / np.sqrt(np.pi), dtype=grad.dtype)
  with ops.control_dependencies([grad.op]):
    x = math_ops.conj(x)
    return grad * minus_two_over_root_pi * math_ops.exp(-math_ops.square(x))


@ops.RegisterGradient("Lgamma")
def _LgammaGrad(op, grad):
  """Returns grad * digamma(x)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad.op]):
    x = math_ops.conj(x)
    return grad * math_ops.digamma(x)


@ops.RegisterGradient("Digamma")
def _DigammaGrad(op, grad):
  """Compute gradient of the digamma function with respect to its argument."""
  x = op.inputs[0]
  with ops.control_dependencies([grad.op]):
    x = math_ops.conj(x)
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
  partial_x = math_ops.exp(-x + (a - 1) * math_ops.log(x) - math_ops.lgamma(a))
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
    x = math_ops.conj(x)
    q = math_ops.conj(q)
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
    n = math_ops.conj(n)
    x = math_ops.conj(x)
    partial_x = math_ops.polygamma(n + 1, x)
    return (None,
            array_ops.reshape(math_ops.reduce_sum(partial_x * grad, rx), sx))


@ops.RegisterGradient("Sigmoid")
def _SigmoidGrad(op, grad):
  """Returns grad * sigmoid(x) * (1 - sigmoid(x))."""
  y = op.outputs[0]  # y = sigmoid(x)
  with ops.control_dependencies([grad.op]):
    y = math_ops.conj(y)
    # pylint: disable=protected-access
    return gen_math_ops._sigmoid_grad(y, grad)


@ops.RegisterGradient("SigmoidGrad")
def _SigmoidGradGrad(op, grad):
  with ops.control_dependencies([grad.op]):
    a = math_ops.conj(op.inputs[0])
    b = math_ops.conj(op.inputs[1])
    gb = grad * b
    # pylint: disable=protected-access
    return gb - 2.0 * gb * a, gen_math_ops._sigmoid_grad(a, grad)


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
    x = math_ops.conj(x)
    return grad * math_ops.cos(x)


@ops.RegisterGradient("Cos")
def _CosGrad(op, grad):
  """Returns grad * -sin(x)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad.op]):
    x = math_ops.conj(x)
    return -grad * math_ops.sin(x)


@ops.RegisterGradient("Tan")
def _TanGrad(op, grad):
  """Returns grad * 1/sec^2(x)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad.op]):
    x = math_ops.conj(x)
    secx = math_ops.reciprocal(math_ops.cos(x))
    secx2 = math_ops.square(secx)
    return grad * secx2


@ops.RegisterGradient("Asin")
def _AsinGrad(op, grad):
  """Returns grad * 1/sqrt(1-x^2)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad.op]):
    x = math_ops.conj(x)
    x2 = math_ops.square(x)
    one = constant_op.constant(1, dtype=grad.dtype)
    den = math_ops.sqrt(math_ops.subtract(one, x2))
    inv = math_ops.reciprocal(den)
    return grad * inv


@ops.RegisterGradient("Acos")
def _AcosGrad(op, grad):
  """Returns grad * -1/sqrt(1-x^2)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad.op]):
    x = math_ops.conj(x)
    x2 = math_ops.square(x)
    one = constant_op.constant(1, dtype=grad.dtype)
    den = math_ops.sqrt(math_ops.subtract(one, x2))
    inv = math_ops.reciprocal(den)
    return -grad * inv


@ops.RegisterGradient("Atan")
def _AtanGrad(op, grad):
  """Returns grad * 1/ (1 + x^2)."""
  x = op.inputs[0]
  with ops.control_dependencies([grad.op]):
    x = math_ops.conj(x)
    x2 = math_ops.square(x)
    one = constant_op.constant(1, dtype=grad.dtype)
    inv = math_ops.reciprocal(math_ops.add(one, x2))
    return grad * inv


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
  x = math_ops.conj(x)
  y = math_ops.conj(y)
  return (array_ops.reshape(math_ops.reduce_sum(grad * y, rx), sx),
          array_ops.reshape(math_ops.reduce_sum(x * grad, ry), sy))


@ops.RegisterGradient("Div")
def _DivGrad(op, grad):
  """The gradient for the Div operator."""
  x = op.inputs[0]
  y = op.inputs[1]
  sx = array_ops.shape(x)
  sy = array_ops.shape(y)
  # pylint: disable=protected-access
  rx, ry = gen_array_ops._broadcast_gradient_args(sx, sy)
  # pylint: enable=protected-access
  x = math_ops.conj(x)
  y = math_ops.conj(y)
  return (array_ops.reshape(math_ops.reduce_sum(math_ops.div(grad, y), rx), sx),
          array_ops.reshape(
              math_ops.reduce_sum(grad * math_ops.div(math_ops.div(-x, y), y),
                                  ry), sy))


@ops.RegisterGradient("FloorDiv")
def _FloorDivGrad(_, unused_grad):
  """The gradient for the FloorDiv operator."""
  return None, None


@ops.RegisterGradient("TruncateDiv")
def _TruncateDivGrad(_, unused_grad):
  return None, None


@ops.RegisterGradient("RealDiv")
def _RealDivGrad(op, grad):
  """RealDiv op gradient."""
  x = op.inputs[0]
  y = op.inputs[1]
  sx = array_ops.shape(x)
  sy = array_ops.shape(y)
  # pylint: disable=protected-access
  rx, ry = gen_array_ops._broadcast_gradient_args(sx, sy)
  # pylint: enable=protected-access
  x = math_ops.conj(x)
  y = math_ops.conj(y)
  return (array_ops.reshape(
      math_ops.reduce_sum(math_ops.realdiv(grad, y), rx),
      sx), array_ops.reshape(
          math_ops.reduce_sum(grad * math_ops.realdiv(math_ops.realdiv(-x, y), y),
                              ry), sy))


@ops.RegisterGradient("Pow")
def _PowGrad(op, grad):
  """Returns grad * (y*x^(y-1), z*log(x))."""
  x = op.inputs[0]
  y = op.inputs[1]
  z = op.outputs[0]
  sx = array_ops.shape(x)
  sy = array_ops.shape(y)
  rx, ry = gen_array_ops._broadcast_gradient_args(sx, sy)
  x = math_ops.conj(x)
  y = math_ops.conj(y)
  z = math_ops.conj(z)
  gx = array_ops.reshape(
      math_ops.reduce_sum(grad * y * math_ops.pow(x, y - 1), rx), sx)
  # Avoid false singularity at x = 0
  if x.dtype.is_complex:
    # real(x) < 0 is fine for the complex case
    log_x = array_ops.where(
        math_ops.not_equal(x, 0), math_ops.log(x), array_ops.zeros_like(x))
  else:
    # There's no sensible real value to return if x < 0, so return 0
    log_x = array_ops.where(x > 0, math_ops.log(x), array_ops.zeros_like(x))
  gy = array_ops.reshape(math_ops.reduce_sum(grad * z * log_x, ry), sy)
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
  xgrad = array_ops.where(xmask, grad, zeros)
  ygrad = array_ops.where(math_ops.logical_not(xmask), grad, zeros)
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
ops.NotDifferentiable("Less")
ops.NotDifferentiable("LessEqual")
ops.NotDifferentiable("Greater")
ops.NotDifferentiable("GreaterEqual")
ops.NotDifferentiable("Equal")
ops.NotDifferentiable("NotEqual")
ops.NotDifferentiable("LogicalAnd")
ops.NotDifferentiable("LogicalOr")
ops.NotDifferentiable("LogicalNot")


@ops.RegisterGradient("Select")
def _SelectGrad(op, grad):
  c = op.inputs[0]
  x = op.inputs[1]
  zeros = array_ops.zeros_like(x)
  return (None, array_ops.where(c, grad, zeros),
          array_ops.where(c, zeros, grad))


@ops.RegisterGradient("MatMul")
def _MatMulGrad(op, grad):
  t_a = op.get_attr("transpose_a")
  t_b = op.get_attr("transpose_b")
  if not t_a and not t_b:
    return (math_ops.matmul(
        grad, op.inputs[1], transpose_b=True), math_ops.matmul(
            op.inputs[0], grad, transpose_a=True))
  elif not t_a and t_b:
    return (math_ops.matmul(grad, op.inputs[1]), math_ops.matmul(
        grad, op.inputs[0], transpose_a=True))
  elif t_a and not t_b:
    return (math_ops.matmul(
        op.inputs[1], grad, transpose_b=True),
            math_ops.matmul(op.inputs[0], grad))
  elif t_a and t_b:
    return (math_ops.matmul(
        op.inputs[1], grad, transpose_a=True, transpose_b=True),
            math_ops.matmul(
                grad, op.inputs[0], transpose_a=True, transpose_b=True))


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

  def _SparseMatMul(t1, t2, out_dtype, transpose_a=False, transpose_b=False):
    """Helper function to create SparseMatMul op."""

    assert t1 in is_sparse and t2 in is_sparse
    t1_sparse = is_sparse[t1]
    t2_sparse = is_sparse[t2]
    if transpose_b:
      t2 = array_ops.transpose(t2)
      transpose_b = False
    prod = math_ops.matmul(
        t1,
        t2,
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
    return (_SparseMatMul(
        grad, op.inputs[1], dtype_a, transpose_b=True), _SparseMatMul(
            op.inputs[0], grad, dtype_b, transpose_a=True))
  elif not t_a and t_b:
    return (_SparseMatMul(grad, op.inputs[1], dtype_a), _SparseMatMul(
        grad, op.inputs[0], dtype_b, transpose_a=True))
  elif t_a and not t_b:
    return (_SparseMatMul(
        op.inputs[1], grad, dtype_a, transpose_b=True),
            _SparseMatMul(op.inputs[0], grad, dtype_b))
  elif t_a and t_b:
    return (_SparseMatMul(
        op.inputs[1], grad, dtype_a, transpose_a=True,
        transpose_b=True), _SparseMatMul(
            grad, op.inputs[0], dtype_b, transpose_a=True, transpose_b=True))


@ops.RegisterGradient("Floor")
def _FloorGrad(_, unused_grad):
  return [None]


@ops.RegisterGradient("Ceil")
def _CeilGrad(_, unused_grad):
  return [None]


@ops.RegisterGradient("Round")
def _RoundGrad(_, unused_grad):
  return [None]


@ops.RegisterGradient("Rint")
def _RintGrad(_, unused_grad):
  # the gradient of Rint is zero
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
      grad_x = math_ops.matmul(grad, y, adjoint_a=False, adjoint_b=True)
      grad_y = math_ops.matmul(x, grad, adjoint_a=True, adjoint_b=False)
    else:
      grad_x = math_ops.matmul(grad, y, adjoint_a=False, adjoint_b=False)
      grad_y = math_ops.matmul(grad, x, adjoint_a=True, adjoint_b=False)
  else:
    if not adj_y:
      grad_x = math_ops.matmul(y, grad, adjoint_a=False, adjoint_b=True)
      grad_y = math_ops.matmul(x, grad, adjoint_a=False, adjoint_b=False)
    else:
      grad_x = math_ops.matmul(y, grad, adjoint_a=True, adjoint_b=True)
      grad_y = math_ops.matmul(grad, x, adjoint_a=True, adjoint_b=True)

  return grad_x, grad_y


ops.NotDifferentiable("Range")
ops.NotDifferentiable("LinSpace")


@ops.RegisterGradient("Complex")
def _ComplexGrad(op, grad):
  """Returns the real and imaginary components of 'grad', respectively."""
  x = op.inputs[0]
  y = op.inputs[1]
  sx = array_ops.shape(x)
  sy = array_ops.shape(y)
  rx, ry = gen_array_ops._broadcast_gradient_args(sx, sy)
  return (array_ops.reshape(math_ops.reduce_sum(math_ops.real(grad), rx), sx),
          array_ops.reshape(math_ops.reduce_sum(math_ops.imag(grad), ry), sy))


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
  t = [
      dtypes.float16, dtypes.float32, dtypes.float64, dtypes.bfloat16,
      dtypes.complex64, dtypes.complex128
  ]
  src_type = op.inputs[0].dtype.base_dtype
  dst_type = grad.dtype.base_dtype
  if src_type in t and dst_type in t:
    return math_ops.cast(grad, src_type)
  else:
    return None


def _FFTSizeForGrad(grad, rank):
  return math_ops.reduce_prod(
      array_ops.slice(
          array_ops.reverse_v2(array_ops.shape(grad), [0]), (0,), (rank,)))


@ops.RegisterGradient("FFT")
def _FFTGrad(_, grad):
  size = math_ops.cast(_FFTSizeForGrad(grad, 1), dtypes.float32)
  return math_ops.ifft(grad) * math_ops.complex(size, 0.)


@ops.RegisterGradient("IFFT")
def _IFFTGrad(_, grad):
  rsize = 1. / math_ops.cast(_FFTSizeForGrad(grad, 1), dtypes.float32)
  return math_ops.fft(grad) * math_ops.complex(rsize, 0.)


@ops.RegisterGradient("FFT2D")
def _FFT2DGrad(_, grad):
  size = math_ops.cast(_FFTSizeForGrad(grad, 2), dtypes.float32)
  return math_ops.ifft2d(grad) * math_ops.complex(size, 0.)


@ops.RegisterGradient("IFFT2D")
def _IFFT2DGrad(_, grad):
  rsize = 1. / math_ops.cast(_FFTSizeForGrad(grad, 2), dtypes.float32)
  return math_ops.fft2d(grad) * math_ops.complex(rsize, 0.)


@ops.RegisterGradient("FFT3D")
def _FFT3DGrad(_, grad):
  size = math_ops.cast(_FFTSizeForGrad(grad, 3), dtypes.float32)
  return math_ops.ifft3d(grad) * math_ops.complex(size, 0.)


@ops.RegisterGradient("IFFT3D")
def _IFFT3DGrad(_, grad):
  rsize = 1. / math_ops.cast(_FFTSizeForGrad(grad, 3), dtypes.float32)
  return math_ops.fft3d(grad) * math_ops.complex(rsize, 0.)


@ops.RegisterGradient("Cross")
def _CrossGrad(op, grad):
  u = op.inputs[0]
  v = op.inputs[1]
  return (math_ops.cross(v, grad), math_ops.cross(grad, u))


@ops.RegisterGradient("Cumsum")
def _CumsumGrad(op, grad):
  axis = op.inputs[1]
  exclusive = op.get_attr("exclusive")
  reverse = op.get_attr("reverse")
  return [
      math_ops.cumsum(
          grad, axis, exclusive=exclusive, reverse=not reverse), None
  ]


@ops.RegisterGradient("Cumprod")
def _CumprodGrad(op, grad):
  x = op.inputs[0]
  axis = op.inputs[1]
  exclusive = op.get_attr("exclusive")
  reverse = op.get_attr("reverse")

  # TODO This fails when x contains 0 and should be fixed
  prod = math_ops.cumprod(x, axis, exclusive=exclusive, reverse=reverse)
  out = math_ops.cumsum(
      prod * grad, axis, exclusive=exclusive, reverse=not reverse)
  return [out / x, None]
