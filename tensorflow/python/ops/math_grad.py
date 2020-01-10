"""Gradients for operators defined in math_ops.py."""

from tensorflow.python.framework import ops
from tensorflow.python.framework import types
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops


def _ReductionGradAssist(op):
  """Reduction grads have much in common, so factor the commonality out."""
  inp = op.inputs[0]                                # Example:
  input_shape = array_ops.shape(inp)                # [2, 3, 5, 7]
  input_rank = array_ops.rank(inp)                  # 4
  indices = op.inputs[1]                            # [1, 2]
  indices_shape = array_ops.shape(indices)          # [2]
  new_output_shape = data_flow_ops.dynamic_stitch(  # [2, 1, 1, 7]
      [math_ops.range(0, input_rank),               # [0, 1, 2, 3]
       indices],                                    # [1, 2]
      [input_shape,                                 # [2, 3, 5, 7]
       array_ops.fill(indices_shape, 1)])           # [1, 1]
  return inp, new_output_shape, input_shape


@ops.RegisterGradient("Sum")
def _SumGrad(op, grad):
  """Gradient for Sum."""
  _, new_output_shape, input_shape = _ReductionGradAssist(op)
  tile_scaling = input_shape / new_output_shape
  grad = array_ops.reshape(grad, new_output_shape)
  return [array_ops.tile(grad, tile_scaling), None]


def _MinOrMaxGrad(op, grad):
  """Gradient for Max or Max. Amazingly it's precisely the same code."""
  inp, new_output_shape, _ = _ReductionGradAssist(op)
  y = op.outputs[0]
  y = array_ops.reshape(y, new_output_shape)
  grad = array_ops.reshape(grad, new_output_shape)
  indicators = math_ops.cast(math_ops.equal(y, inp), grad.dtype)
  return [indicators * grad, None]


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
  factor = (math_ops.reduce_prod(input_shape) /
            math_ops.reduce_prod(output_shape))
  return sum_grad / math_ops.cast(factor, sum_grad.dtype), None


@ops.RegisterGradient("Prod")
def _ProdGrad(op, grad):
  """Gradient for Prod."""
  # TODO(kearnes): this gives NaNs for 0s in the input tensor
  _, new_output_shape, input_shape = _ReductionGradAssist(op)
  tile_scaling = input_shape / new_output_shape
  grad = array_ops.reshape(grad * op.outputs[0], new_output_shape)
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


@ops.RegisterGradient("SegmentMin")
def _SegmentMinGrad(op, grad):
  """Gradient for SegmentMin."""
  zeros = array_ops.zeros(array_ops.shape(op.inputs[0]),
                          dtype=op.inputs[0].dtype)
  gathered_grads = array_ops.gather(grad, op.inputs[1])
  gathered_outputs = array_ops.gather(op.outputs[0], op.inputs[1])
  return math_ops.select(math_ops.greater(op.inputs[0], gathered_outputs),
                         zeros,
                         gathered_grads), None


@ops.RegisterGradient("SegmentMax")
def _SegmentMaxGrad(op, grad):
  """Gradient for SegmentMax."""
  zeros = array_ops.zeros(array_ops.shape(op.inputs[0]),
                          dtype=op.inputs[0].dtype)
  gathered_grads = array_ops.gather(grad, op.inputs[1])
  gathered_outputs = array_ops.gather(op.outputs[0], op.inputs[1])
  return math_ops.select(math_ops.less(op.inputs[0], gathered_outputs),
                         zeros,
                         gathered_grads), None


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
  return grad * (- math_ops.square(y))


@ops.RegisterGradient("Square")
def _SquareGrad(op, grad):
  x = op.inputs[0]
  return grad * (2.0 * x)


@ops.RegisterGradient("Sqrt")
def _SqrtGrad(op, grad):
  y = op.outputs[0]  # y = x^(1/2)
  return grad * (.5 * math_ops.inv(y))


@ops.RegisterGradient("Rsqrt")
def _RsqrtGrad(op, grad):
  x = op.inputs[0]
  y = op.outputs[0]  # y = x^(-1/2)
  return grad * ((-0.5) * math_ops.inv(x) * y)


@ops.RegisterGradient("Exp")
def _ExpGrad(op, grad):
  """Returns grad * exp(x)."""
  y = op.outputs[0]  # y = e^x
  return grad * y


@ops.RegisterGradient("Log")
def _LogGrad(op, grad):
  """Returns grad * (1/x)."""
  x = op.inputs[0]
  return grad * math_ops.inv(x)


@ops.RegisterGradient("Tanh")
def _TanhGrad(op, grad):
  """Returns grad * (1 - tanh(x) * tanh(x))."""
  y = op.outputs[0]  # y = tanh(x)
  return grad * (1 - math_ops.square(y))


@ops.RegisterGradient("Sigmoid")
def _SigmoidGrad(op, grad):
  """Returns grad * sigmoid(x) * (1 - sigmoid(x))."""
  y = op.outputs[0]  # y = sigmoid(x)
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
  return grad * math_ops.cos(x)


@ops.RegisterGradient("Cos")
def _CosGrad(op, grad):
  """Returns grad * -sin(x)."""
  x = op.inputs[0]
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
  x = op.inputs[0]
  y = op.inputs[1]
  assert x.dtype.base_dtype == y.dtype.base_dtype, (x.dtype, " vs. ", y.dtype)
  sx = array_ops.shape(x)
  sy = array_ops.shape(y)
  rx, ry = gen_array_ops._broadcast_gradient_args(sx, sy)
  if x.dtype.base_dtype == types.complex64:
    return (array_ops.reshape(math_ops.reduce_sum(grad * math_ops.conj(y), rx), sx),
            array_ops.reshape(math_ops.reduce_sum(math_ops.conj(x) * grad, ry), sy))
  else:
    return (array_ops.reshape(math_ops.reduce_sum(grad * y, rx), sx),
            array_ops.reshape(math_ops.reduce_sum(x * grad, ry), sy))


@ops.RegisterGradient("Div")
def _DivGrad(op, grad):
  x = op.inputs[0]
  y = op.inputs[1]
  sx = array_ops.shape(x)
  sy = array_ops.shape(y)
  rx, ry = gen_array_ops._broadcast_gradient_args(sx, sy)
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
  gx = array_ops.reshape(math_ops.reduce_sum(grad * y * math_ops.pow(x, y - 1), rx),
                         sx)
  gy = array_ops.reshape(math_ops.reduce_sum(grad * z * math_ops.log(x), ry), sy)
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
  zeros = array_ops.zeros(array_ops.shape(c), dtype=x.dtype)
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
  def _SparseMatMul(t1, t2, transpose_a=False, transpose_b=False):
    """Helper function to create SparseMatMul op."""

    assert t1 in is_sparse and t2 in is_sparse
    t1_sparse = is_sparse[t1]
    t2_sparse = is_sparse[t2]
    if not t1_sparse and not t2_sparse:
      return math_ops.matmul(t1, t2,
                             transpose_a=transpose_a,
                             transpose_b=transpose_b)
    transpose_out = False
    if not t1_sparse:
      transpose_out = True
      t1, t2 = t2, t1
      t1_sparse, t2_sparse = t2_sparse, t1_sparse
      assert t1_sparse
      transpose_a, transpose_b = not transpose_b, not transpose_a

    if transpose_b:
      t2 = array_ops.transpose(t2)
      transpose_b = False
    m = math_ops.matmul(t1, t2,
                        transpose_a=transpose_a,
                        transpose_b=transpose_b,
                        a_is_sparse=t1_sparse,
                        b_is_sparse=t2_sparse)
    if transpose_out:
      m = array_ops.transpose(m)
    return m

  if not t_a and not t_b:
    return (_SparseMatMul(grad, op.inputs[1], transpose_b=True),
            _SparseMatMul(op.inputs[0], grad, transpose_a=True))
  elif not t_a and t_b:
    return (_SparseMatMul(grad, op.inputs[1]),
            _SparseMatMul(grad, op.inputs[0], transpose_a=True))
  elif t_a and not t_b:
    return (_SparseMatMul(op.inputs[1], grad, transpose_b=True),
            _SparseMatMul(op.inputs[0], grad))
  elif t_a and t_b:
    return (_SparseMatMul(op.inputs[1], grad,
                          transpose_a=True, transpose_b=True),
            _SparseMatMul(grad, op.inputs[0],
                          transpose_a=True, transpose_b=True))


@ops.RegisterGradient("Floor")
def _FloorGrad(_, grad):
  return grad


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


@ops.RegisterGradient("Cast")
def _CastGrad(op, grad):
  t = [types.float32, types.float64, types.bfloat16]
  src_type = op.inputs[0].dtype.base_dtype
  dst_type = grad.dtype.base_dtype
  if src_type in t and dst_type in t:
    return math_ops.cast(grad, src_type)
  else:
    return None
