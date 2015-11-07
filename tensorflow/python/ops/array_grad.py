"""Gradients for operators defined in array_ops.py."""

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_array_ops


@ops.RegisterGradient("Pack")
def _PackGrad(op, grad):
  """Gradient for pack op."""
  return array_ops.unpack(grad, num=op.get_attr('N'))


@ops.RegisterGradient("Unpack")
def _UnpackGrad(_, *grads):
  """Gradient for unpack op."""
  return array_ops.pack(grads)


@ops.RegisterGradient("Concat")
def _ConcatGrad(op, grad):
  """Gradient for concat op."""
  assert isinstance(grad, ops.Tensor)
  # Degenerate concatenation, just return grad.
  if len(op.inputs) == 2:
    return [None, grad]
  # Get the inputs' tensor shapes
  sizes = [array_ops.shape(x) for x in op.inputs[1:]]
  concat_dim = op.inputs[0]
  # Since shape is 1-D, shape_of_shape = [rank-of-inputs]
  shape_of_shape = array_ops.shape(sizes[0])
  # Make a vector of length equal to the input's dimensions,
  # with 0's everywhere and 1 in the concat dim position.
  # Note: Can't use sparse_to_dense since it isn't GPU-capable (for now)
  mask = array_ops.concat(0,
                          [array_ops.fill(
                              array_ops.expand_dims(concat_dim, 0), 0), [1],
                           array_ops.fill(shape_of_shape - concat_dim - 1, 0)])
  out_grads = []
  begin = array_ops.fill(shape_of_shape, 0)
  for i in range(len(sizes)):
    out_grads.append(array_ops.slice(grad, begin, sizes[i]))
    # Lint complains begin = begin + ...
    begin = math_ops.add(begin, sizes[i] * mask)
  return [None] + out_grads


@ops.RegisterGradient("Slice")
def _SliceGrad(op, grad):
  """Gradient for Slice op."""
  # Create an Nx2 padding where the first column represents how many
  # zeros are to be prepended for each dimension, and the second
  # column indicates how many zeros are appended.
  #
  # The number of zeros to append is the shape of the input
  # elementwise-subtracted by both the begin vector and sizes vector.
  #
  # Some more reshaping is needed to assemble this tensor with the
  # right dimensions.
  input_vec = op.inputs[0]
  begin_vec = op.inputs[1]
  input_rank = array_ops.rank(input_vec)
  slice_size = array_ops.shape(op.outputs[0])

  shape = array_ops.pack([input_rank, 1])
  before_pad = array_ops.reshape(begin_vec, shape)
  after_pad = array_ops.reshape(
      array_ops.shape(input_vec) - slice_size - begin_vec, shape)
  paddings = array_ops.concat(1, [before_pad, after_pad])
  return array_ops.pad(grad, paddings), None, None


@ops.RegisterGradient("Split")
def _SplitGrad(op, *grads):
  return None, array_ops.concat(op.inputs[0], list(grads))


ops.NoGradient("Const")

# TODO(liqzhang): The gradient for Diag operator would be
# the diagonal of the backprop. Implement if there is a need.
ops.NoGradient("Diag")

# Edit Distance has no gradient (but can be used to eval seq2seq or CTC).
ops.NoGradient("EditDistance")

ops.NoGradient("Fill")


@ops.RegisterGradient("Gather")
def _GatherGrad(op, grad):
  return [
      ops.IndexedSlices(grad, op.inputs[1], array_ops.shape(op.inputs[0])), None
  ]


@ops.RegisterGradient("Identity")
def _IdGrad(_, grad):
  return grad


@ops.RegisterGradient("RefIdentity")
def _RefIdGrad(_, grad):
  return grad


ops.NoGradient("StopGradient")


@ops.RegisterGradient("Reshape")
def _ReshapeGrad(op, grad):
  return [array_ops.reshape(grad, array_ops.shape(op.inputs[0])), None]


ops.NoGradient("InvertPermutation")


def _ReshapeToInput(op, grad):
  """Reshapes the gradient to the shape of the original input."""
  return array_ops.reshape(grad, array_ops.shape(op.inputs[0]))


@ops.RegisterGradient("ExpandDims")
def _ExpandDimsGrad(op, grad):
  return [_ReshapeToInput(op, grad), None]


@ops.RegisterGradient("Squeeze")
def _SqueezeGrad(op, grad):
  return _ReshapeToInput(op, grad)


@ops.RegisterGradient("Transpose")
def _TransposeGrad(op, grad):
  """Returns unshuffle(grad)."""
  p = op.inputs[1]
  return [array_ops.transpose(grad, array_ops.invert_permutation(p)), None]


ops.NoGradient("Shape")


ops.NoGradient("Rank")


ops.NoGradient("Size")


@ops.RegisterGradient("Tile")
def _TileGrad(op, grad):
  """Sum reduces grad along the tiled dimensions."""
  assert isinstance(grad, ops.Tensor)
  return [gen_array_ops._tile_grad(grad, op.inputs[1]), None]


ops.NoGradient("TileGrad")


ops.NoGradient("BroadcastGradientArgs")


@ops.RegisterGradient("Pad")
def _PadGrad(op, grad):
  """Gradient for Pad."""
  # Pad introduces values around the original tensor, so the gradient function
  # slices the original shape out of the gradient."""
  x = op.inputs[0]
  a = op.inputs[1]  # [Rank(x), 2]
  # Takes a slice of a. The 1st column. [Rank(x), 1].
  pad_before = array_ops.slice(a, [0, 0],
                               array_ops.pack([array_ops.rank(x), 1]))
  # Make it a 1-D tensor.
  begin = array_ops.reshape(pad_before, [-1])
  sizes = array_ops.shape(x)
  return array_ops.slice(grad, begin, sizes), None


# ReverseSequence is just a permutation.  The gradient permutes back.
@ops.RegisterGradient("ReverseSequence")
def _ReverseSequenceGrad(op, grad):
  seq_lengths = op.inputs[1]
  return [array_ops.reverse_sequence(grad,
                                    seq_dim=op.get_attr("seq_dim"),
                                    seq_lengths=seq_lengths),
          None]
