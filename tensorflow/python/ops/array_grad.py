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
"""Gradients for operators defined in array_ops.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops


@ops.RegisterGradient("Pack")
def _PackGrad(op, grad):
  """Gradient for pack op."""
  return array_ops.unstack(grad, num=op.get_attr("N"), axis=op.get_attr("axis"))


@ops.RegisterGradient("Unpack")
def _UnpackGrad(op, *grads):
  """Gradient for unpack op."""
  return array_ops.stack(grads, axis=op.get_attr("axis"))


def _ConcatGradHelper(op, grad, start_value_index, end_value_index, dim_index):
  """Gradient for concat op.

  Args:
    op: An operation.
    grad: `Tensor` or `IndexedSlices` representing the gradients with respect
      to each output of the op.
    start_value_index: An integer index of the first value in the op.inputs.
    end_value_index: An integer index of the last value in the op.inputs.
    dim_index: An interger index of concat_dim or axis parameter in op.inputs.

  Returns:
    Tensors representing the partial gradients with respect to each input
    of the op.

  Raises:
    ValueError: if concat_dim/axis is not statically known.
  """

  def _CreateDenseMaskAndBegin(sizes, concat_dim):
    """Create variables for iteratively slicing a dense gradients tensor."""
    # Since shape is 1-D, shape_of_shape = [rank-of-inputs]
    shape_of_shape = array_ops.shape(sizes[0])
    # Make a vector of length equal to the input's dimensions,
    # with 0's everywhere and 1 in the concat dim position.
    # Note: Can't use sparse_to_dense since it isn't GPU-capable (for now)
    mask = array_ops.concat([
        array_ops.fill(array_ops.expand_dims(concat_dim, 0), 0), [1],
        array_ops.fill(shape_of_shape - concat_dim - 1, 0)
    ], 0)
    begin = array_ops.fill(shape_of_shape, 0)
    return mask, begin

  def _ExtractInputShapes(inputs):
    """Extract the shapes of a set of input tensors."""
    if context.executing_eagerly():
      return array_ops.shape_n(inputs)
    sizes = []
    fully_known = True
    for x in inputs:
      input_shape = array_ops.shape(x)
      if not isinstance(input_shape,
                        ops.Tensor) or input_shape.op.type != "Const":
        fully_known = False
        break
      sizes.append(input_shape)

    if fully_known:
      return sizes
    else:
      return array_ops.shape_n(inputs)

  # Degenerate concatenation, just return grad.
  if len(op.inputs) == 2:
    return grad + [None] if end_value_index <= dim_index else [None] + grad

  concat_dim = op.inputs[dim_index]
  input_values = op.inputs[start_value_index:end_value_index]

  out_grads = []
  if isinstance(grad, ops.Tensor):
    if context.executing_eagerly():
      # Using mod here for convenience since concat_dim is already verified
      # in concat implementation to be within the allowed [-rank, rank) range.
      non_neg_concat_dim = (
          concat_dim._numpy().item(0) % input_values[0]._rank())  # pylint: disable=protected-access
      # All inputs are guaranteed to be EagerTensors in eager mode
      sizes = pywrap_tensorflow.TFE_Py_TensorShapeSlice(input_values,
                                                        non_neg_concat_dim)
      out_grads = array_ops.split(grad, sizes, non_neg_concat_dim)
    else:
      if constant_op.is_constant(concat_dim):
        # If concat_dim is a constant defined in a different context,
        # then we duplicate it in the current context to avoid passing it
        # through an Enter node.
        # This is a small optimization in general, but it is required when
        # compiling with XLA, as XLA needs the concat input to be folded into a
        # constant.
        grad_context = control_flow_util.GetOutputContext(grad.op)
        dim_context = control_flow_util.GetOutputContext(concat_dim.op)
        if dim_context != grad_context:
          value = tensor_util.constant_value(concat_dim)
          concat_dim = constant_op.constant(value=value, dtype=concat_dim.dtype)

      # Using mod here for convenience since concat_dim is already verified
      # in concat implementation to be within the allowed [-rank, rank) range.
      non_neg_concat_dim = concat_dim % array_ops.rank(input_values[0])

      # Get the inputs' tensor shapes
      sizes = _ExtractInputShapes(input_values)
      # The magic number of 16 was found through benchmarking a range of sizes
      # on CPUs and a Maxwell TitanX.  A speedup was seen in a large majority of
      # cases when switching implementations at N=16, but it is possible that
      # there will be a small number of performance regressions.
      if len(sizes) > 16:
        # extract the size of each input along the concat dimension
        sizes = array_ops.squeeze(
            array_ops.slice(
                array_ops.stack(sizes, axis=1), [non_neg_concat_dim, 0],
                [1, -1]))
        out_grads = array_ops.split(grad, sizes, non_neg_concat_dim)
      else:
        offset = gen_array_ops.concat_offset(non_neg_concat_dim, sizes)
        for (begin, size) in zip(offset, sizes):
          out_grads.append(array_ops.slice(grad, begin, size))
  elif isinstance(grad, ops.IndexedSlices):
    # Using mod here for convenience since concat_dim is already verified
    # in concat implementation to be within the allowed [-rank, rank) range.
    non_neg_concat_dim = concat_dim % array_ops.rank(input_values[0])
    concat_dim_static = tensor_util.constant_value(concat_dim)
    if concat_dim_static is None:
      raise ValueError("Can only compute IndexedSlices gradient with "
                       "statically-known concat_dim")
    if concat_dim_static < 0:
      rank = tensor_util.constant_value(array_ops.rank(input_values[0]))
      if rank is None:
        raise ValueError("Can only compute IndexedSlices gradient with "
                         "negative concat_dim when first value rank is "
                         "statically-known.")
      concat_dim_static %= rank
    # Get the inputs' tensor shapes
    sizes = [array_ops.shape(x) for x in input_values]
    if concat_dim_static > 0:
      # IndexedSlices, non_neg_concat_dim > 0. Each input gets IndexedSlices
      # gradients with all the indices, but with grad.values sliced accordingly.
      # This is like the Tensor case, except shape(grad.values)[0] is not equal
      # to shape(sizes[i])[0], since only a subset of the dim-0 values are
      # stored.
      mask, begin = _CreateDenseMaskAndBegin(sizes, non_neg_concat_dim)
      for size in sizes:
        new_values = array_ops.slice(
            grad.values, begin,
            array_ops.concat([[-1], array_ops.slice(size, [1], [-1])], 0))
        out_grads.append(ops.IndexedSlices(new_values, grad.indices, size))
        # Lint complains begin = begin + ...
        begin = math_ops.add(begin, size * mask)
    else:
      # IndexedSlices, concat_dim == 0. Each input gets IndexedSlices gradients
      # only for the relevant indices.
      start = constant_op.constant(0, dtype=grad.indices.dtype)
      for size in sizes:
        size_concat_dim = array_ops.gather(size, non_neg_concat_dim)
        if size_concat_dim.dtype != grad.indices.dtype:
          size_concat_dim = math_ops.cast(
              size_concat_dim, dtype=grad.indices.dtype)
        end = start + size_concat_dim
        # Compute the 1-D Tensor of indices relevant for this input.
        indices_to_select = array_ops.squeeze(
            array_ops.where(
                math_ops.logical_and(grad.indices >= start,
                                     grad.indices < end)),
            axis=[1])
        new_indices = array_ops.gather(grad.indices, indices_to_select) - start
        new_values = array_ops.gather(grad.values, indices_to_select)
        out_grads.append(ops.IndexedSlices(new_values, new_indices, size))
        start = end
  else:
    raise TypeError("Expected Tensor or IndexedSlices, got %s" % type(grad))

  return (out_grads + [None]
          if end_value_index <= dim_index else [None] + out_grads)


@ops.RegisterGradient("Concat")
def _ConcatGrad(op, grad):
  return _ConcatGradHelper(
      op,
      grad,
      start_value_index=1,
      end_value_index=len(op.inputs),
      dim_index=0)


@ops.RegisterGradient("ConcatV2")
def _ConcatGradV2(op, grad):
  return _ConcatGradHelper(
      op, grad, start_value_index=0, end_value_index=-1, dim_index=-1)


ops.NotDifferentiable("ConcatOffset")


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

  shape = array_ops.stack([input_rank, 1])
  before_pad = array_ops.reshape(begin_vec, shape)
  after_pad = array_ops.reshape(
      array_ops.shape(input_vec) - slice_size - begin_vec, shape)
  paddings = array_ops.concat([before_pad, after_pad], 1)
  return array_ops.pad(grad, paddings), None, None


@ops.RegisterGradient("StridedSlice")
def _StridedSliceGrad(op, grad):
  """Gradient for StridedSlice op."""
  begin = op.inputs[1]
  end = op.inputs[2]
  strides = op.inputs[3]
  # StridedSliceGrad requires `x`, `begin`, `end` and `strides` to be of the
  # same dtype so we build a shape of the same type as other args.
  # Note that the choice of `begin` for specifying `out_type` is arbitrary.
  # We could choose any of {begin|end|strides}.dtype since they are required to
  # be the same.
  x = array_ops.shape(op.inputs[0], out_type=begin.dtype)

  return array_ops.strided_slice_grad(
      x,
      begin,
      end,
      strides,
      grad,
      begin_mask=op.get_attr("begin_mask"),
      end_mask=op.get_attr("end_mask"),
      ellipsis_mask=op.get_attr("ellipsis_mask"),
      new_axis_mask=op.get_attr("new_axis_mask"),
      shrink_axis_mask=op.get_attr("shrink_axis_mask")), None, None, None


@ops.RegisterGradient("StridedSliceGrad")
def _StridedSliceGradGrad(op, grad):
  """Gradient for StridedSliceGrad op."""
  begin = op.inputs[1]
  end = op.inputs[2]
  strides = op.inputs[3]

  return None, None, None, None, array_ops.strided_slice(
      grad,
      begin,
      end,
      strides,
      begin_mask=op.get_attr("begin_mask"),
      end_mask=op.get_attr("end_mask"),
      ellipsis_mask=op.get_attr("ellipsis_mask"),
      new_axis_mask=op.get_attr("new_axis_mask"),
      shrink_axis_mask=op.get_attr("shrink_axis_mask"))


@ops.RegisterGradient("Split")
def _SplitGrad(op, *grads):
  return None, array_ops.concat(list(grads), op.inputs[0])


@ops.RegisterGradient("SplitV")
def _SplitVGrad(op, *grads):
  returnval = array_ops.concat(list(grads), op.inputs[2])
  returnval = [returnval] + [
      None,
  ] * (
      len(op.inputs) - 1)
  return returnval


ops.NotDifferentiable("Const")


@ops.RegisterGradient("Diag")
def _DiagGrad(_, grad):
  return array_ops.diag_part(grad)


@ops.RegisterGradient("DiagPart")
def _DiagPartGrad(_, grad):
  return array_ops.diag(grad)


@ops.RegisterGradient("MatrixDiag")
def _MatrixDiagGrad(_, grad):
  return array_ops.matrix_diag_part(grad)


@ops.RegisterGradient("MatrixDiagPart")
def _MatrixDiagPartGrad(op, grad):
  matrix_shape = op.inputs[0].get_shape()[-2:]
  if matrix_shape.is_fully_defined() and matrix_shape[0] == matrix_shape[1]:
    return array_ops.matrix_diag(grad)
  else:
    return array_ops.matrix_set_diag(array_ops.zeros_like(op.inputs[0]), grad)


@ops.RegisterGradient("MatrixSetDiag")
def _MatrixSetDiagGrad(op, grad):
  """Gradient for MatrixSetDiag."""
  input_shape = op.inputs[0].get_shape().merge_with(grad.get_shape())
  diag_shape = op.inputs[1].get_shape()
  batch_shape = input_shape[:-2].merge_with(diag_shape[:-1])
  matrix_shape = input_shape[-2:]
  if batch_shape.is_fully_defined() and matrix_shape.is_fully_defined():
    diag_shape = batch_shape.as_list() + [min(matrix_shape.as_list())]
  else:
    with ops.colocate_with(grad):
      grad_shape = array_ops.shape(grad)
      grad_rank = array_ops.rank(grad)
      batch_shape = array_ops.slice(grad_shape, [0], [grad_rank - 2])
      matrix_shape = array_ops.slice(grad_shape, [grad_rank - 2], [2])
      min_dim = math_ops.reduce_min(matrix_shape)
      diag_shape = array_ops.concat([batch_shape, [min_dim]], 0)
  grad_input = array_ops.matrix_set_diag(grad,
                                         array_ops.zeros(
                                             diag_shape, dtype=grad.dtype))
  grad_diag = array_ops.matrix_diag_part(grad)
  return (grad_input, grad_diag)


@ops.RegisterGradient("MatrixBandPart")
def _MatrixBandPartGrad(op, grad):
  num_lower = op.inputs[1]
  num_upper = op.inputs[2]
  return (array_ops.matrix_band_part(grad, num_lower, num_upper), None, None)


# Edit Distance has no gradient (but can be used to eval seq2seq or CTC).
ops.NotDifferentiable("EditDistance")


@ops.RegisterGradient("Fill")
def _FillGrad(_, grad):
  return None, math_ops.reduce_sum(grad)


ops.NotDifferentiable("ZerosLike")
ops.NotDifferentiable("OnesLike")


@ops.RegisterGradient("PreventGradient")
def _PreventGradientGrad(op, _):
  raise LookupError(
      "Gradient explicitly disabled. Reason: %s" % op.get_attr("message"))


@ops.RegisterGradient("Gather")
def _GatherGrad(op, grad):
  """Gradient for Gather op."""
  # params can be large, so colocate the shape calculation with it.
  #
  # params can be very large for sparse model, array_ops.shape raises
  # exception on the Windows platform when any dimension is larger than
  # int32. params_shape is not used in optimizer apply_sparse gradients,
  # so it's fine to convert it back to int32 regardless of truncation.
  params = op.inputs[0]
  with ops.colocate_with(params):
    params_shape = array_ops.shape(params, out_type=ops.dtypes.int64)
    params_shape = math_ops.to_int32(params_shape)

  # Build appropriately shaped IndexedSlices
  indices = op.inputs[1]
  size = array_ops.expand_dims(array_ops.size(indices), 0)
  values_shape = array_ops.concat([size, params_shape[1:]], 0)
  values = array_ops.reshape(grad, values_shape)
  indices = array_ops.reshape(indices, size)
  return [ops.IndexedSlices(values, indices, params_shape), None]


@ops.RegisterGradient("GatherV2")
def _GatherV2Grad(op, grad):
  """Gradient for GatherV2 op."""
  # params can be large, so colocate the shape calculation with it.
  #
  # params can be very large for sparse model, array_ops.shape raises
  # exception on the Windows platform when any dimension is larger than
  # int32. params_shape is not used in optimizer apply_sparse gradients,
  # so it's fine to convert it back to int32 regardless of truncation.
  params = op.inputs[0]
  with ops.colocate_with(params):
    params_shape = array_ops.shape(params, out_type=ops.dtypes.int64)
    params_shape = math_ops.to_int32(params_shape)

  indices = op.inputs[1]
  indices_size = array_ops.expand_dims(array_ops.size(indices), 0)
  axis = op.inputs[2]
  axis_static = tensor_util.constant_value(axis)

  # For axis 0 gathers, build an appropriately shaped IndexedSlices.
  if axis_static == 0:
    if context.executing_eagerly():
      params_tail_shape = params_shape.cpu()[1:]
    else:
      params_tail_shape = params_shape[1:]
    values_shape = array_ops.concat([indices_size, params_tail_shape], 0)
    values = array_ops.reshape(grad, values_shape)
    indices = array_ops.reshape(indices, indices_size)
    return [ops.IndexedSlices(values, indices, params_shape), None, None]

  outer_shape = params_shape[:axis]
  outer_dims = array_ops.size(outer_shape)
  inner_shape = params_shape[axis:][1:]
  inner_dims = array_ops.size(inner_shape)

  outer_axes_indices = math_ops.range(outer_dims)
  inner_axes_indices = math_ops.range(outer_dims + 1,
                                      outer_dims + 1 + inner_dims)

  values_shape = array_ops.concat([outer_shape, indices_size, inner_shape], 0)
  values = array_ops.reshape(grad, values_shape)
  indices = array_ops.reshape(indices, indices_size)

  # We need to sum up every slice `values[..., i, ....]` corresponding to
  # `params[..., indices[i], ...]`. Since `unsorted_segment_sum` does not
  # support an axis parameter, we transpose the gather dimension to the front,
  # then use `unsorted_segment_sum` to build a
  # [gather_axis, outer_axes, inner_axes] tensor with all the gradients
  # affecting each index in `gather_axis` summed up.
  transpose_dims = array_ops.concat(
      [[outer_dims], outer_axes_indices, inner_axes_indices], 0)
  values_transpose = array_ops.transpose(values, transpose_dims)
  num_segments = params_shape[axis]

  params_grad = math_ops.unsorted_segment_sum(values_transpose, indices,
                                              num_segments)

  # Inverts the above transpose by moving dimension 0 back to its original
  # position.
  invert_transpose_dims = array_ops.concat(
      [outer_axes_indices + 1, [0], inner_axes_indices], 0)
  params_grad = array_ops.transpose(params_grad, invert_transpose_dims)
  return [params_grad, None, None]


@ops.RegisterGradient("GatherNd")
def _GatherNdGrad(op, grad):
  ref = op.inputs[0]
  indices = op.inputs[1]
  ref_shape = array_ops.shape(ref, out_type=indices.dtype)
  if indices.shape.ndims == 2 and indices.shape.dims[-1].value == 1:
    ref_grad = ops.IndexedSlices(grad, array_ops.squeeze(indices, axis=-1),
                                 ref_shape)
  else:
    ref_grad = array_ops.scatter_nd(indices, grad, ref_shape)
  return [ref_grad, None]


@ops.RegisterGradient("CheckNumerics")
def _CheckNumericsGrad(op, grad):
  """Gradient for check_numerics op."""
  return array_ops.check_numerics(
      grad,
      "Not a number (NaN) or infinity (Inf) values detected in gradient. %s" %
      op.get_attr("message"))


@ops.RegisterGradient("PlaceholderWithDefault")
@ops.RegisterGradient("Identity")
def _IdGrad(_, grad):
  return grad


@ops.RegisterGradient("RefIdentity")
def _RefIdGrad(_, grad):
  return grad


@ops.RegisterGradient("IdentityN")
def _IdNGrad(_, *grad):
  return grad


ops.NotDifferentiable("StopGradient")


@ops.RegisterGradient("Reshape")
def _ReshapeGrad(op, grad):
  return [array_ops.reshape(grad, array_ops.shape(op.inputs[0])), None]


ops.NotDifferentiable("InvertPermutation")


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


@ops.RegisterGradient("ConjugateTranspose")
def _ConjugateTransposeGrad(op, grad):
  """Returns conj(unshuffle(grad))."""
  p = op.inputs[1]
  return [
      array_ops.transpose(
          grad, array_ops.invert_permutation(p), conjugate=True), None
  ]


ops.NotDifferentiable("Shape")

ops.NotDifferentiable("ShapeN")

ops.NotDifferentiable("Rank")

ops.NotDifferentiable("Size")


@ops.RegisterGradient("Tile")
def _TileGrad(op, grad):
  """Sum reduces grad along the tiled dimensions."""
  input_shape = array_ops.shape(op.inputs[0])
  # We interleave multiples and input_shape to get split_shape,
  # reshape grad to split_shape, and reduce along all even
  # dimensions (the tiled dimensions) to get the result
  # with shape input_shape.  For example
  #   input_shape = [20, 30, 40]
  #   multiples = [2, 3, 4]
  #   split_shape = [2, 20, 3, 30, 4, 40]
  #   axes = [0, 2, 4]
  split_shape = array_ops.reshape(
      array_ops.transpose(array_ops.stack([op.inputs[1], input_shape])), [-1])
  axes = math_ops.range(0, array_ops.size(split_shape), 2)
  # Sum reduces grad along the first dimension for IndexedSlices
  if isinstance(grad, ops.IndexedSlices):
    grad = math_ops.unsorted_segment_sum(
        grad.values,
        math_ops.mod(grad.indices, input_shape[0]),
        input_shape[0])
    split_shape = array_ops.concat([[1], split_shape[1:]], axis=0)
  input_grad = math_ops.reduce_sum(array_ops.reshape(grad, split_shape), axes)
  # Fix shape inference
  if not context.executing_eagerly():
    input_grad.set_shape(op.inputs[0].get_shape())
  return [input_grad, None]


ops.NotDifferentiable("BroadcastGradientArgs")


def _PadGrad(op, grad):
  """Gradient for Pad."""
  # Pad introduces values around the original tensor, so the gradient function
  # slices the original shape out of the gradient."""
  x = op.inputs[0]
  a = op.inputs[1]  # [Rank(x), 2]
  # Takes a slice of a. The 1st column. [Rank(x), 1].
  pad_before = array_ops.slice(a, [0, 0],
                               array_ops.stack([array_ops.rank(x), 1]))
  # Make it a 1-D tensor.
  begin = array_ops.reshape(pad_before, [-1])
  sizes = array_ops.shape(x)
  x_grad = array_ops.slice(grad, begin, sizes)
  if len(op.inputs) == 3:
    return x_grad, None, None
  else:
    return x_grad, None


ops.RegisterGradient("Pad")(_PadGrad)
ops.RegisterGradient("PadV2")(_PadGrad)


# ReverseSequence is just a permutation.  The gradient permutes back.
@ops.RegisterGradient("ReverseSequence")
def _ReverseSequenceGrad(op, grad):
  seq_lengths = op.inputs[1]
  return [
      array_ops.reverse_sequence(
          grad,
          batch_axis=op.get_attr("batch_dim"),
          seq_axis=op.get_attr("seq_dim"),
          seq_lengths=seq_lengths), None
  ]


@ops.RegisterGradient("Reverse")
def _ReverseGrad(op, grad):
  reverse_dims = op.inputs[1]
  return gen_array_ops.reverse(grad, reverse_dims), None


@ops.RegisterGradient("ReverseV2")
def _ReverseV2Grad(op, grad):
  axis = op.inputs[1]
  return array_ops.reverse_v2(grad, axis), None


@ops.RegisterGradient("SpaceToBatch")
def _SpaceToBatchGrad(op, grad):
  # Its gradient is the opposite op: BatchToSpace.
  block_size = op.get_attr("block_size")
  return [
      array_ops.batch_to_space(grad, op.inputs[1], block_size=block_size), None
  ]


@ops.RegisterGradient("SpaceToBatchND")
def _SpaceToBatchNDGrad(op, grad):
  # Its gradient is the opposite op: BatchToSpaceND.
  return [
      array_ops.batch_to_space_nd(grad, op.inputs[1], op.inputs[2]), None, None
  ]


@ops.RegisterGradient("BatchToSpace")
def _BatchToSpaceGrad(op, grad):
  # Its gradient is the opposite op: SpaceToBatch.
  block_size = op.get_attr("block_size")
  return [
      array_ops.space_to_batch(grad, op.inputs[1], block_size=block_size), None
  ]


@ops.RegisterGradient("BatchToSpaceND")
def _BatchToSpaceNDGrad(op, grad):
  # Its gradient is the opposite op: SpaceToBatchND.
  return [
      array_ops.space_to_batch_nd(grad, op.inputs[1], op.inputs[2]), None, None
  ]


@ops.RegisterGradient("SpaceToDepth")
def _SpaceToDepthGrad(op, grad):
  # Its gradient is the opposite op: DepthToSpace.
  block_size = op.get_attr("block_size")
  data_format = op.get_attr("data_format")
  if data_format == "NCHW_VECT_C":
    raise ValueError("Cannot compute SpaceToDepth gradient with NCHW_VECT_C. "
                     "NCHW_VECT_C requires qint8 data type.")
  return array_ops.depth_to_space(grad, block_size, data_format=data_format)


@ops.RegisterGradient("DepthToSpace")
def _DepthToSpaceGrad(op, grad):
  # Its gradient is the opposite op: SpaceToDepth.
  block_size = op.get_attr("block_size")
  data_format = op.get_attr("data_format")
  if data_format == "NCHW_VECT_C":
    raise ValueError("Cannot compute DepthToSpace gradient with NCHW_VECT_C. "
                     "NCHW_VECT_C requires qint8 data type.")
  return array_ops.space_to_depth(grad, block_size, data_format=data_format)


ops.NotDifferentiable("OneHot")


@ops.RegisterGradient("MirrorPad")
def _MirrorPadGrad(op, grad):
  mode = op.get_attr("mode")
  return [gen_array_ops.mirror_pad_grad(grad, op.inputs[1], mode=mode), None]


@ops.RegisterGradient("MirrorPadGrad")
def _MirrorPadGradGrad(op, grad):
  mode = op.get_attr("mode")
  return [gen_array_ops.mirror_pad(grad, op.inputs[1], mode=mode), None]


@ops.RegisterGradient("QuantizeAndDequantize")
def _QuantizeAndDequantizeGrad(_, grad):
  return grad


@ops.RegisterGradient("QuantizeAndDequantizeV2")
def _QuantizeAndDequantizeV2Grad(_, grad):
  return [grad, None, None]


@ops.RegisterGradient("QuantizeAndDequantizeV3")
def _QuantizeAndDequantizeV3Grad(_, grad):
  # Only propagate the gradient for the unquantized input.
  return [grad, None, None, None]


@ops.RegisterGradient("ExtractImagePatches")
def _ExtractImagePatchesGrad(op, grad):
  batch_size, rows_in, cols_in, channels = [
      dim.value for dim in op.inputs[0].shape.dims
  ]
  input_bhwc = array_ops.shape(op.inputs[0])
  batch_size = input_bhwc[0]
  channels = input_bhwc[3]

  # Create indices matrix for input tensor.
  # Note that 0 is preserved for padding location,
  # so indices for input start from 1 to 1 + rows_in * cols_in.
  input_indices_num = 1 + rows_in * cols_in
  input_idx = array_ops.reshape(math_ops.range(1, input_indices_num,
                                               dtype=ops.dtypes.int64),
                                (1, rows_in, cols_in, 1))
  input_idx_patched = gen_array_ops.extract_image_patches(
      input_idx,
      op.get_attr("ksizes"),
      op.get_attr("strides"),
      op.get_attr("rates"),
      op.get_attr("padding"))

  # Create indices matrix for output tensor.
  _, rows_out, cols_out, _ = [dim.value for dim in op.outputs[0].shape.dims]
  _, ksize_r, ksize_c, _ = op.get_attr("ksizes")
  # Indices for output start from 0.
  output_indices_num = rows_out * cols_out * ksize_r * ksize_c
  output_idx = array_ops.reshape(math_ops.range(output_indices_num,
                                                dtype=ops.dtypes.int64),
                                 (1, rows_out, cols_out, ksize_r * ksize_c))

  # Construct mapping table for indices: (input -> output).
  idx_matrix = array_ops.concat(
      [array_ops.expand_dims(input_idx_patched, axis=-1),
       array_ops.expand_dims(output_idx, axis=-1)],
      axis=-1)
  idx_map = array_ops.reshape(idx_matrix, (-1, 2))

  sp_shape = (input_indices_num, output_indices_num)
  sp_mat_full = sparse_tensor.SparseTensor(
      idx_map,
      array_ops.ones([output_indices_num], dtype=grad.dtype),
      sp_shape)
  # Remove all padding locations [0, :].
  sp_mat = sparse_ops.sparse_slice(sp_mat_full,
                                   (1, 0),
                                   (input_indices_num - 1, output_indices_num))

  grad_expanded = array_ops.transpose(
      array_ops.reshape(
          grad, (batch_size, rows_out, cols_out, ksize_r, ksize_c, channels)),
      (1, 2, 3, 4, 0, 5))
  grad_flat = array_ops.reshape(grad_expanded, (-1, batch_size * channels))

  jac = sparse_ops.sparse_tensor_dense_matmul(sp_mat, grad_flat)

  grad_out = array_ops.reshape(jac, (rows_in, cols_in, batch_size, channels))
  grad_out = array_ops.transpose(grad_out, (2, 0, 1, 3))

  return [grad_out]


@ops.RegisterGradient("ExtractVolumePatches")
def _ExtractVolumePatchesGrad(op, grad):
  batch_size, planes_in, rows_in, cols_in, channels = [
      dim.value for dim in op.inputs[0].shape.dims
  ]
  input_bphwc = array_ops.shape(op.inputs[0])
  batch_size = input_bphwc[0]
  channels = input_bphwc[4]

  # Create indices matrix for input tensor.
  # Note that 0 is preserved for padding location,
  # so indices for input start from 1 to 1 + rows_in * cols_in.
  input_indices_num = 1 + planes_in * rows_in * cols_in
  input_idx = array_ops.reshape(math_ops.range(1, input_indices_num,
                                               dtype=ops.dtypes.int64),
                                (1, planes_in, rows_in, cols_in, 1))
  input_idx_patched = gen_array_ops.extract_volume_patches(
      input_idx,
      op.get_attr("ksizes"),
      op.get_attr("strides"),
      op.get_attr("padding"))

  # Create indices matrix for output tensor.
  _, planes_out, rows_out, cols_out, _ = [dim.value for dim in
                                          op.outputs[0].shape.dims]
  _, ksize_p, ksize_r, ksize_c, _ = op.get_attr("ksizes")
  # Indices for output start from 0.
  prc_indices_num = planes_out * rows_out * cols_out
  output_indices_num = prc_indices_num * ksize_p * ksize_r * ksize_c
  output_idx = array_ops.reshape(math_ops.range(output_indices_num,
                                                dtype=ops.dtypes.int64),
                                 (1, planes_out, rows_out, cols_out,
                                  ksize_p * ksize_r * ksize_c))

  # Construct mapping table for indices: (input -> output).
  idx_matrix = array_ops.concat(
      [array_ops.expand_dims(input_idx_patched, axis=-1),
       array_ops.expand_dims(output_idx, axis=-1)],
      axis=-1)
  idx_map = array_ops.reshape(idx_matrix, (-1, 2))

  sp_shape = (input_indices_num, output_indices_num)
  sp_mat_full = sparse_tensor.SparseTensor(
      idx_map,
      array_ops.ones([output_indices_num], dtype=grad.dtype),
      sp_shape)
  # Remove all padding locations [0, :].
  sp_mat = sparse_ops.sparse_slice(sp_mat_full,
                                   (1, 0),
                                   (input_indices_num - 1, output_indices_num))

  grad_expanded = array_ops.transpose(
      array_ops.reshape(
          grad, (batch_size, planes_out, rows_out, cols_out, ksize_p, ksize_r,
                 ksize_c, channels)),
      (1, 2, 3, 4, 5, 6, 0, 7))
  grad_flat = array_ops.reshape(grad_expanded, (-1, batch_size * channels))

  jac = sparse_ops.sparse_tensor_dense_matmul(sp_mat, grad_flat)

  grad_out = array_ops.reshape(jac, (planes_in, rows_in, cols_in, batch_size,
                                     channels))
  grad_out = array_ops.transpose(grad_out, (3, 0, 1, 2, 4))

  return [grad_out]


@ops.RegisterGradient("ScatterNd")
def _ScatterNdGrad(op, grad):
  indices = op.inputs[0]
  updates_grad = array_ops.gather_nd(grad, indices)
  return [None, updates_grad, None]


@ops.RegisterGradient("TensorScatterUpdate")
def _TensorScatterUpdateGrad(op, grad):
  indices = op.inputs[1]
  updates_grad = array_ops.gather_nd(grad, indices)
  tensor_grad = array_ops.tensor_scatter_update(
      array_ops.identity(grad), indices,
      array_ops.zeros_like(op.inputs[2], dtype=grad.dtype))
  return [tensor_grad, None, updates_grad]


@ops.RegisterGradient("TensorScatterAdd")
def _TensorScatterAddGrad(op, grad):
  indices = op.inputs[1]
  updates_grad = array_ops.gather_nd(grad, indices)
  tensor_grad = array_ops.identity(grad)
  return [tensor_grad, None, updates_grad]


@ops.RegisterGradient("TensorScatterSub")
def _TensorScatterSubGrad(op, grad):
  indices = op.inputs[1]
  updates_grad = array_ops.gather_nd(grad, indices)
  tensor_grad = array_ops.identity(grad)
  return [tensor_grad, None, -updates_grad]


@ops.RegisterGradient("ScatterNdNonAliasingAdd")
def _ScatterNdNonAliasingAddGrad(op, grad):
  indices = op.inputs[1]
  updates_grad = array_ops.gather_nd(grad, indices)
  return [grad, None, updates_grad]


@ops.RegisterGradient("BroadcastTo")
def _BroadcastToGrad(op, grad):
  input_value = op.inputs[0]
  broadcast_shape = op.inputs[1]
  # Assign ids for each position in input_value.
  input_value_shape = array_ops.shape(input_value)
  input_value_size = array_ops.size(input_value)
  ids = array_ops.reshape(math_ops.range(input_value_size), input_value_shape)
  broadcast_ids = array_ops.broadcast_to(ids, broadcast_shape)
  # Group by ids and sum its gradients.
  grad_flatten = array_ops.reshape(grad, [-1])
  broadcast_ids_flatten = array_ops.reshape(broadcast_ids, [-1])
  updates_grad_flatten = math_ops.unsorted_segment_sum(grad_flatten,
                                                       broadcast_ids_flatten,
                                                       input_value_size)
  updates_grad = array_ops.reshape(updates_grad_flatten, input_value_shape)
  return [updates_grad, None]
